"""
Module for probabilistic memory models
@author Siddharth Reddy <sgr45@cornell.edu>
"""

from __future__ import division

import sys

from sklearn.linear_model import LogisticRegression
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

import matplotlib as mpl
mpl.rc('savefig', dpi=300)
mpl.rc('text', usetex=True)

import autograd.numpy as np
from autograd import grad

from lentil import datatools
from lentil import models


class EFCModel(models.SkillModel):
    """
    Class for memory models that follow the exponential forgetting curve
    """

    def __init__(self, history, strength_model=None, using_delay=True, 
            using_global_difficulty=True, debug_mode_on=False):
        """
        Initialize memory model object

        :param pd.DataFrame history: Interaction log data. Must contain the following columns.

                tlast
                timestamp
                module_id
                outcome

        :param str|None strength_model: Corresponds to a column in the history dataframe 
            (e.g., 'nreps' or 'deck') or simply None if memory strength is always 1.

        :param bool using_delay: True if the delay term is included in the recall probability, 
            False otherwise.

        :param bool using_global_difficulty: True if item difficulty is a global constant, 
            False if difficulty is an item-specific parameter.

        :param bool debug_mode_on: True if maximum-likelihood estimation should log progress 
            and plot learned difficulty parameters, False otherwise.
        """

        self.history = history[history['module_type']==datatools.AssessmentInteraction.MODULETYPE]
        self.strength_model = strength_model
        self.using_delay = using_delay
        self.using_global_difficulty = using_global_difficulty
        self.debug_mode_on = debug_mode_on

        self.idx_of_module_id = {x: i for i, x in enumerate(self.history['module_id'].unique())}
        self.difficulty = None

    def extract_features(self, df):
        """
        Get delays, memory strengths, and module indices for a set of interactions

        :param pd.DataFrame df: Interaction log data
        :rtype: (np.array,np.array,np.array)
        :return: A tuple of (delays, memory strengths, module indices)
        """

        if self.using_delay:
            timestamps = np.array(df['timestamp'].values)
            previous_review_timestamps = np.array(df['tlast'].values)
            delays = 1 + (timestamps - previous_review_timestamps) / 86400
        else:
            delays = 1
        strengths = 1 if self.strength_model is None else np.array(df[self.strength_model].values)
        module_idxes = 0 if self.using_global_difficulty else np.array(
                df['module_id'].map(self.idx_of_module_id).values)

        return delays, strengths, module_idxes

    def fit(self, learning_rate=0.5, ftol=1e-6, max_iter=1000):
        """
        Learn item difficulty parameter(s) using maximum-likelihood estimation

        Uses batch gradient descent with a fixed learning rate and a fixed threshold on 
            the relative difference between consecutive loss function evaluations 
            as the stopping condition

        Uses the exponentiation trick to enforce the non-negativity constraint on item difficulty

        :param float learning_rate: Fixed learning rate for batch gradient descent
        :param float ftol: If the relative difference between consecutive loss function 
            evaluations falls below this threshold, then gradient descent has 'converged'

        :param int max_iter: If the stopping condition hasn't been met after this many iterations, 
            then stop gradient descent
        """

        delays, strengths, module_idxes = self.extract_features(self.history)
        outcomes = np.array(self.history['outcome'].apply(lambda x: 1 if x else 0).values)

        eps = 1e-9 # smoothing parameter for likelihoods
        def loss(difficulty):
            """
            Compute the average negative log-likelihood of the data given the item difficulty

            :param np.array difficulty: A global item difficulty, 
                or an array of item-specific difficulties

            :rtype: float
            :return: Average negative log-likelihood
            """

            pass_likelihoods = np.exp(-np.exp(-difficulty[module_idxes])*delays/strengths)
            return -np.mean(outcomes*np.log(pass_likelihoods+eps) + (1-outcomes)*np.log(
                1-pass_likelihoods+eps))

        gradient_fun = grad(loss) # differentiate the loss function

        difficulty = np.random.random(1 if self.using_global_difficulty else len(
            self.idx_of_module_id))

        losses = []
        for _ in xrange(max_iter):
            d_difficulty = gradient_fun(difficulty) # evaluate gradient at current difficulty value
            difficulty -= d_difficulty * learning_rate # gradient descent update
            losses.append(loss(difficulty)) # evaluate loss function at current difficulty value
            if len(losses) > 1 and (losses[-2] - losses[-1]) / losses[-2] < ftol: # stopping cond.
                break

        if self.debug_mode_on: # visual check for convergence
            plt.xlabel('Iteration')
            plt.ylabel('Average negative log-likelihood')
            plt.plot(losses)
            plt.show()

        self.difficulty = np.exp(-difficulty) # second part of non-negativity trick

    def assessment_pass_likelihoods(self, df):
        """
        Compute recall likelihoods given the learned item difficulty

        :param pd.DataFrame df: Interaction log data
        :rtype: np.array
        :return: An array of recall likelihoods
        """

        delays, strengths, module_idxes = self.extract_features(df)
        return np.exp(-self.difficulty[module_idxes]*delays/strengths)


class LogisticRegressionModel(models.SkillModel):
    """
    Class for memory model that predicts recall using basic statistics 
    of previous review intervals and outcomes for a user-item pair
    """

    def __init__(self, history, C=1.0, name_of_user_id='user_id'):
        """
        Initialize memory model object

        :param pd.DataFrame history: Interaction log data
        :param float C: Regularization constant
        :param str name_of_user_id:
        """

        self.history = history[history['module_type']==datatools.AssessmentInteraction.MODULETYPE]
        self.C = C
        self.name_of_user_id = name_of_user_id
        self.clf = None
        self.data = None

    def feature_vec(self, x, max_t=sys.maxint):
        intervals, outcomes, ts = x

        i = 0
        while i<len(ts) and ts[i] < max_t:
            i += 1
        outcomes = outcomes[:i]
        intervals = intervals[:max(0, i-1)]

        if len(intervals)==0:
            return [0] * 16

        intervals = np.log(np.array(intervals)+1)
        interval_fvec = [len(intervals), intervals[0], intervals[-1], 
                np.mean(intervals), min(intervals), max(intervals), 
                max(intervals)-min(intervals), sorted(intervals)[len(intervals) // 2]]
        outcome_fvec = [len(outcomes), outcomes[0], outcomes[-1], 
                np.mean(outcomes), min(outcomes), max(outcomes), 
                max(outcomes)-min(outcomes), sorted(outcomes)[len(outcomes) // 2]]
        return np.array(interval_fvec + outcome_fvec)

    def fit(self):
        self.data = {}
        for umid, g in self.history.groupby([self.name_of_user_id, 'module_id']):
            ts = g['timestamp'].values
            intervals = [y-x for x, y in zip(ts[:-1], ts[1:])]
            outcomes = [1 if x else 0 for x in g['outcome']]
            self.data[umid] = (intervals, outcomes, ts)

        X_train = np.array([self.feature_vec((intervals[:i+1], outcomes[:i+1], ts[:i+1])) \
                for intervals, outcomes, ts in self.data.itervalues() \
                for i in xrange(len(intervals))])
        Y_train = np.array([x for intervals, outcomes, ts in self.data.itervalues() \
                for x in outcomes[1:]])

        self.clf = LogisticRegression(C=self.C)
        self.clf.fit(X_train, Y_train)

    def assessment_pass_likelihoods(self, df):
        X_val = np.array([self.feature_vec(self.data[(x[self.name_of_user_id], 
            x['module_id'])], max_t=x['timestamp']) for _, x in df.iterrows()])
        return self.clf.predict_proba(X_val)[:,1]

