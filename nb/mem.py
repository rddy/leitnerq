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
    Class for memory models that predict recall likelihood using the exponential forgetting curve
    """

    def __init__(self, history, strength_model=None, using_delay=True, 
            using_global_difficulty=True, debug_mode_on=False):
        """
        Initialize memory model object

        :param pd.DataFrame history: Interaction log data. Must contain the 'tlast' column,
            in addition to the other columns that belong to the dataframe in a
            lentil.datatools.InteractionHistory object. If strength_model is not None, then
            the history should also contain a column named by the strength_model (e.g., 'nreps' or
            'deck')

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
            difficulty -= gradient_fun(difficulty) * learning_rate # gradient descent update
            losses.append(loss(difficulty)) # evaluate loss function at current difficulty value
            if len(losses) > 1 and (losses[-2] - losses[-1]) / losses[-2] < ftol: # stopping cond.
                break

        self.difficulty = np.exp(-difficulty) # second part of non-negativity trick
        
        if self.debug_mode_on: 
            # visual check for convergence
            plt.xlabel('Iteration')
            plt.ylabel('Average negative log-likelihood')
            plt.plot(losses)
            plt.show()

            if not self.using_global_difficulty:
                # check distribution of learned difficulties
                plt.xlabel(r'Item Difficulty $\theta_i$')
                plt.ylabel('Frequency (Number of Items)')
                plt.hist(self.difficulty)
                plt.show()


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
    Class for a memory model that predicts recall likelihood using basic statistics 
    of previous review intervals and outcomes for a user-item pair
    """

    def __init__(self, history, name_of_user_id='user_id'):
        """
        Initialize memory model object

        :param pd.DataFrame history: Interaction log data. Must contain the 'tlast' column,
            in addition to the other columns that belong to the dataframe in a
            lentil.datatools.InteractionHistory object. If strength_model is not None, then
            the history should also contain a column named by the strength_model (e.g., 'nreps' or
            'deck'). Rows should be sorted in increasing order of timestamp.

        :param str name_of_user_id: Name of column in history that stores user IDs (useful for
            distinguishing between user IDs and user-item pair IDs)
        """

        self.history = history[history['module_type']==datatools.AssessmentInteraction.MODULETYPE]
        self.name_of_user_id = name_of_user_id

        self.clf = None
        self.data = None

    def extract_features(self, review_history, max_time=None):
        """
        Map a sequence of review intervals and outcomes to a fixed-length feature set

        :param (np.array,np.array,np.array) review_history: A tuple of 
            (intervals, outcomes, timestamps) where intervals are the milliseconds elapsed between 
            consecutive reviews, outcomes are binary and timestamps are unix epochs. Note that 
            there is one fewer element in the intervals array than in the outcomes and timestamps.

        :param int max_time: Intervals that occur after this time should not be used to 
            construct the feature set. Outcomes that occur at or after this time should not be
            used in the feature set either.

        :rtype: np.array
        :return: A feature vector for the review history containing the length, first, last,
            mean, min, max, range, and median (in that order) of the log-intervals, concatenated 
            with the length, first, last, mean, min, max, range, and median (in that order) of the 
            outcomes.
        """

        intervals, outcomes, timestamps = review_history

        if max_time is not None:
            # truncate the sequences 
            i = 1
            while i < len(timestamps) and timestamps[i] <= max_time:
                i += 1
            outcomes = outcomes[:i-1]
            intervals = intervals[:i-1]

        if len(intervals) == 0:
            interval_feature_list = [0] * 8
        else:
            intervals = np.log(np.array(intervals)+1)
            interval_feature_list = [len(intervals), intervals[0], intervals[-1], \
                    np.mean(intervals), min(intervals), max(intervals), \
                    max(intervals)-min(intervals), sorted(intervals)[len(intervals) // 2]]

        if len(outcomes) == 0:
            outcome_feature_list = [0] * 8
        else:
            outcome_feature_list = [len(outcomes), outcomes[0], outcomes[-1], \
                    np.mean(outcomes), min(outcomes), max(outcomes), \
                    max(outcomes)-min(outcomes), sorted(outcomes)[len(outcomes) // 2]]

        return np.array(interval_feature_list + outcome_feature_list)

    def fit(self, C=1.0):
        """
        Estimate the coefficients of a logistic regression model with a bias term and an L2 penalty
        
        :param float C: Regularization constant. Inverse of regularization strength.
        """
        
        self.data = {}
        for user_item_pair_id, group in self.history.groupby([self.name_of_user_id, 'module_id']):
            if len(group) <= 1:
                continue
            timestamps = np.array(group['timestamp'].values)
            intervals = timestamps[1:] - timestamps[:-1]
            outcomes = np.array(group['outcome'].apply(lambda x: 1 if x else 0).values)
            self.data[user_item_pair_id] = (intervals, outcomes, timestamps)

        X_train = np.array([self.extract_features(
            (intervals[:i+1], outcomes[:i+1], timestamps[:i+1])) \
                for intervals, outcomes, timestamps in self.data.itervalues() \
                for i in xrange(len(intervals))])
        Y_train = np.array([x for intervals, outcomes, timestamps in self.data.itervalues() \
                for x in outcomes[1:]])

        self.clf = LogisticRegression(C=C)
        self.clf.fit(X_train, Y_train)

    def assessment_pass_likelihoods(self, df):
        """
        Compute recall likelihoods given the learned coefficients

        :param pd.DataFrame df: Interaction log data
        :rtype: np.array
        :return: An array of recall likelihoods
        """
        
        X = np.array([self.extract_features(self.data[(x[self.name_of_user_id], 
            x['module_id'])], max_time=x['timestamp']) for _, x in df.iterrows()])
        return self.clf.predict_proba(X)[:,1]

