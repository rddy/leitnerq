"""
Module for Leitner Queue Network schedulers
@author Siddharth Reddy <sgr45@cornell.edu>
"""

from __future__ import division

import random
import time
import copy

import numpy as np


class ExhaustedError(Exception):
    def __init__(self):
        pass


class ExtLQNScheduler(object):
    """
    Class for stateful scheduler that uses the Extended Leitner Queue Network to decide
    which item to review at any given time, and updates its state given a stream of review log data
    """

    def __init__(self, init_data, processor_sharing=False):
        """
        Initialize scheduler object

        :param dict[str,object] init_data: A dictionary for general scheduling parameters, i.e.,
            values that don't depend on the user history. These are the expected data types:

            init_data['arrival_time_of_item'] : OrderedDict[object,int] = the unix epoch when each
                item arrives into the system
            init_data['review_rates'] : np.ndarray = the review rate for each system-deck pair 
                (must sum to 1)
            init_data['difficulty_of_item'] : dict[object,float] = decay rates for 
                item recall probabilities
            init_data['difficulty_rate'] : float = rate parameter for exponential distribution 
                of item difficulties

        :param bool processor_sharing: True if we should use the processor sharing
            service discipline, False if we should use first-in-first-out
        """

        self.__dict__.update(copy.deepcopy(init_data))
        self.processor_sharing = processor_sharing
        self.num_systems, self.num_decks = self.review_rates.shape

        # -ln(1-x)/rate is the inverse cdf of the exponential distribution
        boundaries = -np.log(1 - np.arange(0, 1, self.num_systems)) / self.difficulty_rate
        # linear search (no need for a proper binary search, since num_systems is typically small)
        def place_item(difficulty):
            for i, x in enumerate(boundaries[::-1]):
                if difficulty >= x:
                    return self.num_systems - i
        self.system_of_item = {item: place_item(difficulty) \
                for item, difficulty in self.difficulty_of_item.iteritems()}

        self.deck_of_item = {item: (system, 0) for item, system in self.system_of_item.iteritems()}
        self.latest_timestamp_of_item = copy.copy(self.arrival_time_of_item)
        self.items_of_deck = {(i, j): set() \
                for i in xrange(1, self.num_systems + 1) for j in xrange(1, self.num_decks + 1)}

    def next_item(self, current_time=None):
        """
        Select the next item to present to the user

        If all the items that have arrived have been mastered, an ExhaustedError is raised.

        :param int|None current_time: Leave it as None if you want to use
            current_time = int(time.time()), otherwise supply the desired unix epoch that we are
            going to pretend is the current time

        :rtype: int
        :return: The index of the next item to show
        """

        if current_time is None:
            current_time = int(time.time())

        # handle arrivals
        kvs = self.arrival_time_of_item.items()
        for item, arrival_time in kvs:
            if arrival_time > current_time:
                break
            deck = (self.system_of_item[item], 1)
            self.deck_of_item[item] = deck
            self.items_of_deck[deck].add(item)
            del self.arrival_time_of_item[item]

        if all(deck == 0 or deck > self.num_decks \
                for (system, deck) in self.deck_of_item.itervalues()):
            raise ExhaustedError # all items that have arrived have been mastered

        # sample deck
        normalize = lambda x: np.array(x) / sum(x)
        decks = [(x, y) for x in xrange(1, self.num_systems + 1) \
                for y in xrange(1, self.num_decks + 1)]
        sampled_deck_idx = np.random.choice(
            range(len(decks)),
            p=normalize([self.review_rates[x-1, y-1] \
                    if self.items_of_deck[(x, y)] != set() else 0 \
                    for x in xrange(1, self.num_systems + 1) \
                    for y in xrange(1, self.num_decks + 1)]))
        sampled_deck = decks[sampled_deck_idx]

        if self.processor_sharing:
            # select an item from the queue uniformly at random
            return random.choice(self.items_of_deck[sampled_deck])
        else:
            # select the item at the front of the queue (i.e., the one with the longest delay)
            return min(self.items_of_deck[sampled_deck], key=self.latest_timestamp_of_item.get)

    def update(self, item, outcome, timestamp):
        """
        Update item and deck states

        :param int item: Index of an item
        :param int outcome: 1 if recalled, 0 if forgotten
        :param int timestamp: Timestamp of interaction
        """

        system, current_deck = self.deck_of_item[item]
        new_deck = max(1, current_deck + 2 * outcome - 1)

        self.deck_of_item[item] = (system, new_deck)
        if current_deck >= 1:
            self.items_of_deck[(system, current_deck)].remove(item)
        if new_deck <= self.num_decks:
            self.items_of_deck[(system, new_deck)].add(item)

        self.latest_timestamp_of_item[item] = timestamp

