"""
Module for Leitner Queue Network schedulers
@author Siddharth Reddy <sgr45@cornell.edu>
"""

from __future__ import division

from collections import OrderedDict
import random
import time
import copy

import numpy as np


class ExhaustedError(Exception):
    def __init__(self):
        pass


class StatelessLQNScheduler(object):

    def __init__(self, init_data, processor_sharing=False):
        """
        Initialize scheduler object

        :param dict[str,object] init_data: A dictionary for general scheduling parameters, i.e.,
            values that don't depend on the user history. These are the expected data types:

            init_data['arrival_time_of_item'] : dict[str,int] = the unix epoch when each
                item arrives into the system
            init_data['review_rates'] : list[float] = the review rate for each deck (must sum to 1)

        :param bool processor_sharing: True if we should use the processor sharing
            service discipline, False if we should use first-in-first-out
        """

        self.__dict__.update(init_data)
        self.processor_sharing = processor_sharing
        self.num_decks = len(self.review_rates)

    def next_item(self, history, current_time=None):
        """
        Select the next item to present to the user

        If all the items that have arrived have been mastered, an ExhaustedError is raised.

        :param list[dict[str,object]] history: The logs for a single user
            Each element of the list should contain the following key-value pairs:
                history[i]['item_id'] : str = the id of an item
                history[i]['outcome'] : int = 0 (forgot) or 1 (recalled)
                history[i]['timestamp'] : int = unix epoch time (seconds)

        :param int|None current_time: Leave it as None if you want to use
            current_time = int(time.time()), otherwise supply the desired unix epoch that we are
            going to pretend is the current time

        :rtype: int
        :return: The index of the next item to show
        """

        if current_time is None:
            current_time = int(time.time())

        # items that haven't arrived belong to deck 0, and all other items start at deck 1
        deck_of_item = {item: (1 if arrival_time <= current_time else 0) \
                for item, arrival_time in self.arrival_time_of_item.iteritems()}

        # compute the current deck of each item, based on the logs
        latest_timestamp_of_item = copy.copy(self.arrival_time_of_item)
        for ixn in history:
            item = ixn['item_id']
            deck_of_item[item] = max(1, deck_of_item[item] + 2 * ixn['outcome'] - 1)
            latest_timestamp_of_item[item] = max(latest_timestamp_of_item[item], ixn['timestamp'])

        if all(deck == 0 or deck > self.num_decks for deck in deck_of_item.itervalues()):
            raise ExhaustedError # all items that have arrived have been mastered

        items_of_deck = {i: [] for i in xrange(1, self.num_decks + 1)}
        for item, deck in deck_of_item.iteritems():
            if deck >= 1 and deck <= self.num_decks:
                items_of_deck[deck].append(item)

        # sample deck
        normalize = lambda x: np.array(x) / sum(x)
        sampled_deck = np.random.choice(
            range(1, self.num_decks + 1),
            p=normalize([x if items_of_deck[i+1] != [] else 0 \
                    for i, x in enumerate(self.review_rates)]))

        if self.processor_sharing:
            # select an item from the queue uniformly at random
            return np.random.choice(items_of_deck[sampled_deck])
        else:
            # select the item at the front of the queue (i.e., the one with the longest delay)
            return min(items_of_deck[sampled_deck], key=latest_timestamp_of_item.get)


class StatefulLQNScheduler(object):

    def __init__(self, init_data, processor_sharing=False):
        """
        Initialize scheduler object

        :param dict[str,object] init_data: A dictionary for general scheduling parameters, i.e.,
            values that don't depend on the user history. These are the expected data types:

            init_data['arrival_time_of_item'] : OrderedDict[str,int] = the unix epoch when each
                item arrives into the system
            init_data['review_rates'] : list[float] = the review rate for each deck (must sum to 1)

        :param bool processor_sharing: True if we should use the processor sharing
            service discipline, False if we should use first-in-first-out
        """

        self.__dict__.update(copy.deepcopy(init_data))
        self.processor_sharing = processor_sharing
        self.num_decks = len(self.review_rates)

        self.deck_of_item = {item: 0 for item in self.arrival_time_of_item}
        self.latest_timestamp_of_item = copy.copy(self.arrival_time_of_item)
        self.items_of_deck = {i: set() for i in xrange(1, self.num_decks + 1)}

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
            self.deck_of_item[item] = 1
            self.items_of_deck[1].add(item)
            del self.arrival_time_of_item[item]

        if all(deck == 0 or deck > self.num_decks for deck in self.deck_of_item.itervalues()):
            raise ExhaustedError # all items that have arrived have been mastered

        # sample deck
        normalize = lambda x: np.array(x) / sum(x)
        sampled_deck = np.random.choice(
            range(1, self.num_decks + 1),
            p=normalize([x if self.items_of_deck[i+1] != set() else 0 \
                    for i, x in enumerate(self.review_rates)]))

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

        current_deck = self.deck_of_item[item]
        new_deck = max(1, current_deck + 2 * outcome - 1)

        self.deck_of_item[item] = new_deck
        if current_deck >= 1:
            self.items_of_deck[current_deck].remove(item)
        if new_deck <= self.num_decks:
            self.items_of_deck[new_deck].add(item)

        self.latest_timestamp_of_item[item] = timestamp


def sample_arrival_times(all_items, arrival_rate, start_time):
    """
    Sample item arrival times for init_data['arrival_time_of_item'],
    which gets passed to the StatelessLQNScheduler constructor

    :param set[str] all_items: A set of item ids
    :param float arrival_rate: The arrival rate for the Poisson process
    :param int start_time: Start time (unix epoch) for the arrival process
    """

    all_items = list(all_items)
    random.shuffle(all_items)
    inter_arrival_times = np.random.exponential(1 / arrival_rate, len(all_items))
    arrival_times = start_time + np.cumsum(inter_arrival_times, axis=0).astype(int)
    return OrderedDict(zip(all_items, arrival_times))

