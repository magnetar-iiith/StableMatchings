"""This module finds the man optimal matching
    via the Gale-Shapley Algorithm"""

import numpy as np
from collections import deque

def print_matching(matching):
    """Prints matching"""
    for man, woman in enumerate(matching):
        print("Man", man, "is matched with Woman", woman)
        # print the marriage

# def gale_shapley(preflist):
#     """Executes Gale-Shapley Algorithm"""
#     # this runs slower
#     proposer_ranklist = preflist[0] # preference list of the proposer (men)
#     rejector_ranklist = preflist[1] # preference list of the rejectors (women)
#     num_agents = np.shape(proposer_ranklist)[0] # number of agents on each side
#     free_proposers = np.arange(num_agents) # numpy array of proposers who are not engaged yet
#     free_proposer_idx = 0 # index tracking position in free_proposers array
#     partners_of_rejectors = -1 * np.ones(num_agents, dtype=int) # list of partners of rejectors, -1 means not engaged
#     partners_of_proposers = -1 * np.ones(num_agents, dtype=int) # list of partners of proposers, -1 means not engaged
#     to_be_proposed = np.zeros(num_agents, dtype=int) # index of next rejector in proposers preference
#     # list to be proposed by each proposer

#     # Some preprocessing of ranks
#     rejector_preferences = np.zeros((num_agents, num_agents), dtype=int) # rank of each proposer in the
#     # preference list of each rejector
#     for rejector in range(num_agents): # iterating over rejectors
#         for rank, proposer in enumerate(rejector_ranklist[rejector]): # iterating over proposers in the
#             # rejector's preference list
#             rejector_preferences[rejector][proposer] = rank # assigning a rank to each proposer in
#             # rejector's preference list

#     while free_proposer_idx < len(free_proposers):
#         proposer = free_proposers[free_proposer_idx] # Get the first free proposer
#         # Check if proposer has exhausted all their preferences
#         # if to_be_proposed[proposer] >= len(proposer_ranklist[proposer]):
#         #     free_proposer_idx += 1
#         #     continue
#         rejector = proposer_ranklist[proposer][to_be_proposed[proposer]] # get the next rejector that proposer
#         # is going to propose to
#         # print("Proposer", proposer, "proposes to Rejector", rejector)
#         # print the proposal
#         if partners_of_rejectors[rejector] == -1: # If rejector is free, accept the proposal
#             partners_of_rejectors[rejector] = proposer # partner of rejector becomes proposer
#             partners_of_proposers[proposer] = rejector
#             # print("Proposer", proposer, "is now matched with Rejector", rejector)
#             # print the engagement
#             free_proposer_idx += 1
#         else: # If rejector is already engaged
#             current_partner = partners_of_rejectors[rejector] # store the current partner of the rejector
#             current_partner_rank = rejector_preferences[rejector][current_partner] # store current partner's rank in the rejector's preference list
#             proposer_rank = rejector_preferences[rejector][proposer] # store the proposer's rank in the rejectors preference list
#             if proposer_rank < current_partner_rank: # If proposer is preferred over current partner
#                 partners_of_rejectors[rejector] = proposer # ditch the current partner
#                 # and engage with new proposer
#                 partners_of_proposers[proposer] = rejector
#                 # print("Rejector", rejector, "dumps her current partner", current_partner)
#                 # print the dump
#                 # print("Proposer", proposer, "s now matched with Rejector", rejector)
#                 # print the engagement
#                 free_proposers = np.append(free_proposers, current_partner) # current partner becomes free
#                 partners_of_proposers[current_partner] = -1
#                 free_proposer_idx += 1
#             else: # If rejector prefers current partner over proposer
#                 # print("Proposer", proposer, "is rejected by Rejector", rejector)
#                 # print the rejection
#                 # free_proposers = np.append(free_proposers, proposer) # proposer is again free
#                 # need to add back as he was
#                 # popped out of the list
#                 to_be_proposed[proposer] += 1 # increase index of rejector to be proposed next
#     return partners_of_proposers # return the result

def gale_shapley(preflist):
    """Executes Gale-Shapley Algorithm"""
    # this runs faster
    proposer_ranklist = preflist[0]
    # preference list of the proposer (men)
    rejector_ranklist = preflist[1]
    # preference list of the rejectors (women)

    num_agents = len(proposer_ranklist)
    # number of agents on each side
    free_proposers = list(range(num_agents))
    head = 0
    # free_proposers = deque(range(num_agents))
    # list of proposers who are not engaged yet
    # partners_of_rejectors = -1 * np.ones(num_agents, dtype=int)
    partners_of_rejectors = [-1] * num_agents
    # list of partners of rejectors, -1 means not engaged
    # partners_of_proposers = -1 * np.ones(num_agents, dtype=int)
    partners_of_proposers = [-1] * num_agents
    # list of partners of proposers, -1 means not engaged
    # to_be_proposed = np.zeros(num_agents, dtype=int)
    to_be_proposed = [0] * num_agents
    # index of next rejector in proposers preference
    # list to be proposed by each proposer

    # Some preprocessing of ranks
    # rejector_preferences = np.zeros((num_agents, num_agents), dtype=int)
    rejector_preferences = [[0] * num_agents for _ in range(num_agents)]
    # rank of each proposer in the
    # preference list of each rejector
    for woman, woman_prefs in enumerate(rejector_ranklist):
        pref = rejector_preferences[woman]
        for rank, man in enumerate(woman_prefs):
            pref[man] = rank
    # for rejector in range(num_agents):
    #     # iterating over rejectors
    #     for rank, proposer in enumerate(rejector_ranklist[rejector]):
    #         # iterating over proposers in the
    #         # rejector's preference list
    #         rejector_preferences[rejector][proposer] = rank
    #         # assigning a rank to each proposer in
    #         rejector_preferences[rejector][proposer] = rank # assigning a rank to each proposer in
    #         # rejector's preference list

    while head < len(free_proposers):
        proposer = free_proposers[head]
        # proposer = free_proposers.popleft()
        # Get the first free proposer
        rejector = proposer_ranklist[proposer][to_be_proposed[proposer]]
        # get the next rejector that proposer
        # is going to propose to
        # print("Proposer", proposer, "proposes to Rejector", rejector)
        # print the proposal
        if partners_of_rejectors[rejector] == -1:
            # If rejector is free, accept the proposal
            partners_of_rejectors[rejector] = proposer
            # partner of rejector becomes proposer
            partners_of_proposers[proposer] = rejector
            head += 1
            # print("Proposer", proposer, "is now matched with Rejector", rejector)
            # print the engagement
        else:
            # If rejector is already engaged
            current_partner = partners_of_rejectors[rejector]
            # store the current partner of the rejector
            rejector_preflist = rejector_preferences[rejector]
            # store current partner's rank in the rejector's preference list
            # proposer_rank = rejector_preferences[rejector][proposer]
            # store the proposer's rank in the rejectors preference list
            if  rejector_preflist[proposer] < rejector_preflist[current_partner]:
                # If proposer is preferred over current partner
                partners_of_rejectors[rejector] = proposer
                # ditch the current partner
                # and engage with new proposer
                partners_of_proposers[proposer] = rejector
                # print("Rejector", rejector, "dumps her current partner", current_partner)
                # print the dump
                # print("Proposer", proposer, "s now matched with Rejector", rejector)
                # print the engagement
                free_proposers.append(current_partner)
                head += 1
                # current partner becomes free
                partners_of_proposers[current_partner] = -1
            # else:
                # If rejector prefers current partner over proposer
                # print("Proposer", proposer, "is rejected by Rejector", rejector)
                # print the rejection
                # free_proposers.append(proposer)
                # to_be_proposed[proposer] += 1
                # proposer is again free
                # need to add back as he was
                # popped out of the list
        to_be_proposed[proposer] += 1
        # increase index of rejector to be proposed next

    return partners_of_proposers
    # return the result

def blocking_pairs(matching, preflist):
    """Returns all blocking pairs"""
    num_agents = len(matching)
    blockingpairs = []
    reverse_matching = [-1] * num_agents
    for m_i in range(num_agents):
        reverse_matching[matching[m_i]] = m_i
    for m_i in range(num_agents):
        for w_j in range(num_agents):
            blocking_pair = (m_i, w_j)
            if matching[m_i] == w_j:
                # if i is matched with j, skip
                continue
            if w_j in preflist[0][m_i] and m_i in preflist[1][w_j]:
                w_i = matching[m_i]
                m_j = reverse_matching[w_j]
                mr_m_i_w_i = preflist[0][m_i].index(w_i)
                mr_m_i_w_j = preflist[0][m_i].index(w_j)
                wr_w_j_m_i = preflist[1][w_j].index(m_i)
                wr_w_j_m_j = preflist[1][w_j].index(m_j)
                if mr_m_i_w_j < mr_m_i_w_i:
                    if wr_w_j_m_i < wr_w_j_m_j:
                        blockingpairs.append(blocking_pair)

    return blockingpairs
