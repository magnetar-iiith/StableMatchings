import numpy as np

def print_matching(matching):
    for man, woman in enumerate(matching):
        print("Man", man, "is matched with Woman", woman)
        # print the marriage

def gale_shapley(preflist):
    proposer_ranklist = preflist[0]
    # preference list of the proposer (men)
    rejector_ranklist = preflist[1]
    # preference list of the rejectors (women)

    num_agents = len(proposer_ranklist)
    # number of agents on each side
    free_proposers = list(range(num_agents))
    # list of proposers who are not engaged yet
    partners_of_rejectors = -1 * np.ones(num_agents, dtype=int)
    # list of partners of rejectors, -1 means not engaged
    partners_of_proposers = -1 * np.ones(num_agents, dtype=int)
    # list of partners of proposers, -1 means not engaged
    to_be_proposed = np.zeros(num_agents, dtype=int)
    # index of next rejector in proposers preference
    # list to be proposed by each proposer
    
    # Some preprocessing of ranks
    rejector_preferences = np.zeros((num_agents, num_agents), dtype=int)
    # rank of each proposer in the 
    # preference list of each rejector
    for rejector in range(num_agents):
        # iterating over rejectors
        for rank, proposer in enumerate(rejector_ranklist[rejector]):
            # iterating over proposers in the 
            # rejector's preference list
            rejector_preferences[rejector][proposer] = rank
            # assigning a rank to each proposer in 
            # rejector's preference list

    while free_proposers:
        proposer = free_proposers.pop(0) 
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
            # print("Proposer", proposer, "is now matched with Rejector", rejector)
            # print the engagement
        else:
            # If rejector is already engaged
            current_partner = partners_of_rejectors[rejector]
            # store the current partner of the rejector
            current_partner_rank = rejector_preferences[rejector][current_partner]
            # store current partner's rank in the rejector's preference list
            proposer_rank = rejector_preferences[rejector][proposer]
            # store the proposer's rank in the rejectors preference list
            if  proposer_rank < current_partner_rank:
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
                # current partner becomes free
                partners_of_proposers[current_partner] = -1
            else:
                # If rejector prefers current partner over proposer
                # print("Proposer", proposer, "is rejected by Rejector", rejector)
                # print the rejection
                free_proposers.append(proposer)
                # proposer is again free
                # need to add back as he was 
                # popped out of the list
        to_be_proposed[proposer] += 1
        # increase index of rejector to be proposed next

    return partners_of_proposers
    # return the result

def blocking_pairs(matching, preflist):
    num_agents = len(matching)
    blockingpairs = []
    reverse_matching = [-1] * num_agents
    for m_i in range(num_agents):
        reverse_matching[matching[m_i]] = m_i
    for m_i in range(num_agents):
        for w_j in range(num_agents):
            bp = (m_i, w_j)
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
                        blockingpairs.append(bp)

    return blockingpairs
