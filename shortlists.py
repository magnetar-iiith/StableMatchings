"""This creates shortlists"""

import copy

def print_shortlists(shortlist):
    """prints mens or womens shortlist"""
    for i, preferences in enumerate(shortlist):
        # print reduced preference list
        # of each agent
        print("Shortlist of Agent", i, ":", preferences)
        # prints the preferences

def create_shortlists(preflist, matching):
    """given the original preference list
    and men optimal matching
    create the man-oriented shortlists"""
    num_agents = len(preflist[0])
    # number of agents
    men_shortlists = copy.deepcopy(preflist[0])
    women_shortlists = copy.deepcopy(preflist[1])

    for man, woman in enumerate(matching):
        # iterating over the couples
        idx = women_shortlists[woman].index(man)
        # idx is rank of man in woman's
        # preference list
        women_shortlists[woman] = women_shortlists[woman][:idx + 1]
        # women woman's reduced preference list
        # is all men before and including man

    for man in range(num_agents):
        # iterating over all men
        for woman in men_shortlists[man][:]:
            # iterating over all women woman
            # in man's shortlist
            if man not in women_shortlists[woman]:
                # If man is not in woman's shortlist
                men_shortlists[man].remove(woman)
                # remove woman from man's shortlist

    return men_shortlists, women_shortlists
# return the reduced preference lists called as shortlists
