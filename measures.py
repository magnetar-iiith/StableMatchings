"""This module has the welfare measures calculations"""

import numpy as np

def regret(matching, preflist):
    """calculate regret of the matching"""
    regret_measure = 0
    for i, match in enumerate(matching):
        rank_man = preflist[0][i].index(match) + 1
        rank_woman = preflist[1][match].index(i) + 1
        regret_measure = max(rank_man, rank_woman, regret_measure)
    return regret_measure

def egalitarian(matching, preflist):
    """calculate summation of ranks"""
    d_man, d_woman, num_agents = 0, 0, len(preflist[0])
    for man, woman in enumerate(matching):
        rank_man = preflist[0][man].index(woman) + 1
        rank_woman = preflist[1][woman].index(man) + 1
        d_man += rank_man
        d_woman += rank_woman
    return (d_man + d_woman)/(2*num_agents)

def egalitarian_welfare(matching, preflist):
    """calculate social welfare"""
    d_man, d_woman, num_agents = 0, 0, len(preflist[0])
    for man, woman in enumerate(matching):
        rank_man = num_agents - preflist[0][man].index(woman)
        rank_woman = num_agents - preflist[1][woman].index(man)
        d_man += rank_man
        d_woman += rank_woman
    return (d_man + d_woman)

def disparity(matching, preflist):
    """calculate disparity"""
    sum_of_ranks_men, sum_of_ranks_women, num_agents \
        = 0, 0, len(preflist[0])
    for i, match in enumerate(matching):
        rank_man = preflist[0][i].index(match)
        rank_woman = preflist[1][match].index(i)
        sum_of_ranks_men += rank_man
        sum_of_ranks_women += rank_woman
    return 1 + (np.abs(\
        sum_of_ranks_men - sum_of_ranks_women)/num_agents)

def nash_welfare(matching, preflist):
    """calculate nash social welfare"""
    d_man, d_woman, num_aqents = 1, 1, len(preflist[0])
    for man, woman in enumerate(matching):
        rank_man = num_aqents - preflist[0][man].index(woman)
        rank_woman = num_aqents - preflist[1][woman].index(man)
        d_man *= rank_man**(1/(2*num_aqents))
        d_woman *= rank_woman**(1/(2*num_aqents))
    return d_man*d_woman
