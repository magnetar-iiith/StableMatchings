"""This creates shortlists"""

import numpy as np

def print_shortlists(shortlist):
    """prints mens or womens shortlist"""
    for i, preferences in enumerate(shortlist):
        # print reduced preference list
        # of each agent
        print("Shortlist of Agent", i, ":", preferences)
        # prints the preferences

def create_shortlists(preflist, matching):
    men_prefs, women_prefs = preflist

    # Deep copy rows (your .copy() only copies outer list)
    women_shortlists = [row[:] for row in women_prefs]

    # Reduce women's lists
    for man, woman in enumerate(matching):
        idx = women_shortlists[woman].index(man)
        women_shortlists[woman] = women_shortlists[woman][:idx + 1]

    # Convert to sets for fast membership tests
    women_sets = [set(lst) for lst in women_shortlists]

    # Build men's shortlists directly
    men_shortlists = [
        [woman for woman in men_prefs[man]
         if man in women_sets[woman]]
        for man in range(len(men_prefs))
    ]

    return men_shortlists, women_shortlists

# def create_shortlists(preflist, matching):
#     """given the original preference list
#     and men optimal matching
#     create the man-oriented shortlists"""
#     num_agents = len(preflist[0])
#     # number of agents
#     men_shortlists = preflist[0].copy()
#     women_shortlists = preflist[1].copy()

#     for man, woman in enumerate(matching):
#         # iterating over the couples
#         idx = women_shortlists[woman].index(man)
#         # idx is rank of man in woman's
#         # preference list
#         women_shortlists[woman] = women_shortlists[woman][:idx + 1]
#         # women woman's reduced preference list
#         # is all men before and including man

#     for man in range(num_agents):
#         # iterating over all men
#         for woman in men_shortlists[man][:]:
#             # iterating over all women woman
#             # in man's shortlist
#             if man not in women_shortlists[woman]:
#                 # If man is not in woman's shortlist
#                 men_shortlists[man].remove(woman)
#                 # remove woman from man's shortlist

#     return men_shortlists, women_shortlists
# # return the reduced preference lists called as shortlists

# def create_shortlists(preflist, matching):
#     """given the original preference list
#     and men optimal matching
#     create the man-oriented shortlists"""
#     num_agents = len(preflist[0])
#     # number of agents
#     men_shortlists = np.array(preflist[0], copy=True)
#     women_shortlists = np.array(preflist[1], copy=True)

#     # Build women shortlists
#     women_masks = np.zeros_like(women_shortlists, dtype=bool)

#     for man, woman in enumerate(matching):
#         # iterating over the couples
#         idx = np.where(women_shortlists[woman] == man)[0][0]
#         # idx is rank of man in woman's
#         # preference list

#         women_masks[woman, :idx + 1] = True
#         # women woman's reduced preference list
#         # is all men before and including man

#     # Remove women from men's lists when the man
#     # does not appear in the corresponding woman's shortlist
#     men_masks = np.ones_like(men_shortlists, dtype=bool)

#     for man in range(num_agents):
#         # iterating over all men
#         for pos, woman in enumerate(men_shortlists[man]):
#             # iterating over all women woman
#             # in man's shortlist
#             if not np.any(
#                 women_masks[woman] &
#                 (women_shortlists[woman] == man)
#             ):
#                 # If man is not in woman's shortlist
#                 men_masks[man, pos] = False
#                 # remove woman from man's shortlist

#     men_shortlists = [row[mask].tolist()
#                       for row, mask in zip(men_shortlists, men_masks)]

#     women_shortlists = [row[mask].tolist()
#                         for row, mask in zip(women_shortlists, women_masks)]

#     return men_shortlists, women_shortlists
# # return the reduced preference lists called as shortlists
