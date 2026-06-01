import copy

def print_shortlists(shortlist):
    # print mens or womens shortlist
    for i, preferences in enumerate(shortlist):
        # print reduced preference list
        # of each agent 
        print("Shortlist of Agent", i, ":", preferences)
        # prints the preferences

def create_shortlists(preflist, matching):
    # given the original preference list
    # and men optimal matching
    # create the man-oriented shortlists
    num_agents = len(preflist[0])
    # number of agents
    men_shortlists = copy.deepcopy(preflist[0])
    women_shortlists = copy.deepcopy(preflist[1])

    for m, w in enumerate(matching):
        # iterating over the couples
        idx = women_shortlists[w].index(m) 
        # idx is rank of man m in woman w's 
        # preference list                                                      
        women_shortlists[w] = women_shortlists[w][:idx + 1]
        # women w's reduced preference list
        # is all men beore and including m

    for m in range(num_agents):
        # iterating over all men
        for w in men_shortlists[m][:]:
            # iterating over all women w 
            # in man m's shortlist
            if m not in women_shortlists[w]:
                # If m is not in w's shortlist
                men_shortlists[m].remove(w)
                # remove w from m's shortlist

    return men_shortlists, women_shortlists
# return the reduced preference lists called as shortlists