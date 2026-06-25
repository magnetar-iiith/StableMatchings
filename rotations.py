def identify_rotation(start_man, men_shortlists):
    # identify the current rotation exposed in the shortlists
    position = {}
    # position stores the man being processed
    rotation = []
    # rotation stores the current rotation
    visited = set()
    # visited is a set of nodes
    first_choice_to_man = {
        man: None for man in ()
    }
    first_choice_to_man = {
        prefs[0]: man
        for man, prefs in enumerate(men_shortlists)
        if prefs
    }

    current_man = start_man
    # initial node is start_man

    while current_man not in visited:
        # while the current node is not visited
        visited.add(current_man)
        prefs = men_shortlists[current_man]

        # add the current node in the visited set
        if len(prefs) < 2:
            # if current node has less than 2 preferences
            rotation = None
            break
        position[current_man] = len(rotation)
        # add the current node in the rotation index
        rotation.append((current_man, prefs[0]))
        # add the current pair (m_i, w_i) in the rotation
        # var rotation is a rotation - a list of tuples
        # next_man = None
        # # next node is initialized as None
        # for iter_man, prefs in enumerate(men_shortlists):
        #     # iterating over men in mens shortlists
        #     # prefs is iter_man's preferences in mens shortlists
        #     if prefs:
        #         # if his preference list is not empty
        #         if len(men_shortlists[current_man]) > 1:
        #             # if current_man has a second preference
        #             # in his preference list
        #             if prefs[0] == men_shortlists[current_man][1]:
        #                 # if first preference of iter_man is
        #                 # second preference of current_man m_i
        #                 next_man = iter_man
        #                 # next node to visit is iter_man
        #                 break
        #             # move on
        next_man = first_choice_to_man.get(prefs[1])

        
        if next_man ==  None:
            # if no next man found, means no rotation exposed
            # in the shortlists
            return None
        # stop
        if next_man in visited:
            # else if next_man is already visited
            rotation = rotation[position[next_man]:]
            # rotation formed is (m_t, w_t) to (m_s-1, w_s-1)
            break
        # stop
        current_man = next_man
        # else continue appending the rotation
    # if the rotation is empty, return None
    if not rotation:
        return None
    if len(rotation) < 2:
        # if rotation is less than 2, return None
        return None
    # else return the rotation
    return rotation

def eliminate_rotation(rotation, men_shortlists, women_shortlists):
    # function to eliminate a rotation from the shortlists
    r = len(rotation)
    # let r be the length of the rotation
    for i in range(r):
        # iterate over the rotation
        j = (i + 1) % r
        # j is the next index in the rotation
        m_i, w_j = rotation[i][0], rotation[j][1] 
        # m_i, w_{i+1}
        w_i = rotation[i][1]
        # w_i
        if w_i in men_shortlists[m_i]:
            # if w_i is in m_i's shortlist
            men_shortlists[m_i].remove(w_i)
            # remove w_i from m_i's shortlist
        if m_i in women_shortlists[w_i]:
            # if m_i is in w_i's shortlist
            women_shortlists[w_i].remove(m_i)
            # remove m_i from w_i's shortlist
        index = None
        # index is initialized as None
        if m_i in women_shortlists[w_j]:
            # if m_i is in w_{i+1}'s shortlist
            index = women_shortlists[w_j].index(m_i)
            # find index of m_i in w_{i+1}'s shortlist
        if index is None:
        # if m_i is not in w_{i+1}'s shortlist
            continue
        # continue to next iteration
        # this happens when an element/man is
        # not part of any exposed rotation
        else:
            # if exposed rotation is found
            for idx, x in enumerate(women_shortlists[w_j]):
                # iterate over w_{i+1}'s shortlist
                if idx > index:
                    # if idx (rank of x)
                    # is greater than index
                    # then remove x from womans shortlist
                    # and vice versa
                    if w_j in men_shortlists[x]:
                        # sanity check
                        id = men_shortlists[x].index(w_j)
                        # find index of w_{i+1} in x's shortlist
                        men_shortlists[x].pop(id)
                        # remove that id element
            women_shortlists[w_j] = women_shortlists[w_j][:index+1]
            # reduce w_{i+1}'s shortlist from 0 to index (inclusive)

def find_a_rotation(men_shortlists):
    # Find the first exposed rotation
    found_rotation = None
    # initialize the found rotation as None
    for man in range(len(men_shortlists)):
        # iterate over men in mens shortlists
        if len(men_shortlists[man]) > 1:  
        # Exposed if man has a second choice
            found_rotation = identify_rotation(man, men_shortlists)
            # identify the rotation exposed in the shortlists
            break
        
    return found_rotation
    # return the rotation

def print_rotations(rotations, weights):
    print("Rotations:")
    for i, rotation in enumerate(rotations):
        print(f"Rotation {i}: {rotation}, Weight = {weights[i]}")
