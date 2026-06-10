import numpy as np

def reg(matching, preflist):
    # calculate regret of the matching
    regret = 0
    for i, match in enumerate(matching):
        rank_m = preflist[0][i].index(match) + 1
        rank_w = preflist[1][match].index(i) + 1
        regret = max(rank_m, rank_w, regret)
    return regret

def eg(matching, preflist):
    # calculate summation of ranks
    d_m, d_w, n = 0, 0, len(preflist[0])
    for m, w in enumerate(matching):
        rank_m = preflist[0][m].index(w) + 1
        rank_w = preflist[1][w].index(m) + 1
        d_m += rank_m
        d_w += rank_w
    return (d_m + d_w)/(2*n)

def disp(matching, preflist):
    # calculate disparity
    sum_of_ranks_men, sum_of_ranks_women, n = 0, 0, len(preflist[0])
    for i, match in enumerate(matching):
        rank_m = preflist[0][i].index(match)
        rank_w = preflist[1][match].index(i)
        sum_of_ranks_men += rank_m
        sum_of_ranks_women += rank_w
    return 1 + (np.abs(sum_of_ranks_men - sum_of_ranks_women)/n)

def nsw(matching, preflist):
    # calculate nash social welfare
    d_m, d_w, n = 1, 1, len(preflist[0])
    for m, w in enumerate(matching):
        rank_m = n - preflist[0][m].index(w) # Rank starts from 1
        rank_w = n - preflist[1][w].index(m) # Rank starts from 1
        d_m *= rank_m**(1/(2*n))
        d_w *= rank_w**(1/(2*n))
    return (d_m*d_w)

def statistics(c_1_list, c_2_list, c_3_list, c_4_list, d_1_list, d_2_list, d_3_list, d_4_list, e_1_list, e_2_list, e_3_list, e_4_list, nsw_1_list, nsw_2_list, nsw_3_list, nsw_4_list):
    # calculate variance of each
    c_1_avg = np.mean(c_1_list)
    c_2_avg = np.mean(c_2_list)
    c_3_avg = np.mean(c_3_list)
    c_4_avg = np.mean(c_4_list)
    c_1_var = np.var(c_1_list)
    c_2_var = np.var(c_2_list)
    c_3_var = np.var(c_3_list)
    c_4_var = np.var(c_4_list)
    d_1_avg = np.mean(d_1_list)
    d_2_avg = np.mean(d_2_list)
    d_3_avg = np.mean(d_3_list)
    d_4_avg = np.mean(d_4_list)
    d_1_var = np.var(d_1_list)
    d_2_var = np.var(d_2_list)
    d_3_var = np.var(d_3_list)
    d_4_var = np.var(d_4_list)
    e_1_avg = np.mean(e_1_list)
    e_2_avg = np.mean(e_2_list)
    e_3_avg = np.mean(e_3_list)
    e_4_avg = np.mean(e_4_list)
    e_1_var = np.var(e_1_list)
    e_2_var = np.var(e_2_list)
    e_3_var = np.var(e_3_list)
    e_4_var = np.var(e_4_list)
    nsw_1_avg = np.mean(nsw_1_list)
    nsw_2_avg = np.mean(nsw_2_list)
    nsw_3_avg = np.mean(nsw_3_list)
    nsw_4_avg = np.mean(nsw_4_list)
    nsw_1_var = np.var(nsw_1_list)
    nsw_2_var = np.var(nsw_2_list)
    nsw_3_var = np.var(nsw_3_list)
    nsw_4_var = np.var(nsw_4_list)

    return  [[  c_1_avg, c_2_avg, c_3_avg, c_4_avg, \
                d_1_avg, d_2_avg, d_3_avg, d_4_avg, \
                e_1_avg, e_2_avg, e_3_avg, e_4_avg, \
                nsw_1_avg, nsw_2_avg, nsw_3_avg, nsw_4_avg],\
            [   c_1_var, c_2_var, c_3_var, c_4_var, \
                d_1_var, d_2_var, d_3_var, d_4_var, \
                e_1_var, e_2_var, e_3_var, e_4_var, \
                nsw_1_var, nsw_2_var, nsw_3_var, nsw_4_var]]