from __future__ import division
import numpy as np

def triangularizer(Qk, rk, Nk, Lk):
    # Triangularize the data for bandit k according to the labeling L
    # Inputs:
    # Qk = numpy array representing the rate matrix for bandit k
    # rk = numpy array representing the reward vector for bandit k
    # Nk = dictionary with keys denoting the state names, and values giving the corresponding row number in Qk and rk
    # L = dictionary with keys denoting state names and values giving the corresponding label, for each state in Nk
    # Outputs:
    # tilde_Qk = finalized transition rates for bandit k
    # tilde_rk = finalized rewards for bandit k

    nk = len(Nk) # number of states in bandit k
    tableau = np.concatenate((np.eye(nk)-Qk, rk.T), axis=1)
    M = Nk.copy() # make a copy of Nk
    Lktemp = Lk.copy() # make a copy of L
    
    while len(M) > 0:
        i = min(Lktemp, key=Lktemp.get) # state i in bandit k whose label L(i) is smallest
        i_ind = M[i] # state i's row in Qk and rk
        alpha = 1/(tableau[i_ind, i_ind])
        tableau[i_ind] = alpha * tableau[i_ind] # update state i's row in the tableau
        del M[i] # remove state i from M
        del Lktemp[i] # remove state i from L
        for j in M: # update the other rows in the tableau
            tableau[M[j]] = tableau[M[j]] - tableau[M[j],i_ind] * tableau[i_ind]
    tilde_Qk = np.eye(nk) - tableau[:,:-1] # finalized transition rates
    tilde_rk = tableau[:,-1] # finalized rewards
    return (tilde_Qk, tilde_rk)

def evaluator(bandits, s, L):
    # Evaluate the total reward earned under the policy keyed to the labeling L starting from a given initial multi-state
    # Inputs:
    # bandits = dictionary where bandit k is identified by key k, whose corresponding value is the tuple (Qk, rk, Nk) giving bandit k's transition rate matrix Qk, reward vector rk, and dictionary Nk with keys denoting state names and values giving the corresponding row number in Qk and rk
    # s = dictionary denoting the initial multi-state (bandit : state)
    # L = dictionary with keys denoting state names and values giving the corresponding labels
    
    y = {bandit: {state : 0 for state in bandits[bandit][2].keys()} for bandit in bandits.keys()}
    for bandit in bandits: # Initialize y for each bandit
        y[bandit][s[bandit]] = 1
    finTableaus = {bandit : None for bandit in bandits.keys()} # dictionary to store finalized tableaus for each bandit
    for bandit in bandits: # Triangularize each bandit according to L
        Qk,rk,Nk = bandits[bandit]
        Lk = {state : L[state] for state in bandits[bandit][2].keys()}
        finTableaus[bandit] = triangularizer(Qk, rk, Nk, Lk)
    return finTableaus # test output
    V = 0 # initialize V
    n = 1
    while n <= len(L):
        i = [state for state in L.keys() if L[state] == n]
        if len(i) != 1: # if more than one state's label is n
            print "Error: Invalid labeling"
            return
        else:
            i = i[0]
            for bandit in bandits: # find the bandit k that state i belongs to
                if i in set(bandits[bandit][2].keys()):
                    k = bandit
                    break
            rTilde_i = finTableaus[k][1][bandits[k][2][i]]
            yk_i = y[k][i]
            prod = 1
            for p in set(bandits.keys()).difference(set[k])):
                prod *= sum(y[p].values())
            V += rTilde_i * yk_i * prod # update V
            i_ind = bandits[k][2][i]
            for state in set(bandits[k][2].keys()).difference(set[i])):
                state_ind = bandits[k][2][state]
                y[k][state] += yk_i * bandits[k][0][i_ind, state_ind]
            y[k][i] = 0
