from __future__ import division
import numpy as np

def Triangularizer(Qk, rk, Nk, Lk):
    # Triangularize the data for bandit k according to the labeling L
    # Qk = numpy array representing the rate matrix for bandit k
    # rk = numpy array representing the reward vector for bandit k
    # Nk = dictionary with keys denoting the state number, and values giving the corresponding row number in Qk and rk
    # L = dictionary with keys denoting the state number, and values giving the corresponding label, for each state in Nk

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
    return tableau

def Evaluator():
