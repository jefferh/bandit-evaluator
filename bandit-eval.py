from __future__ import division
import numpy as np

def Triangularizer(Qk, rk, Nk, L):
    # Triangularize the data for bandit k according to the labeling L
    # Qk = numpy array representing the rate matrix for bandit k
    # rk = numpy array representing the reward vector for bandit k
    # Nk = dictionary with keys denoting the state number, and values giving the corresponding row number in Qk and rk
    # L = dictionary with keys denoting the state number, and values giving the corresponding label

    nk = len(Nk) # number of states in bandit k
    tableau = np.concatenate((np.eye(nk)-Qk, rk.T), axis=1)
    M = Nk
    
    while len(M) > 0:
        i = min(L, key=L.get) # state i in bandit k whose label L(i) is smallest
        i_ind = M[i] # state i's row in Qk and rk
        alpha = 1/(1 - Qk[i_ind,i_ind])
        tableau[i_ind] = alpha * tableau[i_ind] # update state i's row in the tableau
        del M[i] # remove state i from M
        for j in M: # update the other rows in the tableau
            tableau[M[j]] = tableau[M[j]] + Qk[M[j],i_ind] * tableau[i_ind]
