# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
#1-D, 2-D and 3-D systems for quantum entanglement
#------------------------------------------------------------------------------

import numpy as np
import scipy as sp
import scipy.linalg as spl

#------------------------------------------------------------------------------
# Defining the matrices - creation, annihilation, identity
# The complete basis for each site consists of two elements, an up-spin and
# a down-spin. AN up-spin is represented by the column vector (1,0) and the
# down spin by the column vector (0,1)
#------------------------------------------------------------------------------

I = np.array([[1,0],        # Identity matrix
              [0,1]])
a_ud = np.array([[0,1],     # Creation operator for up-spin electron
                [0,0]])
a_u = np.array([[0,0],      # Annihilation operator for up-spin electron
                [1,0]])
a_dd = a_u                  # Creation operator for down-spin electron
a_d = a_ud                  # Annihilation operator for down-spin electron
n_u = np.dot(a_ud,a_u)      # The number operator for up-spin
n_d = np.dot(a_dd,a_d)      # The number operator for down-spin

#------------------------------------------------------------------------------
# Defining the on-site energy and the hopping parameter
#------------------------------------------------------------------------------
h = 10.                     # The on-site energy fo rthe presence of 1 electron
t = 4.                      # The hopping energy
#%%
#------------------------------------------------------------------------------
# Creating the matrix of connection. 1 denotes that the two sites (i,j) are 
# connected. It is simple to see that the matrix is symmetric
#------------------------------------------------------------------------------

n = 4       # n denotes the number of sites

#------------------------------------------------------------------------------
# C for a 1-D system where the sites are connected  like 1 -- 2 -- 3 -- 4
#------------------------------------------------------------------------------
#C = np.array([[1,1,0,0],
#              [1,1,1,0],
#              [0,1,1,1],
#              [0,0,1,1]])

#------------------------------------------------------------------------------
# C for a 2-D system where the sites are connected in the following fashion:
# 1 -- 2
# |    |
# 4 -- 3
#------------------------------------------------------------------------------
#C = np.array([[1,1,0,1],
#              [1,1,1,0],
#              [0,1,1,1],
#              [1,0,1,1]])

#------------------------------------------------------------------------------
# C for a 3-D system where the sites are connected in the following fashion:
# 1 ---- 2
# \  \/ /   You can visualize this like a tetrahedron
#  \ 3 /
#   \|/
#    4
#------------------------------------------------------------------------------
C = np.array([[1,1,1,1],
              [1,1,1,1],
              [1,1,1,1],
              [1,1,1,1]])

#%%
#------------------------------------------------------------------------------
#Function for creating a matrix element for the Hamiltonian
#------------------------------------------------------------------------------
def H_el(n,s1,s2,A,B):      #A = Operator corresponding to site s1
                            #B = Operator corresponding to site s2
    X = np.array([[1]])
    I = np.identity(2, dtype = float)
    i = 0
    while(i<s1):
        X = np.kron(X,I)
        i = i+1
    X = np.kron(X,A)
    i = s1
    while(i<s2-1):
        X = np.kron(X,I)
        i = i+1
    if(s1!=s2):
        X = np.kron(X,B)
    i = s2
    while(i<n-1):
        X = np.kron(X,I)
        i = i+1
    return X

#%%
#------------------------------------------------------------------------------
# Creating the Hamiltonian given the connection matrix C
# The Hamiltonian is given by:
# H = on-site energy terms for the orbitals + hopping terms
#------------------------------------------------------------------------------

H = np.zeros((np.power(2,4), np.power(2,4)), dtype = float)
for i in range(0,n):
    for j in range(i,n):
        if(C[i,j] == 1):
            if(i==j):
                H = H + h*H_el(n,i,j,n_u,n_u) + h*H_el(n,i,j,n_d,n_d)
            else:
                H = H + t*(H_el(n,i,j,a_u,a_ud) + H_el(n,i,j,a_ud,a_u)
                    + H_el(n,i,j,a_dd,a_d) + H_el(n,i,j,a_d,a_dd))


eigvals, eigvecs = np.linalg.eig(H)
w = np.argsort(eigvals)
eigvals = eigvals[w]
eigvecs = eigvecs[:,w]

#------------------------------------------------------------------------------
# Computing density matrices and entanglement entropy
#------------------------------------------------------------------------------
k = 2       # Defining the sub-system size for calculating RDM
sub_sys = np.power(2,k)

# Reshaping the ground-state in order to form RDM
gs = np.reshape(eigvecs[:,0], (sub_sys,int(np.power(2,n)/sub_sys)))
#%%
rdm = np.dot(gs, np.transpose(gs))          # Forming the RDM

# Calculation of entanglement entropy
S = -1*np.trace(np.dot(rdm, spl.logm(rdm)))

#%%
# Checking the validity of the rdm. The trace of the rdm should give 1
P1 = np.trace(rdm)
# Checking if the rdm is a pure or mixed state
P2 = np.trace(np.dot(rdm,rdm))
