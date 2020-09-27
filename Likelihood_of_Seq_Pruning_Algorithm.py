import numpy as np
from scipy.linalg import expm
from copy import deepcopy

def get_K2P_Q_matrix(a, b):
    # order: T, C, A, G
    Q = np.array([[-(a+2*b), a, b, b],
                  [a, -(a+2*b), b, b],
                  [b, b, -(a+2*b), a],
                  [b, b, a, -(a+2*b)]])
    return Q

def get_P_t_matrix_from_Q_matrix(Q, t):
    P_t = expm(Q*t)
    return P_t

def get_p_t_dictionary_from_P_t_matrix(P_t):
    p_t_dict = {"pTT":P_t[0][0],"pTC":P_t[0][1],"pTA":P_t[0][2],"pTG":P_t[0][3],
                "pCT":P_t[1][0],"pCC":P_t[1][1],"pCA":P_t[1][2],"pCG":P_t[1][3],
                "pAT":P_t[2][0],"pAC":P_t[2][1],"pAA":P_t[2][2],"pAG":P_t[2][3],
                "pGT":P_t[3][0],"pGC":P_t[3][1],"pGA":P_t[3][2],"pGG":P_t[3][3]}

def Likelihood_of_site_h(x_h, tree_dict, Q, pi):
    n_nodes = n_seq + len(tree_dict)
    L_vectors_h = np.zeros((n_nodes, 4))
    # L_vectors of tip nodes
    for node_idx in range(n_seq):
        tip_nt = x_h[node_idx]
        if tip_nt == "T": L_vectors_h[node_idx] = [1, 0, 0, 0]
        if tip_nt == "C": L_vectors_h[node_idx] = [0, 1, 0, 0]
        if tip_nt == "A": L_vectors_h[node_idx] = [0, 0, 1, 0]
        if tip_nt == "G": L_vectors_h[node_idx] = [0, 0, 0, 1]
    # L_vectors of internal nodes
    temp_tree = deepcopy(tree_dict)
    while len(temp_tree) != 0:
        node_list = [ ]
        for node_i in temp_tree:
            ((node_j, t_j), (node_k, t_k)) = temp_tree[node_i]
            check_sum = np.sum(L_vectors_h[node_j])*np.sum(L_vectors_h[node_k])
            if check_sum == 0:
                pass
            else:
                L_vector_j = L_vectors_h[node_j].reshape((4,1))
                L_vector_k = L_vectors_h[node_k].reshape((4,1))
                P_t_j = get_P_t_matrix_from_Q_matrix(Q, t_j)
                P_t_k = get_P_t_matrix_from_Q_matrix(Q, t_k)
                sum_vector_for_j = np.dot(P_t_j, L_vector_j)
                sum_vector_for_k = np.dot(P_t_k, L_vector_k)
                L_vector_i = sum_vector_for_j * sum_vector_for_k
                L_vector_i = L_vector_i.reshape((4,))
                L_vectors_h[node_i] = L_vector_i
                node_list.append(node_i)
        for node in node_list:
            del temp_tree[node]
    # Likelihood of site h
    root_node_idx = n_nodes - 1
    Likelihood_h = np.dot(pi,L_vectors_h[root_node_idx])
    #print()
    #print(L_vectors_h)
    #print()
    #print("Likelihood of site h:", Likelihood_h)
    #print("log-Likelihood of site h:", np.log(Likelihood_h))
    return Likelihood_h

# === Define Substitution Model
a  = 2 /4  # K2P alpha; then t = d
b  = 1 /4  # K2P beta;  then t = d
Q  = get_K2P_Q_matrix(a,b)          # rate matrix
pi = np.array([1/4, 1/4, 1/4, 1/4]) # stationary distribution

# === Define X (sequence) and Tree
X  = np.array([["T", "C", "T"],  # each row is a sequence
               ["C", "C", "C"],
               ["A", "C", "A"],
               ["C", "C", "C"],
               ["C", "C", "T"]])
             # Each row is a sequence
tree_dict = {5:((6,0.1),(2,0.2)),
             6:((0,0.2),(1,0.2)),
             7:((3,0.2),(4,0.2)),
             8:((5,0.1),(7,0.1))}
             # internal nodes only
             # rooted tree; root node => last node!!!
n_seq = X.shape[0]
print("n_seq:", n_seq)
print("n_internal_nodes:", len(tree_dict))

# === Main
L = X.shape[1]
C = X.T
# x_0 = ["T", "C", "A", "C", "C"]
# observed nucleotides at tip nodes.
# Note that seq #1's node idx will be zero!!!
l_h = [ ] # list of log likelihood
for h in range(L):
    x_h = C[h]
    L_h = Likelihood_of_site_h(x_h, tree_dict, Q, pi)
    l_h = np.log(L_h)
    # For the calculation of the likelihood of sequence,
    # always use log_likelihood.
l = np.sum(l_h)
print("Log-likelihood of seq:", l)
