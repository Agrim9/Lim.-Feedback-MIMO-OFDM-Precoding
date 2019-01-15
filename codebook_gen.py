#-----------------------------------------------------------------------------
# Code to obtain the base codebook: We consider 10,000 independent channel 
# evolutions. For each one, we use the evolution of the channel over 10 OFDM 
# frames’ duration to obtain a reliable prediction from both methods. 
# We collect the vectors representing the skew Hermitian tangents for vector 
# quantisation (recall that Cayley Exp. map has skew Hermitian matrices as images 
# (Section III of paper), which form a vector space). 
# This is done to get 10,000 independent predictions that give the same number 
# of independent vectors in the collection.  (Done in vec_collec_gen.py code)
# The code here applies k-means algorithm to get 
# a 6 bit base codebook for, for both the hopping based and time based schemes
#-----------------------------------------------------------------------------

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import itpp
from mimo_tdl_channel import *
from sklearn.cluster import KMeans
import sys
import signal
import pdb

#---------------------------------------------------------------------------
#Lambda Functions
frob_norm= lambda A:np.linalg.norm(A, 'fro')
diff_frob_norm = lambda A,B:np.linalg.norm(A-B, 'fro')
#---------------------------------------------------------------------------
#SIGINT handler
def sigint_handler(signal, frame):
    pdb.set_trace()
    sys.exit(0)

vec_list=np.load('./Codebooks/Pred_qt/base_quantcoll.npy')
vec_list_norm1=[vec/la.norm(vec) for vec in vec_list]
kmeans64=KMeans(n_clusters=64).fit(vec_list_norm1)
np.save('./Codebooks/Pred_qt/base_quant_cb.npy',kmeans64.cluster_centers_)
pdb.set_trace()