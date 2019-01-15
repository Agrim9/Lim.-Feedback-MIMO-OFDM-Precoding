#-----------------------------------------------------------------------------
# Code to generate codebooks for quantization of singular values
# Done via VQ performed over collection of sigma values
#-----------------------------------------------------------------------------

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import itpp
from mimo_tdl_channel import *
from sumrate_BER import leakage_analysis, calculate_BER_performance_QAM256
import precoder_interpolation
from sklearn.cluster import KMeans
import sys
import signal
import pdb
import copy
import time
np.random.seed(81)
#---------------------------------------------------------------------------
#Lambda Functions
frob_norm= lambda A:np.linalg.norm(A, 'fro')
diff_frob_norm = lambda A,B:np.linalg.norm(A-B, 'fro')
#---------------------------------------------------------------------------
#SIGINT handler
def sigint_handler(signal, frame):
    # print("Avg norm Error for new scheme: "+str(norm_err/count))
    # print("Avg norm Error for naive TD scheme"+str(norm_err_td/count))
    # print("Norm Improvements "+str(norm_impr/count))
    pdb.set_trace()
    sys.exit(0)
#---------------------------------------------------------------------------
#Channel Params
signal.signal(signal.SIGINT, sigint_handler)
itpp.RNG_randomize()
c_spec=itpp.comm.Channel_Specification(itpp.comm.CHANNEL_PROFILE.ITU_Vehicular_A)
#c_spec=itpp.comm.Channel_Specification(itpp.comm.CHANNEL_PROFILE.ITU_Pedestrian_A)
Ts=5e-8
num_subcarriers=1024
Nt=4
Nr=2
B=2
cb_size=2**B
class_obj=MIMO_TDL_Channel(Nt,Nr,c_spec,Ts,num_subcarriers)
number_simulations=10000
sigma_col=[]
for simulation_index in range(number_simulations):
    # Print statement to indicate progress
    if ((simulation_index%100)==0):
        print ("Starting sim: "+str(simulation_index)+" : of "+ str(number_simulations) + " # of total simulations")
    # Generate Channel matrices with the above specs
    class_obj.generate()
    # Get the channel matrix for num_subcarriers
    H_list=class_obj.get_Hlist()
    # Get the matrices
    for H_matrix in H_list:
        H_matrix=np.array(H_matrix)
        U, S, V = np.linalg.svd(H_matrix,full_matrices=0)
        sigma_col.append(S)

kmeans64=KMeans(n_clusters=cb_size).fit(sigma_col)
#np.save('./Results/18_09_18/Codebooks/sigma_cb_2bits_10000.npy',kmeans64.cluster_centers_)
pdb.set_trace()