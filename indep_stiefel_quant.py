#-----------------------------------------------------------------------------
# Code to generate independent codebooks for quantization on Stiefel Manifold
# The LLoyd type algorithm to do so is taken from the paper
# "Joint Grassmann-Stiefel Quantization for MIMO Product Codebooks"
# By Renaud-Alexandre Pitaval and Olav Tirkkonen
# https://www.researchgate.net/publication/260722239_Joint_Grassmann-Stiefel_Quantization_for_MIMO_Product_Codebooks
#-----------------------------------------------------------------------------

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import itpp
from mimo_tdl_channel import *
from sumrate_BER import leakage_analysis, calculate_BER_performance_QAM256
import precoder_interpolation
import sys
import signal
import pdb
import copy
import time
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
B=6
cb_size=2**B
class_obj=MIMO_TDL_Channel(Nt,Nr,c_spec,Ts,num_subcarriers)
number_simulations=1000
#---------------------------------------------------------------------------
# Step 1: Random Initialization of Codebook
ran_init=False
if(ran_init):
    orth_codebook=[np.matrix(rand_stiefel(Nt,Nr)) for i in range(cb_size)]
    full_codebook=[np.matrix(unitary(Nt)) for i in range(cb_size)]
else:
    print("Initialised from pre existing codebooks")
    orth_codebook=np.load('./Results/17_09_18/Codebooks/orth_cb_1000_20.npy')
    full_codebook=np.load('./Results/17_09_18/Codebooks/full_cb_1000_20.npy')
orthU_coll=[]
fullU_coll=[]
num_iters=10
start_time=time.time()
for iter_index in range(num_iters):
    #---------------------------------------------------------------------------
    # Step 2: Generating matrices lying on the manifold to be quantized
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
            V = np.transpose(np.conjugate(V))
            newU=np.matmul(U,np.diag(np.exp(-1j*np.angle(U[0,:]))))
            orthU_coll.append(np.matrix(newU))
            U, S, V = np.linalg.svd(H_matrix,full_matrices=1)
            newU=np.matmul(U,np.diag(np.exp(-1j*np.angle(U[0,:]))))
            fullU_coll.append(np.matrix(newU))
    print("Starting Step 3 of codebook updation "+ str(time.strftime("Elapsed Time %H:%M:%S",time.gmtime(time.time()-start_time))))
    print(time.strftime("Current Time %H:%M:%S",time.localtime(time.time())))
    #---------------------------------------------------------------------------
    # Step 3: Nearest Neighbour Characterisation
    orthNN_list=[np.argmin([diff_frob_norm(orthU_coll[i],orth_codebook[j]) for j in range(cb_size)])\
                for i in range(len(orthU_coll))]
    fullNN_list=[np.argmin([diff_frob_norm(fullU_coll[i],full_codebook[j]) for j in range(cb_size)])\
                for i in range(len(fullU_coll))]
    #---------------------------------------------------------------------------
    # Step 4a: Computer Centeroid for each Voronoi Cell
    print("Starting Step 4 (Centroid) of codebook updation " + str(time.strftime("Elapsed Time %H:%M:%S",time.gmtime(time.time()-start_time))))
    count=np.zeros(cb_size)
    orthCOM=np.zeros((cb_size,Nt,Nr),dtype=complex)
    for i in range(len(orthU_coll)):
        orthCOM[orthNN_list[i]]=(count[orthNN_list[i]]*orthCOM[orthNN_list[i]]+np.matrix(orthU_coll[i]))/(count[orthNN_list[i]]+1)
        count[orthNN_list[i]]+=1
    count=np.zeros(cb_size)
    fullCOM=np.zeros((cb_size,Nt,Nt),dtype=complex)
    for i in range(len(fullU_coll)):
        fullCOM[fullNN_list[i]]=(count[fullNN_list[i]]*fullCOM[fullNN_list[i]]+np.matrix(fullU_coll[i]))/(count[fullNN_list[i]]+1)
        count[fullNN_list[i]]+=1
    orthCentroid=[np.matrix(la.polar(orthCOM[i])[0]) for i in range(cb_size)]
    fullCentroid=[np.matrix(la.polar(fullCOM[i])[0]) for i in range(cb_size)]
    #Finding closest grassmanaian plane to te centroid found
    orth_closestDgplane=[np.matrix(orth_codebook[np.argmin([chordal_dist(orthCentroid[i],orth_codebook[j]) for j in range(cb_size)])])\
                        for i in range(cb_size)]
    full_closestDgplane=[np.matrix(full_codebook[np.argmin([chordal_dist(fullCentroid[i],full_codebook[j]) for j in range(cb_size)])])\
                        for i in range(cb_size)]
    #Solving Procrutes problem to get the best rotation from grassmanian plane to stiefel manifold
    orth_procrutes_soln=[np.matrix(la.polar(orth_closestDgplane[i].H*orthCentroid[i])[0]) for i in range(cb_size)]
    full_procrutes_soln=[np.matrix(la.polar(full_closestDgplane[i].H*fullCentroid[i])[0]) for i in range(cb_size)]
    #Update codewords accordingly
    orth_codebook=[orth_codebook[i]*orth_procrutes_soln[i] for i in range(cb_size)]
    full_codebook=[full_codebook[i]*full_procrutes_soln[i] for i in range(cb_size)]
    print("End of Interation :" +str(iter_index)+ str(time.strftime(" Elapsed Time %H:%M:%S",time.gmtime(time.time()-start_time))))


pdb.set_trace()

# np.save('./Results/12_09_18/Interpolation_comparison/Vehicular, 3.5 1e-4/pUwb.npy',iU)
# np.save('./Results/12_09_18/Interpolation_comparison/Vehicular, 3.5 1e-4/U_mats.npy',U_mats)
# #np.save('./Results/12_09_18/Interpolation_comparison/Vehicular, 3.5 1e-4/qU_mats.npy',qU)
# np.save('./Results/12_09_18/Interpolation_comparison/Vehicular, 3.5 1e-4/Q_error.npy',interp_norm_error)
# #np.save('./Results/12_09_18/Interpolation_comparison/Vehicular, 3.5 1e-4/P_error.npy',quant_interp_norm_error)

'''
#-------------------------------------------------------------------------
# Single hops
time_dom_map=[np.arange(max(index-4,0),min(index+4,num_subcarriers))\
              for index in range(num_subcarriers)]
freq_dom_map=[np.arange(max(index-5,0),min(index+2,num_subcarriers))\
              for index in range(num_subcarriers)]
'''