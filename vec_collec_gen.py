#-----------------------------------------------------------------------------
# Code to obtain the base codebook: We consider 10,000 independent channel 
# evolutions. For each one, we use the evolution of the channel over 10 OFDM 
# frames’ duration to obtain a reliable prediction from both methods. 
# We collect the vectors representing the skew Hermitian tangents for vector 
# quantisation (recall that Cayley Exp. map has skew Hermitian matrices as images 
# (Section III of paper), which form a vector space). 
# This is done to get 10,000 independent predictions that give the same number 
# of independent vectors in the collection.  (Done in this code)
# The corresponding code in codebook_gen.py applies k-means algorithm to get 
# a 6 bit base codebook for, for both the hopping based and time based schemes
#-----------------------------------------------------------------------------

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import itpp
from mimo_tdl_channel import *
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
B=6
mat_list=[2*np.matrix(np.random.rand(Nt,Nt)-np.ones((Nt,Nt))-1j*(2*np.random.rand(Nt,Nt)-np.ones((Nt,Nt)))) for i in range(2**B)]
sH_list=[(mat-mat.H)/2 for mat in mat_list]
if(Nt!=Nr):
    sHt_list=[]
    for mat in sH_list:
        mat[Nr:][:,np.arange(Nr,Nt)]=np.zeros((Nt-Nr,Nt-Nr))
        sHt_list.append(mat)
else:
    sHt_list=sH_list
number_simulations=1024
feedback_mats=32
freq_quanta=16
time_quanta=1
# Run Time variables
# Stores the previously known estimates of the subcarriers
past_vals=3
#prev_list=np.zeros((num_subcarriers,Nt,Nr),dtype=complex)
fb_list=np.zeros((2*feedback_mats,past_vals,Nt,Nr),dtype=complex)
# Stores the time instant each subcarrier was communicated
time_coef=np.zeros((2*feedback_mats,past_vals))
vec_collection=[]
# Proxy count variable
count=0
num_preds=10000
noise=1
qtCodebook=np.load('./Codebooks/Independent/orth_cb_1000_20.npy')
start_time=time.time()
#---------------------------------------------------------------------------
# Main simulation loop for the algorithm
for preds in range(num_preds):
    # Print statement to indicate progress
    if ((preds%100)==0):
        print ("Starting sim: "+str(preds)+" : of "+ str(num_preds) + " # of total simulations")
        print(str(time.strftime("Elapsed Time %H:%M:%S",time.gmtime(time.time()-start_time)))\
            +str(time.strftime(" Current Time %H:%M:%S",time.localtime(time.time()))))
    class_obj=MIMO_TDL_Channel(Nt,Nr,c_spec,Ts,num_subcarriers)
    class_obj.set_norm_doppler(3.5*1e-4)
    # Rolling variable
    roll_var=0
    skew_roll_var=0
    for time_index in range(2*past_vals+1):
        # Generate Channel matrices with the above specs
        class_obj.generate()
        # Get the channel matrix for num_subcarriers
        H_list=class_obj.get_Hlist()
        # Take SVD to obtain U matrices
        V_list, U_list,sigma_list= find_precoder_list(H_list)
        # Set appropriate roll variable for the time index (even, odd)
        if(time_index%2==0):
            s_roll_var=roll_var
            o_roll_var=skew_roll_var
        else:
            s_roll_var=skew_roll_var
            o_roll_var=roll_var

        if(time_index==2*past_vals):
            # Predict the appropriate feedback mat matrices 
            for pred_indice in range(feedback_mats):     
                fb_index=time_index%2+2*pred_indice
                test_index=time_index%2*(feedback_mats//2)+pred_indice*feedback_mats
                rU=U_list[test_index]
                #Populate the prev_list useful for prediction
                if(fb_index!=(2*feedback_mats-1)):
                    #Initiate with zeros
                    prev_list=np.zeros((2*past_vals,Nt,Nr),dtype=complex)
                    #Store Time indices here
                    time_indices=np.zeros(2*past_vals)
                    #Store Freq indices here
                    freq_indices=np.zeros(2*past_vals)
                    #---------------------------------------------------------------------------
                    # Store time separated but same subcarrier 
                    prev_list[0:past_vals]=fb_list[fb_index][[(past_vals-i)\
                     for i in range(1,past_vals+1)]]
                    time_indices[0:past_vals]=time_coef[fb_index][[(past_vals-i)\
                     for i in range(1,past_vals+1)]]
                    freq_indices[0:past_vals]=np.array([0]*past_vals)
                    # !!!! Center arbitarily set to 0, should be adaptive in the final version !!!!
                    center_index=0
                    #---------------------------------------------------------------------------
                    # Store time separated and subcarrier separated here (+1) 
                    prev_list[past_vals:2*past_vals]=fb_list[fb_index+1][[(past_vals-i)\
                     for i in range(1,past_vals+1)]]
                    time_indices[past_vals:2*past_vals]=time_coef[fb_index+1][[(past_vals-i)\
                     for i in range(1,past_vals+1)]]
                    freq_indices[past_vals:2*past_vals]=np.array([1]*past_vals)
                    #---------------------------------------------------------------------------
                else:
                    #Initiate with zeros
                    prev_list=np.zeros((2*past_vals,Nt,Nr),dtype=complex)
                    #Store Time indices here
                    time_indices=np.zeros(2*past_vals)
                    #Store Freq indices here
                    freq_indices=np.zeros(2*past_vals)
                    #---------------------------------------------------------------------------
                    # Store time separated but same subcarrier 
                    prev_list[0:past_vals]=fb_list[fb_index][[(past_vals-i)\
                     for i in range(1,past_vals+1)]]
                    time_indices[0:past_vals]=time_coef[fb_index][[(past_vals-i)\
                     for i in range(1,past_vals+1)]]
                    freq_indices[0:past_vals]=np.array([0]*past_vals)
                    # !!!! Center arbitarily set to 0, should be adaptive in the final version !!!!
                    center_index=0
                    #---------------------------------------------------------------------------
                    # Store time separated and subcarrier separated here (-1) 
                    prev_list[past_vals:2*past_vals]=fb_list[fb_index-1][[(past_vals-i)\
                     for i in range(1,past_vals+1)]]
                    time_indices[past_vals:2*past_vals]=time_coef[fb_index-1][[(past_vals-i)\
                     for i in range(1,past_vals+1)]]
                    freq_indices[past_vals:2*past_vals]=np.array([-1]*past_vals)
                    #---------------------------------------------------------------------------

                
                # Convert indices to multipliers
                freq_multipliers=(freq_indices)*freq_quanta
                time_multipliers=(time_indices-time_indices[center_index])*time_quanta
                t_mult=(time_index-time_indices[center_index])*time_quanta
                f_mult=0
                # Predict and store the values in pU array for post analysis
                predU,F,T=MMSE_tangent_predict(prev_list,time_multipliers,freq_multipliers,center_index,rU,t_mult,f_mult,past_vals,False)
                #pdb.set_trace()
                rT,vecrT=sH_lift(predU,rU,True)
                # Store in vec_collec
                vec_collection.append(vecrT/la.norm(vecrT))
                
        # Initialisation criteria
        else:
            for i in range(feedback_mats):
                oU=U_list[feedback_mats*i+time_index%2*(feedback_mats//2)]
                qU=qtCodebook[np.argmin([diff_frob_norm(oU,codeword) for codeword in qtCodebook])]
                fb_list[2*i+time_index%2][s_roll_var]=qU
                time_coef[2*i+time_index%2][s_roll_var]=time_index
        
        if(time_index%2==0):
            roll_var=(roll_var+1)%past_vals
        else:
            skew_roll_var=(skew_roll_var+1)%past_vals


pdb.set_trace()
