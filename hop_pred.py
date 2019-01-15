# Code to generate the generates precoders corresponding to 
# 100 channel evolutions for 10 independent channel realizations
# using both the time based and hopping based schemes 
# (i.e. fill 10 independent 100*`num_subcarriers` 
# time-frequency bins matrix with generated precoders), and
# save them as .npy files
#---------------------------------------------------------------------------
#Import Statements
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import itpp
from mimo_tdl_channel import *
import sys
import signal
import pdb
import time
import copy
from sumrate_BER import leakage_analysis, calculate_BER_performance_QPSK
# np.random.seed(81)
#---------------------------------------------------------------------------
#SIGINT handler
def sigint_handler(signal, frame):
    #Do something while breaking
    pdb.set_trace()
    sys.exit(0)
#---------------------------------------------------------------------------
#Lambda Functions
frob_norm= lambda A:np.linalg.norm(A, 'fro')
diff_frob_norm = lambda A,B:np.linalg.norm(A-B, 'fro')
#---------------------------------------------------------------------------
#Channel Params
signal.signal(signal.SIGINT, sigint_handler)
itpp.RNG_randomize()
#c_spec=itpp.comm.Channel_Specification(itpp.comm.CHANNEL_PROFILE.ITU_Vehicular_A)
c_spec=itpp.comm.Channel_Specification(itpp.comm.CHANNEL_PROFILE.ITU_Pedestrian_A)
Ts=5e-8
num_subcarriers=64
Nt=4
Nr=2
B=6
#---------------------------------------------------------------------------
#Codebook Variables
codebook_init=True
#Random Base Codewords Initialization. sHt_list stores codebook of Nt*Nt skew 
# hermitian marices
if(not codebook_init):
    mat_list=[2*np.matrix(np.random.rand(Nt,Nt)-np.ones((Nt,Nt))-1j*(2*np.random.rand(Nt,Nt)-np.ones((Nt,Nt)))) for i in range(2**B)]
    sH_list=[(mat-mat.H)/2 for mat in mat_list]
    if(Nt!=Nr):
        sHt_list=[]
        for mat in sH_list:
            mat[Nr:][:,np.arange(Nr,Nt)]=np.zeros((Nt-Nr,Nt-Nr))
            sHt_list.append(mat)
    else:
        sHt_list=sH_list
#VQ Based Initialisation
else:
    vec_list=np.load('./Codebooks/Pred_qt/base_quant_cb.npy')
    sHt_list=[vec_to_tangent(vec,Nt,Nr) for vec in vec_list]
#Standalone codebooks for setting initialization
qtCodebook=np.load('./Codebooks/Independent_qt/orth_cb_1000_20.npy')
sigma_cb=np.load('./Codebooks/Independent_qt/sigma_cb_2bits_10000.npy')
#---------------------------------------------------------------------------
# Simulation Parameters
number_simulations=100
feedback_mats=8
freq_jump=(num_subcarriers-1)//(feedback_mats-1)
freq_quanta=9
time_quanta=1
# for cold start
ran_init=False
# Number of channels simulated for
num_chan_realisations=10
count=0
chan_offset=10
fdts=1e-4
#---------------------------------------------------------------------------
# Run Time variables
# Stores the number of previously known estimates of the subcarriers
past_vals=3
# Hopping feedback list
fb_list=np.zeros((2*feedback_mats-1,past_vals,Nt,Nr),dtype=complex)
# Time based feedback list
onlyt_fb_list=np.zeros((feedback_mats,2*past_vals,Nt,Nr),dtype=complex)
# Stores the time instant each subcarrier was communicated
time_coef=np.zeros((2*feedback_mats-1,past_vals))
# Store the quantized values here
Q_error=np.zeros((number_simulations//2,2*feedback_mats-1),dtype=np.float64)
onlyt_Q_error=np.zeros((number_simulations,feedback_mats),dtype=np.float64)
# Use Stiefel Chordal Distance as norm
norm_fn='stiefCD'
start_time=time.time()
save=False
for chan_index in range(num_chan_realisations):
    print("-----------------------------------------------------------------------")
    print ("Starting Chan Realisation: "+str(chan_index)+" : of "+ str(num_chan_realisations) + " # of total channel realisations for "+str(fdts))
    print(str(time.strftime("Elapsed Time %H:%M:%S",time.gmtime(time.time()-start_time)))\
        +str(time.strftime(" Current Time %H:%M:%S",time.localtime(time.time()))))
    # Variables to store quantised U values for number_simulations
    allU=np.zeros((number_simulations,num_subcarriers,Nt,Nr),dtype=complex)
    onlyt_allU=np.zeros((number_simulations,num_subcarriers,Nt,Nr),dtype=complex)
    # Variables to store unquantised U and H values for number_simulations
    tH_allU=np.zeros((number_simulations,num_subcarriers,Nt,Nr),dtype=complex)
    tH_allH=np.zeros((number_simulations,num_subcarriers,Nt,Nr),dtype=complex)
    # Variables to store quantised sigma values
    interpS=np.zeros((number_simulations,num_subcarriers,Nr),dtype=complex)
    # Variables to store unquantised sigma values
    tHS=np.zeros((number_simulations,num_subcarriers,Nr),dtype=complex)
    # Generate Channels
    class_obj=MIMO_TDL_Channel(Nt,Nr,c_spec,Ts,num_subcarriers)
    class_obj.set_norm_doppler(fdts)
    # Roll variables to help for hopping code 
    roll_var=0
    skew_roll_var=0
    ot_roll_var=0
    #---------------------------------------------------------------------------
    # Main simulation loop for the algorithm 
    for simulation_index in range(number_simulations):
        # Print statement to indicate progress
        if(simulation_index%10==0):
            print ("Starting Gen sim: "+str(simulation_index)+" : of "+ str(number_simulations) + " # of total simulations")
        # Generate Channel matrices with the above specs
        class_obj.generate()
        # Get the channel matrix for num_subcarriers
        H_list=class_obj.get_Hlist()
        tH_allH[simulation_index]=H_list
        V_list, U_list,sigma_list= find_precoder_list(H_list,False)
        # if(simulation_index>=1):
        #     diff_err.append(np.mean([stiefCD(U_list[i],prev_Ulist[i])\
        #      for i in range(num_subcarriers)]))
        #     # print("Differential Channel Variation "+str(diff_err[-1]))
        prev_Ulist=U_list
        
        # Decide which roll variable to choose
        if(simulation_index%2==0):
            s_roll_var=roll_var
            o_roll_var=skew_roll_var
        else:
            s_roll_var=skew_roll_var
            o_roll_var=roll_var

        # Check if the system has been initialised 
        if(simulation_index>=2*past_vals):
            # Predict the appropriate feedback mat matrices 
            for pred_indice in range(feedback_mats):
                if((simulation_index%2==1) and (pred_indice==feedback_mats-1)):
                    onlyt_fb_index=pred_indice
                    onlyt_test_index=pred_indice*freq_jump
                    onlyt_rU=U_list[onlyt_test_index]
                    onlyt_prev_list=onlyt_fb_list[onlyt_fb_index][[(ot_roll_var-i-1)%(2*past_vals)\
                        for i in range(2*past_vals)]]
                    onlyt_predU=onlyT_pred(onlyt_prev_list[0],onlyt_prev_list)
                    onlyt_qtiz_err,onlyt_qtiz_U=qtisn(onlyt_predU,onlyt_rU,1.5,20,sHt_list,norm_fn,sk=0.0)
                    onlyt_Q_error[simulation_index-past_vals][onlyt_fb_index]= onlyt_qtiz_err
                    onlyt_fb_list[onlyt_fb_index][ot_roll_var]=onlyt_qtiz_U
                    onlyt_allU[simulation_index][onlyt_test_index]=onlyt_qtiz_U
                    interpS[simulation_index][test_index]=sigma_list[onlyt_test_index]            
                    continue

                fb_index=simulation_index%2+2*pred_indice
                onlyt_fb_index=pred_indice
                test_index=simulation_index%2*(freq_jump//2)+pred_indice*freq_jump
                onlyt_test_index=pred_indice*freq_jump
                rU=U_list[test_index]
                onlyt_rU=U_list[onlyt_test_index]
                #Populate the prev_list useful for prediction
                #Initiate with zeros
                prev_list=np.zeros((2*past_vals,Nt,Nr),dtype=complex)
                onlyt_prev_list=np.zeros((2*past_vals,Nt,Nr),dtype=complex)
                #Store Time indices here
                time_indices=np.zeros(2*past_vals)
                #Store Freq indices here
                freq_indices=np.zeros(2*past_vals)
                #---------------------------------------------------------------------------
                # Store time separated but same subcarrier 
                prev_list[0:past_vals]=fb_list[fb_index][[(s_roll_var-i-1)%past_vals\
                 for i in range(past_vals)]]
                onlyt_prev_list=onlyt_fb_list[onlyt_fb_index][[(ot_roll_var-i-1)%(2*past_vals)\
                 for i in range(2*past_vals)]]
                time_indices[0:past_vals]=time_coef[fb_index][[(s_roll_var-i-1)%past_vals\
                 for i in range(past_vals)]]
                freq_indices[0:past_vals]=np.array([0]*past_vals)
                
                center_index=0
                if(fb_index!=(2*feedback_mats-2)):
                    #---------------------------------------------------------------------------
                    # Store time separated and subcarrier separated here (+1) 
                    prev_list[past_vals:2*past_vals]=fb_list[fb_index+1][[(o_roll_var-i-1)%past_vals\
                     for i in range(past_vals)]]
                    time_indices[past_vals:2*past_vals]=time_coef[fb_index+1][[(o_roll_var-i-1)%past_vals\
                     for i in range(past_vals)]]
                    freq_indices[past_vals:2*past_vals]=np.array([1]*past_vals)
                    #---------------------------------------------------------------------------
                else:
                    #---------------------------------------------------------------------------
                    # Store time separated and subcarrier separated here (-1) 
                    prev_list[past_vals:2*past_vals]=fb_list[fb_index-1][[(o_roll_var-i-1)%past_vals\
                     for i in range(past_vals)]]
                    time_indices[past_vals:2*past_vals]=time_coef[fb_index-1][[(o_roll_var-i-1)%past_vals\
                     for i in range(past_vals)]]
                    freq_indices[past_vals:2*past_vals]=np.array([-1]*past_vals)
                    #---------------------------------------------------------------------------
                
                # Convert indices to multipliers
                freq_multipliers=(freq_indices)*freq_quanta
                time_multipliers=(time_indices-time_indices[center_index])*time_quanta
                t_mult=(simulation_index-time_indices[center_index])*time_quanta
                f_mult=0
                # Predict and store the values in pU array for post analysis
                predU,F,T=MMSE_tangent_predict(prev_list,time_multipliers,freq_multipliers,center_index,rU,t_mult,f_mult,past_vals,False)
                onlyt_predU=onlyT_pred(onlyt_prev_list[0],onlyt_prev_list)            
                # Quantize and store the values in qU array for post analysis
                qtiz_err,qtiz_U=qtisn(predU,rU,1.5,20,sHt_list,"stiefCD",sk=0.0)
                onlyt_qtiz_err,onlyt_qtiz_U=qtisn(onlyt_predU,onlyt_rU,1.5,20,sHt_list,"stiefCD",sk=0.0)

                Q_error[simulation_index//2-past_vals][fb_index]= qtiz_err
                onlyt_Q_error[simulation_index-past_vals][onlyt_fb_index]= onlyt_qtiz_err
                #Update information for subsequent predictions
                fb_list[fb_index][s_roll_var]=qtiz_U
                time_coef[fb_index][s_roll_var]=simulation_index
                onlyt_fb_list[onlyt_fb_index][ot_roll_var]=onlyt_qtiz_U
                #Store Quantized Values in All U for end 
                allU[simulation_index][test_index]=qtiz_U
                onlyt_allU[simulation_index][onlyt_test_index]=onlyt_qtiz_U
                interpS[simulation_index][test_index]=sigma_list[onlyt_test_index]
                

            if(simulation_index%2==1):
                # pdb.set_trace()
                print("---------------------------------------------------------------------------")
                print("Simulation Index: " +str(simulation_index))
                print("Hop QT Error: "+str(np.mean(Q_error[simulation_index//2-past_vals]))+\
                "Only T QT Error: "+str(np.mean(onlyt_Q_error[simulation_index-past_vals])))
                print(str(time.strftime("Elapsed Time %H:%M:%S",time.gmtime(time.time()-start_time)))\
                +str(time.strftime(" Current Time %H:%M:%S",time.localtime(time.time()))))        
        
        # Initialisation criteria
        else:
            for i in range(feedback_mats):
                fb_index=2*i+simulation_index%2
                test_index=simulation_index%2*(freq_jump//2)+i*freq_jump
                # In odd instances communicate only 31 matrices (middle)
                if((simulation_index%2==1) and (i==feedback_mats-1)):
                    onlyt_fb_index=i                
                    onlyt_test_index=freq_jump*i
                    # Initialise only time predictions
                    onlyt_oU=U_list[freq_jump*i]
                    if(not ran_init):
                        qU=qtCodebook[np.argmin([diff_frob_norm(onlyt_oU,codeword) for codeword in qtCodebook])]
                        onlyt_fb_list[i][ot_roll_var]=qU
                        onlyt_allU[simulation_index][onlyt_test_index]=qU
                    else:
                        rand_stmat=rand_stiefel(Nt,Nr)
                        onlyt_fb_list[i][ot_roll_var]=rand_stmat
                        onlyt_allU[simulation_index][onlyt_test_index]=rand_stmat
                    continue
                
                oU=U_list[test_index]
                # 
                if(not ran_init):
                    qU=qtCodebook[np.argmin([diff_frob_norm(oU,codeword) for codeword in qtCodebook])]
                    fb_list[fb_index][s_roll_var]=qU
                    allU[simulation_index][test_index]=qU
                else:
                    rand_stmat=rand_stiefel(Nt,Nr)
                    fb_list[fb_index][s_roll_var]=rand_stmat
                    allU[simulation_index][test_index]=rand_stmat
                
                time_coef[fb_index][s_roll_var]=simulation_index
                
                onlyt_fb_index=i                
                onlyt_test_index=freq_jump*i
                # Initialise only time predictions
                onlyt_oU=U_list[freq_jump*i]
                if(not ran_init):
                    qU=qtCodebook[np.argmin([diff_frob_norm(onlyt_oU,codeword) for codeword in qtCodebook])]
                    onlyt_fb_list[i][ot_roll_var]=qU
                    onlyt_allU[simulation_index][onlyt_test_index]=qU
                else:
                    rand_stmat=rand_stiefel(Nt,Nr)
                    onlyt_fb_list[i][ot_roll_var]=rand_stmat
                    onlyt_allU[simulation_index][onlyt_test_index]=rand_stmat
        
        # Update the roll variables
        tH_allU[simulation_index]=U_list
        tHS[simulation_index]=sigma_list
        ot_roll_var=(ot_roll_var+1)%(2*past_vals)
        if(simulation_index%2==0):
            roll_var=(roll_var+1)%past_vals
        else:
            skew_roll_var=(skew_roll_var+1)%past_vals
    # pdb.set_trace()
    # Save Channel Generation variables for BER and sumrate evaluation by eavl.py
    if(save==True):
        np.save('./Precoders_generated/Pedestrian/'+str(fdts)+'/th_allH_'+str(chan_index+chan_offset)+'.npy',tH_allH)
        np.save('./Precoders_generated/Pedestrian/'+str(fdts)+'/th_allU_'+str(chan_index+chan_offset)+'.npy',tH_allU)
        np.save('./Precoders_generated/Pedestrian/'+str(fdts)+'/thS_'+str(chan_index+chan_offset)+'.npy',tHS)
        np.save('./Precoders_generated/Pedestrian/'+str(fdts)+'/allU_'+str(chan_index+chan_offset)+'.npy',allU)
        np.save('./Precoders_generated/Pedestrian/'+str(fdts)+'/onlyt_allU_'+str(chan_index+chan_offset)+'.npy',onlyt_allU)
        np.save('./Precoders_generated/Pedestrian/'+str(fdts)+'/interpS_'+str(chan_index+chan_offset)+'.npy',interpS)

pdb.set_trace()

