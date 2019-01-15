# Code to generate the generates precoders corresponding to 
# 100 channel evolutions for 10 independent channel realizations
# using both the time based and hopping based schemes 
# (i.e. fill 10 independent 100*`num_subcarriers` 
# time-frequency bins matrix with generated precoders), and
# save them as .npy files
#---------------------------------------------------------------------------
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
from sumrate_BER import leakage_analysis, calculate_BER_performance_QPSK,calculate_BER_performance_QAM256,waterfilling
#SIGINT handler
def sigint_handler(signal, frame):
    #Do something while breaking
    pdb.set_trace()
    sys.exit(0)
#---------------------------------------------------------------------------
#Channel Params
signal.signal(signal.SIGINT, sigint_handler)
#number of db values for which logsum rate will be calculated
num_Cappar=7
avg_hpcap=np.zeros(num_Cappar)
avg_otcap=np.zeros(num_Cappar)
avg_maxcap=np.zeros(num_Cappar)
# Eb_N0_dB=np.arange(-6,6,2)
Eb_N0_dB=np.arange(-6,20,3)
#Store BER here
hpBER_QPSK=np.zeros(Eb_N0_dB.shape[0])
otBER_QPSK=np.zeros(Eb_N0_dB.shape[0])
Ts=5e-8
num_subcarriers=64
Nt=4
Nr=2
freq_quanta=4
time_quanta=1
#---------------------------------------------------------------------------
# Run Time variables
# Stores the previously known estimates of the subcarriers
past_vals=3
count=0
num_chan=10
feedback_mats=8
freq_jump=(num_subcarriers-1)//(feedback_mats-1)
fb_indices=freq_jump*np.arange(feedback_mats)
prev_indices=freq_jump*np.arange(feedback_mats-1)+freq_jump//2
net_fb_indices=np.sort(np.append(fb_indices,prev_indices))
interp_nbr_vals=2
steady_start_index=6
sil_BER=0
number_simulations=100
start_time=time.time()
sigma_cb=np.load('./Codebooks/Independent_qt/sigma_cb_2bits_10000.npy')
hp_Qerr=np.zeros(number_simulations-interp_nbr_vals-2)
ot_Qerr=np.zeros(number_simulations-interp_nbr_vals-2)
hp_Neterr=np.zeros(number_simulations-interp_nbr_vals-2)
ot_Neterr=np.zeros(number_simulations-interp_nbr_vals-2)
chan_offset=0
for chan_index in range(num_chan):

    tH_allH=np.load('./Precoders_generated/Pedestrian/3.5e-04/th_allH_'+str(chan_index+chan_offset)+'.npy')
    tH_allU=np.load('./Precoders_generated/Pedestrian/3.5e-04/th_allU_'+str(chan_index+chan_offset)+'.npy')
    tHS=np.load('./Precoders_generated/Pedestrian/3.5e-04/thS_'+str(chan_index+chan_offset)+'.npy')
    allU=np.load('./Precoders_generated/Pedestrian/3.5e-04/allU_'+str(chan_index+chan_offset)+'.npy')
    onlyt_allU=np.load('./Precoders_generated/Pedestrian/3.5e-04/onlyt_allU_'+str(chan_index+chan_offset)+'.npy')
    interpS=np.load('./Precoders_generated/Pedestrian/3.5e-04/interpS_'+str(chan_index+chan_offset)+'.npy')

    for simulation_index in range(0,number_simulations-interp_nbr_vals-2):
        print("---------------------------------------------------------------------------")
        print ("Starting Eval sim: "+str(simulation_index)+" : of "+ str(number_simulations) + " # of total simulations")
        print(str(time.strftime("Elapsed Time %H:%M:%S",time.gmtime(time.time()-start_time)))\
            +str(time.strftime(" Current Time %H:%M:%S",time.localtime(time.time()))))
        curr_fb_indices=np.arange(feedback_mats)*2 if simulation_index%2==0 else np.arange(feedback_mats-1)*2+1
        hp_qterr=np.mean([stiefCD(tH_allU[simulation_index][net_fb_indices[i]],allU[simulation_index][net_fb_indices[i]])\
            for i in curr_fb_indices])
        hp_Qerr[simulation_index]=(chan_index*hp_Qerr[simulation_index]+hp_qterr)/(chan_index+1)
        ot_qterr=np.mean([stiefCD(tH_allU[simulation_index][i],onlyt_allU[simulation_index][i])\
            for i in fb_indices])
        ot_Qerr[simulation_index]=(chan_index*ot_Qerr[simulation_index]+ot_qterr)/(chan_index+1)
        print("Qtisn Error: Hop -> "+str(hp_qterr) + " OnlyT -> "+str(ot_qterr))
        if(simulation_index>=steady_start_index-1):
            for indice_index in curr_fb_indices:
                #Check for extreme cases
                if(indice_index!=0 and indice_index!=(2*feedback_mats-2)):
                    time_index=simulation_index
                    freq_index=net_fb_indices[indice_index]
                    next_freq_index=net_fb_indices[indice_index+1]
                    prev_freq_index=net_fb_indices[indice_index-1]
                    center=allU[time_index][freq_index]
                    time_indices=np.zeros(2*interp_nbr_vals)
                    #Store Freq indices here
                    freq_indices=np.zeros(2*interp_nbr_vals)
                    #List on basis of which local maps will be computed
                    pred_list=np.zeros((2*interp_nbr_vals,Nt,Nr),dtype=complex)
                    # time_index+2, time_index+4 by hopping 2 times
                    pred_list[0:interp_nbr_vals]=allU[:,freq_index][time_index+2:time_index+2*interp_nbr_vals+2:2]
                    time_indices[0:interp_nbr_vals]=np.arange(2,interp_nbr_vals+2,2)*time_quanta
                    freq_indices[0:interp_nbr_vals]=np.zeros(interp_nbr_vals)*freq_quanta
                    # Get Left Freq map
                    pred_list[interp_nbr_vals:2*interp_nbr_vals]=allU[:,prev_freq_index][time_index+1:time_index+2*interp_nbr_vals+1:2]
                    time_indices[interp_nbr_vals:2*interp_nbr_vals]=np.arange(1,2*interp_nbr_vals+1,2)*time_quanta
                    freq_indices[interp_nbr_vals:2*interp_nbr_vals]=np.ones(interp_nbr_vals)*freq_quanta
                    #Get left frequency map
                    Tm,lF=get_tangent_maps(center,pred_list,time_indices,freq_indices)
                    for pred_freq_indice in range(freq_index - freq_jump//4,freq_index):
                        allU[time_index][pred_freq_indice]=sH_retract(center,lF*(freq_index - pred_freq_indice)/diff_freq)
                        allU[time_index+1][pred_freq_indice]=sH_retract(center,lF*(freq_index - pred_freq_indice)/diff_freq+Tm)
                    
                    # Make one time hop prediction
                    allU[time_index+1][freq_index]=sH_retract(center,Tm)
                        
                    #Get Right Frequency map
                    pred_list[interp_nbr_vals:2*interp_nbr_vals]=allU[:,next_freq_index][time_index+1:time_index+2*interp_nbr_vals+1:2]
                    Tm,rF=get_tangent_maps(center,pred_list,time_indices,freq_indices)
                    
                    for pred_freq_indice in range(freq_index+1,freq_index+freq_jump//4+simulation_index%2):
                        allU[time_index][pred_freq_indice]=sH_retract(center,rF*(pred_freq_indice-freq_index)/diff_freq)
                        allU[time_index+1][pred_freq_indice]=sH_retract(center,rF*(pred_freq_indice-freq_index)/diff_freq+Tm)


                else:
                    if(indice_index==0):
                        time_index=simulation_index
                        freq_index=net_fb_indices[indice_index]
                        next_freq_index=net_fb_indices[indice_index+1]
                        center=allU[time_index][freq_index]
                        
                        time_indices=np.zeros(2*interp_nbr_vals)
                        #Store Freq indices here
                        freq_indices=np.zeros(2*interp_nbr_vals)
                        #List on basis of which local maps will be computed
                        pred_list=np.zeros((2*interp_nbr_vals,Nt,Nr),dtype=complex)
                        pred_list[0:interp_nbr_vals]=allU[:,freq_index][time_index+2:time_index+2*interp_nbr_vals+2:2]
                        time_indices[0:interp_nbr_vals]=np.arange(2,2*interp_nbr_vals+2,2)*time_quanta
                        freq_indices[0:interp_nbr_vals]=np.zeros(interp_nbr_vals)*freq_quanta
                        pred_list[interp_nbr_vals:2*interp_nbr_vals]=allU[:,next_freq_index][time_index+1:time_index+2*interp_nbr_vals+1:2]
                        time_indices[interp_nbr_vals:2*interp_nbr_vals]=np.arange(1,2*interp_nbr_vals+1,2)*time_quanta
                        freq_indices[interp_nbr_vals:2*interp_nbr_vals]=np.ones(interp_nbr_vals)*freq_quanta
                        #Get Right Frequency map
                        Tm,rF=get_tangent_maps(center,pred_list,time_indices,freq_indices)
                        diff_freq=next_freq_index - freq_index
                        for pred_freq_indice in range(freq_index+1,freq_index+freq_jump//4+simulation_index%2):
                            allU[time_index][pred_freq_indice]=sH_retract(center,rF*(pred_freq_indice-freq_index)/diff_freq)
                            allU[time_index+1][pred_freq_indice]=sH_retract(center,rF*(pred_freq_indice-freq_index)/diff_freq+Tm)
                        # Make one time hop prediction
                        allU[time_index+1][freq_index]=sH_retract(center,Tm)

                    else:
                        time_index=simulation_index
                        freq_index=net_fb_indices[indice_index]
                        prev_freq_index=net_fb_indices[indice_index-1]
                        center=allU[time_index][freq_index]
                        
                        # Time Indices
                        time_indices=np.zeros(2*interp_nbr_vals)
                        #Store Freq indices here
                        freq_indices=np.zeros(2*interp_nbr_vals)
                        #List on basis of which local maps will be computed
                        pred_list=np.zeros((2*interp_nbr_vals,Nt,Nr),dtype=complex)
                        # Select time_index-4, time_index-2, time_index, time_index+2, time_index+4 by hopping 2 times
                        pred_list[0:interp_nbr_vals]=allU[:,freq_index][time_index+2:time_index+2*interp_nbr_vals+2:2]
                        time_indices[0:interp_nbr_vals]=np.arange(2,2*interp_nbr_vals+2,2)*time_quanta
                        freq_indices[0:interp_nbr_vals]=np.zeros(interp_nbr_vals)*freq_quanta
                        pred_list[interp_nbr_vals:2*interp_nbr_vals]=allU[:,prev_freq_index][time_index+1:time_index+2*interp_nbr_vals+1:2]
                        time_indices[interp_nbr_vals:2*interp_nbr_vals]=np.arange(1,2*interp_nbr_vals+1,2)*time_quanta
                        freq_indices[interp_nbr_vals:2*interp_nbr_vals]=np.ones(interp_nbr_vals)*freq_quanta
                        #Get left frequency map
                        Tm,lF=get_tangent_maps(center,pred_list,time_indices,freq_indices)
                        diff_freq=freq_index - prev_freq_index
                        for pred_freq_indice in range(freq_index - freq_jump//4,freq_index):
                            allU[time_index][pred_freq_indice]=sH_retract(center,lF*(freq_index - pred_freq_indice)/diff_freq)
                            allU[time_index+1][pred_freq_indice]=sH_retract(center,lF*(freq_index - pred_freq_indice)/diff_freq+Tm)
                        # Make one time hop prediction
                        allU[time_index+1][freq_index]=sH_retract(center,Tm)
                        

        if(simulation_index<=steady_start_index-1):
            if(simulation_index==0):
                for indice_index in range(fb_indices.shape[0]-1):
                    curr_freq_index=fb_indices[indice_index]
                    next_freq_index=fb_indices[indice_index+1]
                    diff_freq=next_freq_index - curr_freq_index
                    curr_U=allU[simulation_index][curr_freq_index]
                    next_U=allU[simulation_index][next_freq_index]
                    interpolatingT=sH_lift(curr_U,next_U)
                    allU[simulation_index][curr_freq_index+1:next_freq_index]=[sH_retract(curr_U,(t/diff_freq)*interpolatingT)\
                     for t in range(1,diff_freq)]
            else:
                for indice_index in range(0,net_fb_indices.shape[0]-1,2):
                    curr_freq_index=net_fb_indices[indice_index]
                    next_freq_index=net_fb_indices[indice_index+1]
                    diff_freq=next_freq_index - curr_freq_index
                    if(simulation_index%2==0):
                        curr_U=allU[simulation_index][curr_freq_index]    
                        next_U=allU[simulation_index-1][next_freq_index]
                        #Use the last value as well
                        allU[simulation_index][next_freq_index]=next_U
                    else:
                        curr_U=allU[simulation_index-1][curr_freq_index]    
                        next_U=allU[simulation_index][next_freq_index]
                        #Use the last value as well
                        allU[simulation_index][curr_freq_index]=curr_U

                    interpolatingT=sH_lift(curr_U,next_U)
                    allU[simulation_index][curr_freq_index+1:next_freq_index]=[sH_retract(curr_U,(t/diff_freq)*interpolatingT)\
                     for t in range(1,diff_freq)]
                    curr_freq_index=net_fb_indices[indice_index+1]
                    
                    next_freq_index=net_fb_indices[indice_index+2]
                    diff_freq=next_freq_index - curr_freq_index
                    if(simulation_index%2==0):
                        curr_U=allU[simulation_index-1][curr_freq_index]    
                        next_U=allU[simulation_index][next_freq_index]
                    else:
                        curr_U=allU[simulation_index][curr_freq_index]    
                        next_U=allU[simulation_index-1][next_freq_index]
                        #Use the last value as well
                        allU[simulation_index][next_freq_index]=next_U
                    
                    interpolatingT=sH_lift(curr_U,np.matrix(next_U))
                    allU[simulation_index][curr_freq_index+1:next_freq_index]=[sH_retract(curr_U,(t/diff_freq)*interpolatingT)\
                     for t in range(1,diff_freq)]
        
        for indice_index in range(fb_indices.shape[0]-1):
            curr_freq_index=fb_indices[indice_index]
            next_freq_index=fb_indices[indice_index+1]
            diff_freq=next_freq_index - curr_freq_index
            curr_U=onlyt_allU[simulation_index][curr_freq_index]
            next_U=onlyt_allU[simulation_index][next_freq_index]
            interpolatingT=sH_lift(curr_U,np.matrix(next_U))
            onlyt_allU[simulation_index][curr_freq_index+1:next_freq_index]=[sH_retract(curr_U,(t/diff_freq)*interpolatingT)\
             for t in range(1,diff_freq)]
            curr_S=interpS[simulation_index][curr_freq_index]
            next_S=interpS[simulation_index][next_freq_index]
            qcurr_S=sigma_cb[np.argmin([la.norm(curr_S-codeword) for codeword in sigma_cb])]
            qnext_S=sigma_cb[np.argmin([la.norm(next_S-codeword) for codeword in sigma_cb])]
            interpS[simulation_index][curr_freq_index+1:next_freq_index]=[(1-(t/diff_freq))*qcurr_S+(t/diff_freq)*qnext_S\
             for t in range(1,diff_freq)]
        
        
        
        ot_cap=[np.mean(leakage_analysis(tH_allH[simulation_index],tH_allU[simulation_index],\
            onlyt_allU[simulation_index],num_subcarriers,\
            waterfilling(tHS[simulation_index].flatten(),10**(0.1*p_dB)*num_subcarriers),\
            waterfilling(interpS[simulation_index].flatten(),10**(0.1*p_dB)*num_subcarriers),\
            Nt,Nr,ret_abs=True)) for p_dB in 5*np.arange(num_Cappar)]
        hp_cap=[np.mean(leakage_analysis(tH_allH[simulation_index],tH_allU[simulation_index],\
            allU[simulation_index],num_subcarriers,\
            waterfilling(tHS[simulation_index].flatten(),10**(0.1*p_dB)*num_subcarriers),\
            waterfilling(interpS[simulation_index].flatten(),10**(0.1*p_dB)*num_subcarriers),\
            Nt,Nr,ret_abs=True)) for p_dB in 5*np.arange(num_Cappar)]
        avg_otcap=(count*avg_otcap+np.array(ot_cap))/(count+1)
        avg_hpcap=(count*avg_hpcap+np.array(hp_cap))/(count+1)
        print("Avg. Onlyt Capacity " +str(repr(avg_otcap)))
        print("Avg. Freq Hopping Capacity "+str(repr(avg_hpcap)))
        hp_neterr=np.mean([stiefCD(tH_allU[simulation_index][i],allU[simulation_index][i])\
            for i in range(num_subcarriers)])
        hp_Neterr[simulation_index]=(chan_index*hp_Qerr[simulation_index]+hp_qterr)/(chan_index+1)
        ot_neterr=np.mean([stiefCD(tH_allU[simulation_index][i],onlyt_allU[simulation_index][i])\
            for i in range(num_subcarriers)])
        ot_Neterr[simulation_index]=(chan_index*ot_Qerr[simulation_index]+ot_qterr)/(chan_index+1)
        
        #----------------------------------------------------------------------
        #BER tests
        if(sil_BER==0):
            BER_onlyt_QPSK=np.zeros(Eb_N0_dB.shape[0])
            BER_freqhop_QPSK=np.zeros(Eb_N0_dB.shape[0])
            for i in range(Eb_N0_dB.shape[0]):
                BER_onlyt_QPSK[i]=calculate_BER_performance_QPSK(np.array(tH_allH[simulation_index]),onlyt_allU[simulation_index],Eb_N0_dB[i])
                BER_freqhop_QPSK[i]=calculate_BER_performance_QPSK(np.array(tH_allH[simulation_index]),allU[simulation_index],Eb_N0_dB[i])
                #print(BER_quasigeodesic_list_QAM[i])
            hpBER_QPSK=(count*hpBER_QPSK+BER_freqhop_QPSK)/(count+1)
            otBER_QPSK=(count*otBER_QPSK+BER_onlyt_QPSK)/(count+1)
            print("ot_pred = np."+str(repr(otBER_QPSK)))
            print("hp_pred = np."+str(repr(hpBER_QPSK)))
        
        count=count+1
    

pdb.set_trace()