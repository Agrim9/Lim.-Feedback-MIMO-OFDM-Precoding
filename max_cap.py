from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import itpp
from mimo_tdl_channel import *
from sumrate_BER import max_leakage_analysis,waterfilling
import precoder_interpolation
import sys
import signal
import pdb
import copy
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
#c_spec=itpp.comm.Channel_Specification(itpp.comm.CHANNEL_PROFILE.ITU_Vehicular_A)
c_spec=itpp.comm.Channel_Specification(itpp.comm.CHANNEL_PROFILE.ITU_Pedestrian_A)
Ts=5e-8
num_subcarriers=64
Nt=4
Nr=2
B=6
class_obj=MIMO_TDL_Channel(Nt,Nr,c_spec,Ts,num_subcarriers)
number_simulations=1000
max_avg=np.zeros(7)
count=0
for simulation_index in range(number_simulations):
    # Print statement to indicate progress
    if ((simulation_index%20)==0):
        print ("Starting sim: "+str(simulation_index)+" : of "+ str(number_simulations) + " # of total simulations")
    # Generate Channel matrices with the above specs
    class_obj.generate()
    # Get the channel matrix for num_subcarriers
    H_list=class_obj.get_Hlist()
    V_list, U_list,sigma_list= find_precoder_list(H_list,False)

    curr_max=np.mean([max_leakage_analysis(H_list,U_list,waterfilling(sigma_list.flatten(),10**(SNR*0.1)*num_subcarriers),
        num_subcarriers,Nt,Nr) for SNR in 5*(np.arange(7))],axis=1)
    max_avg=(count*max_avg+curr_max)/(count+1)
    count=count+1

pdb.set_trace()