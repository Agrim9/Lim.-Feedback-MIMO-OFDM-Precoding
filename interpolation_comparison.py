#-----------------------------------------------------------------------------
# Code to compare interpolation approaches (Caayley Exp. and Geodesic)
#-----------------------------------------------------------------------------

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import itpp
from mimo_tdl_channel import *
from sumrate_BER import waterfilling,leakage_analysis, calculate_BER_performance_QAM256, calculate_BER_performance_QPSK
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
class_obj=MIMO_TDL_Channel(Nt,Nr,c_spec,Ts,num_subcarriers)
number_simulations=1000
feedback_mats=32
# Why ((num_subcarriers-1)//(feedback_mats-1)) ?
# We have to cover indices upto (self.num_subcarriers-1) with (self.num_fb_points-1)
# Since, we have 0 as an implicit input
fb_indices=np.arange(feedback_mats)*((num_subcarriers-1)//(feedback_mats-1))
#----------------------------------------------------------------------
#Leakage arr
avg_quasigeodesic_leakage=np.zeros(num_subcarriers)
avg_geodesic_leakage=np.zeros(num_subcarriers)
count=0
Eb_N0_dB=np.arange(-10,24,4)
gdBER_QPSK=np.zeros(Eb_N0_dB.shape[0])
qgdBER_QPSK=np.zeros(Eb_N0_dB.shape[0])
use_cb=True
if(use_cb==True):
    orth_cb=np.load('./Codebooks/Independent_qt/orth_cb_1000_20.npy')
    full_cb=np.load('./Codebooks/Independent_qt/full_cb_1000_20.npy')
    sigma_cb=np.load('./Codebooks/Independent_qt/sigma_cb_2bits_10000.npy')
max_cap=2.09838799 #Find max_cap using max_cap.py
#---------------------------------------------------------------------------
# Main simulation loop for the algorithm 
for simulation_index in range(number_simulations):
    # Print statement to indicate progress
    if ((simulation_index%100)==0):
        print ("Starting sim: "+str(simulation_index)+" : of "+ str(number_simulations) + " # of total simulations")
    # Generate Channel matrices with the above specs
    class_obj.generate()
    # Get the channel matrix for num_subcarriers
    H_list=class_obj.get_Hlist()
    V_list, U_list,sigma_list,fU_list,fsigma_list,fV_list= find_precoder_list(H_list,True)
    Interpolator_obj=precoder_interpolation.Interpolator(1024,32)
    if(not use_cb):
        quasigeodesic_interpolated_U=np.array(Interpolator_obj.apply_interpolation(U_list,Nt,Nr,'quasiGeodesic'))
        geodesic_interpolated_U=np.array(Interpolator_obj.apply_interpolation(fU_list,Nt,Nr,'Geodesic'))
    else:
        quasigeodesic_interpolated_U=np.array(Interpolator_obj.apply_interpolation(U_list,Nt,Nr,'quasiGeodesic',qtCodebook=orth_cb,retQTnoise=False))
        geodesic_interpolated_U=np.array(Interpolator_obj.apply_interpolation(fU_list,Nt,Nr,'Geodesic',qtCodebook=full_cb,retQTnoise=False))
        interpolated_sigma=np.array(Interpolator_obj.apply_sigmainterp(sigma_list,Nr,qtCodebook=sigma_cb))


    #----------------------------------------------------------------------
    #Leakage analysis
    P=1
    wf_sigma_list=waterfilling(sigma_list.flatten(),P*num_subcarriers)
    wf_interp_S=waterfilling(interpolated_sigma.flatten(),P*num_subcarriers)
    quasigeodesic_leakage=leakage_analysis(H_list,U_list,quasigeodesic_interpolated_U,num_subcarriers,wf_sigma_list,wf_interp_S,Nt,Nr,ret_abs=True)
    geodesic_leakage=leakage_analysis(H_list,fU_list,geodesic_interpolated_U,num_subcarriers,wf_sigma_list,wf_interp_S,Nt,Nr,ret_abs=True)
    avg_geodesic_leakage=(count*avg_geodesic_leakage+geodesic_leakage)/(count+1)
    avg_quasigeodesic_leakage=(count*avg_quasigeodesic_leakage+quasigeodesic_leakage)/(count+1)
    print("Average Geodesic Leakage "+str(np.mean(avg_geodesic_leakage)/max_cap))
    print("Average quasiGeodesic Leakage "+str(np.mean(avg_quasigeodesic_leakage)/max_cap))
    
    #BER_geodesic_list_QPSK=numpy.zeros(Eb_N0_dB.shape[0])
    #BER_quasigeodesic_list_QPSK=numpy.zeros(Eb_N0_dB.shape[0])
    BER_geodesic_list_QPSK=np.zeros(Eb_N0_dB.shape[0])
    BER_quasigeodesic_list_QPSK=np.zeros(Eb_N0_dB.shape[0])
    for i in range(Eb_N0_dB.shape[0]):
        BER_geodesic_list_QPSK[i]=calculate_BER_performance_QPSK(np.array(H_list),geodesic_interpolated_U,Eb_N0_dB[i])
        BER_quasigeodesic_list_QPSK[i]=calculate_BER_performance_QPSK(np.array(H_list),quasigeodesic_interpolated_U,Eb_N0_dB[i])
        #print(BER_quasigeodesic_list_QAM[i])
    qgdBER_QPSK=(count*qgdBER_QPSK+BER_quasigeodesic_list_QPSK)/(count+1)
    gdBER_QPSK=(count*gdBER_QPSK+BER_geodesic_list_QPSK)/(count+1)
    print("QuasiGeo "+str(repr(qgdBER_QPSK)))
    print("Geo "+str(repr(gdBER_QPSK)))
    count=count+1
        

# Plot the Results
fb_indices=np.arange(feedback_mats)*((num_subcarriers-1)//(feedback_mats-1))
print("Average Geodesic Leakage "+str(np.mean(avg_geodesic_leakage)))
print("Average quasiGeodesic Leakage "+str(np.mean(avg_quasigeodesic_leakage)))
ax=plt.gca()
ax.xaxis.set_ticks(np.arange(0,num_subcarriers,50))
ax.grid(which='both', axis='both', linestyle='--')
ax.plot((avg_quasigeodesic_leakage/max_cap)*100)
ax.plot((avg_geodesic_leakage/max_cap)*100)
plt.xlabel('Subcarrier Frequency Index',size=25)
plt.ylabel("% of maximum capacity utilized at SNR=0dB",size=25)
ax.legend(['Cayley Exp. Interpolation','Unitary Geodesic Interpolation'],prop={'size': 20})
plt.xticks(fontsize=15)
plt.yticks(fontsize=20)
plt.show()
pdb.set_trace()
plt.plot(Eb_N0_dB,qgdBER_QAM)
plt.plot(Eb_N0_dB,gdBER_QAM)
plt.legend(['QuasiGeodesic','Geodesic'])
plt.show()
pdb.set_trace()