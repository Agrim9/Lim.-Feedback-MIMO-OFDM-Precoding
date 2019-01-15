#The python script [sumrate\_BER.py](./sumrate_BER.py) 
#contains the codes used to obtain the BER and achievable rate

from __future__ import division
import numpy as np
import scipy.linalg as la
import pdb

diff_frobenius_norm = lambda A,B:np.linalg.norm(A-B, 'fro')

def waterfilling(alpha,P=1):
	n=alpha.size
	ind=np.argsort(alpha)
	lambda1=0
	for m in range(n):
		t=0
		for j in range(m,n):
			t=t+(1/alpha[ind[j]])
		t=(t+P)/(n-m)
		lambda1=1/t
		if(lambda1<alpha[ind[m]]):
			break
	result=[((1/lambda1)-(1/alpha[j])) if (lambda1<alpha[j]) else 0 for j in range(n)]
	return result

def leakage_analysis(H_list,U_list,interpolated_U,num_subcarriers,wf_sigma_list,wf_interp_S,Nt,Nr,ret_abs=False):
    fig_merit=np.zeros(num_subcarriers)
    subopt_cap=np.zeros(num_subcarriers)
    for i in range(num_subcarriers):
        H=np.matrix(H_list[i])
        U=np.matrix(U_list[i][:,np.arange(Nr)])
        U_tilde=np.matrix(interpolated_U[i])
        opt_vec=wf_sigma_list[Nr*i:Nr*(i+1)]
        interpopt_vec=wf_interp_S[Nr*i:Nr*(i+1)]
        #Get Maximal Capacity
        Q=U*np.diag(opt_vec)*U.H
        det_mat=np.identity(Nr)+H.H*Q*H
        max_cap=np.real(np.log(la.det(det_mat)))
        #Get Suboptimal Capacity
        Q_tilde=U_tilde*np.diag(interpopt_vec)*U_tilde.H
        det_mat=np.identity(Nr)+H.H*Q_tilde*H
        subopt_cap[i]=np.real(np.log(la.det(det_mat)))
        #print("SubOptCap also is: "+str(subopt_cap))
        fig_merit[i]=(subopt_cap[i]/max_cap)*100
    #print(norm_err)
    if(ret_abs):
        return subopt_cap
    else:
        return fig_merit

def max_leakage_analysis(H_list,U_list,wf_sigma_list,num_subcarriers,Nt,Nr,P=1):
    
    max_cap=np.zeros(num_subcarriers)
    
    for i in range(num_subcarriers):
        H=np.matrix(H_list[i])
        U=np.matrix(U_list[i][:,np.arange(Nr)])
        opt_vec=wf_sigma_list[i*Nr:(i+1)*Nr]
        #print("Max Capacity: "+str(max_cap))
        Q=U*np.diag(opt_vec)*U.H
        det_mat=np.identity(Nr)+H.H*Q*H
        max_cap[i]=np.real(np.log(la.det(det_mat)))

    return max_cap

def bin2conv(ctr):
    ret_val=np.zeros(2)
    if(ctr==0):
        return ret_val
    elif(ctr==1):
        ret_val[0]=1
        return ret_val
    elif(ctr==2):
        ret_val[1]=1
        return ret_val
    elif(ctr==3):
        ret_val[0]=1
        ret_val[1]=1
        return ret_val
    else:
        print("Error encountered in bin2conv")
    
def calculate_BER_performance_QPSK(H_sequence, precoder_sequence, Eb_N0_dB,flag=1):
    # Get H*Vp
    H_H_sequence=[np.matrix(H).H for H in H_sequence]
    H_F_sequence=np.matmul(H_H_sequence, precoder_sequence)
    # Get rows of H
    N_subcarrier=H_sequence.shape[0]
    # Num Bits and modulated stream
    Number_bits_each_stream=N_subcarrier*2*100
    num_comp_each_stream=N_subcarrier*100
    number_streams=precoder_sequence.shape[2]
    total_number_bits=number_streams*Number_bits_each_stream
    Eb_N0=10**(0.1*Eb_N0_dB)
    N0=1.0
    #Constellation_pts=np.array([((2*i+1)+1j*(2*j+1))*Eb_N0/16 for i in range(-8,8) for j in range(-8,8)])    
    Constellation_pts=np.array([-1-1j,-1+1j,1-1j,+1+1j])*(Eb_N0)/np.sqrt(2)
    rand_arr=np.random.randint(4,size=(1,number_streams*num_comp_each_stream)).flatten()
    QPSK_stream=np.array([2*Constellation_pts[rand_arr[i]] for i in range(rand_arr.shape[0])]).reshape(number_streams,num_comp_each_stream)
    QPSK_stream_to_bits=np.ravel([bin2conv(rand_arr[i])\
    for i in range(number_streams*num_comp_each_stream)
    ])
    
    #print(QAM_stream.shape)
    Es=2*(np.sum([np.abs(Constellation_pt) for Constellation_pt in Constellation_pts]))*N0
    #Chan Out
    channel_out=np.matmul(H_F_sequence, np.array(np.hsplit(QPSK_stream, N_subcarrier)))
    noise=(N0/2)*np.random.randn(channel_out.shape[0], channel_out.shape[1], channel_out.shape[2])+(1.0j)*(N0/2)*np.random.randn(channel_out.shape[0], channel_out.shape[1], channel_out.shape[2])
    channel_out_with_noise=channel_out+noise
    est_output=apply_MMSE_decoder(H_F_sequence, Es, channel_out_with_noise,N0)
    #MMSE
    #print(est_output.shape)
    flattened_op=est_output
    flattened_op=flattened_op.flatten()
    bins=[0]
    
    real_arr=np.digitize(flattened_op.real,bins)
    imag_arr=np.digitize(flattened_op.imag,bins)
    
    ml_estimate=np.ravel([np.array([imag_arr[k],real_arr[k]])\
         for k in range(number_streams*num_comp_each_stream)])
    error_rate=1-(np.float128(np.sum(ml_estimate==QPSK_stream_to_bits))/np.float128(ml_estimate.shape[0]))
    #print(error_rate)
    return error_rate

def calculate_BER_performance_QAM256(H_sequence, precoder_sequence, Eb_N0_dB,flag=1):
    # Get H*Vp
    H_H_sequence=[np.matrix(H).H for H in H_sequence]
    H_F_sequence=np.matmul(H_H_sequence, precoder_sequence)
    # Get rows of H
    N_subcarrier=H_sequence.shape[0]
    # Num Bits and modulated stream
    Number_bits_each_stream=N_subcarrier*8*100
    num_comp_each_stream=N_subcarrier*4*100
    number_streams=precoder_sequence.shape[2]
    total_number_bits=number_streams*Number_bits_each_stream
    Eb_N0=10**(0.1*Eb_N0_dB)
    N0=1.0
    Constellation_pts=np.array([((2*i+1)+1j*(2*j+1))*Eb_N0/la.norm([(2*i+1),(2*j+1)]) for i in range(-8,8) for j in range(-8,8)])    
    
    rand_arr=np.random.randint(256,size=(1,number_streams*num_comp_each_stream)).flatten()
    QAM_stream=np.array([Constellation_pts[rand_arr[i]] for i in range(rand_arr.shape[0])]).reshape(number_streams,num_comp_each_stream)
    qam_stream_to_bits=np.ravel([np.unpackbits(np.uint8(rand_arr[i]))\
    for i in range(number_streams*num_comp_each_stream)])
    
    #print(QAM_stream.shape)
    Es=8*(np.sum([np.abs(Constellation_pt) for Constellation_pt in Constellation_pts]))*N0
    #Chan Out
    channel_out=np.matmul(H_F_sequence, np.array(np.hsplit(QAM_stream, N_subcarrier)))
    noise=(N0/2)*np.random.randn(channel_out.shape[0], channel_out.shape[1], channel_out.shape[2])+(1.0j)*(N0/2)*np.random.randn(channel_out.shape[0], channel_out.shape[1], channel_out.shape[2])
    channel_out_with_noise=channel_out+noise
    est_output=apply_MMSE_decoder(H_F_sequence, Es, channel_out_with_noise,N0)
    #MMSE
    #print(est_output.shape)
    flattened_op=est_output
    flattened_op=flattened_op.flatten()
    bins=[2*i*Eb_N0/16 for i in range(-7,7)]
    
    real_arr=np.digitize(flattened_op.real,bins)-8
    imag_arr=np.digitize(flattened_op.imag,bins)-8
    
    ml_estimate=np.ravel([np.unpackbits(np.uint8((real_arr[k]+8)*16+(imag_arr[k]+8)))\
         for k in range(number_streams*num_comp_each_stream)])
    error_rate=1-(np.float128(np.sum(ml_estimate==qam_stream_to_bits))/np.float128(ml_estimate.shape[0]))
    #print(error_rate)
    return error_rate

def calculate_BER_performance_QAM16(H_sequence, precoder_sequence, Eb_N0_dB,flag=1):
    # Get H*Vp
    H_H_sequence=[np.matrix(H).H for H in H_sequence]
    H_F_sequence=np.matmul(H_H_sequence, precoder_sequence)
    # Get rows of H
    N_subcarrier=H_sequence.shape[0]
    # Num Bits and modulated stream
    Number_bits_each_stream=N_subcarrier*4*100
    num_comp_each_stream=N_subcarrier*2*100
    number_streams=precoder_sequence.shape[2]
    total_number_bits=number_streams*Number_bits_each_stream
    Eb_N0=10**(0.1*Eb_N0_dB)
    N0=1.0
    Constellation_pts=np.array([((2*i+1)+1j*(2*j+1))*Eb_N0/la.norm([(2*i+1),(2*j+1)]) for i in range(-2,2) for j in range(-2,2)])    
    
    rand_arr=np.random.randint(16,size=(1,number_streams*num_comp_each_stream)).flatten()
    QAM_stream=np.array([4*Constellation_pts[rand_arr[i]] for i in range(rand_arr.shape[0])]).reshape(number_streams,num_comp_each_stream)
    qam_stream_to_bits=np.ravel([np.unpackbits(np.uint8(rand_arr[i]))\
    for i in range(number_streams*num_comp_each_stream)])
    
    #print(QAM_stream.shape)
    Es=4*(np.sum([np.abs(Constellation_pt) for Constellation_pt in Constellation_pts]))*N0
    #Chan Out
    channel_out=np.matmul(H_F_sequence, np.array(np.hsplit(QAM_stream, N_subcarrier)))
    noise=(N0/2)*np.random.randn(channel_out.shape[0], channel_out.shape[1], channel_out.shape[2])+(1.0j)*(N0/2)*np.random.randn(channel_out.shape[0], channel_out.shape[1], channel_out.shape[2])
    channel_out_with_noise=channel_out+noise
    est_output=apply_MMSE_decoder(H_F_sequence, Es, channel_out_with_noise,N0)
    #MMSE
    #print(est_output.shape)
    flattened_op=est_output
    flattened_op=flattened_op.flatten()
    bins=[2*i*Eb_N0/4 for i in range(-1,1)]
    
    real_arr=np.digitize(flattened_op.real,bins)-8
    imag_arr=np.digitize(flattened_op.imag,bins)-8
    
    ml_estimate=np.ravel([np.unpackbits(np.uint8((real_arr[k]+8)*16+(imag_arr[k]+8)))\
         for k in range(number_streams*num_comp_each_stream)])
    error_rate=1-(np.float128(np.sum(ml_estimate==qam_stream_to_bits))/np.float128(ml_estimate.shape[0]))
    #print(error_rate)
    return error_rate

def apply_ZF_decoder(H_F_sequence, Es, input_sequence, N0=1.0):
    N_subcarriers=H_F_sequence.shape[0]
    H_F_h_sequence=np.conjugate(np.transpose(H_F_sequence, (0,2,1)))
    A_sequence=np.matmul(H_F_h_sequence, H_F_sequence)
    C_sequence=np.linalg.inv(A_sequence)
    G_sequence=np.matmul(C_sequence, H_F_sequence)
    output_sequence=np.matmul(G_sequence, input_sequence)
    return np.concatenate(output_sequence,axis=1)

def apply_MMSE_decoder(H_F_sequence, Es, input_sequence, N0=1.0):
    N_subcarriers=H_F_sequence.shape[0]
    H_F_h_sequence=np.conjugate(np.transpose(H_F_sequence, (0,2,1)))
    A_sequence=np.matmul(H_F_h_sequence, H_F_sequence)
    B_sequence=np.tile((input_sequence.shape[1])*(N0/Es)*np.eye(input_sequence.shape[1]), (N_subcarriers,1,1))
    C_sequence=np.linalg.inv(A_sequence+B_sequence)
    G_sequence=np.matmul(C_sequence, H_F_h_sequence)
    output_sequence=np.matmul(G_sequence, input_sequence)
    return np.concatenate(output_sequence,axis=1)

def get_MMSE(P,Vp,SNR,flag=1):
    N_subcarriers=P.shape[0]
    P_h=np.conjugate(np.transpose(P, (0,2,1)))
    A_sequence=np.matmul(P_h,P)
    B_sequence=np.identity(P.shape[2])/SNR
    C_sequence=np.linalg.inv(A_sequence)
    W=np.matmul(C_sequence, P_h)
    WP=np.matmul(W, P)
    if(flag==0):
        return W 
    return WP

def apply_ML_decoder(H_F_sequence,Es,input_sequence):
    for i in range(input_sequence.shape[0]):
        print "ML Decoding for Precoder #: "+ str(i)
        split_op=np.hsplit(input_sequence[i],input_sequence.shape[2])
        candidates=get_candidates(H_F_sequence[i],Es)
        ml_argmins=[np.argmin([la.norm(candidates[k]-split_op[j]) for k in range(256)]) for j in range(input_sequence.shape[2])]
        ml_bits=[(np.array(list(np.binary_repr(argmins).zfill(8))).astype(np.int8)) for argmins in ml_argmins]
        if(i==0):
            ret_bits=np.array(ml_bits)
        else:
            ret_bits=np.append(ret_bits,ml_bits)
        #print ret_bits.shape

    return ret_bits

def get_candidates(H,Es):
    candidates=np.zeros(shape=(256,4),dtype=np.complex128)
    for i in range(256):
        bit_stream=np.array(list(np.binary_repr(i).zfill(8))).astype(np.int8)
        bpsk_stream=2*bit_stream-1
        combined=np.hsplit(bpsk_stream,4)
        comp_vec=np.array([(combined[j][0]+1j*combined[j][1])*(Es/np.sqrt(2)) for j in range(4)])
        candidates[i]=np.matmul(H,comp_vec)
	return candidates
