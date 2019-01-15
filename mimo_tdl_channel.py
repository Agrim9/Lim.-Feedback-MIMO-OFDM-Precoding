# This python script contains codes for channel generation, lifting 
# and precoder prediction

from __future__ import division
import itpp
import numpy as np
import scipy.linalg as la
import pdb


def unitary(n):
    X=(np.random.rand(n,n)+1j*np.random.rand(n,n))/np.sqrt(2)
    [Q,R]=np.linalg.qr(X)
    T=np.diag(np.diag(R)/np.abs(np.diag(R)))
    U=np.matrix(np.matmul(Q,T))
    # Verify print (np.matmul(U,U.H))
    return U    
    
def rand_stiefel(n,p):
    H=(np.random.rand(n,p)+1j*np.random.rand(n,p))/np.sqrt(2)
    U, S, V = np.linalg.svd(H,full_matrices=0)
    return U

def rand_SH(n):
    X=2*np.random.rand(n,n)-np.ones((n,n))+1j*(2*np.random.rand(n,n)-np.ones((n,n)))
    A=np.matrix(X-X.T.conjugate())/2
    # Verify print(A-A.H)
    return A

def qtisn(pU,rU,g,num_iter,sH_list,norm_fn,sk=0.0):
    diff_frob_norm = lambda A,B:np.linalg.norm(A-B, 'fro')
    trials=0
    sp=0.0
    sm=0.0
    while(trials<num_iter):
        sp=g**(np.min(sk+1,0))
        sm=g**(sk-1)
        # Qp=[np.matmul(la.expm(sp*sH),pU) for sH in sH_list]
        # Qm=[np.matmul(la.expm(sm*sH),pU) for sH in sH_list]
        Qp=[sH_retract(pU,sp*sH) for sH in sH_list]
        Qm=[sH_retract(pU,sm*sH) for sH in sH_list]
        Q=Qp+Qm
        if(norm_fn=="diff_frob_norm"):
            chordal_dists=np.array([diff_frob_norm(Q[i],rU) for i in range(len(Q))])
        else:
            chordal_dists=np.array([stiefCD(Q[i],rU) for i in range(len(Q))])

        trials=trials+1
        if(np.argmin(chordal_dists)<len(Q)//2):
            sk=np.min(sk+1,0)
        else:
            sk=sk-1
    #print("Qtisn error: "+str(np.min(chordal_dists)))
    return np.min(chordal_dists),Q[np.argmin(chordal_dists)]

def MMSE_tangent_predict(prev_list,a_arr,b_arr,center_index,rU,t_mult,f_mult,past_vals,bias=True,posterior=True):
    cent_mats=[prev_list[point] for point in range(past_vals) if(point!=center_index)]
    init_center=prev_list[center_index]
    center=it_avg_center(init_center,cent_mats,rU,posterior)
    sigma_ai=np.sum(a_arr)
    sigma_bi=np.sum(b_arr)
    num=a_arr.shape[0]
    sigma_ai2=np.sum(a_arr**2)
    sigma_bi2=np.sum(b_arr**2)
    sigma_aibi=np.sum(a_arr*b_arr)
    if(bias):
        mat=np.matrix([[sigma_ai,sigma_bi,num],[sigma_ai2,sigma_aibi,sigma_ai],[sigma_aibi,sigma_bi2,sigma_bi]])
        invA=la.inv(mat)
        Tangent_list=[lift(center,prev_list[point]) for point in range(len(prev_list))]
        vec=[]
        vec.append(np.sum(Tangent_list,axis=0))
        vec.append(np.sum([a_arr[i]*Tangent_list[i] for i in range(len(Tangent_list))],axis=0))
        vec.append(np.sum([b_arr[i]*Tangent_list[i] for i in range(len(Tangent_list))],axis=0))
        T1=np.sum([invA[0][i]*vec[i] for i in range(3)],axis=0)
        T2=np.sum([invA[1][i]*vec[i] for i in range(3)],axis=0)
        T0=np.sum([invA[2][i]*vec[i] for i in range(3)],axis=0)
        T_pred=t_mult*T1+f_mult*T2+T0
        pU=retract(center,T_pred)
        return pU,T0,T1,T2
    else:
        mat=np.matrix([[sigma_ai2,sigma_aibi],[sigma_aibi,sigma_bi2]])
        invA=la.inv(mat)
        Tangent_list=[lift(center,prev_list[point]) for point in range(len(prev_list))]
        vec=[]
        vec.append(np.sum([a_arr[i]*Tangent_list[i] for i in range(len(Tangent_list))],axis=0))
        vec.append(np.sum([b_arr[i]*Tangent_list[i] for i in range(len(Tangent_list))],axis=0))
        T1=np.sum([invA[0][i]*vec[i] for i in range(2)],axis=0)
        T2=np.sum([invA[1][i]*vec[i] for i in range(2)],axis=0)
        T_pred=t_mult*T1+f_mult*T2
        pU=retract(center,T_pred)
        return pU,T2,T1

def get_tangent_maps(center,pred_list,a_arr,b_arr):
    sigma_ai=np.sum(a_arr)
    sigma_bi=np.sum(b_arr)
    num=a_arr.shape[0]
    sigma_ai2=np.sum(a_arr**2)
    sigma_bi2=np.sum(b_arr**2)
    sigma_aibi=np.sum(a_arr*b_arr)
    mat=np.matrix([[sigma_ai2,sigma_aibi],[sigma_aibi,sigma_bi2]])
    invA=la.inv(mat)
    Tangent_list=[sH_lift(center,pred_list[point]) for point in range(len(pred_list))]
    vec=[]
    vec.append(np.sum([a_arr[i]*Tangent_list[i] for i in range(len(Tangent_list))],axis=0))
    vec.append(np.sum([b_arr[i]*Tangent_list[i] for i in range(len(Tangent_list))],axis=0))
    T=np.sum([invA[0][i]*vec[i] for i in range(2)],axis=0)
    F=np.sum([invA[1][i]*vec[i] for i in range(2)],axis=0)
    return T,F

def onlyT_pred(center,pred_list):
    num_iter=20
    i=0
    Np=pred_list.shape[0]
    while (i<num_iter):
        tangent_list=np.array([lift(center,manifold_pt) for manifold_pt in pred_list])
        new_tangent=np.sum(tangent_list,axis=0)/Np
        center=retract(center,new_tangent)
        i+=1
    tangent_list=np.array([lift(center,manifold_pt) for manifold_pt in pred_list])
    sum_itp=np.sum(np.array([(i+1)*tangent_list[i] for i in range(Np)]),axis=0)
    sum_tp=np.sum(tangent_list,axis=0)
    T_1=((-sum_itp+((1+Np)/2)*sum_tp)*12)/((Np-1)*(Np**2+Np))
    T_0=(sum_tp-((Np-1)*Np/2)*T_1)/Np
    pU=retract(center,T_0+Np*T_1)
    return pU


def it_avg_center(center,prev_list,realU,posterior=True):
    num_iter=50
    i=0
    Np=len(prev_list)
    init_CD=chordal_dist(center,realU)
    init_center=center
    while (i<num_iter):
        # if(i%49==0):
        #        print("Iteration: "+ str(i)+ " CD: "+str(chordal_dist(center,realU)))
        if(i==49):
            final_CD=chordal_dist(center,realU)
        tangent_list=np.array([lift(center,manifold_pt) for manifold_pt in prev_list])
        new_tangent=np.sum(tangent_list,axis=0)/Np
        center=retract(center,new_tangent)
        i+=1
    if((init_CD<final_CD) and posterior):
        center=init_center
    # else:
    #     print("Sucessfull avging")
    return center 


def euc_inn(T1,T2):
    T1_mat=np.matrix(T1)
    T2_mat=np.matrix(T2)
    return np.trace(T1_mat.H*T2_mat)

def predict(alpha,beta,center,T1,T2,realU):
    diff_frob_norm = lambda A,B:np.linalg.norm(A-B, 'fro')
    T=(np.cos(alpha)*T1+np.cos(beta)*T2)
    predU=retract(center,T)
    return diff_frob_norm(predU,realU)

def skew_lift(A,B):
    p=A.shape[0]
    n=A.shape[1]
    A_mat=np.matrix(A)
    B_mat=np.matrix(B)
    T=0.5*A_mat*(A_mat.H*B_mat-B_mat.H*A_mat)
    return T

def lift(A,B):
    p=A.shape[0]
    n=A.shape[1]
    A_mat=np.matrix(A)
    B_mat=np.matrix(B)
    T=(np.identity(p)-A_mat*A_mat.H)*B_mat+0.5*A_mat*(A_mat.H*B_mat-B_mat.H*A_mat)
    return T

def retract(A,V):
    p=A.shape[0]
    n=A.shape[1]
    A_mat=np.matrix(A)
    V_mat=np.matrix(V)
    S=(A_mat+V_mat)*la.sqrtm(la.inv((np.identity(n)+V_mat.H*V)))
    return S


def sH_lift(A,B,ret_vec=False):
    p=A.shape[1]
    n=A.shape[0]
    X=np.matrix(A)
    Y=np.matrix(B)
    Xu=X[0:p]
    Xl=X[p:n]
    Yu=Y[0:p]
    Yl=Y[p:n]
    C=2*la.inv((Xu.H+Yu.H))*skew(Yu.H*Xu+Xl.H*Yl)*la.inv(Xu+Yu)
    B=(Yl-Xl)*la.inv(Xu+Yu)
    T=np.bmat([[C,-B.H],[B,np.zeros((n-p,n-p))]])
    cvecC=np.array(C[np.triu_indices(C.shape[0],1)]).flatten()
    rvecC=np.append(np.imag(np.diagonal(C)),np.append(np.imag(cvecC),np.real(cvecC)))
    cvecB=np.squeeze(np.asarray(np.reshape(B,(1,(n-p)*p))))
    rvecB=np.append(np.imag(cvecB),np.real(cvecB))
    vecT=np.append(rvecC,rvecB)
    if(ret_vec):
        return T,vecT
    else:
        return T

def vec_to_tangent(vec,n,p):
    C=np.diag(1j*vec[:p])
    C[np.triu_indices(C.shape[0],1)]=1j*vec[p:p+(p*(p-1)/2)]+vec[p+(p*(p-1)/2):p+p*(p-1)]
    C[np.tril_indices(C.shape[0],-1)]=1j*vec[p:p+(p*(p-1)/2)]-vec[p+(p*(p-1)/2):p+p*(p-1)]
    vec_recon=1j*vec[p+p*(p-1):p+p*(p-1)+p*(n-p)]+vec[p+p*(p-1)+p*(n-p):p+p*(p-1)+2*p*(n-p)]
    B=np.matrix(np.reshape(vec_recon,(n-p,p)))
    T=np.bmat([[C,-B.H],[B,np.zeros((n-p,n-p))]])
    return T

def sH_retract(A,B):
    p=A.shape[1]
    n=A.shape[0]
    X=np.matrix(A)
    W=np.matrix(B)
    Cay_W=(np.identity(n)+W)*la.inv(np.identity(n)-W)
    un_normQT= Cay_W*X
    norm_Qt= un_normQT/la.norm(un_normQT,axis=0)
    return norm_Qt

def skew(A):
    A_mat=np.matrix(A)
    return 0.5*(A.H-A)

def Ds_metric(A,B):
    A_mat=np.matrix(A)
    B_mat=np.matrix(B)
    return A.shape[1]-np.trace(np.square(np.abs(A_mat.H*B_mat)))

def chordal_dist(A,B):
    A_mat=np.matrix(A)
    B_mat=np.matrix(B)
    return np.sqrt(np.abs(A.shape[1]-np.linalg.norm(A_mat.H*B_mat,'fro')**2))

def grassCD_2(A,B):
    C=np.vdot(A,B)
    rho=np.real(C*np.conjugate(C))
    return (1-rho)

def stiefCD(A,B):
    A_mat=np.matrix(A)
    B_mat=np.matrix(B)
    CD_2=np.sum([grassCD_2(np.array(A_mat[:,i]),np.array(B_mat[:,i])) for i in range(A_mat.shape[1])])
    if(CD_2<0):
        pdb.set_trace()
        return -1
    return np.sqrt(CD_2)


def find_orientation_matrix(U0, U1):
    A_current= np.diag(np.matmul(U1.T.conj(),U0))
    orientaion_matrix=np.diag((A_current/np.absolute(A_current)))
    return orientaion_matrix
# def chordal_dist(A,B):
#     A_mat=np.matrix(A)
#     B_mat=np.matrix(B)
#     return np.linalg.norm(A_mat*A_mat.H-B_mat*B_mat.H, 'fro')/2**0.5

def vec_corr(A,B):
    num_elem=A.shape[0]*A.shape[1]
    A_vec=np.squeeze(np.reshape(A,(num_elem,1)))
    B_vec=np.squeeze(np.reshape(B,(num_elem,1)))
    return np.abs(np.sum(A_vec*np.conjugate(B_vec)))

def find_precoder_list(H_list,ret_full=False):
    V_list = []
    U_list = []
    sigma_list=[]
    if(ret_full):
        fV_list=[]
        fU_list=[]
        fsigma_list=[]
    for H_matrix in H_list:
        H_matrix=np.array(H_matrix)
        U, S, V = np.linalg.svd(H_matrix,full_matrices=0)
        V = np.transpose(np.conjugate(V))
        V_list.append(np.matrix(V))
        newU=np.matmul(U,np.diag(np.exp(-1j*np.angle(U[0,:]))))
        U_list.append(np.matrix(newU))
        sigma_list.append(S)
        if(ret_full):
            U, S, V = np.linalg.svd(H_matrix,full_matrices=1)
            V = np.transpose(np.conjugate(V))
            newU=np.matmul(U,np.diag(np.exp(-1j*np.angle(U[0,:]))))
            fU_list.append(np.matrix(newU))
            fsigma_list.append(S)
            fV_list.append(V)
    if(ret_full):
        return np.array(V_list),np.array(U_list),np.array(sigma_list),np.array(fU_list),np.array(fsigma_list),np.array(fV_list)
    else:
        return np.array(V_list),np.array(U_list),np.array(sigma_list)
# Class for generating a MIMO TDL channel
class MIMO_TDL_Channel():
    # Class Constructor
    def __init__(self,Nt,Nr,c_spec,sampling_time,Num_subcarriers):
        # Num Tx antenna
        self.Nt=Nt
        # Num Rx antenna
        self.Nr=Nr
        # Sampling Time
        self.sampling_time = sampling_time
        # FFT size will be the number of subcarriers in OFDM frame
        self.fft_size = Num_subcarriers
        # Declare Nt*Nr channels        
        self.tdl_channels=[]
        # Initialise using the TDL Channel method to get the channel length
        # Delay profile in cspec will be discretized with sampling_time in seconds
        channel = itpp.comm.TDL_Channel(c_spec, sampling_time)
        self.channel_length=channel.taps()
        # Initialise the Nt*Nr TDL channels
        for i in range(self.Nt*self.Nr):
            self.tdl_channels.append(itpp.comm.TDL_Channel(c_spec,sampling_time))

    # Set Doppler for each Nt*Nr channel
    def set_norm_doppler(self,norm_doppler):
        for i in range(self.Nt*self.Nr):
            self.tdl_channels[i].set_norm_doppler(norm_doppler)

    # Generate method to 
    def generate(self):
        # Declare a CMAT channel matrix of Nt*Nr size
        self.channel=itpp.cmat()
        self.channel.set_size(self.Nt*self.Nr,self.channel_length,False)
        # Declare another temp CMAT
        channel_coef_one=itpp.cmat()
        # Initialise the matrix element wise
        for i in range(self.Nt):
            for j in range(self.Nr):
                # Generate "1" sample values of the channel,  
                # channel_coef_one has one tap per column                 
                self.tdl_channels[i*self.Nr+j].generate(1,channel_coef_one)
                for l in range(self.channel_length):
                    self.channel.set(i*self.Nr+j,l,channel_coef_one(0,l))

    # Function to get Precoder List
    def get_Hlist(self):
        # Np array to store `FFT_size` number of Nt*Nr channels
        chan_freq=np.zeros(shape=(self.fft_size,self.Nt,self.Nr),dtype=complex)
        # Initialise the matrices        
        for i in range(self.Nt*self.Nr):
            col_idx=i%self.Nr
            row_idx=i//self.Nr
            freq_resp=itpp.cmat()
            inp_resp=itpp.cmat()
            #Get Impulse Response
            inp_resp.set_size(1,self.channel_length,False)
            inp_resp.set_row(0,self.channel.get_row(i))
            # print(inp_resp)
            #Calculate Frequency Response for each Nt*Nr channel (64 length array)
            self.tdl_channels[i].calc_frequency_response(inp_resp,freq_resp , 2*self.fft_size)
            # Store it in chan_freq in appropriate places across 64 matrices
            # print(freq_resp.to_numpy_ndarray().flatten()[0:64]-freq_resp.to_numpy_ndarray().flatten()[64:128])
            chan_freq[:,row_idx][:,col_idx]=freq_resp.to_numpy_ndarray().flatten()[0:self.fft_size]
        
        chan_freq_list=[]
        for i in range(self.fft_size):
            chan_freq_list.append(chan_freq[i])
        #print(chan_freq_list[0])
        return(chan_freq_list)


'''
	##Archived Generate Code

    # Generate method to 
    def generate(self):
        # Declare List to hold `channel_length` number of channel instances
        self.py_channel=[]
        # Declare a CMAT channel matrix of Nt*Nr size
        self.channel=itpp.cmat()
        self.channel.set_size(self.Nt*self.Nr,self.channel_length,False)
        # Declare channel instance till we get to channel length number of instances
        for i in range(self.channel_length):
            chan_instance=np.zeros(shape=(self.Nt,self.Nr),dtype=complex)
            self.py_channel.append(chan_instance)
        # Initialise the py_channel list with channel insantces
        channel_coef_one=itpp.cmat()
        for i in range(self.Nr):
            for j in range(self.Nt):                
                self.tdl_channels[i*self.Nr+j].generate(1,channel_coef_one)
                for l in range(self.channel_length):
                    self.py_channel[l][i][j]=channel_coef_one(0,l)
                    self.channel.set(i*self.Nr+j,l,channel_coef_one(0,l))
        #print(self.py_channel[0])
        return self.py_channel
'''