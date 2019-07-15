import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import pdb
from mimo_tdl_channel import sH_lift,sH_retract,lift,retract


def unitary(n):
    X=(np.random.rand(n,n)+1j*np.random.rand(n,n))/np.sqrt(2)
    [Q,R]=np.linalg.qr(X)
    T=np.diag(np.diag(R)/np.abs(np.diag(R)))
    U=np.matrix(np.matmul(Q,T))
    # Verify print (np.matmul(U,U.H))
    return U  

def rand_SH(n):
    X=2*np.random.rand(n,n)-np.ones((n,n))+1j*(2*np.random.rand(n,n)-np.ones((n,n)))
    A=np.matrix(X-X.T.conjugate())/2
    # Verify print(A-A.H)
    return A


class Interpolator():

    def __init__(self, num_subcarriers=1024, num_fb_points=32):
        self.num_subcarriers=num_subcarriers
        self.num_fb_points=num_fb_points
        # Why ((self.num_subcarriers-1)//(self.num_fb_points-1)) ?
        # We have to cover indices upto (self.num_subcarriers-1) with (self.num_fb_points-1)
        # Since, we have 0 as an implicit input
        if(num_fb_points==63):
            freq_jump=(num_subcarriers-1)//31
            currfb_indices=freq_jump*np.arange(32)
            prev_indices=freq_jump*np.arange(31)+freq_jump//2
            self.fb_indices=np.sort(np.append(currfb_indices,prev_indices))
        else:
            if(num_fb_points==15):
                freq_jump=(num_subcarriers-1)//7
                currfb_indices=freq_jump*np.arange(8)
                prev_indices=freq_jump*np.arange(7)+freq_jump//2
                self.fb_indices=np.sort(np.append(currfb_indices,prev_indices))
                pdb.set_trace()   
            else:
                self.fb_indices=np.arange(self.num_fb_points)*((self.num_subcarriers-1)//(self.num_fb_points-1))


    def apply_sigmainterp(self,sigma_list,Nr,qtCodebook=None):
        feedback_indices=self.fb_indices
        interpolated_S_list=np.zeros((self.num_subcarriers, Nr),dtype=complex)
        for i in range(self.num_fb_points-1):
            thS_current=sigma_list[feedback_indices[i]]
            thS_next=sigma_list[feedback_indices[i+1]]
            if(qtCodebook is not None):
                S_current=qtCodebook[np.argmin([la.norm(thS_current-codeword) for codeword in qtCodebook])]
                S_next=qtCodebook[np.argmin([la.norm(thS_next-codeword)\
                    for codeword in qtCodebook])]
            else:
                S_current=thS_current
                S_next=thS_next
            num_indices_to_be_filled=feedback_indices[i+1]-feedback_indices[i]-1
            if(i==self.num_fb_points-2):
                interpolated_S_list[feedback_indices[i]:feedback_indices[i+1]+1]=\
                    self.convex_interpolation(S_current, S_next, num_indices_to_be_filled, last_fill_flag=True)
            else:
                interpolated_S_list[feedback_indices[i]:feedback_indices[i+1]]=\
                    self.convex_interpolation(S_current, S_next, num_indices_to_be_filled, last_fill_flag=False)
        return interpolated_S_list



    def apply_interpolation(self, feedback_list, Nt, Nr,interpType,qtCodebook=None,retQTnoise=False):
        diff_frob_norm = lambda A,B:np.linalg.norm(A-B, 'fro')
        feedback_indices=self.fb_indices
        interpolated_U_list=np.zeros((self.num_subcarriers, Nt, Nr),dtype=complex)
        qtNoise=np.zeros(feedback_indices.shape[0])
        for i in range(self.num_fb_points-1):
            if(qtCodebook is not None):
                thU_curr=feedback_list[feedback_indices[i]]
                thU_next=feedback_list[feedback_indices[i+1]]
                U_current=qtCodebook[np.argmin([diff_frob_norm(thU_curr,codeword)\
                    for codeword in qtCodebook])]
                U_next=qtCodebook[np.argmin([diff_frob_norm(thU_next,codeword)\
                 for codeword in qtCodebook])]
                qtNoise[i]+=diff_frob_norm(thU_curr,U_current)
                if(i==self.num_fb_points-2):
                    qtNoise[i+1]+=diff_frob_norm(thU_next,U_next)
            else:            
                U_current=feedback_list[feedback_indices[i]]
                U_next=feedback_list[feedback_indices[i+1]]
            num_indices_to_be_filled=feedback_indices[i+1]-feedback_indices[i]-1
            if(i==self.num_fb_points-2):
                if(interpType=='Geodesic'):
                    interpolated_U_list[feedback_indices[i]:feedback_indices[i+1]+1]=\
                    self.Geodesic_interpolation(U_current, U_next, num_indices_to_be_filled,Nr, last_fill_flag=True)
                if(interpType=='quasiGeodesic'):
                    interpolated_U_list[feedback_indices[i]:feedback_indices[i+1]+1]=\
                    self.quasiGeodesic_interpolation(U_current, U_next, num_indices_to_be_filled, last_fill_flag=True)
                if(interpType=='orthLifting'):
                    interpolated_U_list[feedback_indices[i]:feedback_indices[i+1]+1]=\
                    self.orthLifting_interpolation(U_current, U_next, num_indices_to_be_filled, last_fill_flag=True)
            else:
                if(interpType=='Geodesic'):
                    interpolated_U_list[feedback_indices[i]:feedback_indices[i+1]]=\
                    self.Geodesic_interpolation(U_current, U_next, num_indices_to_be_filled,Nr, last_fill_flag=False)
                if(interpType=='quasiGeodesic'):
                    interpolated_U_list[feedback_indices[i]:feedback_indices[i+1]]=\
                    self.quasiGeodesic_interpolation(U_current, U_next, num_indices_to_be_filled, last_fill_flag=False)
                if(interpType=='orthLifting'):
                    interpolated_U_list[feedback_indices[i]:feedback_indices[i+1]]=\
                    self.orthLifting_interpolation(U_current, U_next, num_indices_to_be_filled, last_fill_flag=False)                    
        if(retQTnoise):
            return interpolated_U_list,qtNoise
        else:
            return interpolated_U_list
    
    def Geodesic_interpolation(self, U_current, U_next, num_indices_to_be_filled, Nr, last_fill_flag=False):
        orientation_matrix=self.find_orientation_matrix(U_current, U_next)
        M=np.matrix(la.inv(U_current))*np.matrix(U_next)*np.matrix(orientation_matrix)
        S=la.logm(M)
        t=np.linspace(0,1,num=num_indices_to_be_filled+1, endpoint=False)
        interpolation_fn= lambda  t:(np.matrix(U_current)*np.matrix(la.expm(t*S)))[:,np.arange(Nr)]
        U_interpolate=list(map(interpolation_fn, t))
        if(last_fill_flag):
            U_interpolate.append(U_next[:,np.arange(Nr)])
        return np.array(U_interpolate)

    def quasiGeodesic_interpolation(self, U_current, U_next, num_indices_to_be_filled, last_fill_flag=False):
        # orientation_matrix=self.find_orientation_matrix(U_current, U_next)
        M=np.matrix(U_next)
        S=sH_lift(U_current,M)
        t=np.linspace(0,1,num=num_indices_to_be_filled+1, endpoint=False)
        interpolation_fn= lambda  t:sH_retract(U_current,t*S)
        U_interpolate=list(map(interpolation_fn, t))
        if(last_fill_flag):
            U_interpolate.append(U_next)
        return np.array(U_interpolate)

    def orthLifting_interpolation(self, U_current, U_next, num_indices_to_be_filled, last_fill_flag=False):
        orientation_matrix=self.find_orientation_matrix(U_current, U_next)
        M=np.matrix(U_next)*np.matrix(orientation_matrix)
        S=lift(U_current,M)
        t=np.linspace(0,1,num=num_indices_to_be_filled+1, endpoint=False)
        interpolation_fn= lambda  t:retract(U_current,t*S)
        U_interpolate=list(map(interpolation_fn, t))
        if(last_fill_flag):
            U_interpolate.append(U_next)
        return np.array(U_interpolate)

    def find_orientation_matrix(self, U0, U1):
        A_current= np.diag(np.matmul(U1.T.conj(),U0))
        orientaion_matrix=np.diag((A_current/np.absolute(A_current)))
        return orientaion_matrix

    def convex_interpolation(self, S_current, S_next, num_indices_to_be_filled, last_fill_flag=False):
        t=np.linspace(0,1,num=num_indices_to_be_filled+1, endpoint=False)
        interpolation_fn= lambda  t:(1-t)*S_current+t*S_next
        S_interpolate=list(map(interpolation_fn,t))
        if(last_fill_flag):
            S_interpolate.append(S_next)
        return np.array(S_interpolate)
