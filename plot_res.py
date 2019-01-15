import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import matplotlib as mplb


def rand_stiefel(n,p):
    H=(np.random.rand(n,p)+1j*np.random.rand(n,p))/np.sqrt(2)
    U, S, V = np.linalg.svd(H,full_matrices=0)
    return U

def unitary(n):
    X=(np.random.rand(n,n)+1j*np.random.rand(n,n))/np.sqrt(2)
    [Q,R]=np.linalg.qr(X)
    T=np.diag(np.diag(R)/np.abs(np.diag(R)))
    U=np.matrix(np.matmul(Q,T))
    # Verify print (np.matmul(U,U.H))
    return U    

def skew_hermitian(n):
    X=np.random.rand(n,n)+1j*np.random.rand(n,n)
    A=np.matrix(X-X.T.conjugate())/2
    # Verify print(A-A.H)
    return A

def hermitian(n):
    X=np.random.rand(n,n)+1j*np.random.rand(n,n)
    A=np.matrix(X+X.T.conjugate())/2
    # Verify print(A-A.H)
    return A

def CDF(data):
	
	sorted_data = np.sort(data)

	yvals=np.arange(len(sorted_data))/float(len(sorted_data)-1)

	plt.plot(sorted_data,yvals)

	plt.show()

def grassCD_2(A,B):
    C=np.vdot(A,B)
    rho=np.real(C*np.conjugate(C))
    return (1-rho)

def stiefCD(A,B):
    A_mat=np.matrix(A)
    B_mat=np.matrix(B)
    return np.sqrt(np.sum([grassCD_2(np.array(A_mat[:,i]),np.array(B_mat[:,i])) for i in range(A_mat.shape[1])]))



def plt_interp_res_63():
    Eb_N0_dB=np.arange(-6,20,3)
    plt.figure(figsize=(24, 18))
    ax = plt.gca()
    # ax.yaxis.set_ticks()
    # plt.figure()

    ax.grid(which='both', axis='both', linestyle='--')
    ax.semilogy(Eb_N0_dB,qgd_63,linestyle='-', marker='^',markersize=32,lw=8)
    ax.semilogy(Eb_N0_dB,gd_63,linestyle='-', marker='o',markersize=32,lw=8)
    ax.semilogy(Eb_N0_dB,uqgd_63,linestyle='-', marker='s',markersize=32,lw=8)
    ax.semilogy(Eb_N0_dB,ugd_63,linestyle='-', marker='*',markersize=32,lw=8)
    ax.set_xlim(np.min(Eb_N0_dB),np.max(Eb_N0_dB)+1)
    plt.xlabel('E$_b$/N$_0$ (dB)',size=80)
    plt.ylabel('BER',size=80)
    plt.legend([r'Cayley Exp. map ($32\times6$ bits)',r'Geodesic ($32\times6$ bits)','Cayley Exp. map (ideal)','Geodesic (ideal)'],prop={'size': 45})
    plt.xticks(np.arange(np.min(Eb_N0_dB)+2,np.max(Eb_N0_dB)+1,4),fontsize=80)
    plt.yticks([1e-1,1e-2,1e-3,1e-4,1e-5],fontsize=80)
    plt.savefig('/home/agrim/DDP_Results/interpBER63.svg')
    plt.show()


def plt_ped_ber_64():
    Eb_N0_dB=np.arange(-6,20,3)
    plt.figure(figsize=(24, 18))
    ax = plt.gca()
    ax.grid(which='both', axis='both', linestyle='--')
    ax.semilogy(Eb_N0_dB,qgd_ped,linestyle='-', marker='*',markersize=32,lw=8)
    ax.semilogy(Eb_N0_dB,ot_pred_ped,linestyle='-', marker='^',markersize=32,lw=8)
    ax.semilogy(Eb_N0_dB,hp_pred_ped,linestyle='-', marker='o',markersize=32,lw=8)
    ax.semilogy(Eb_N0_dB,uqgd_ped,linestyle='-', marker='s',markersize=32,lw=8)
    ax.semilogy(Eb_N0_dB,uqgd_2_ped,linestyle='-', marker='D',markersize=32,lw=8)
    
    ax.set_xlim(np.min(Eb_N0_dB),np.max(Eb_N0_dB)+1)
    plt.xlabel('E$_b$/N$_0$ (dB)',size=80)
    plt.ylabel('BER',size=80)
    plt.legend([r'Cayley Exp. map ($8\times6$ bits)',r'Time based predictive qt. ($8\times6$ bits)',\
        r'Hopping predictive qt. ($7.5\times6$ bits)',r'Cayley Exp. map (8, Ideal)',r'Cayley Exp. map (15, Ideal)'],prop={'size': 35})
    plt.xticks(np.arange(np.min(Eb_N0_dB)+2,np.max(Eb_N0_dB)+1,4),fontsize=80)
    plt.yticks([1e-1,1e-2,1e-3,1e-4,1e-5,1e-6],fontsize=80)
    plt.savefig('/home/agrim/DDP_Results/BER_Ped64.svg')
    # plt.show()

def plt_ped_ber_1024():
    Eb_N0_dB=np.arange(-6,20,3)
    plt.figure(figsize=(24, 18))
    ax = plt.gca()
    ax.grid(which='both', axis='both', linestyle='--')
    ax.semilogy(Eb_N0_dB,ot_pred_ped1024,linestyle='-', marker='^',markersize=32,lw=8)
    ax.semilogy(Eb_N0_dB,hp_pred_ped1024,linestyle='-', marker='o',markersize=32,lw=8)
    ax.set_xlim(np.min(Eb_N0_dB),np.max(Eb_N0_dB)+1)
    plt.xlabel('E$_b$/N$_0$ (dB)',size=80)
    plt.ylabel('BER',size=80)
    plt.legend(['Time based','Hopping Based'],prop={'size': 45})
    plt.xticks(np.arange(np.min(Eb_N0_dB)+2,np.max(Eb_N0_dB)+1,4),fontsize=80)
    plt.yticks([1e-1,1e-2,1e-3,1e-4,1e-5],fontsize=80)
    plt.savefig('/home/agrim/DDP_Results/BER_Ped1024.svg')

def plt_interp_res():
    Eb_N0_dB=np.arange(-10,24,4)
    plt.figure(figsize=(24, 18))
    ax = plt.gca()
    # ax.yaxis.set_ticks()
    # plt.figure()
    ax.grid(which='both', axis='both', linestyle='--')
    ax.semilogy(Eb_N0_dB,qgd,linestyle='-', marker='^',markersize=32,lw=8)
    ax.semilogy(Eb_N0_dB,gd,linestyle='-', marker='o',markersize=32,lw=8)
    ax.semilogy(Eb_N0_dB,uqgd,linestyle='-', marker='s',markersize=32,lw=8)
    ax.semilogy(Eb_N0_dB,ugd,linestyle='-', marker='*',markersize=32,lw=8)
    ax.set_xlim(np.min(Eb_N0_dB),np.max(Eb_N0_dB)+1)
    plt.xlabel('E$_b$/N$_0$ (dB)',size=80)
    plt.ylabel('BER',size=80)
    plt.legend([r'Cayley Exp. map ($32\times6$ bits)',r'Geodesic ($32\times6$ bits)',\
        'Cayley Exp. map (ideal)','Geodesic (ideal)'],prop={'size': 38})
    plt.xticks(np.arange(np.min(Eb_N0_dB)+2,np.max(Eb_N0_dB)+1,4),fontsize=80)
    plt.yticks([1e-1,1e-2,1e-3,1e-4,1e-5,1e-6],fontsize=80)
    plt.savefig('/home/agrim/DDP_Results/interpBER.svg')
    #plt.show()


def plt_all():
    plt.figure(figsize=(24, 18))
    Eb_N0_dB=np.arange(-6,20,3)
    ax = plt.gca() 
    ax.grid(which='both', axis='both', linestyle='--')
    ax.semilogy(Eb_N0_dB,qgd,linestyle='-', marker='^',markersize=32,lw=8)
    ax.semilogy(Eb_N0_dB,ot_pred,linestyle='-', marker='D',markersize=32,lw=8)
    ax.semilogy(Eb_N0_dB,uqgd,linestyle='-', marker='*',markersize=32,lw=8)
    ax.semilogy(Eb_N0_dB,hp_pred,linestyle='-', marker='s',markersize=32,lw=8)
    ax.semilogy(Eb_N0_dB,uqgd_2,linestyle='-', marker='o',markersize=32,lw=8)
    ax.set_xlim(np.min(Eb_N0_dB),np.max(Eb_N0_dB)+1)
    plt.xlabel('E$_b$/N$_0$ (dB)',size=80)
    plt.ylabel('BER',size=80)
    plt.legend([r'Cayley Exp. map ($32\times6$ bits)',r'Time based predictive qt. ($32\times6$ bits)',r'Cayley Exp. map ($32$, ideal)',\
         r'Hopping predictive qt. ($31.5\times6$ bits)',r'Cayley Exp. map ($63$, ideal)'],prop={'size': 38})
    plt.xticks(np.arange(np.min(Eb_N0_dB)+3,np.max(Eb_N0_dB)+1,3),fontsize=80)
    plt.yticks([1e-1,1e-2,1e-3,1e-4,1e-5,1e-6],fontsize=80)
    plt.savefig('/home/agrim/DDP_Results/hopBER.svg')


def plt_logsumrate():
    plt.figure(figsize=(24, 18))
    SNR=np.arange(7)*5
    ax = plt.gca()
    ax.xaxis.set_ticks(np.arange(np.min(SNR),np.max(SNR)+1))
    ax.grid(which='both', axis='both', linestyle='--')
    ax.plot(SNR,100*(hp_4/max_cap),linestyle='-', marker='^',markersize=32,lw=8,color='r',label='$f_{d}T_{s}=10^{-4}$,hop')
    ax.plot(SNR,100*(ot_4/max_cap),linestyle='-.', marker='^',markersize=32,lw=8,color='r',label='$f_{d}T_{s}=10^{-4}$,time')
    ax.plot(SNR,100*(hp_2/max_cap),linestyle='-', marker='o',markersize=32,lw=8,color='b',label='$f_{d}T_{s}=10^{-2}$,hop')
    ax.plot(SNR,100*(ot_2/max_cap),linestyle='-.', marker='o',markersize=32,lw=8,color='b',label='$f_{d}T_{s}=10^{-2}$,time')
    ax.plot(SNR,100*(hp_1/max_cap),linestyle='-', marker='s',markersize=32,lw=8,color='g',label='$f_{d}T_{s}=10^{-1}$,hop')
    ax.plot(SNR,100*(ot_1/max_cap),linestyle='-.', marker='s',markersize=32,lw=8,color='g',label='$f_{d}T_{s}=10^{-1}$,time')
    ax.set_xlim(np.min(SNR),np.max(SNR)+1)
        # ax.plot(SNR,100*(hp_1/max_cap),linestyle='-', marker='*',markersize=8,color='c',label='1e-1,hop')
    # ax.plot(SNR,100*(ot_1/max_cap),linestyle='-.', marker='*',markersize=8,color='c',label='1e-1,time')
    # ax.plot(SNR,max_cap,linestyle='--', marker='D',markersize=8,color='black',label='Max Capacity')
    plt.legend(prop={'size': 50})
    plt.xlabel('SNR (dB)',size=80)
    plt.ylabel(r'Achievable rate (\% max. capacity)',size=80)
    plt.xticks(SNR,fontsize=80)
    plt.yticks(fontsize=80)
    plt.savefig('/home/agrim/DDP_Results/hopRate.svg')

def plt_logsumrate_ped():
    plt.figure(figsize=(24, 18))
    SNR=np.arange(7)*5
    ax = plt.gca()
    ax.xaxis.set_ticks(np.arange(np.min(SNR),np.max(SNR)+1))
    ax.grid(which='both', axis='both', linestyle='--')
    ax.plot(SNR,100*(hp_4_ped/max_cap),linestyle='-', marker='^',markersize=32,lw=8,color='r',label='$f_{d}T_{s}=10^{-4}$,hop')
    ax.plot(SNR,100*(ot_4_ped/max_cap),linestyle='-.', marker='^',markersize=32,lw=8,color='r',label='$f_{d}T_{s}=10^{-4}$,time')
    ax.plot(SNR,100*(hp_2_ped/max_cap),linestyle='-', marker='o',markersize=32,lw=8,color='b',label='$f_{d}T_{s}=10^{-2}$,hop')
    ax.plot(SNR,100*(ot_2_ped/max_cap),linestyle='-.', marker='o',markersize=32,lw=8,color='b',label='$f_{d}T_{s}=10^{-2}$,time')
    ax.plot(SNR,100*(hp_1_ped/max_cap),linestyle='-', marker='s',markersize=32,lw=8,color='g',label='$f_{d}T_{s}=10^{-1}$,hop')
    ax.plot(SNR,100*(ot_1_ped/max_cap),linestyle='-.', marker='s',markersize=32,lw=8,color='g',label='$f_{d}T_{s}=10^{-1}$,time')
    ax.set_xlim(np.min(SNR),np.max(SNR)+1)
        # ax.plot(SNR,100*(hp_1/max_cap),linestyle='-', marker='*',markersize=8,color='c',label='1e-1,hop')
    # ax.plot(SNR,100*(ot_1/max_cap),linestyle='-.', marker='*',markersize=8,color='c',label='1e-1,time')
    # ax.plot(SNR,max_cap,linestyle='--', marker='D',markersize=8,color='black',label='Max Capacity')
    plt.legend(prop={'size': 50})
    plt.xlabel('SNR (dB)',size=80)
    plt.ylabel(r'Achievable rate (\% max. capacity)',size=80)
    plt.xticks(SNR,fontsize=80)
    plt.yticks(fontsize=80)
    plt.savefig('/home/agrim/DDP_Results/hopRate_ped.svg')
    # plt.show()

def plt_one_logsumrate():
    plt.figure(figsize=(24, 18))
    SNR=np.arange(7)*5
    ax = plt.gca()
    ax.xaxis.set_ticks(np.arange(np.min(SNR),np.max(SNR)+1))
    ax.grid(which='both', axis='both', linestyle='--')
    ax.plot(SNR,100*(hp_cap/max_cap),linestyle='-', marker='s',markersize=32,lw=8,color='g',label='$f_{d}T_{s}=10^{-5}$,hop')
    ax.plot(SNR,100*(ot_cap/max_cap),linestyle='-.', marker='s',markersize=32,lw=8,color='g',label='$f_{d}T_{s}=10^{-5}$,time')
    ax.set_xlim(np.min(SNR),np.max(SNR)+1)
        # ax.plot(SNR,100*(hp_1/max_cap),linestyle='-', marker='*',markersize=8,color='c',label='1e-1,hop')
    # ax.plot(SNR,100*(ot_1/max_cap),linestyle='-.', marker='*',markersize=8,color='c',label='1e-1,time')
    # ax.plot(SNR,max_cap,linestyle='--', marker='D',markersize=8,color='black',label='Max Capacity')
    plt.legend(prop={'size': 50})
    plt.xlabel('SNR (dB)',size=80)
    plt.ylabel(r'Achievable rate (\% max. capacity)',size=80)
    plt.xticks(SNR,fontsize=80)
    plt.yticks(fontsize=80)
    # plt.savefig('/home/agrim/DDP_Results/hopRate.svg')
    plt.show()

# def plt_interprate():
#     plt.figure(figsize=(24, 18))
#     percqg_rate=np.load('/home/agrim/DDP_Results/interp_res_qg_rate.npy')
#     percg_rate=np.load('/home/agrim/DDP_Results/interp_res_g_rate.npy')
#     ax=plt.gca()
#     num_subcarriers=1024
#     ax.xaxis.set_ticks(np.arange(0,num_subcarriers,100))
#     ax.grid(which='both', axis='both', linestyle='--')
#     ax.plot(percqg_rate)
#     ax.plot(percg_rate)
#     plt.xlabel('Subcarrier Frequency Index',size=30)
#     plt.ylabel(r"\% of maximum capacity utilized at SNR=0dB",size=25)
#     ax.legend(['Cayley Exp. Interpolation','Unitary Geodesic Interpolation'],prop={'size': 30})
#     plt.xticks(fontsize=30)
#     plt.yticks(fontsize=30)
#     plt.show()

def plt_dt_error():
    plt.figure(figsize=(24, 18))
    hp_Qerr=np.load('/home/agrim/DDP_Results/hp_Neterr.npy')
    ot_Qerr=np.load('/home/agrim/DDP_Results/ot_Neterr.npy')
    ax=plt.gca()
    ax.grid(which='both', axis='both', linestyle='--')
    ax.plot(hp_Qerr,linestyle='-.',lw=8)
    ax.plot(ot_Qerr,linestyle='--',lw=8)
    plt.xlabel('Time index',size=80)
    plt.ylabel(r"Chordal distance error ($d[t]$)",size=80)
    ax.legend(["Hopping based predictive quantization","Time based predictive quantization"],prop={'size': 50})
    plt.xticks(fontsize=80)
    plt.yticks(fontsize=80)
    plt.savefig('/home/agrim/DDP_Results/hopErr.svg')


mplb.rc('text',usetex=True)
ot_pred = np.array([  2.95155404e-01,   1.63160112e-01,   5.47891764e-02,
         8.25974263e-03,   6.01548666e-04,   9.89581299e-05,
         3.61293538e-05,   1.60498047e-05,   7.91137695e-06])
hp_pred = np.array([  2.60527310e-01,   1.21329251e-01,   2.89104551e-02,
         2.66645141e-03,   1.98200887e-04,   3.62886556e-05,
         1.22676595e-05,   5.26774089e-06,   2.40478515e-06])

# >>> ot_pred = np.array([  2.94931354e-01,   1.59993899e-01,   4.93282262e-02,
# ...          7.05493596e-03,   5.97342936e-04,   4.71595764e-05,
# ...          5.21825155e-06,   1.62506104e-07,   0.00000000e+00])
# >>> hp_pred = np.array([  2.86789015e-01,   1.49748993e-01,   4.30961662e-02,
# ...          5.35938543e-03,   4.27747345e-04,   4.06715393e-05,
# ...          7.80944824e-06,   2.02051799e-06,   3.69008382e-07])

qgd_63 = np.array([  2.97795999e-01,   1.77910472e-01,   7.24426461e-02,
         2.24024347e-02,   7.35978190e-03,   2.91501034e-03,
         1.28784180e-03,   6.00064147e-04,   2.80723422e-04])
gd_63 = np.array([ 0.29486527,  0.17658951,  0.0735702 ,  0.02342051,  0.00774663,
        0.0030637 ,  0.00138425,  0.00068729,  0.00036671])
uqgd_63 = np.array([  2.16981804e-01,   8.39996218e-02,   1.60830126e-02,
         1.21418313e-03,   1.32046569e-04,   4.10491345e-05,
         1.24990426e-05,   3.45626532e-06,   1.86695772e-06])
ugd_63 = np. array([  2.15785309e-01,   8.26273122e-02,   1.53498583e-02,
         1.10360179e-03,   1.09030331e-04,   2.28247549e-05,
         4.16475184e-06,   2.01056985e-07,   0.00000000e+00])



qgd_2 = np.array([  3.06703025e-01,   1.76699312e-01,   5.91567041e-02,
         1.05639014e-02,   1.65912109e-03,   3.97292480e-04,
         1.29030762e-04,   4.35620117e-05,   1.57421875e-05])
uqgd_2 = np.array([  2.60814126e-01,   1.19592854e-01,   2.73023047e-02,
         2.20062988e-03,   8.78564453e-05,   6.83349609e-06,
         2.00927734e-06,   9.52685547e-07,   4.49218750e-07])
qgd = np.array([  3.09741226e-01,   1.81388149e-01,   6.33838818e-02,
         1.26018408e-02,   2.28333496e-03,   5.62272949e-04,
         1.84970703e-04,   6.95019531e-05,   2.93017578e-05])
gd = np.array([  3.21277842e-01,   2.01335337e-01,   8.20790332e-02,
         2.17456860e-02,   5.38198242e-03,   1.66379395e-03,
         6.42656250e-04,   2.75383301e-04,   1.26713867e-04])
uqgd = np.array([  2.64613201e-01,   1.24440117e-01,   3.02768774e-02,
         3.38722168e-03,   3.59001465e-04,   6.37646484e-05,
         1.56494141e-05,   5.20751953e-06,   1.49414063e-06])
ugd = np. array([  2.63786794e-01,   1.23296599e-01,   2.94851611e-02,
         3.17135010e-03,   3.38178711e-04,   6.44580078e-05,
         1.97631836e-05,   7.00195313e-06,   2.24853516e-06])


uqgd_ped = np.array([  1.13420938e-01,   2.32688672e-02,   1.45410156e-03,
         4.52734375e-05,   7.81250000e-07,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00])
uqgd_2_ped = np. array([  1.12714023e-01,   2.29441016e-02,   1.33414063e-03,
         2.89843750e-05,   1.56250000e-07,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00])

qgd_ped = np.array([  1.98651211e-01,   8.53144922e-02,   2.75089453e-02,
         9.09378906e-03,   3.47718750e-03,   1.49406250e-03,
         6.98007812e-04,   2.97695312e-04,   1.79687500e-04])




ot_35=np.array([  1.85361089,   3.30526429,   5.20591557,   7.32644549,
         9.53897195,  11.78567642,  14.04436933])
hp_35=np.array([  1.92866456,   3.42733048,   5.35457058,   7.48965721,
         9.70895414,  11.95848168,  14.21825625])

ot_4=np.array([  1.84784996,   3.33174066,   5.23529662,   7.35756468,
         9.57083991,  11.81782344,  14.07661394])
hp_4=np.array([  1.9331859 ,   3.45406168,   5.3841723 ,   7.52090597,
         9.74088225,  11.99065314,  14.25051006])



ot_2=np.array([  1.81032343,   3.2306555 ,   5.0913624 ,   7.19286137,
         9.39805084,  11.64222403,  13.90007412])
hp_2=np.array([  1.79841769,   3.22067371,   5.08187212,   7.18323417,
         9.38815523,  11.63210521,  13.8898092 ])


# ot_2=np.array([  1.71348309,   3.110657  ,   4.94854389,   7.03477409,
#          9.2319213 ,  11.47240193,  13.72863985])
# hp_2=np.array([  1.71188821,   3.11372713,   4.95657955,   7.04680417,
#          9.24648039,  11.48829116,  13.74513831])


ot_1=np.array([  1.69953756,   3.07992737,   4.90801593,   6.99084028,
         9.18742747,  11.42808573,  13.68458777])
hp_1=np.array([  1.50937304,   2.80379998,   4.54083677,   6.55734481,
         8.71633311,  10.93884528,  13.18745491])


max_cap=np.array([  2.09549676,   3.62698007,   5.58622217,   7.76312496,
        10.02360512,  12.31259012,  14.61084268])


ot_pred_ped = np.array([  1.34874186e-01,   3.42179362e-02,   4.11722819e-03,
         4.88566081e-04,   1.60725911e-04,   6.88069661e-05,
         2.93375651e-05,   1.57877604e-05,   4.39453125e-06])
hp_pred_ped = np.array([  1.20995646e-01,   2.54759928e-02,   1.68432617e-03,
         1.23738607e-04,   5.28564453e-05,   2.86865234e-05,
         1.66422526e-05,   7.56835937e-06,   3.29589844e-06])



# ot_pred_ped1024 = np.array([  2.94931354e-01,   1.59993899e-01,   4.93282262e-02,
#           7.05493596e-03,   5.97342936e-04,   4.71595764e-05,
#           5.21825155e-06,   1.62506104e-06,   2.84452293e-07])
# hp_pred_ped1024 = np.array([  2.86789015e-01,   1.49748993e-01,   4.30961662e-02,
#           5.35938543e-03,   4.27747345e-04,   4.06715393e-05,
#           7.80944824e-06,   2.02051799e-06,   3.69008382e-07])

ot_pred_ped1024 = np.array([  1.60932185e-01,   5.09019699e-02,   8.08418783e-03,
         9.56489563e-04,   1.69423421e-04,   5.12491862e-05,
         2.14818319e-05,   1.18433634e-05,   6.49261475e-06])
hp_pred_ped1024 = np.array([  1.48738599e-01,   4.26868413e-02,   5.37896729e-03,
         4.73258972e-04,   6.06867472e-05,   1.43712362e-05,
         2.47955322e-06,   1.15203857e-06,   4.39961751e-07])



ot_cap=np.array([  1.99466309,   3.42933474,   5.24620274,   7.26275924,
         9.36404126,  11.49709733,  13.64123109])
hp_cap=np.array([  2.12990324,   3.61252586,   5.46187545,   7.49545363,
         9.60433187,  11.74043605,  13.88571704])

ot_1_ped=np.array([  1.60870138,   2.91443126,   4.6417556 ,   6.61274664,
         8.69456507,  10.81999997,  12.96129036])
hp_1_ped=np.array([  1.5038306 ,   2.7625781 ,   4.43681652,   6.36748911,
         8.42580392,  10.53980167,  12.6762114 ])

ot_4_ped=np.array([  1.78726381,   3.16232458,   4.93665787,   6.93134492,
         9.02293148,  11.15178996,  13.29402576])
hp_4_ped=np.array([  1.91625854,   3.3561822 ,   5.18131706,   7.2057403 ,
         9.31149496,  11.44627299,  13.59076989])

ot_2_ped=np.array([  1.765853  ,   3.12912225,   4.90047557,   6.89594988,
         8.98886609,  11.11874358,  13.26167457])
hp_2_ped=np.array([  1.81316967,   3.20260242,   4.99393771,   7.00091914,
         9.09931626,  11.23151031,  13.37536874])



