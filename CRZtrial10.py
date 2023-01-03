import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import warnings
from matplotlib.cbook import MatplotlibDeprecationWarning
warnings.filterwarnings('ignore', category=MatplotlibDeprecationWarning)
import cmath
from scipy import linalg
from numpy.linalg import solve
import scipy.stats
from qulacs import QuantumState
from qulacs import QuantumCircuit
from qulacs.gate import DenseMatrix
from qulacs.circuit import QuantumCircuitOptimizer
from multiprocessing import Pool
from multiprocessing import Process
from qulacs.gate import Identity, X,Y,Z 
from qulacs.gate import H,S,Sdag, sqrtX,sqrtXdag,sqrtY,sqrtYdag
from qulacs.gate import T,Tdag
from qulacs.gate import RX,RY,RZ
from qulacs.gate import CNOT, CZ, SWAP
import time
def bin4(m,a,b):
    m=int(m)
    l = [*map(int,f"{format(m,'b'):>0{a}}")]
    return l[a-1-b]

def bin5(m,a):
    m=int(m)
    l = [*map(int,f"{format(m,'b'):>0{a}}")]
    return l
def bin6(m,a):
    m=int(m)
    l = [*map(int,f"{format(m,'b'):>0{a}}")]
    l=str(l).replace('[','').replace(']','').replace(',','').replace(' ','')
    return l
def decimal(n,a):
    answer=0
    for i in range(n):
        answer+=pow(2,n-1-i)*a[i]
    return answer
import sympy
pi = sympy.pi
E = sympy.E
I = sympy.I
from qulacs.gate import DenseMatrix,to_matrix_gate
def MCRZ(bit,n,state,degree):
    m=sum(bit[0:n])
    control_index=[0 for i in range(m+1)]
    count=0
    for i in range(n):
        if bit[i]==1:
            control_index[count]=i
            count+=1
    control_index[m]=bit[n]
    index = control_index[m]
    rz_gate = RZ(index,degree)
    mcrz_mat_gate = to_matrix_gate(rz_gate)
    control_with_value = 1
    for i in range(m):
        mcrz_mat_gate.add_control_qubit(control_index[i], control_with_value)
    mcrz_mat_gate.update_quantum_state(state)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()
zeros_original=np.zeros((100,28,28))
ones_original=np.zeros((100,28,28))
i=0
j=0
k=0
while i<100 or j<100:
    if train_labels[k]==0 and i<100:
        zeros_original[i]=train_images[k]
        i+=1
    elif train_labels[k]==1 and j<100:
        ones_original[j]=train_images[k]
        j+=1
    k+=1

zeros_32=np.zeros((100,32,32))
ones_32=np.zeros((100,32,32))
for i in range(0,100):
   for j in range(0,28):
       for k in range(0,28):
           zeros_32[i][j+2][k+2]=zeros_original[i][j][k]
           ones_32[i][j+2][k+2]=ones_original[i][j][k]
        
zeros_row_32=np.zeros((100,1024))
ones_row_32=np.zeros((100,1024))
for i in range(100):
    for j in range(32):
        for k in range(32):
            zeros_row_32[i][j*32+k]=zeros_32[i][j][k]
            ones_row_32[i][j*32+k]=ones_32[i][j][k]
            
from PIL import Image
zeros_8=np.zeros((100,8, 8))
ones_8=np.zeros((100,8,8))
for i in range(100):
    zeros_8[i]=np.asarray(Image.fromarray(np.uint8(zeros_original[i])).resize((8,8)))
    ones_8[i]=np.asarray(Image.fromarray(np.uint8(ones_original[i])).resize((8,8)))

zeros_row_8=np.zeros((100,64))
ones_row_8=np.zeros((100,64))
for i in range(100):
    for j in range(8):
        for k in range(8):
            zeros_row_8[i][j*8+k]=zeros_8[i][j][k]-128
            ones_row_8[i][j*8+k]=ones_8[i][j][k]-128
            
def function_n2(n,m,ki,cycle):
    # print(ki)
    # time.sleep(100)
    vertical_of_A=np.zeros(n)
    global kw_272
    bit=np.array([[0 for i in range(n)] for j in range(pow(2,n))], dtype=object)
    gate=np.array([[0 for i in range(n)] for j in range(pow(2,n))], dtype=object)
    gate2=np.array([[0 for i in range(n+1)] for j in range(pow(2,n))], dtype=object)
    FLAG=np.array([0 for i in range(pow(2,m))])
    FLAG2=np.array([[1 for i in range(n+m+1)] for j in range(2**m)], dtype=object)
    FLAG3=np.array([[0 for i in range(m)] for j in range(2**m)], dtype=object)
    # kw=np.array([pi/4 for i in range(2**m)])
    for i in range(0,pow(2,n)):
        for j in range(0,n):
            bit[i][n-1-j]=bin4(i,n,j)
    a=0
    for i in range(1,n+1):
        for j in range(0,2**n):
            if sum(bit[2**n-j-1])==i:
                gate[a]=bit[2**n-j-1]
                a+=1
    a=0
    for i in range(0,2**n):
        for j in range(0,n):
            if gate[i][j]==0:
                for k in range(n):
                    gate2[a][k]=gate[i][k]
                gate2[a][n]=j
                a+=1
                break
        if a==2**n:
                break
        else:
            continue
    for i in range(2**m):
        for j in range(m):
            FLAG3[i][j]=bin4(i,m,j)
    for i in range(2**m):
        FLAG[i]=i%m
    for i in range(2**m):
        FLAG2[i][n+FLAG[i]]=0
        FLAG2[i][n+m]=n+FLAG[i]

    nqubits = n+m
    state = QuantumState(nqubits)
    state.set_zero_state()
    # print("159")
    # for i in range(len(ki)):
        # print(ki[i])
    # time.sleep(5)
    state.load(ki)
    # print(state)
    # time.sleep(5)
    # print(ki)
    for i in range(n):
        H(i).update_quantum_state(state)

    for i in range(2**m):
        for j in range(m):
            if(FLAG3[i][j]==0):
                X(n-m+j).update_quantum_state(state)
                X(n+j).update_quantum_state(state)
        # print(f"今ここ{i}")
        if(cycle==0):
            MCRZ(FLAG2[i],n+m,state,kw_272[i])
        else:
            MCRZ(FLAG2[i],n+m,state,kw_272[i+256])#qubit数変えるときここも変える
        # print(f"終わった{i}")
        for j in range(m):
            if(FLAG3[i][j]==0):
                X(n-m+j).update_quantum_state(state)
                X(n+j).update_quantum_state(state)

    for i in range(n,n+m):
        H(i).update_quantum_state(state)  
    result = state.sampling(nqubits)
    num_ones=np.array([0.0 for i in range(m)])
    # print(state)
    for i in range(n,n+m):
        if(sum(result[n:n+m])==0):
            num_ones[i-n] = 1/m
        else:
            num_ones[i-n] = result[i]/sum(result[n:n+m])
    return num_ones
def multi(n):
    # p = Pool(32)
    # result = p.map(main_calculation, range(n))
    for i in range(n):
        main_calculation(i)
    # return n,result
def multi_ki(n):
    # p = Pool(32)
    # result = p.map(ki_processor, range(n))
    for i in range(n):
        ki_processor(i)
    # return n,result
def multi_cycle(n):
    p = Pool(32)
    result = p.map(cost_calculator, range(n))
    # for i in range(n):
    #     ki_processor(i)
    return n,result

def main_cycle():
    i,data = multi_cycle(100)
    result_cycle=0
    for j in data:
        result_cycle=result_cycle+j
    return result_cycle

result1=np.array([0.0 for i in range (100)])
result2=np.array([0.0 for i in range (100)])
bottom=np.array([0.0 for i in range (100)])
ki0=np.array([[1.0*2**(-9)+0.0j for i in range(4*16*16*256)] for j in range(100)])
ki1=np.array([[1.0*2**(-9)+0.0j for i in range(4*16*16*256)] for j in range(100)])
# kw_272=np.array([0.0 for j in range (272)])
kw_272=[ 0.0937922,   0.02135668,  0.07413795, -0.06516433, -0.04831931,  0.02232466,
 -0.06181498,  0.08023291,  0.06744392, -0.05129845,  0.10933415, -0.03384373,
 -0.06172077, -0.0417346 ,  0.08483436,  0.08916708, -0.10737238,  0.07312065,
 -0.0608648 ,  0.01092846,  0.13842292,  0.0038599 , -0.03076372,  0.05755226,
  0.04781802, -0.23015385, -0.12404386, -0.01140812,  0.02358028,  0.06792332,
  0.04042729,  0.05575872,  0.06227697, -0.00120455, -0.00519412,  0.09394512,
 -0.00676361,  0.01209514,  0.03304014, -0.01381739,  0.03292656,  0.08834996,
 -0.03765578,  0.05988379, -0.01221018,  0.05768145, -0.05639764, -0.00215268,
 -0.02374533, -0.0999133 , -0.08975107, -0.0794213 ,  0.01910694, -0.00608501,
  0.00260088,  0.07526798,  0.10546851, -0.13810327,  0.02539216,  0.07196691,
 -0.09637994,  0.03596846, -0.05066871, -0.15279034, -0.04163996,  0.02627058,
 -0.01230303,  0.04537876,  0.07868614,  0.09978842, -0.17895806, -0.14724627,
  0.01179418,  0.04555829, -0.02688513, -0.08807502, -0.029095  ,  0.07499018,
 -0.01823565, -0.06111151, -0.08977127, -0.02572132,  0.09178421, -0.04903558,
 -0.02880602,  0.22157972,  0.12116314,  0.04696881, -0.18112477,  0.01113569,
  0.01253641,  0.05918666, -0.0084077 ,  0.01485545,  0.03303404, -0.17681077,
 -0.01141812,  0.07747216,  0.09859892, -0.07044708, -0.08201113,  0.19434645,
  0.02662773, -0.06070106,  0.06506816,  0.04258801,  0.0063996 ,  0.12588676,
  0.01242621,  0.16290898,  0.03552281,  0.01358086,  0.02126872,  0.08530726,
 -0.01906885,  0.03853189, -0.04747431,  0.09035858,  0.04760751, -0.10627007,
  0.11584553,  0.0430011 ,  0.10166858,  0.11303318,  0.01092015, -0.06647347,
  0.04589654,  0.04933209, -0.06693865,  0.03278821, -0.0381204 ,  0.0919395,
 -0.07510804, -0.01720538, -0.02618928,  0.02975736, -0.00064717, -0.05723987,
 -0.21985562, -0.01966881, -0.03384062,  0.09587226, -0.02359765, -0.12182386,
  0.14174144, -0.03102507, -0.00994362, -0.10865004,  0.01257307,  0.04101844,
 -0.10870434, -0.01362907,  0.00444668, -0.11612999,  0.08349079,  0.05958448,
  0.12328816,  0.11161   , -0.0339004 , -0.09511003,  0.10745625,  0.16832974,
  0.07002496,  0.13814331, -0.02967101, -0.06125285,  0.0399296 , -0.08727134,
 -0.01990449,  0.07344452,  0.23902009, -0.03487358,  0.04445324,  0.11245612,
  0.02672872, -0.05593732, -0.0294391 ,  0.01221114, -0.03530062,  0.02106865,
  0.04466159,  0.07613173, -0.00276649,  0.07432033,  0.00587088,  0.08841427,
 -0.03047358,  0.01897122,  0.01050681,  0.0516422 ,  0.03793384, -0.0227941,
  0.08275675,  0.01540308, -0.02840539, -0.00069415,  0.04880542, -0.07487142,
  0.08137497, -0.00311765, -0.07739469, -0.09108017, -0.05470311, -0.00402281,
 -0.00397528, -0.03335127, -0.06701306,  0.0983727 ,  0.08042162, -0.10315099,
  0.05510956,  0.12282867, -0.00043559, -0.07908009,  0.06702596, -0.03829121,
  0.08018652,  0.00874924,  0.10031289, -0.01884892,  0.07059784, -0.00697807,
 -0.1296639 , -0.05789054, -0.10278986, -0.13492332,  0.04678594,  0.03541046,
  0.04028827,  0.05023843,  0.03180973,  0.06767536,  0.01723863,  0.00700823,
  0.01548002,  0.06025029,  0.02722475,  0.06207291,  0.15115986, -0.07104655,
 -0.01920243,  0.02569906, -0.00223317, -0.06430086,  0.03217637, -0.03061147,
  0.04284069, -0.02246788, -0.05035489,  0.0172711 , -0.06543973,  0.04677353,
  0.15730045, -0.02209389, -0.05220712,  0.0893942 , -0.03828736,  0.06815497,
 -0.05026485,  0.01132108, -0.01702544, -0.00570728, -0.05268029,  0.15771667,
  0.00301718, -0.02564368, -0.01002566,  0.04396459, -0.161464  ,  0.10528113,
 -0.0895227 , -0.12006831]
kw_4=np.array([1.0 for i in range(4)])
# kw_4=[ 0.4893939,  -0.23480339,  0.59227593,  0.68247309]
kw_4[1]=-1
# kw_4[3]=-1
s_272=np.zeros((100,272))
s_4=np.zeros((100,4))
for i in range(100):
    for j in range(272):
        s_272[i][j]=1-2*scipy.stats.bernoulli.rvs(0.5)
    for j in range(4):
        s_4[i][j]=1-2*scipy.stats.bernoulli.rvs(0.5)

def cost_calculator(i):
    global kw_4, kw_272, s_272, s_4
    # print(f"in cost_caluculation {i}")
    results_one_cycle=np.dot(one_cycle(ki0[i]),kw_4)**2+(1-np.dot(one_cycle(ki1[i]),kw_4))**2
    return results_one_cycle

def main_calculation(i):
    global kw_4, kw_272, s_272, s_4
    # print("start main_cycle")
    # results[i]=main_cycle()
    # bottom[i]=i/100
    # if(0<i and i<99):
    #     LOSS=results[i]-results[i-1]
    #     kw_272=kw_272-LOSS*s_272[i]/(i/10+10)/100#kw_272=kw_272-(results[i]-results[i-1])*s_272[i]/(i/10+10)/100
    #     kw_4=kw_4-LOSS*s_4[i]/(i+100)/10
    # print(f"i={i}:{results[i]}")
    # return results[i]

    result1[i]=main_cycle()
    c1=1/20
    c2=1/50
    eta=1/(1000+10*i)
    kw_272=kw_272+s_272[i]*c1
    kw_4=kw_4+s_4[i]*c2
    result2[i]=main_cycle()
    kw_272=kw_272-s_272[i]*c1-(result2[i]-result1[i])*s_272[i]*eta/c1
    kw_4=kw_4-s_4[i]*c2-(result2[i]-result1[i])*s_4[i]*eta/c2
    print(f"i={i}:{result1[i]}")
    return result1[i]

def ki_processor(i):
    print(i)
    for j in range(4):
        for k in range(8):
            for l in range(8):
                for o in range(256):
                    # print((E**(I*(zeros_8[i][k][l]*pi/256)))/512)
                    ki0[i][j*16*16*256+(j//2+3+k)*16*256+(j%2+3+l)*256+o]=(np.e**(1j*((zeros_8[i][k][l]-128)*np.pi/256)))/512
                    ki1[i][j*16*16*256+(j//2+3+k)*16*256+(j%2+3+l)*256+o]=(np.e**(1j*((ones_8[i][k][l]-128)*np.pi/256)))/512
                    # ki0[i][k*8*16+l*16+o]=(np.e**(1j*((zeros_8[i][k][l]-8)*np.pi/16)))/32
                    # ki1[i][k*8*16+l*16+o]=(np.e**(1j*((ones_8[i][k][l]-8)*np.pi/16)))/32

def step(x):
    return 1.0 * (x > 0.0)
    
def one_cycle(input):
    num_ones=np.array([0.0 for i in range(8)])
    original_num_ones=function_n2(10,8,input,0)
    # print(original_num_ones)
    num_ones=(original_num_ones-np.average(original_num_ones))*2*pi/sum(original_num_ones)
    ki2=np.array([np.e**(1j*0)/32 for i in range(1024)])
    ki2[2]=ki2[2]*(np.e**(1j*num_ones[0]))
    ki2[5]=ki2[5]*(np.e**(1j*num_ones[1]))
    ki2[6]=ki2[6]*(np.e**(1j*num_ones[2]))
    ki2[7]=ki2[7]*(np.e**(1j*num_ones[3]))
    ki2[8]=ki2[8]*(np.e**(1j*num_ones[4]))
    ki2[9]=ki2[9]*(np.e**(1j*num_ones[5]))
    ki2[10]=ki2[10]*(np.e**(1j*num_ones[6])) 
    ki2[13]=ki2[13]*(np.e**(1j*num_ones[7]))
    ki2[29]=ki2[29]*(np.e**(1j*num_ones[0]))
    ki2[18]=ki2[18]*(np.e**(1j*num_ones[1]))
    ki2[21]=ki2[21]*(np.e**(1j*num_ones[2]))
    ki2[22]=ki2[22]*(np.e**(1j*num_ones[3]))
    ki2[23]=ki2[23]*(np.e**(1j*num_ones[4]))
    ki2[24]=ki2[24]*(np.e**(1j*num_ones[5]))
    ki2[25]=ki2[25]*(np.e**(1j*num_ones[6]))
    ki2[26]=ki2[26]*(np.e**(1j*num_ones[7]))
    ki2[42]=ki2[42]*(np.e**(1j*num_ones[0]))
    ki2[45]=ki2[45]*(np.e**(1j*num_ones[1]))
    ki2[34]=ki2[34]*(np.e**(1j*num_ones[2]))
    ki2[37]=ki2[37]*(np.e**(1j*num_ones[3]))
    ki2[38]=ki2[38]*(np.e**(1j*num_ones[4]))
    ki2[39]=ki2[39]*(np.e**(1j*num_ones[5]))
    ki2[40]=ki2[40]*(np.e**(1j*num_ones[6]))
    ki2[41]=ki2[41]*(np.e**(1j*num_ones[7]))
    ki2[57]=ki2[57]*(np.e**(1j*num_ones[0]))
    ki2[58]=ki2[58]*(np.e**(1j*num_ones[1]))
    ki2[61]=ki2[61]*(np.e**(1j*num_ones[2]))
    ki2[50]=ki2[50]*(np.e**(1j*num_ones[3]))
    ki2[53]=ki2[53]*(np.e**(1j*num_ones[4]))
    ki2[54]=ki2[54]*(np.e**(1j*num_ones[5]))
    ki2[55]=ki2[55]*(np.e**(1j*num_ones[6]))
    ki2[56]=ki2[56]*(np.e**(1j*num_ones[7]))
    for i in range(64):
        for j in range(16):
            ki2[(63-i)*16+j]=ki2[63-i]
    original_num_ones=function_n2(6,4,ki2,1)
    # print(original_num_ones)
    num_ones=np.array([0.0 for i in range(4)])
    num_ones=original_num_ones
    # print(num_ones)
    # print(sum(num_ones))
    return num_ones

multi_ki(100)
multi(100)
# plt.plot(bottom,results)
print(kw_272)
print(kw_4)
print(results)
