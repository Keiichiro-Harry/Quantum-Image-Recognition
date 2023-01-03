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
            
def function_n2(n,m,ki):
    # print(ki)
    # time.sleep(100)
    vertical_of_A=np.zeros(n)
    bit=np.array([[0 for i in range(n)] for j in range(pow(2,n))], dtype=object)
    gate=np.array([[0 for i in range(n)] for j in range(pow(2,n))], dtype=object)
    gate2=np.array([[0 for i in range(n+1)] for j in range(pow(2,n))], dtype=object)
    FLAG=np.array([0 for i in range(pow(2,m))])
    FLAG2=np.array([[1 for i in range(n+m+1)] for j in range(2**m)], dtype=object)
    FLAG3=np.array([[0 for i in range(m)] for j in range(2**m)], dtype=object)
    kw=np.array([pi/4 for i in range(2**m)])
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
        MCRZ(FLAG2[i],n+m,state,kw[i])
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
def main_ki():
    i,data = multi_ki(100)
    for j in data:
        pass
results=np.array([0.0 for i in range (100)])
bottom=np.array([0.0 for i in range (100)])
ki0=np.array([[1.0*2**(-9)+0.0j for i in range(4*16*16*256)] for j in range(100)])
ki1=np.array([[1.0*2**(-9)+0.0j for i in range(4*16*16*256)] for j in range(100)])
kw_272=np.array([0.0 for j in range (272)])
kw_4=np.array([1.0 for i in range(4)])
kw_4[1]=-1
# kw_4[3]=-1
s_272=np.zeros((100,272))
s_4=np.zeros((100,4))
for i in range(100):
    for j in range(272):
        s_272[i][j]=1-2*scipy.stats.bernoulli.rvs(0.5)
    for j in range(4):
        s_4[i][j]=1-2*scipy.stats.bernoulli.rvs(0.5)

def main_calculation(i):
    global kw_4, kw_272, s_272, s_4
    results[i]=np.dot(one_cycle(ki0[i]),kw_4)**2+(1-np.dot(one_cycle(ki1[i]),kw_4))**2
    bottom[i]=i/100
    if(0<i and i<99):
        LOSS=results[i]-results[i-1]
        kw_272=kw_272-LOSS*s_272[i]/(i+100)
        kw_4=kw_4-LOSS*s_4[i]/(i+100)
    print(f"i={i}:{results[i]}")
    return results[i]

def ki_processor(i):
    print(i)
    for j in range(4):
        for k in range(8):
            for l in range(8):
                for o in range(256):
                    # print((E**(I*(zeros_8[i][k][l]*pi/256)))/512)
                    ki0[i][j*16*16*256+(j//2+3+k)*16*256+(j%2+3+l)*256+o]=(np.e**(1j*((zeros_8[i][k][l]-128)*np.pi/256)))/512
                    ki1[i][j*16*16*256+(j//2+3+k)*16*256+(j%2+3+l)*256+o]=(np.e**(1j*((ones_8[i][k][l]-128)*np.pi/256)))/512


def step(x):
    return 1.0 * (x > 0.0)
    
def one_cycle(input):
    num_ones=np.array([0.0 for i in range(8)])
    original_num_ones=function_n2(10,8,input)
    print(original_num_ones)
    num_ones=(original_num_ones-np.average(original_num_ones))*2*pi/sum(original_num_ones)
    ki2=np.array([E**(I*0)/32 for i in range(1024)])
    ki2[2]=ki2[2]*(E**(I*num_ones[0]))
    ki2[5]=ki2[5]*(E**(I*num_ones[1]))
    ki2[6]=ki2[6]*(E**(I*num_ones[2]))
    ki2[7]=ki2[7]*(E**(I*num_ones[3]))
    ki2[8]=ki2[8]*(E**(I*num_ones[4]))
    ki2[9]=ki2[9]*(E**(I*num_ones[5]))
    ki2[10]=ki2[10]*(E**(I*num_ones[6])) 
    ki2[13]=ki2[13]*(E**(I*num_ones[7]))
    ki2[29]=ki2[29]*(E**(I*num_ones[0]))
    ki2[18]=ki2[18]*(E**(I*num_ones[1]))
    ki2[21]=ki2[21]*(E**(I*num_ones[2]))
    ki2[22]=ki2[22]*(E**(I*num_ones[3]))
    ki2[23]=ki2[23]*(E**(I*num_ones[4]))
    ki2[24]=ki2[24]*(E**(I*num_ones[5]))
    ki2[25]=ki2[25]*(E**(I*num_ones[6]))
    ki2[26]=ki2[26]*(E**(I*num_ones[7]))
    ki2[42]=ki2[42]*(E**(I*num_ones[0]))
    ki2[45]=ki2[45]*(E**(I*num_ones[1]))
    ki2[34]=ki2[34]*(E**(I*num_ones[2]))
    ki2[37]=ki2[37]*(E**(I*num_ones[3]))
    ki2[38]=ki2[38]*(E**(I*num_ones[4]))
    ki2[39]=ki2[39]*(E**(I*num_ones[5]))
    ki2[40]=ki2[40]*(E**(I*num_ones[6]))
    ki2[41]=ki2[41]*(E**(I*num_ones[7]))
    ki2[57]=ki2[57]*(E**(I*num_ones[0]))
    ki2[58]=ki2[58]*(E**(I*num_ones[1]))
    ki2[61]=ki2[61]*(E**(I*num_ones[2]))
    ki2[50]=ki2[50]*(E**(I*num_ones[3]))
    ki2[53]=ki2[53]*(E**(I*num_ones[4]))
    ki2[54]=ki2[54]*(E**(I*num_ones[5]))
    ki2[55]=ki2[55]*(E**(I*num_ones[6]))
    ki2[56]=ki2[56]*(E**(I*num_ones[7]))
    for i in range(64):
        for j in range(16):
            ki2[(63-i)*16+j]=ki2[63-i]
    original_num_ones=function_n2(6,4,ki2)
    print(original_num_ones)
    num_ones=np.array([0.0 for i in range(4)])
    num_ones=original_num_ones
    print(num_ones)
    # print(sum(num_ones))
    return num_ones

multi_ki(100)
multi(100)
# plt.plot(bottom,results)
print(results)
