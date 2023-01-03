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
from sklearn import datasets
from sklearn.model_selection import train_test_split 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset
from torch.nn import functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
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
zeros_7=np.zeros((100,7,7))
ones_7=np.zeros((100,7,7))
for i in range(100):
    zeros_7[i]=np.asarray(Image.fromarray(np.uint8(zeros_original[i])).resize((7,7)))
    ones_7[i]=np.asarray(Image.fromarray(np.uint8(ones_original[i])).resize((7,7)))

zeros_row_7=np.zeros((100,49))
ones_row_7=np.zeros((100,49))
for i in range(100):
    for j in range(7):
        for k in range(7):
            zeros_row_7[i][j*7+k]=zeros_7[i][j][k]-128
            ones_row_7[i][j*7+k]=ones_7[i][j][k]-128
            
def function_n2(n,m,ki):
    # print(ki)
    # time.sleep(100)
    vertical_of_A=np.zeros(n)
    global kw_64
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
        MCRZ(FLAG2[i],n+m,state,kw_64[i])#qubit数変えるときここも変える
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
    # result = p.map(cost_calculator, range(n))
    res= p.map(cost_calculator, range(n))
    # for i in range(n):
    #     ki_processor(i)
    return n,res

def main_cycle():
    i,data= multi_cycle(100)
    result_cycle=0
    train_X=[]
    label=[]
    for j in data:
        train_X.append(j[0])
        train_X.append(j[1])
        label.append(0)
        label.append(1)
    train_X = torch.FloatTensor(train_X) #float32のtensorに変換
    train_Y = torch.FloatTensor(label) 
    train = TensorDataset(train_X, train_Y)
    #ミニバッチのサイズ
    BATCH_SIZE = 8 
    #訓練用データのDataloaderを作成
    train_dataloader = DataLoader(
        train,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    for epoch in range(100):
        for batch, label in train_dataloader: #エポックのループの内側で、さらにデータローダーによるループ
            optimizer.zero_grad()
            t_p = net(batch)
            label = label.unsqueeze(1) #損失関数に代入するために次元を調節する処理(気にしなくて大丈夫です)
            loss = criterion(t_p,label)
            loss.backward()
            optimizer.step()
        # y_axis_list.append(loss.detach().numpy())#プロット用のy軸方向リストに損失の値を代入
        if epoch % 10 == 0:#10エポック毎に損失の値を表示
          print("epoch: %d  loss: %f" % (epoch+1 ,float(loss)))

    with torch.no_grad():# 試験用データでは勾配を計算しない
        pred_labels = [] # 各バッチごとの結果格納用
        preds=[]
        for x in train_X:
            pred = net(x)
            preds.append(pred)
            pred_label = discretize(pred) #離散化する
            pred_labels.append(pred_label[0])
    pred_labels = np.array(pred_labels) #numpy arrayに変換
    accuracy=0
    for j in range(len(train_X)):
        if(pred_labels[j]==train_Y[j]):accuracy+=1
        result_cycle+=(train_Y[j]-preds[j])**2
        print(train_Y[j],preds[j])

    print(f"accuracy={accuracy}")

    return result_cycle

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6, 10)
        self.fc2 = nn.Linear(10, 1)
        self.sigmoid = torch.nn.Sigmoid() #出力層の活性化関数はsigmoid関数を使用

    # 順伝播
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x    

# インスタンス化
net = Net()
# 損失関数の設定(BCE損失)
criterion = nn.BCELoss()
# 最適化手法の選択(SGD)
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

result1=np.array([0.0 for i in range (100)])
result2=np.array([0.0 for i in range (100)])
bottom=np.array([0.0 for i in range (100)])
ki0=np.array([[1.0*2**(-7)+0.0j for i in range(4*8*8*64)] for j in range(100)])
ki1=np.array([[1.0*2**(-7)+0.0j for i in range(4*8*8*64)] for j in range(100)])
kw_64=np.array([0.0 for j in range (64)])
s_64=np.zeros((100,64))
for i in range(100):
    for j in range(64):
        s_64[i][j]=1-2*scipy.stats.bernoulli.rvs(0.5)

# 離散化を行う関数
def discretize(proba):
    threshold = torch.Tensor([0.5]) # 0か1かを分ける閾値を0.5に設定
    discretized = (proba >= threshold).int() # 閾値未満で0、以上で1に変換
    return discretized

def cost_calculator(i):
    global kw_64, s_64
    # print(f"in cost_caluculation {i}")
    return [function_n2(8,6,ki0[i]),function_n2(8,6,ki1[i])]

def cost_progress(i):
    global kw_64, s_64
    num_ones=np.array([0.0 for i in range(8)])
    original_num_ones=function_n2(8,6,ki1[i],0)
    # print(original_num_ones)
    num_ones=(original_num_ones-np.average(original_num_ones))*2*pi/sum(original_num_ones)
    #print(f"{i}:{num_ones}")


def main_calculation(i):
    global kw_64, s_64
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
    eta=1/(1000+10*i)
    kw_64=kw_64+s_64[i]*c1
    result2[i]=main_cycle()
    kw_64=kw_64-s_64[i]*c1-(result2[i]-result1[i])*s_64[i]*eta/c1
    # kw_4=kw_4-s_4[i]*c2-(result2[i]-result1[i])*s_4[i]*eta/c2ここ変えました！！！
    print(f"i={i}:{result1[i]}")
    return result1[i]

def ki_processor(i):
    print(i)
    for j in range(4):
        for k in range(7):
            for l in range(7):
                for o in range(64):
                    # print((E**(I*(zeros_8[i][k][l]*pi/256)))/512)
                    ki0[i][j*8*8*64+(j//2+k)*8*64+(j%2+l)*64+o]=(np.e**(1j*((zeros_7[i][k][l]-128)*np.pi/256)))/128
                    ki1[i][j*8*8*64+(j//2+k)*8*64+(j%2+l)*64+o]=(np.e**(1j*((ones_7[i][k][l]-128)*np.pi/256)))/128
                    # ki0[i][k*8*16+l*16+o]=(np.e**(1j*((zeros_8[i][k][l]-8)*np.pi/16)))/32
                    # ki1[i][k*8*16+l*16+o]=(np.e**(1j*((ones_8[i][k][l]-8)*np.pi/16)))/32

def step(x):
    return 1.0 * (x > 0.0)
    
multi_ki(100)
multi(100)
# main_cycle()
# plt.plot(bottom,results)
print(kw_64)
print(result1)
#最後の4qubitだけ