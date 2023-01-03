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
from multiprocessing import Pool
from multiprocessing import Process
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
            
def main_cycle():
    result_cycle=0
    train_X=[]
    label=[]
    for j in zeros_row_7:
        train_X.append(j)
        label.append(0)
    for j in ones_row_7:
        train_X.append(j)
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

    for epoch in range(200):
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

    print(f"accuracy={accuracy/len(train_X)}")

    return result_cycle

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(49, 100)
        self.fc2 = nn.Linear(100, 1)
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

# 離散化を行う関数
def discretize(proba):
    threshold = torch.Tensor([0.5]) # 0か1かを分ける閾値を0.5に設定
    discretized = (proba >= threshold).int() # 閾値未満で0、以上で1に変換
    return discretized

main_cycle()