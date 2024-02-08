# 10/31　最新　モデルを学習させる用のコード　l
import csv
import math
from re import M
import numpy as np
import torch
import torch.nn.functional
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
#from torchmetrics.functional import r2_score
import sys
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
#from torch.optim import LBFGS 
import pandas as pd
from torch.utils.data import random_split
import torch
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings
import flwr as fl
from collections import OrderedDict
import data_loading

warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self,):
        #  親クラスnn.Modulesの初期化呼び出し
        super().__init__()

        # 出力層の定義
        self.l1 = nn.Linear(16, 30) 
        self.l2 = nn.Linear(30, 1)
        
    # 予測関数の定義
    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        #x = x.unsqueeze(1) 
        return x

def load_data():
    #データの読み込み
    df = pd.read_csv("3km_traindata.csv")
    #説明変数と目的変数に分割
    X = df.drop(["time","accumulated_time", 'ave_move_pace'], axis = 1)#axisって何？
    Y = df["time"]

    
    #訓練データとテストデータに分割
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,  random_state = 0)
    
    #標準化処理
    scaler_train = StandardScaler()
    X_train = scaler_train.fit_transform(X_train)
    mean_train = scaler_train.mean_
    var_train = scaler_train.var_
    std_train = np.sqrt(var_train)
    X_train = torch.from_numpy(X_train).float()
    Y_train = torch.tensor(Y_train.values) 
    Y_train = Y_train.to(device)

    scaler_test = StandardScaler()
    X_test = scaler_test.fit_transform(X_test)
    mean_test = scaler_test.mean_
    var_test = scaler_test.var_
    std_test = np.sqrt(var_test)
    X_test = torch.from_numpy(X_test).float()
    Y_test = torch.tensor(Y_test.values) 
    Y_test = Y_test.to(device)

    trainset = TensorDataset(X_train, Y_train)
    testset = TensorDataset(X_test, Y_test)
    return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)

def train(net, trainloader, epochs):
    loss_func = torch.nn.functional.cross_entropy 
    #定義
    net.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(),lr=0.001)
    
    # 学習
    for epoch in range(epochs):
        for i, data in enumerate(trainloader, 0):
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))
            # Zero the gradients
            optimizer.zero_grad()
            # Perform forward pass
            outputs = net(inputs)
            # Compute loss
            loss = criterion(outputs, targets)
            # Perform backward pass
            loss.backward()
            optimizer.step() 
        
        #X_train, Y_train = trainloader
        #prediction_train = net(X_train.to(device)).squeeze(axis = 1)
        #loss = criterion(prediction_train, Y_train)
        
        #print('Epoch [%d/%d], Loss: %.4f' %(epoch+1, epochs, loss))
        # R2score
        #R2_score = r2_score(Y_train.detach().numpy(), prediction_train.detach().numpy())
        #print("\ntrain R2score:",R2_score)
        torch.save(net.state_dict(), 'model.pth')

def test(net, testloader):
    loss_func = torch.nn.functional.cross_entropy 
    #定義
    net.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(),lr=0.001)
    #Test
    test_loss = 0
    epoch = 0
    correct = 0
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            inputs, targets = data
            #inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss
            #print('\nTest loss (avg)', loss)
            #pred = outputs.argmax(dim=1, keepdim=True)
            epoch = epoch+1

    #X_test, Y_test = testloader
    #test_loss = test_loss/a
    #print('\nTest loss (avg): {}, Accuracy: {}'.format(test_loss,
    #                                                     correct / 10000))

    #prediction_test = net(X_test.to(device)).squeeze(axis = 1)
    #loss = criterion(prediction_test, Y_test)
    # R2score
    #R2_score = r2_score(Y_test.detach().numpy(), prediction_test.detach().numpy())
    #print("\ntest R2score:",R2_score)
    #print("\nLoss:", loss)
    return test_loss/epoch, correct

# インスタンス生成
def load_model():
    return Net().to(device)

trainloader, testloader = load_data()
if __name__ == "__main__":
    net = load_model()
    criterion = nn.MSELoss()
    loss_func = torch.nn.functional.cross_entropy
    trainloader, testloader, X_train, X_test, Y_train, Y_test = data_loading.load_train(all_data = 'personal_data/all_members_data_original.csv', test_data = '3km_testdata.csv')
    train(net, trainloader, 50000)
    torch.save(net.state_dict(), 'model.pth')

    prediction_test = net(X_test.to(device)).squeeze(axis = 1)
    loss = criterion(prediction_test, Y_test)
    # R2score
    R2_score = r2_score(Y_test.detach().numpy(), prediction_test.detach().numpy())
    print("\ntest R2score:",R2_score)
    print("\nLoss:", loss)
    print(f"Loss:{loss:.5f}")


