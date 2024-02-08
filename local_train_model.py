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
import data_loading

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
    
class training_test:
    num_epoch = 500
    # 350: 102.92738342285156
    # 380: 89.85020446777344
    # 390: Test loss (avg): 75.04996490478516
    # 400: Test loss (avg): 92.92153930664062

    loss_func = torch.nn.functional.cross_entropy
    loss = 0
    min_loss=10000
    #def __init__(self, net, X_train, Y_train, device):
        # 損失関数など設定 初期化するとこある？
    def training(self, net, X_train, Y_train, device):
        net.train()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(net.parameters(),lr=0.001)
        # 学習率
        #dataを機械学習するために変換する
        train = TensorDataset(X_train, Y_train)
        trainloader = DataLoader(train, batch_size=1, shuffle=True)
    
        for epoch in range(self.num_epoch):
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

            prediction_train = net(X_train.to(device)).squeeze(axis = 1)
            loss = criterion(prediction_train, Y_train)
            
            print('Epoch [%d/%d], Loss: %.4f' %(epoch+1, self.num_epoch, loss))
            # R2score
            R2_score = r2_score(Y_train.detach().numpy(), prediction_train.detach().numpy())
            print("\ntrain R2score:",R2_score)

        return net

    def test(self,net, X, Y, max, min, dynamics_number, mean, std, filename):
        input_data = TensorDataset(X, Y)
        data_loader = DataLoader(input_data, batch_size=1)
        prediction_time = 10000
        best_time = Y
        print("mean", mean)
        print("\nstd", std)
        best_dynamics = X[0][dynamics_number]
        # 空のDataFrameを作成
        df = pd.DataFrame()
        #データを変更しながら、よいものを探していく
        #値を変更していくプログラム(3kmを1kmずつに分解するため、どうやって出すべきか、できれば足し合わせたくない)
        #値の条件があるから、最初にマイナスした状態から初めて、一定の上限値まで(純粋にこれまでのデータ等の最大値から最小値までを計算していけばよいのかも)
        for dynamics in range(int(min), int(max)):#いったん適当に範囲を決めて変更してみる
            #もしも、ピッチを変える場合
            if dynamics_number == 8:
                dynamics = float(dynamics)*0.01
                #mean[dynamics_number] =  mean[dynamics_number]*0.01
                X[0][dynamics_number] = dynamics
            elif dynamics_number == 7:
                dynamics = float(dynamics)*0.001
                #mean[dynamics_number] =  mean[dynamics_number]*0.01
                X[0][dynamics_number] = dynamics

            elif dynamics_number == 3 or dynamics_number == 4:
                dynamics = float(dynamics)*0.1
                #mean[dynamics_number] =  mean[dynamics_number]*0.1
                X[0][dynamics_number] = dynamics
            else:
                dynamics = float(dynamics)
                X[0][dynamics_number] = dynamics


            with torch.no_grad():                
                for e, data in enumerate(data_loader, 0):
                    inputs, targets = data
                    #標準化
                    #print("\ninputs",inputs)
                    inputs[0][dynamics_number] = (inputs[0][dynamics_number] - mean[dynamics_number])/std[dynamics_number]
                    #print("std[dynamics_number]", std[dynamics_number])
                    #print("mean[dynamics_number]", mean[dynamics_number])
                    #print("\ninputs標準化",inputs)
                    inputs, targets = inputs.float(), targets.float()
                    targets = targets.reshape((targets.shape[0], 1))
                    outputs = net(inputs)
                    #print('\ndynamics:{} targets:{} output: {}'.format(dynamics, targets, outputs))
                    prediction_time = outputs
                    prediction_time_numpy = prediction_time.numpy().item()
                    # DataFrameに追加するデータ
                    data = {'dynamics': [dynamics], 'prediction_time': [prediction_time_numpy]}
                    # DataFrameを作成
                    df = df.append(pd.DataFrame(data), ignore_index=True)
                    # Excelファイルに追記モードで書き込み

                if best_time > prediction_time:
                    best_time = prediction_time
                    best_dynamics = X[0][dynamics_number]
                    #print("\nbest_time:", best_time, "\nbest_dynamics:", best_dynamics)
        
        #print("best_time:", best_time, "\nbest_pitch:", best_pitch)
        df.to_excel(filename, index=False)
        return net, best_time, best_dynamics
        
    def GCT_balance_test(self,net, X, Y, max, min, dynamics_number, mean, std, filename):
  
        input_data = TensorDataset(X, Y)
        data_loader = DataLoader(input_data, batch_size=1)
        prediction_time = 10000
        best_time = Y
        print("mean", mean)
        print("\nstd", std)
        best_dynamics = X[0][dynamics_number]
        # 空のDataFrameを作成
        df = pd.DataFrame()
        #データを変更しながら、よいものを探していく
        #値を変更していくプログラム(3kmを1kmずつに分解するため、どうやって出すべきか、できれば足し合わせたくない)
        #値の条件があるから、最初にマイナスした状態から初めて、一定の上限値まで(純粋にこれまでのデータ等の最大値から最小値までを計算していけばよいのかも)
        for dynamics in range(int(min), int(max)):#いったん適当に範囲を決めて変更してみる
            dynamics = float(dynamics)*0.1
            #mean[dynamics_number] =  mean[dynamics_number]*0.01
            X[0][dynamics_number] = dynamics
            if dynamics_number==5:
                dynamics2 = float(100-dynamics)
                X[0][dynamics_number+1] = dynamics2
            elif dynamics_number==6:
                dynamics2 = float(100-dynamics)
                X[0][dynamics_number-1] = dynamics2

            with torch.no_grad():                
                for e, data in enumerate(data_loader, 0):
                    inputs, targets = data
                    #標準化
                    #print("\ninputs",inputs)
                    if dynamics_number==5:
                        inputs[0][dynamics_number] = (inputs[0][dynamics_number] - mean[dynamics_number])/std[dynamics_number]
                        inputs[0][dynamics_number+1] = (inputs[0][dynamics_number+1] - mean[dynamics_number+1])/std[dynamics_number+1]
                    elif dynamics_number==6:
                        inputs[0][dynamics_number] = (inputs[0][dynamics_number] - mean[dynamics_number])/std[dynamics_number]
                        inputs[0][dynamics_number-1] = (inputs[0][dynamics_number-1] - mean[dynamics_number-1])/std[dynamics_number-1]
                    #print("\ninputs標準化",inputs)
                    inputs, targets = inputs.float(), targets.float()
                    targets = targets.reshape((targets.shape[0], 1))
                    outputs = net(inputs)
                    #print('\ndynamics1:{} dyamics2:{} targets:{} output: {}'.format(dynamics, dynamics2, targets, outputs))
                    prediction_time = outputs
                    prediction_time_numpy = prediction_time.numpy().item()
                    # DataFrameに追加するデータ
                    data = {'dynamics': [dynamics], 'prediction_time': [prediction_time_numpy]}
                    # DataFrameを作成
                    df = df.append(pd.DataFrame(data), ignore_index=True)
                    # Excelファイルに追記モードで書き込み

                if best_time > prediction_time:
                    best_time = prediction_time
                    best_dynamics = X[0][dynamics_number]
                    #print("\nbest_time:", best_time, "\nbest_dynamics:", best_dynamics)
        
        #print("best_time:", best_time, "\nbest_pitch:", best_pitch)
        df.to_(filename, index=False)
        return net, best_time, best_dynamics

def test2(X_test, Y_test):
    #Test
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    test = TensorDataset(X_test, Y_test)
    testloader = DataLoader(test, batch_size=1, shuffle=True)

    net = Net()
    net.load_state_dict(torch.load('model.pth'))
    net.eval()

    test_loss = 0
    correct = 0
    i = 0
    a = 0
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            inputs, targets = data
            #inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            if loss > 100:
                print('\ndata', data)
                print('loss', loss)
            test_loss += loss
            #print('\nTest loss (avg)', loss)
            #pred = outputs.argmax(dim=1, keepdim=True)
            a = a+1
            
    test_loss = test_loss/a
    print('\nTest loss (avg): {}, Accuracy: {}'.format(test_loss,
                                                         correct / 10000))

    prediction_test = net(X_test.to(device)).squeeze(axis = 1)
    loss = criterion(prediction_test, Y_test)
    # R2score
    R2_score = r2_score(Y_test.detach().numpy(), prediction_test.detach().numpy())
    print("\ntest R2score:",R2_score)
    print("\nLoss:", loss)

def csv_to_csv(csv_file, output_csv_file):
    # CSVファイルを読み込む
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)

        # CSVファイルにデータを追加する
        with open(output_csv_file, 'a', newline='') as file:
            csv_writer = csv.writer(file)

            for row in csv_reader:
                # データをCSVファイルに書き込む
                csv_writer.writerow(row)

    print("変換が完了しました。")


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # インスタンス生成
    net = Net()
    # 損失関数： 平均2乗誤差
    criterion = nn.MSELoss()
    loss_func = torch.nn.functional.cross_entropy
    #def dataloaderから整形されたデータを受け取る
    
    #テスト用データと訓練用データに分割しない
    #X_train01, Y_train01, X1_train, Y1_train, X2_train, Y2_train, X3_train, Y3_train = data_loading(all_data = 'personal_data/all_members_data1.csv', test_data = '3km_traindata.csv')
     #テスト用データと訓練用データを７対３に分割したデータが、X_train00, X_test00, Y_train00, Y_test00　
    trainloader, testloader, standard_train, standard_test = data_loading.load(all_data = 'personal_data/all_members_data_original.csv', test_data = '3km_traindata.csv')
   


    train = training_test()
    net = train.training(net, trainloader, device)
    torch.save(net.state_dict(), 'model.pth')


    net = Net()
    net.load_state_dict(torch.load('model.pth'))
    net.eval()

    test_loss = 0
    correct = 0
    i = 0
    a = 0
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            inputs, targets = data
            #inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            if loss > 100:
                print('\ndata', data)
                print('loss', loss)
            test_loss += loss
            #print('\nTest loss (avg)', loss)
            #pred = outputs.argmax(dim=1, keepdim=True)
            a = a+1
            
    test_loss = test_loss/a
    print('\nTest loss (avg): {}, Accuracy: {}'.format(test_loss,
                                                         correct / 10000))

    prediction_test = net(X_test.to(device)).squeeze(axis = 1)
    loss = criterion(prediction_test, Y_test)
    # R2score
    R2_score = r2_score(Y_test.detach().numpy(), prediction_test.detach().numpy())
    print("\ntest R2score:",R2_score)
    print("\nLoss:", loss)

 

if __name__ == '__main__':
    main()
