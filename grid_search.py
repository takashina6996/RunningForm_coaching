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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
#from torch.optim import LBFGS 

import warnings
warnings.simplefilter('ignore', FutureWarning)
import pandas as pd

from torch.utils.data import random_split
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

def test(net, X, Y, max, min, dynamics_number, mean, std, filename):
    input_data = TensorDataset(X, Y)
    data_loader = DataLoader(input_data, batch_size=1)
    prediction_time = 10000
    best_time = 1000000
    best_dynamics_number = 0
    number = 0
    #print("mean", mean)
    #print("\nstd", std)
    best_dynamics = X[0][dynamics_number]
    # 空のlistを作成
    df = []
    #データを変更しながら、よいものを探していく
    #値を変更していくプログラム(3kmを1kmずつに分解するため、どうやって出すべきか、できれば足し合わせたくない)
    #値の条件があるから、最初にマイナスした状態から初めて、一定の上限値まで(純粋にこれまでのデータ等の最大値から最小値までを計算していけばよいのかも)
    if dynamics_number == 7:
        for dynamics in range(int(min), int(max)):#いったん適当に範囲を決めて変更してみる
            #もしも、ピッチを変える場合
            
            #dynamics = int(float(dynamics)*0.01)
            #mean[dynamics_number] =  mean[dynamics_number]*0.01
            dynamics = float(dynamics)*0.01
            X[0][dynamics_number] = dynamics

            with torch.no_grad():                
                for e, data in enumerate(data_loader, 0):
                    inputs, targets = data
                    #標準化
                    #print("\ninputs",inputs)
                    inputs[0][dynamics_number] = (inputs[0][dynamics_number] - mean[dynamics_number])/std[dynamics_number]
                    inputs, targets = inputs.float(), targets.float()
                    targets = targets.reshape((targets.shape[0], 1))
                    outputs = net(inputs)
                    #print('\ndynamics:{} targets:{} output: {}'.format(dynamics, targets, outputs))
                    prediction_time = outputs
                    prediction_time_numpy = prediction_time.numpy().item()
                    # DataFrameに追加するデータ
                    data = {'dynamics': [int(dynamics*100)], 'prediction_time': [prediction_time_numpy]}
                  #listの作成
                    df = []
                    df.append(data)
                if best_time > prediction_time:
                    best_time = prediction_time
                    best_dynamics = X[0][dynamics_number]
                    best_dynamics_number = number
                    #print("best_dynamics_number", best_dynamics_number)                        

                        #print("\nbest_time:", best_time, "\nbest_dynamics:", best_dynamics)
            number = number + 1
        result = pd.DataFrame(df)
        result.to_csv(filename, index=False)
        return net, best_time, best_dynamics, best_dynamics_number
    else:
        for dynamics in range(int(min), int(max)):
            if dynamics_number == 8:
                dynamics = float(dynamics)*0.1

                #mean[dynamics_number] =  mean[dynamics_number]*0.01
                X[0][dynamics_number] = dynamics

            elif dynamics_number == 3 or dynamics_number == 4:
                #dynamics = int(float(dynamics)*0.1)
                #mean[dynamics_number] =  mean[dynamics_number]*0.1
                X[0][dynamics_number] = dynamics
            else:
                #dynamics = int(float(dynamics))
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
                    if dynamics_number == 8:
                       data = {'dynamics': [dynamics*100], 'prediction_time': [prediction_time_numpy]}
                    else:
                       data = {'dynamics': [dynamics], 'prediction_time': [prediction_time_numpy]}                    
                    # listを作成
                    df_else = []
                    df_else.append(data)
                if best_time > prediction_time:
                    best_time = prediction_time
                    best_dynamics = X[0][dynamics_number]
                    best_dynamics_number = number
            number = number + 1
        #dataframeに変換
        result = pd.DataFrame(df_else)
        # csvファイルに追記モードで書き込み
        result.to_csv(filename, index=False)
        return net, best_time, best_dynamics, best_dynamics_number

def GCT_balance_test(net, X, Y, max, min, dynamics_number, mean, std, filename):
        input_data = TensorDataset(X, Y)
        data_loader = DataLoader(input_data, batch_size=1)
        prediction_time = 10000
        best_time = 1000000
        best_dynamics_number = 0
        number = 0
        #print("mean", mean)
        #print("\nstd", std)
        best_dynamics = X[0][dynamics_number]
        #データを変更しながら、よいものを探していく
        #値を変更していくプログラム(3kmを1kmずつに分解するため、どうやって出すべきか、できれば足し合わせたくない)
        #値の条件があるから、最初にマイナスした状態から初めて、一定の上限値まで(純粋にこれまでのデータ等の最大値から最小値までを計算していけばよいのかも)
        for dynamics in range(int(min), int(max)):#いったん適当に範囲を決めて変更してみる
            dynamics = float(dynamics)
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
                    df_GCT = []
                    df_GCT.append(data)

                    # csvファイルに追記モードで書き込み

                if best_time > prediction_time:
                    best_time = prediction_time
                    best_dynamics = X[0][dynamics_number]
                    best_dynamics_number = number
            number = number + 1
        result = pd.DataFrame(df_GCT)
        # csvファイルに追記モードで書き込み
        result.to_csv(filename, index=False)
        #if dynamics_number==5:
            #print("右")
        #elif dynamics_number==6:
            #print("左")
        return net, best_time, best_dynamics, best_dynamics_number

def main(trials_number, runner_height, X_train01, Y_train01):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # インスタンス生成
    net = Net()
    runner_height = str(runner_height)
    # 損失関数： 平均2乗誤差
    criterion = nn.MSELoss()
    loss_func = torch.nn.functional.cross_entropy
    #"Permission error 13" エラーが出てしまう問題で、search_optimal_form.pyにて、データを作成することにした

    #numpyに変換
    X_train02 = X_train01.to_numpy()
    Y_train = Y_train01.to_numpy()
    #X_trainを標準化する。Yはしない
    scaler = StandardScaler()
    X_train= scaler.fit_transform(X_train02)
    mean = scaler.mean_
    var = scaler.var_
    std = np.sqrt(var)
    X_train= torch.from_numpy(X_train).float()
    Y_train = torch.from_numpy(Y_train).float()
    #dataを機械学習するために変換する
    train = TensorDataset(X_train, Y_train)
    trainloader = DataLoader(train, batch_size=1, shuffle=True)

    optimizer = optim.Adam(net.parameters(),lr=0.001)
    net = Net()
    net.load_state_dict(torch.load('model.pth'))
    net.eval()
    #X, Y, X1, Y1, X2, Y2, X3, Y3 = input_data_loading()
    X_test01, Y_test01, standard_train, standard_test = data_loading.load_test(all_data = 'personal_data/'+ runner_height + 'cm_data.csv', test_data = "test_data/" + runner_height + "cm_testdata.csv")
    #X_test01, Y_test01, X1_test, Y1_test, X2_test, Y2_test, X3_test, Y3_test = data_loading(all_data = 'personal_data/sibata_data.csv', test_data = 'kyoshiro_testdata.csv')
    
    #numpyに変換
    X_test02 = X_test01.to_numpy()
    Y_test = Y_test01.to_numpy()

    for i in range(len(X_test02)):
        for e in range(16):
            X_test02[i][e] = (X_test02[i][e] - mean[e])/std[e]
    
    X_test= torch.from_numpy(X_test02).float()
    Y_test = torch.from_numpy(Y_test).float()
    X = []

    X.append(X_test[trials_number])
    X.append(X_test[trials_number])
    X.append(X_test[trials_number])
    Y = []
    Y.append(Y_test[trials_number])
    Y.append(Y_test[trials_number])
    Y.append(Y_test[trials_number])
    X = torch.stack(X)
    Y = torch.stack(Y) 
    input_data = TensorDataset(X, Y)
    data_loader = DataLoader(input_data, batch_size=1)

    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))
            outputs = net(inputs)
            #print('\npitch:{}, targets: {}, output: {}'.format(inputs[0][1], targets,
            #                                                    outputs))
            loss = criterion(outputs, targets)
    lap = 2
    # pitch
    X_train_new= torch.from_numpy(X_train02).float()
    #print("\n---------------------------------------------------------------------------------------------")
    #print("pitch")
    X = []

    X.append(X_test[trials_number])
    X.append(X_test[trials_number])
    X.append(X_test[trials_number])
    Y = []
    Y.append(Y_test[trials_number])
    Y.append(Y_test[trials_number])
    Y.append(Y_test[trials_number])
    X = torch.stack(X)
    Y = torch.stack(Y) 
    X1 = X[lap].view(1, -1)


    Y1 = Y[lap].view(1, -1)    
    net = Net()
    net.load_state_dict(torch.load('model.pth'))
    net.eval()
    dynamics_number = 3
    pitch_maxmin = X_train01.at[0, 'ave_pitch']*0.1
    max1 = X_train01.at[0, 'ave_pitch'] + pitch_maxmin
    min1 = X_train01.at[0, 'ave_pitch'] - pitch_maxmin
 

    filename = 'prediction_data/pitch_output_' + runner_height + '.csv'
    net, best_time, best_pitch, best_pitch_number = test(net, X1, Y1, max1, min1, dynamics_number, mean, std, filename)
    #print("best_time:", best_time, "\nbest_pitch:", best_pitch)

    #ground_time
    #print("---------------------------------------------------------------------------------------------")
    #print("ground_time")
    X = []
 
    X.append(X_test[trials_number])
    X.append(X_test[trials_number])
    X.append(X_test[trials_number])
    Y = []
    Y.append(Y_test[trials_number])
    Y.append(Y_test[trials_number])
    Y.append(Y_test[trials_number])
    X = torch.stack(X)
    Y = torch.stack(Y)    
    X2 = X[lap].view(1, -1)
    Y2 = Y[lap].view(1, -1)    

    net = Net()
    net.load_state_dict(torch.load('model.pth'))
    net.eval()
    #train = training_test()
    dynamics_number = 4
    ground_time_maxmin = X_train01.at[0, 'ave_grond']*0.1
    max2 = X_train01.at[0, 'ave_grond'] + ground_time_maxmin
    min2 = X_train01.at[0, 'ave_grond'] - ground_time_maxmin

    filename = 'prediction_data/ground_time_output_' + runner_height + '.csv'
    net, best_time, best_ground, best_ground_number = test(net, X2, Y2, max2, min2, dynamics_number, mean, std, filename)
    #print("best_time:", best_time, "\nbest_ground_time:", best_ground)

    #stride
    #print("---------------------------------------------------------------------------------------------")
    #print("stride")
    X = []
    X.append(X_test[trials_number])
    X.append(X_test[trials_number])
    X.append(X_test[trials_number])
    Y = []
    Y.append(Y_test[trials_number])
    Y.append(Y_test[trials_number])
    Y.append(Y_test[trials_number])
    X = torch.stack(X)
    Y = torch.stack(Y)    
    X3 = X[lap].view(1, -1)
    Y3 = Y[lap].view(1, -1)    
    net = Net()
    net.load_state_dict(torch.load('model.pth'))
    net.eval()
    dynamics_number = 7
    stride_maxmin = X_train01.at[0, 'ave_stride']*0.1
    max3 = X_train01.at[0, 'ave_stride'] + stride_maxmin
    min3 = X_train01.at[0, 'ave_stride'] - stride_maxmin
    max3 = max3*100
    min3 = min3*100
    filename = 'prediction_data/stride_output_' + runner_height + '.csv'
    net, best_time, best_stride, best_stride_number = test(net, X3, Y3, max3, min3, dynamics_number, mean, std, filename)
    #print("best_time:", best_time, "\nbest_stride:", best_stride)

    #vertical_motion
    #print("---------------------------------------------------------------------------------------------")
    #print("vertical_motion")
    X = []
    X.append(X_test[trials_number])
    X.append(X_test[trials_number])
    X.append(X_test[trials_number])
    Y = []
    Y.append(Y_test[trials_number])
    Y.append(Y_test[trials_number])
    Y.append(Y_test[trials_number])
    X = torch.stack(X)
    Y = torch.stack(Y)     
    X4 = X[lap].view(1, -1)
    Y4 = Y[lap].view(1, -1)    
    net = Net()
    dynamics_number = 8
    net.load_state_dict(torch.load('model.pth'))
    net.eval()
    stride_maxmin = X_train01.at[0, 'ave_vertical_motion']*0.1
    max4 = X_train01.at[0, 'ave_vertical_motion'] + stride_maxmin
    min4 = X_train01.at[0, 'ave_vertical_motion'] - stride_maxmin
    max4 = max4*10
    min4 = min4*10
    #print("max:{} min{}".format(max4, min4))
    filename = 'prediction_data/vertical_motion_output_' + runner_height + '.csv'
    net, best_time, best_ave_vertical_motion, best_ave_vertical_motion_number = test(net, X4, Y4, max4, min4, dynamics_number, mean, std, filename)
    #print("best_time:", best_time, "\nbest_ave_vertical_motion:", best_ave_vertical_motion)

    #GCT balance
    #print("---------------------------------------------------------------------------------------------")
    #print("GCT_balance")
    X = []
    X.append(X_test[trials_number])
    X.append(X_test[trials_number])
    X.append(X_test[trials_number])
    Y = []
    Y.append(Y_test[trials_number])
    Y.append(Y_test[trials_number])
    Y.append(Y_test[trials_number])
    X = torch.stack(X)
    Y = torch.stack(Y)     
    X5 = X[lap].view(1, -1)
    Y5 = Y[lap].view(1, -1)    
    net = Net()
    net.load_state_dict(torch.load('model.pth'))
    net.eval()
    max5 = 53
    right_min5 = min(X_train01["right_GCT"])#最小値
    left_min5 = min(X_train01["left_GCT"])#最小値

    if right_min5 > left_min5:
        min5 =  left_min5
        dynamics_number = 6
    else:
        min5 =  right_min5
        dynamics_number = 5
    GCT_dynamics_number = dynamics_number
    #print("max:{} min{}".format(max5, min5))
    filename = 'prediction_data/GCT_balance_output_' + runner_height + '.csv'
    net, best_time, best_GCT_balance, best_GCT_balance_number = GCT_balance_test(net, X5, Y5, max5, min5, dynamics_number, mean, std, filename)
    #print("best_time:", best_time, "\nGCT_balance:", best_GCT_balance)

    return best_GCT_balance, best_ground, best_pitch, best_stride, best_ave_vertical_motion, best_GCT_balance_number, best_ground_number, best_pitch_number, best_stride_number, best_ave_vertical_motion_number, GCT_dynamics_number

if __name__ == '__main__':
    trials_number = 0
    X_train01, Y_train01 = data_loading.load(all_data = 'personal_data/all_members_data_original.csv', test_data = '3km_testdata.csv')
    best_GCT_balance, best_ground, best_pitch, best_stride, best_ave_vertical_motion, best_GCT_balance_number, best_ground_number, best_pitch_number, best_stride_number, best_ave_vertical_motion_number, GCT_dynamics_number= main(trials_number, str(177), X_train01, Y_train01)
