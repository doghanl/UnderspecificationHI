# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 17:44:53 2024

@author: zihanli1
"""

import torch
#import os
print(torch.__version__)
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from torch import nn
#from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import KFold
import torch.optim as optim
from statistics import mean
import pandas as pd
from sklearn import preprocessing
import scipy.linalg as la
from numpy.linalg import pinv
from scipy.linalg import eig
#from re import L
#import multiprocessing
#from multiprocessing import Pool, Manager, Queue
#import autograd.numpy as np

#%%

class customerDataset():

    def Import(self, FD):
        # read training data - It is the aircraft engine run-to-failure data.
        train_df = pd.read_csv('/Users/lizihan/Desktop/Domain-guided nonlinear health index construction/Data/c-mapss/c-mapss/train_FD00' + FD + '.txt', sep=" ", header=None)
        train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)
        train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                            's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                            's15', 's16', 's17', 's18', 's19', 's20', 's21']

        train_df = train_df.sort_values(['id','cycle'])

        # read test data - It is the aircraft engine operating data without failure events recorded.
        test_df = pd.read_csv('/Users/lizihan/Desktop/Domain-guided nonlinear health index construction/Data/c-mapss/c-mapss/test_FD00'+FD+'.txt', sep=" ", header=None)
        test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)
        test_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                            's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                            's15', 's16', 's17', 's18', 's19', 's20', 's21']

        # read ground truth data - It contains the information of true remaining cycles for each engine in the testing data.
        truth_df = pd.read_csv('/Users/lizihan/Desktop/Domain-guided nonlinear health index construction/Data/c-mapss/c-mapss/RUL_FD00'+FD+'.txt', sep=" ", header=None)
        truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)

        ##################################
        # Data Preprocessing
        ##################################

        #######
        # TRAIN
        #######
        # MinMax normalization (from 0 to 1)
        train_df['cycle_norm'] = train_df['cycle']
        cols_normalize = train_df.columns.difference(['id','cycle','RUL'])
        min_max_scaler = preprocessing.MinMaxScaler()
        norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_df[cols_normalize]),
                                    columns=cols_normalize,
                                    index=train_df.index)
        join_df = train_df[train_df.columns.difference(cols_normalize)].join(norm_train_df)
        train_df = join_df.reindex(columns = train_df.columns)

        ######
        # TEST
        ######
        # MinMax normalization (from 0 to 1)
        test_df['cycle_norm'] = test_df['cycle']
        norm_test_df = pd.DataFrame(min_max_scaler.transform(test_df[cols_normalize]),
                                    columns=cols_normalize,
                                    index=test_df.index)
        test_join_df = test_df[test_df.columns.difference(cols_normalize)].join(norm_test_df)
        test_df = test_join_df.reindex(columns = test_df.columns)
        test_df = test_df.reset_index(drop=True)

        # We use the ground truth dataset to generate labels for the test data.
        # generate column max for test data
        rul = pd.DataFrame(test_df.groupby('id')['cycle'].max()).reset_index()
        rul.columns = ['id', 'max']
        truth_df.columns = ['more']
        truth_df['id'] = truth_df.index + 1
        truth_df['max'] = truth_df['more'] #rul['max'] +
        truth_df.drop('more', axis=1, inplace=True)

        return train_df[['id','cycle', 's2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']], test_df[['id','cycle', 's2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']], truth_df
    def __gettrain__(self):
        train_df, test_df, truth_df = self.Import('1')
        units = range(1, train_df['id'].nunique() + 1, 1)
        multiUnitDataset = []
        for i in units:
            signals = np.transpose(train_df[train_df['id'] == i].iloc[:, 2: 16].values)
            idx = train_df[(train_df['id'] == i)].iloc[:, 0].values
            time = train_df[(train_df['id'] == i)].iloc[:, 1].values
            T = max(train_df[(train_df['id'] == i)].iloc[:, 1].values)
            multiUnitDataset.append(np.array([signals, time, T], dtype=object))

        return np.array(multiUnitDataset)

    def __gettest__(self):
        train_df, test_df, truth_df = self.Import('1')
        units = range(1, train_df['id'].nunique() + 1, 1)
        multiUnitDataset = []
        for i in units:
            signals = np.transpose(test_df[test_df['id'] == i].iloc[:, 2: 16].values)
            idx = test_df[(test_df['id'] == i)].iloc[:, 0].values
            time = test_df[(test_df['id'] == i)].iloc[:, 1].values
            T = max(test_df[(test_df['id'] == i)].iloc[:, 1].values) + truth_df[(truth_df['id'] == i)].iloc[:, 1].values
            rul = truth_df[(truth_df['id'] == i)]
            multiUnitDataset.append(np.array([signals, time, T, rul], dtype=object))

        return np.array(multiUnitDataset)

    def __gettruth__(self):
        train_df, test_df, truth_df = self.Import('1')
        return truth_df
    
#%%    

class PreDegradationDataset(Dataset):
    def __init__(self, unit_data, transform=None, target_transform=None, seq_len = 50):
        # Initialize with the degradation data for a specific unit
        self.transform = transform
        self.target_transform = target_transform
        self.unit_data = unit_data
        self.signal_data = self.unit_data[0].transpose()  # Get the 'signal' data
        self.time = self.unit_data[1]
        self.failure_time = self.unit_data[2]
        self.seq_len = seq_len

    def __len__(self):
        return len(self.time) - self.seq_len

    def __getitem__(self, index):
        # Return a single sample from the degradation data for the unit

        signal_sample = self.signal_data[index : index + self.seq_len]
        time = self.time[index : index + self.seq_len]
        T = self.failure_time
        labels = 1 - (1 - np.power((time / T), 1))

        return signal_sample, time, T, labels #passes the HI, signals, time, and failure time
print("##")

#%%
    
class DegradationDataset(Dataset):
    def __init__(self, unit_data, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.unit_data = unit_data
        self.signal_data = self.unit_data[0].transpose()  # Get the 'signal' data
        self.time = self.unit_data[1]
        self.failure_time = self.unit_data[2]

    def __len__(self):
        # Return the length of the entire degradation time series
        return len(self.time)

    def __getitem__(self, idx):
        # Return a single sample from the entire degradation time series
        signal_sample = self.signal_data
        time = self.time
        T = self.failure_time  # Assuming it's the same for the entire test set
        labels = (np.power((time / T), 1/2))
        return signal_sample, time, T, labels
    
#%%    

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()

        # Define MLP layers
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        x = x.view(batch_size * seq_len, -1)

        x = nn.functional.tanh(x)

        x = nn.functional.tanh(self.fc1(x))

        x = nn.functional.tanh(self.fc2(x))

        x = nn.functional.sigmoid(self.fc3(x))

        x = x.view(batch_size, seq_len, -1)

        return x    
    
#%%    

class CNN(nn.Module):
    def __init__(self, input_channels, output_size):
        super(CNN, self).__init__()

        # Define CNN layers
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate the input size for the fully connected layer
        self.fc_input_size = 16 * (input_channels // 8)  # Assuming input_size is divisible by 8 after max pooling
        #print(self.fc_input_size)
        self.fc = nn.Linear(self.fc_input_size, output_size)

    def forward(self, x):
        batch_size, seq_len, input_channels = x.shape
        
        # Permute input dimensions for convolution
        x = x.permute(0, 2, 1)

        # Apply convolutional and pooling layers
        x = nn.functional.relu(self.conv1(x))
        x = self.pool(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        x = nn.functional.relu(self.conv3(x))
        x = self.pool(x)

        # Flatten the output of convolutional layers
        x = x.view(batch_size, -1)
        print(x.shape)
        # Apply fully connected layer
        x = self.fc(x)

        return x
#%%

class EarlyStopping():
    def __init__(self, patience=5, delta=0.0001, verbose=False, path='checkpoint.pt'):

        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):

        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)

        elif mean(val_loss) > mean(self.best_score) + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'Validation loss increased ({self.best_score:.6f} --> {val_loss:.6f}).')
            if self.counter >= self.patience:
                if self.verbose:
                    print('Early stopping.')
                self.early_stop = True

        else:
            if mean(val_loss) < mean(self.best_score) - self.delta:
                self.best_score = val_loss
                self.save_checkpoint(val_loss, model)
            self.counter = 0

        return self.early_stop

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.best_score:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        
#%%
    
class GenericHILoss():
    def __init__(self, l, T, time, outputs, lembda):
        self.lembda = lembda
        self.time = time
        self.outputs = outputs
        ones = torch.ones_like(time)
        Psi = torch.stack((ones, time.float(), pow(time.float(), 2)), dim=2)
        Psi_T_Psi = Psi.transpose(1, 2).matmul(Psi)
        Psi_T_Psi_inv = torch.linalg.inv(Psi_T_Psi).float()
        psi = torch.stack((torch.ones_like(T), T.float(), pow(T.float(), 2)), dim=1).unsqueeze(1)

        gamma = Psi_T_Psi_inv @ Psi.transpose(1, 2) @ outputs
        z_hat = Psi @ gamma

        fa = outputs[:,-1,:]
        fb = outputs[:,0,:]
        
        downsampled_outputs = outputs[:, ::10, :]
        diff = downsampled_outputs[:,:-1,:]-downsampled_outputs[:,1:,:]
        diff2 = downsampled_outputs[:,2:,:] - 2 * downsampled_outputs[:,1:-1,:] + downsampled_outputs[:,:-2,:]
        # Calculate loss1 for the entire batch
        self.loss2 = torch.sum(gamma[:,0,:])#torch.sum(nn.functional.relu(diff.exp()-1))#
        self.loss3 = torch.sum(gamma[:,2,:])#torch.sum(diff2.exp())# #torch.sum(nn.functional.relu(diff2.exp()-1))
        self.loss4 = torch.sum(torch.abs(z_hat - outputs)**2)
        self.loss5 = torch.sum(fa - fb)
        self.loss6 = torch.sum((psi @ Psi_T_Psi_inv @ Psi.transpose(1, 2) @ outputs - 1)**2)

    def __lossFunction__(self):
        lembda = self.lembda
        loss = lembda[0] * self.loss2 + lembda[1] * self.loss3+ lembda[2] * self.loss4 - lembda[3] * self.loss5 + lembda[4] * self.loss6
        return loss
    
#%%
import numpy as np
from numpy.linalg import pinv
from scipy.linalg import eig
import numpy as np
from scipy.stats import norm

class maintenanceloss():

    def __init__(self, idx, ts, test_idx, test_ts, i, output, health_index, cost, T, partial):

        signals = np.concatenate(health_index)
        output = output.detach().numpy()
        rld = self.rldtraining(signals = signals, idx = idx, ts = ts, psi = [0, 1, 2])
        self.costpredict(output, test_ts[test_idx == i], rld, 0.8, cost, T, partial, True)

    def design_matrix(self, obs, psi):
        obs = obs.T


        p = len(psi)
        Psi = np.ones((len(obs), p))  # Design matrix
        for j in range(p):
            Psi[:, j] = np.power(obs, psi[j])
        return Psi


    def wtlm(self, X, y, ts):
        X = X
        n_rows = X.shape[0]
        c = np.eye(n_rows)
        A = X.T @ c @ c @ X
        b = X.T @ c @ c @ y

        beta = np.linalg.solve(A, b)

        return beta.T


    def rldtraining(self, signals, idx, ts, psi):
        p = len(psi)
        n_unit = max(idx)
        Gamma = np.zeros((n_unit, p))
        noisevar = np.zeros(n_unit)
        varthetaeps = np.zeros((p, p, n_unit))

        for i in range(1, n_unit + 1):
            signal = signals[idx == i]
            Psi = self.design_matrix(ts[idx == i], psi)
            wtlm_value = self.wtlm(Psi, signal, ts[idx == i])
            Gamma[i - 1] = wtlm_value

            singlenoise = signal - (Psi @ Gamma[i - 1].T).reshape(-1, 1)

            k = np.sum(singlenoise ** 2) / (len(signal) - p)

            noisevar[i - 1] = np.sum(np.square(singlenoise)) / (len(signal) - p)
            varthetaeps[:, :, i - 1] = noisevar[i - 1] * pinv(np.dot(Psi.T, Psi))

        rld = {}
        rld['mu'] = np.median(Gamma, axis=0).reshape(-1, 1)
        Ma = np.cov(Gamma.T)
        Mb = np.mean(varthetaeps, axis=2)
        D, V = la.eig(Ma, Mb)
        if np.sum(np.diag(D) >= 1) == p:
            rld['Sigma'] = Ma - Mb

        elif np.sum(np.diag(D) < 1) == p:
            rld['Sigma'] = np.zeros((p, p))

        else:
            for i in range(V.shape[1]):
                V[:, i] = V[:, i] / np.sqrt(np.dot(V[:, i].T, np.dot(Mb, V[:, i])))

            Theta = pinv(V.T)
            colidx = np.diag(D) >= 1
            print(Theta.shape)
            rld['Sigma'] = np.dot(Theta[:, colidx], np.dot(D[colidx][:, colidx], Theta[:, colidx].T))

        if np.linalg.cond(rld['Sigma']) < 1e-10:
            rld['Sigma'] = rld['Sigma'] + np.eye(p) * 1e-5

        rld['psi'] = psi
        rld['sigma2'] = np.mean(noisevar)
        rld['trend'] = np.sign(rld['mu'][-1])

        return rld

    def costpredict(self, signal, ts, rld, sthres, cost, T, partial, returnRUL=True):
        ni = len(signal)
        if partial:
            percentage = 0.6
            index = int(len(ts) * percentage)
            ts = ts[:index]
       
        Psi = design_matrix(ts, rld['psi'])
        ci = (ts - ts[0]) / (ts[-1] - ts[0]) * 0.98 + 0.01
        ci = np.diag(np.sqrt(ci / np.sum(ci) * ni))
        post_sigma = np.linalg.pinv(Psi.T @ ci @ ci @ Psi / rld['sigma2'] + np.linalg.pinv(rld['Sigma']))
        post_mu = post_sigma @ (Psi.T @ ci @ ci @ signal / rld['sigma2'] + np.linalg.solve(rld['Sigma'], rld['mu']))
        
        t = np.arange(0, T, 1)
        gt = g(t, max(ts), post_mu, post_sigma, rld, sthres)
        Pt = (norm.cdf(gt) - norm.cdf(gt[0])) / (1 - norm.cdf(gt[0]))
        ind = np.argmin(np.abs(Pt - 0.5))
        y = t[ind]
        self.TotalCost = (cost[1] * (T - y) + cost[2] * (1 - norm.cdf(self.g(T, 0, post_mu, post_sigma, rld, sthres)))) / T
    
    def g(self, t, tni, mu, sigma, rld, sthres):
        Psi = design_matrix(t + tni, rld['psi'])
        pred_mu = Psi @ mu
        
        pred_sigma = np.sum(np.multiply(Psi @ sigma, Psi), axis=1).reshape(-1, 1)
        y = np.divide((pred_mu - sthres), np.sqrt(pred_sigma))
        y = y * rld['trend']
    
        return y

    def __lossFunction__(self):
        return self.TotalCost    

#%%

##############
##TRAINING
##############

import numpy as np
from numpy.linalg import pinv
from scipy.linalg import eig


def design_matrix(obs, psi):
    obs = obs.T

    p = len(psi)
    Psi = np.ones((len(obs), p))  # Design matrix
    for j in range(p):
        Psi[:, j] = np.power(obs, psi[j])
    return Psi


def wtlm(X, y, ts):
    X = X
    # Calculate beta
    n_rows = X.shape[0]
    c = np.eye(n_rows)
    A = X.T @ c @ c @ X
    b = X.T @ c @ c @ y
    
    beta = np.linalg.solve(A, b)

    return beta.T


def rldtraining(signals, idx, ts, psi):
    p = len(psi)
    n_unit = max(idx)
    Gamma = np.zeros((n_unit, p))
    noisevar = np.zeros(n_unit)
    varthetaeps = np.zeros((p, p, n_unit))

    for i in range(1, n_unit + 1):
        signal = signals[idx == i]
        Psi = design_matrix(ts[idx == i], psi)
        
        wtlm_value = wtlm(Psi, signal, ts[idx == i])
        
        Gamma[i - 1] = wtlm_value
        singlenoise = signal - (Psi @ Gamma[i - 1].T).reshape(-1, 1)
        k = np.sum(singlenoise ** 2) / (len(signal) - p)
        noisevar[i - 1] = np.sum(np.square(singlenoise)) / (len(signal) - p)
        varthetaeps[:, :, i - 1] = noisevar[i - 1] * pinv(np.dot(Psi.T, Psi))

    rld = {}
    rld['mu'] = np.median(Gamma, axis=0).reshape(-1, 1)
    Ma = np.cov(Gamma.T)
    Mb = np.mean(varthetaeps, axis=2)
    D, V = la.eig(Ma, Mb)
    if np.sum(np.diag(D) >= 1) == p:
        rld['Sigma'] = Ma - Mb

    elif np.sum(np.diag(D) < 1) == p:
        rld['Sigma'] = np.zeros((p, p))

    else:
        for i in range(V.shape[1]):
            V[:, i] = V[:, i] / np.sqrt(np.dot(V[:, i].T, np.dot(Mb, V[:, i])))

        Theta = pinv(V.T)
        colidx = np.diag(D) >= 1
        rld['Sigma'] = np.dot(Theta[:, colidx], np.dot(D[colidx][:, colidx], Theta[:, colidx].T))

    if np.linalg.cond(rld['Sigma']) < 1e-10:
        rld['Sigma'] = rld['Sigma'] + np.eye(p) * 1e-5

    rld['psi'] = psi
    rld['sigma2'] = np.mean(noisevar)
    rld['trend'] = np.sign(rld['mu'][-1])

    return rld

import numpy as np
from scipy.stats import norm

def rldpredict(signal, ts, rld, fthres, returnRUL=True):
    ni = len(signal)
    
    Psi = design_matrix(ts, rld['psi'])
    # Calculation of ci
    ci = (ts - ts[0]) / (ts[-1] - ts[0]) * 0.98 + 0.01
    ci = np.diag(np.sqrt(ci / np.sum(ci) * ni))
    # Variance of the posterior distribution of Gamma
    post_sigma = np.linalg.pinv(Psi.T @ ci @ ci @ Psi / rld['sigma2'] + np.linalg.pinv(rld['Sigma']))
    
    post_mu = post_sigma @ (Psi.T @ ci @ ci @ signal / rld['sigma2'] + np.linalg.solve(rld['Sigma'], rld['mu']))
    
    t = np.arange(0, 300, 1)
    gt = g(t, max(ts), post_mu, post_sigma, rld, fthres)
    Pt = (norm.cdf(gt) - norm.cdf(gt[0])) / (1 - norm.cdf(gt[0]))
    ind = np.argmin(np.abs(Pt - 0.5))
    y = t[ind]
    if not returnRUL:
        y += ts[-1]

    return y


def g(t, tni, mu, sigma, rld, fthres):
    Psi = design_matrix(t + tni, rld['psi'])
    pred_mu = Psi @ mu
    
    pred_sigma = np.sum(np.multiply(Psi @ sigma, Psi), axis=1).reshape(-1, 1)
    y = np.divide((pred_mu - fthres), np.sqrt(pred_sigma))
    y = y * rld['trend']

    return y

def costCal(train_unit_data, test_unit_data, train_ts, train_idx, test_ts, test_idx, rul_truth, lstm_model1, costs1):
    costs = costs1
    HIs = []
    Ts = []
    for time_series_data in train_unit_data:
        test_dataset = DegradationDataset(time_series_data)
        test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False)

        signal_sample, time, T, labels = next(iter(test_loader))
        inputs = signal_sample
        ts = time[0].numpy()
        output = lstm_model1(inputs.to(torch.float32))
        output = output[0].detach().numpy()
        output
        HIs.append(output)
        
    signals = np.concatenate(HIs)    
    rld = rldtraining(signals, train_idx, train_ts, psi = [0, 1, 2])
        
    i = 1
    params_l = 1
    all_cost = []
    preds = []
    for time_series_data in test_unit_data:
        test_dataset = DegradationDataset(time_series_data)
        test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False)
        
        signal_sample, time, T, labels = next(iter(test_loader))
        Ts.append(T)
        inputs = signal_sample

        output = lstm_model1(inputs.to(torch.float32))
        output1 = output
        output = output[0]
        
        cost = maintenanceloss(train_idx, train_ts, test_idx, test_ts, i, labels.T, HIs, costs, T, partial = False).__lossFunction__()
        all_cost.append(cost)
        output = output.detach().numpy()
        preds.append(rldpredict(output, test_ts[test_idx == i], rld, params_l, True))
        i = i + 1
        
    true = rul_truth[max].values.reshape(-1, 1)
    preds = np.array(preds).reshape(-1, 1)
    Ts = np.array(Ts).reshape(-1, 1)    
    percentageError = sum(abs(preds - true)/(Ts))
    
    return percentageError, sum(all_cost)

def preTrain(unit_data, lstm_model1, lembdas):
    
    lstm_model = lstm_model1
    for fold, (train_ids, test_ids) in enumerate(kfold.split(unit_data)):

        print(f'PRE-FOLD {fold}')
        print('--------------------------------')

        optimizer = optim.Adam(lstm_model.parameters(), lr = 0.001)
        train_unit_data = [unit_data[i] for i in train_ids]
        test_unit_data = [unit_data[i] for i in test_ids]

        early_stopping = EarlyStopping(patience=10)
        num_epochs = 100
        
        for epoch in range(num_epochs):
            test_results = []
            running_loss = []

            for time_series_data in train_unit_data:
                train_dataset = PreDegradationDataset(time_series_data)
                train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True)
                signal_sample, time, T, labels = next(iter(train_loader))
                inputs = signal_sample

                outputs = lstm_model(inputs.to(torch.float32))
                labels = labels.to(torch.float32).unsqueeze(2)
            
                loss_function = nn.CrossEntropyLoss()
                
                loss = loss_function(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
         
    for fold, (train_ids, test_ids) in enumerate(kfold.split(unit_data)):

        print(f'FOLD {fold}')
        print('--------------------------------')
        optimizer = optim.SGD(lstm_model.parameters(), lr = 1e-7)

        train_unit_data = [unit_data[i] for i in train_ids] # Good!
        test_unit_data = [unit_data[i] for i in test_ids]

        early_stopping = EarlyStopping(patience=10)
        num_epochs = 1
        for epoch in range(num_epochs):
            test_results = []
            running_loss = []

            for time_series_data in train_unit_data:
                train_dataset = DegradationDataset(time_series_data)
                train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
                signal_sample, time, T, label = next(iter(train_loader))
                inputs = signal_sample

                outputs = lstm_model(inputs.to(torch.float32))

                loss = GenericHILoss(l / l, T, time, outputs, lembdas).__lossFunction__()

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                running_loss.append(loss.data.item())

            val_loss = []
            with torch.no_grad():
                for time_series_data in test_unit_data:
                    test_dataset = DegradationDataset(time_series_data)
                    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = True)
                    signal_sample, time, T, label = next(iter(train_loader))
                    inputs = signal_sample
                    
                    outputs = lstm_model(inputs.to(torch.float32))

                    loss = GenericHILoss(l / l, T, time, outputs, lembdas).__lossFunction__()
                    val_loss.append(loss.data.item())

                    test_results.append(outputs)

            if early_stopping(val_loss, lstm_model):
                print(f"Early stopping at epoch {epoch + 1} due to no improvement in validation loss.")
                break

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/100], Loss: {mean(running_loss)}, Val Loss: {mean(val_loss)}")
         
    return lstm_model            


def fineTuneTraining(unit_data, test_unit_data1, train_idx, train_ts, test_ts1, test_idx1, rul_truth, kfold, costs, lstm_model1, err0, tol, lembdas):
    lstm_model = lstm_model1
    for fold, (train_ids, test_ids) in enumerate(kfold.split(unit_data)): #use 3-fold CV split 3 unit;

        print(f'FOLD {fold}')
        print('--------------------------------')
        optimizer = optim.SGD(lstm_model.parameters(), lr = 1e-5)

        train_unit_data = [unit_data[i] for i in train_ids]
        test_unit_data = test_unit_data1
        test_unit_data = [unit_data[i] for i in test_ids]

        early_stopping = EarlyStopping(patience=10)
        num_epochs = 100
        for epoch in range(num_epochs):
            test_results = []
            running_loss = []
            i = 10
            for time_series_data in unit_data:
                train_dataset = DegradationDataset(time_series_data)
                train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True)
                signal_sample, time, T, label = next(iter(train_loader))
                inputs = signal_sample
                
                #model running
                output = lstm_model(inputs.to(torch.float32))
                loss = GenericHILoss(l / l, T, time, output, lembdas).__lossFunction__()
                
                #backpropgation
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                running_loss.append(loss.data.item())

            val_loss = []
            with torch.no_grad():
                for time_series_data in test_unit_data:
                    test_dataset = DegradationDataset(time_series_data)
                    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = True)
                    signal_sample, time, T, label = next(iter(train_loader))
                    inputs = signal_sample
                    
                    outputs = lstm_model(inputs.to(torch.float32))
                    
                    loss = GenericHILoss(l / l, T, time, outputs, lembdas).__lossFunction__()
                    val_loss.append(loss.data.item())

                    test_results.append(outputs)

            if early_stopping(val_loss, lstm_model):
                print(f"Early stopping at epoch {epoch + 1} due to no improvement in validation loss.")
                break
            
    return lstm_model            

def plot(lstm_model, unit_data, name):
    HIs = []
    Ts = []
    plt.ylim(0, 1.2)
    for time_series_data in unit_data:
        test_dataset = DegradationDataset(time_series_data)
        test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False)
        sample = next(iter(test_loader))  # Get the single item in the test_loader
    
        signal_sample, time, T, labels = sample
        inputs = signal_sample
        ts = time[0].numpy()
        output = lstm_model(inputs.to(torch.float32))
        output = output[0].detach().numpy()
        plt.plot(output)
        HIs.append(output)
        
    plt.plot(HIs[10])
    plt.savefig(name)
    plt.show()


l = 1
batch_size = 2**5
unit_data = customerDataset().__gettrain__()

kfold = KFold(n_splits = 10, shuffle = True)
INPUT_FEATURES_NUM = 14
OUTPUT_FEATURES_NUM = 1
best_val_loss = float('inf')

lstm_model = MLP(input_size = INPUT_FEATURES_NUM, hidden_size1 = 4, hidden_size2 = 4, output_size = OUTPUT_FEATURES_NUM)
lstm_model.eval()
print('MLP model:', lstm_model)
print('model.parameters:', lstm_model.parameters)

costs = [0, 1, 100]
lembdas = [0, 0, 50, 50, 50]

train_df, test_df, truth_df = customerDataset().Import('1')

train_idx = train_df.iloc[:, 0].values
train_ts = train_df.iloc[:, 1].values

test_idx = test_df.iloc[:, 0].values
test_ts = test_df.iloc[:, 1].values

rul_truth = customerDataset().__gettruth__()

train_unit_data = customerDataset().__gettrain__()
test_unit_data = customerDataset().__gettest__()
    
name = f'convex_pretrain'
tol = 1
#lstm_model = preTrain(unit_data, lstm_model, lembdas) #depends on if you are suing pretraing
percentageError_0 = 100
lstm_model = fineTuneTraining(unit_data, test_unit_data, train_idx, train_ts, test_ts, test_idx, rul_truth, kfold, costs, lstm_model, 100, tol, lembdas)

plot(lstm_model, unit_data, name)
percentageError_0, cost_0 = costCal(train_unit_data, test_unit_data, train_ts, train_idx, test_ts, test_idx, rul_truth, lstm_model, costs)
print(f"[{percentageError_0}, {cost_0}]")
model_scripted = torch.jit.script(lstm_model)
model_scripted.save(f'/Users/lizihan/Desktop/models/BBBBBcost+RUL_2.pt')
