# -*- coding: utf-8 -*-
import torch
import os
print(torch.__version__)
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from torch import nn
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import KFold
import torch.optim as optim
from statistics import mean
import pandas as pd
from sklearn import preprocessing
import scipy.linalg as la
from numpy.linalg import pinv
from scipy.linalg import eig
import torch.nn.functional as F

class MLP_Dynamic(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP_Dynamic, self).__init__()

        # Define MLP layers structure, not weights
        self.fc1 = nn.Linear(input_size, hidden_size1, bias=False)  # Assume bias is handled externally if needed
        self.fc2 = nn.Linear(hidden_size1, hidden_size2, bias=False)
        self.fc3 = nn.Linear(hidden_size2, output_size, bias=False)

    def forward(self, x, weights=None):
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size * seq_len, -1)

        # Apply activations and use external weights if provided
        x = F.tanh(x)

        if weights is not None:
            x = F.tanh(F.linear(x, weights['fc1.weight'], weights['fc1.bias']))
            x = F.tanh(F.linear(x, weights['fc2.weight'], weights['fc2.bias']))
            x = F.sigmoid(F.linear(x, weights['fc3.weight'], weights['fc3.bias']))
        else:
            x = F.tanh(self.fc1(x))
            x = F.tanh(self.fc2(x))
            x = F.sigmoid(self.fc3(x))

        x = x.view(batch_size, seq_len, -1)
        return x 
    
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

        #x = self.fc3(x)

        x = x.view(batch_size, seq_len, -1)

        return x

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
        labels = (np.power((time / T), 2))

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
        #print(type(time))
        #print(time.shape)
        labels = (np.power((time / T), 2))
        return signal_sample, time, T, labels  
    
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
        
        self.loss2 = torch.sum(gamma[:,0,:])
        self.loss3 = torch.sum(gamma[:,2,:])
        self.loss4 = torch.sum(torch.abs(z_hat - outputs)**2)
        self.loss5 = torch.sum(fa - fb)
    
        self.loss6 = torch.sum((psi @ Psi_T_Psi_inv @ Psi.transpose(1, 2) @ outputs - 1)**2)

    def __lossFunction__(self):
        lembda = self.lembda
        loss = lembda[0] * self.loss2 + lembda[1] * self.loss3+ lembda[2] * self.loss4 - lembda[3] * self.loss5 + lembda[4] * self.loss6
        return loss
        
train_unit_data = customerDataset().__gettrain__()
INPUT_FEATURES_NUM = 14
OUTPUT_FEATURES_NUM = 1
l = 1
lembdas = [0,0,50,50,100]
#%%

def bezier_interpolation(modelA, modelB, control, t):
    new_state_dict = {}
    t = torch.tensor(t, requires_grad=True)
    for key in modelA.state_dict():
        w1 = modelA.state_dict()[key]
        w2 = modelB.state_dict()[key]
        # Assuming control is a dict with the same structure but different values
        theta = control[key]

        # Bezier interpolation for weights
        new_weight = (1.0 - t)**2.0 * w1 + 2.0 * (1.0 - t) * t * theta + t**2.0 * w2
        new_state_dict[key] = new_weight
    return new_state_dict


model1 = torch.jit.load('/Users/lizihan/Desktop/models/BBBBBlinear_8.pt')

model2 = torch.jit.load('/Users/lizihan/Desktop/models/BBBBBconvex_8.pt')

control_point = {key: torch.nn.Parameter((model1.state_dict()[key] + model2.state_dict()[key]) / 2) for key in model1.state_dict()}

model = MLP_Dynamic(input_size = INPUT_FEATURES_NUM, hidden_size1 = 4, hidden_size2 = 4, output_size = OUTPUT_FEATURES_NUM)

optimizer = torch.optim.SGD(control_point.values(), lr=1e-5)

num_epochs = 200
for epoch in range(num_epochs):
    epoch_loss = []
    for time_series_data in train_unit_data:
        train_dataset = DegradationDataset(time_series_data)
        train_loader = DataLoader(train_dataset, batch_size = 1, shuffle = False)
        signal_sample, time, T, label = next(iter(train_loader))
        #get input
        inputs = signal_sample
        optimizer.zero_grad()
        losses = []
        # Generate a model for a given value of t
        # Random t for each batch to sample along the curve
        
        for t in torch.linspace(0, 1, steps=20):
            
            new_model_weights = bezier_interpolation(model1, model2, control_point, t)
            
            output = model(inputs.to(torch.float32), new_model_weights)
            loss = GenericHILoss(l / l, T, time, output, lembdas).__lossFunction__()
            loss.backward()
            optimizer.step()
            losses.append(loss)
        
        total_loss = sum(losses)
        
        epoch_loss.append(total_loss.item())
        
    if (epoch + 1) % 10 == 0:    
        print(f'Epoch: {epoch + 1}/100, Loss: {mean(epoch_loss)}')

#%%
import numpy as np
import matplotlib.pyplot as plt

def evaluate_model_performance(model, train_unit_data):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    with torch.no_grad():
        for time_series_data in train_unit_data:
            train_dataset = DegradationDataset(time_series_data)
            train_loader = DataLoader(train_dataset, batch_size = 1, shuffle = False)
            signal_sample, time, T, label = next(iter(train_loader))
            #get input
            inputs = signal_sample
            
            output = model(inputs.to(torch.float32))
            loss = GenericHILoss(l / l, T, time, output, lembdas).__lossFunction__()
            #print(loss.item())
            total_loss += loss.item()  # Sum up batch loss
        
    return total_loss / len(train_unit_data)

def plot_curve_performance(modelA, modelB, control_point, test_loader, steps=50, device='cpu'):
    t_values = np.linspace(0, 1, steps)
    losses = []

    for t in t_values:
        interpolated_weights = {key: (1 - t)**2 * modelA.state_dict()[key] + 2 * (1 - t) * t * control_point[key] + t**2 * modelB.state_dict()[key] for key in modelA.state_dict()}
        
        model_temp = MLP(input_size = INPUT_FEATURES_NUM, hidden_size1 = 4, hidden_size2 = 4, output_size = OUTPUT_FEATURES_NUM)  # Assuming a function to initialize the model
        model_temp.load_state_dict(interpolated_weights)
        
        loss = evaluate_model_performance(model_temp, test_loader)
        losses.append(loss)
        
    return losses

#%%

def plot_line_performance(modelA, modelB, test_loader, steps=50, device='cpu'):
    t_values = np.linspace(0, 1, steps)
    losses = []
    
    for t in t_values:
        seg_point = {key: torch.nn.Parameter((t * model1.state_dict()[key] + (1 - t) * model2.state_dict()[key])) for key in model1.state_dict()}
        #print(seg_point)
        model_temp = MLP(input_size = INPUT_FEATURES_NUM, hidden_size1 = 4, hidden_size2 = 4, output_size = OUTPUT_FEATURES_NUM)
        model_temp.load_state_dict(seg_point)

        loss = evaluate_model_performance(model_temp, test_loader)
        losses.append(loss)


    return losses


loss1 = plot_curve_performance(model2, model1, control_point, train_unit_data)
loss2 = plot_line_performance(model1, model2, train_unit_data)
#%%
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
#%%
unit_data = customerDataset().__gettrain__()
test_unit_data = customerDataset().__gettest__()
def plot_curve_test(modelB, modelA, control_point, steps=50, device='cpu'):
    
    t_values = np.linspace(0, 1, steps)
    losses = []
    
    for t in t_values:
        interpolated_weights = {key: (1 - t)**2 * modelA.state_dict()[key] + 2 * (1 - t) * t * control_point[key] + t**2 * modelB.state_dict()[key] for key in modelA.state_dict()}
        
        model_temp = MLP(input_size = INPUT_FEATURES_NUM, hidden_size1 = 4, hidden_size2 = 4, output_size = OUTPUT_FEATURES_NUM)  # Assuming a function to initialize the model
        model_temp.load_state_dict(interpolated_weights)
        
        HIs = []
        Ts = []
        for time_series_data in unit_data:
            test_dataset = DegradationDataset(time_series_data)
            test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False)
            sample = next(iter(test_loader))  # Get the single item in the test_loader

            signal_sample, time, T, labels = sample
            inputs = signal_sample
            ts = time[0].numpy()
            output = model_temp(inputs.to(torch.float32))
            output = output[0].detach().numpy()
            HIs.append(output)
            
        train_df, test_df, truth_df = customerDataset().Import('1')

        idx = train_df.iloc[:, 0].values
        ts = train_df.iloc[:, 1].values
        signals = np.concatenate(HIs)
        rld = rldtraining(signals = signals, idx = idx, ts = ts, psi = [0, 1, 2])

        rul_truth = customerDataset().__gettruth__()
        true = rul_truth[max].values.reshape(-1, 1)
        idx = test_df.iloc[:, 0].values
        ts = test_df.iloc[:, 1].values
        params_l = 1.0
        preds = []
        i = 1
        for time_series_data in test_unit_data:
            test_dataset = DegradationDataset(time_series_data)
            test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False)
            sample = next(iter(test_loader)) # Get the single item in the test_loader
            signal_sample, time, T, labels = sample 
            Ts.append(T)
            inputs = signal_sample

            output = model_temp(inputs.to(torch.float32))
            output = output[0].detach().numpy()

            preds.append(rldpredict(output, ts[idx == i], rld, params_l, True))
            i = i + 1
        true = rul_truth[max].values.reshape(-1, 1)
        preds = np.array(preds).reshape(-1, 1)
        Ts = np.array(Ts).reshape(-1, 1)    
        percentageError = sum(abs(preds - true)/(Ts))
        losses.append(percentageError)
    return losses


def plot_line_test(modelA, modelB, steps=50, device='cpu'):
    
    t_values = np.linspace(0, 1, steps)
    losses = []
    
    for t in t_values:
        seg_point = {key: torch.nn.Parameter((t * model1.state_dict()[key] + (1 - t) * model2.state_dict()[key])) for key in model1.state_dict()}
        
        model_temp = MLP(input_size = INPUT_FEATURES_NUM, hidden_size1 = 4, hidden_size2 = 4, output_size = OUTPUT_FEATURES_NUM)  # Assuming a function to initialize the model
        model_temp.load_state_dict(seg_point)
        
        HIs = []
        Ts = []
        for time_series_data in unit_data:
            test_dataset = DegradationDataset(time_series_data)
            test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False)
            sample = next(iter(test_loader))  # Get the single item in the test_loader

            signal_sample, time, T, labels = sample
            inputs = signal_sample
            ts = time[0].numpy()
            output = model_temp(inputs.to(torch.float32))
            output = output[0].detach().numpy()
            HIs.append(output)
            
        train_df, test_df, truth_df = customerDataset().Import('1')

        idx = train_df.iloc[:, 0].values
        ts = train_df.iloc[:, 1].values
        signals = np.concatenate(HIs)
        rld = rldtraining(signals = signals, idx = idx, ts = ts, psi = [0, 1, 2])

        rul_truth = customerDataset().__gettruth__()
        true = rul_truth[max].values.reshape(-1, 1)

        idx = test_df.iloc[:, 0].values
        ts = test_df.iloc[:, 1].values
        params_l = 1.0
        preds = []
        i = 1
        for time_series_data in test_unit_data:
            test_dataset = DegradationDataset(time_series_data)
            test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False)
            sample = next(iter(test_loader)) # Get the single item in the test_loader
            signal_sample, time, T, labels = sample 
            Ts.append(T)
            inputs = signal_sample

            output = model_temp(inputs.to(torch.float32))
            output = output[0].detach().numpy()

            preds.append(rldpredict(output, ts[idx == i], rld, params_l, True))
            i = i + 1
        true = rul_truth[max].values.reshape(-1, 1)
        preds = np.array(preds).reshape(-1, 1)
        Ts = np.array(Ts).reshape(-1, 1)    
        percentageError = sum(abs(preds - true)/(Ts))
        losses.append(percentageError)
    return losses


test1 = plot_curve_test(model1, model2, control_point)
test2 = plot_line_test(model1, model2)
t_values = np.linspace(0, 1, 50)


#%%
plt.figure
plt.plot(t_values, loss1, label = "Bezier curve", color = "#DACEC2")
plt.plot(t_values, loss2, label = "Linear path", color = "#3F6561")
plt.xlabel("$\\beta$")
plt.ylabel("Train loss")
plt.legend()
plt.savefig("/Users/lizihan/Desktop/curveloss.pdf", format="pdf", bbox_inches="tight")

plt.figure()
plt.plot(t_values, test1, label = "Bezier curve", color = "#DACEC2")
plt.plot(t_values, test2, label = "Linear path", color = "#3F6561")
plt.xlabel("$\\beta$")
plt.ylabel("Test error (%, RUL percentage error)")
plt.legend()
plt.savefig("/Users/lizihan/Desktop/curveTest.pdf", format="pdf", bbox_inches="tight")

#%%
class maintenanceloss():

    def __init__(self, idx, ts, test_idx, test_ts, i, output, health_index, cost, T, partial):

        #print(outputs.shape)
        signals = np.concatenate(health_index)
        output = output.detach().numpy()
        rld = self.rldtraining(signals = signals, idx = idx, ts = ts, psi = [0, 1, 2])
        self.costpredict(output, test_ts[test_idx == i], rld, 0.7, cost, T, partial, True)

    def design_matrix(self, obs, psi):
        #print(obs)
        
        obs = obs.T


        p = len(psi)
        Psi = np.ones((len(obs), p))  # Design matrix
        #print(psi)
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
        #print(ts.shape)
        # Calculation of ci
        ci = (ts - ts[0]) / (ts[-1] - ts[0]) * 0.98 + 0.01
        ci = np.diag(np.sqrt(ci / np.sum(ci) * ni))
        #print(ci)
        # Variance of the posterior distribution of Gamma
        post_sigma = np.linalg.pinv(Psi.T @ ci @ ci @ Psi / rld['sigma2'] + np.linalg.pinv(rld['Sigma']))
        #print(type(signal))
        post_mu = post_sigma @ (Psi.T @ ci @ ci @ signal / rld['sigma2'] + np.linalg.solve(rld['Sigma'], rld['mu']))
        
        t = np.arange(0, T, 1)
        gt = g(t, max(ts), post_mu, post_sigma, rld, sthres)
        Pt = (norm.cdf(gt) - norm.cdf(gt[0])) / (1 - norm.cdf(gt[0]))
        ind = np.argmin(np.abs(Pt - 0.5))
        y = t[ind]
        #print(y)
        #print(1 - norm.cdf(self.g(T, 0, post_mu, post_sigma, rld, sthres)))
        self.TotalCost = (cost[1] * (T - y) + cost[2] * (1 - norm.cdf(self.g(T, 0, post_mu, post_sigma, rld, sthres)))) / T
    
    def g(self, t, tni, mu, sigma, rld, sthres):
        Psi = design_matrix(t + tni, rld['psi'])
        pred_mu = Psi @ mu
        
        pred_sigma = np.sum(np.multiply(Psi @ sigma, Psi), axis=1).reshape(-1, 1)
        y = np.divide((pred_mu - sthres), np.sqrt(pred_sigma))
        y = y * rld['trend']
    
        return y

    def __lossFunction__(self):
        #cost = self.TotalCost.requires_grad_(requires_grad=True).squeeze()
        #T = self.T
        return self.TotalCost    

#%%

costs = [0, 1, 1]

train_df, test_df, truth_df = customerDataset().Import('1')
train_idx = train_df.iloc[:, 0].values
train_ts = train_df.iloc[:, 1].values

test_idx = test_df.iloc[:, 0].values
test_ts = test_df.iloc[:, 1].values

rul_truth = customerDataset().__gettruth__()
train_unit_data = customerDataset().__gettrain__()
test_unit_data = customerDataset().__gettest__()


#%%

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

c = [[0, 1, 1], [0, 1, 100], [0, 1, 300], [0, 1, 500], [0, 1, 800], [0, 1, 1000]]
c_value = []
for costs in c:
    cost_values = []
    for t in np.linspace(0, 1, 100):
        
        model_weights = bezier_interpolation(model1, model2, control_point, t)
        
        model_test = MLP(input_size = INPUT_FEATURES_NUM, hidden_size1 = 4, hidden_size2 = 4, output_size = OUTPUT_FEATURES_NUM)
        model_test.load_state_dict(model_weights)
        
        
        percentageError_0, cost_0 = costCal(train_unit_data, test_unit_data, train_ts, train_idx, test_ts, test_idx, rul_truth, model_test, costs)
        cost_values.append(cost_0.item())
    print(cost_values)
    t_1000 = np.argmin(cost_values)
    print(t_1000)
    print(min(cost_values))
    print(np.linspace(0, 1, 100)[np.argmin(cost_values)])
    c_value.append(cost_values)
#%%
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs[0, 0].plot(np.linspace(0, 1, 100), c_value[0], label = "[1, 1]", color = "#76758B")
axs[0, 0].set_title("[1, 1]")
#axs[0, 0].xlabel('Time cycles (t)')
axs[0, 0].axvline(x=np.linspace(0, 1, 100)[np.argmin(c_value[0])], color='black', linestyle='--')
axs[0, 1].plot(np.linspace(0, 1, 100), c_value[1], label = "[1, 100]", color = "#76758B")
axs[0, 1].set_title("[1, 100]")
axs[0, 1].axvline(x=np.linspace(0, 1, 100)[np.argmin(c_value[1])], color='black', linestyle='--')
axs[0, 2].plot(np.linspace(0, 1, 100), c_value[2], label = "[1, 300]", color = "#76758B")
axs[0, 2].set_title("[1, 300]")
axs[0, 2].axvline(x=np.linspace(0, 1, 100)[np.argmin(c_value[2])], color='black', linestyle='--')

axs[1, 0].plot(np.linspace(0, 1, 100), c_value[3], label = "[1, 500]", color = "#76758B")
axs[1, 0].set_title("[1, 500]")
axs[1, 0].axvline(x=np.linspace(0, 1, 100)[np.argmin(c_value[3])], color='black', linestyle='--')
axs[1, 1].plot(np.linspace(0, 1, 100), c_value[4], label = "[1, 800]", color = "#76758B")
axs[1, 1].set_title("[1, 800]")
axs[1, 1].axvline(x=np.linspace(0, 1, 100)[np.argmin(c_value[4])], color='black', linestyle='--')
axs[1, 2].plot(np.linspace(0, 1, 100), c_value[5], label = "[1, 1000]", color = "#76758B")
axs[1, 2].set_title("[1, 1000]")
axs[1, 2].axvline(x=np.linspace(0, 1, 100)[np.argmin(c_value[5])], color='black', linestyle='--')