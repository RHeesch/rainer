import random
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

o_fit = True

### hparam ###
hparam = {
    "SEED":1,
    "BATCH_SIZE": 20,
    "LEARNING_RATE":1e-3,
    "WEIGHT_DECAY":1e-8,
    "EPOCHS": 1000,

    "SET_LEN":1000,
    "FEAT":5,
    "PAR_FIX":3,
    "PAR_VAR":1
         }

PATH = "model_FliPSi.pth"

# Datagen
def data_generator(setlength, features, number_fix_Parameters, number_variable_Parameters):
    dimensions_Products = (setlength, features)
    dimensions_fix_Parameters = (setlength, number_fix_Parameters)
    dimensions_var_Parameters = (setlength, number_variable_Parameters)

    Product_in = np.zeros(dimensions_Products)
    Product_out = np.zeros(dimensions_Products)
    Parameters_fix = np.zeros(dimensions_fix_Parameters)
    Parameters_var = np.zeros(dimensions_var_Parameters)

    Parameter_var = ["energy_input"]
    Parameter_fix = ["volume", "density", "specific_heat_capacity"]
    Features = ["coloured", "temperature", "milled", "drilled", "rounded"]

    bool_ls = [0, 10]

    cf_dens = 1000
    cf_vol = 1
    cf_shc = 100
    cf_hc = 10000000

    volume_top = 2
    volume_down = 1
    density_top = 10000
    density_down = 2000
    specific_heat_capacity_top = 1000
    specific_heat_capacity_down = 250

    energy_input_down = 0

    Product_in_temp_top = 100
    Product_in_temp_down = 0

    for i in range(setlength):
        Product_in[i][0] = random.choice(bool_ls)
        Product_out[i][0] = Product_in[i][0]
        Product_in[i][2] = random.choice(bool_ls)
        Product_out[i][2] = Product_in[i][2]
        Product_in[i][3] = random.choice(bool_ls)
        Product_out[i][3] = Product_in[i][3]
        Product_in[i][4] = random.choice(bool_ls)
        Product_out[i][4] = Product_in[i][4]

    if o_fit == True: 
        for i in range(setlength):
            Parameters_fix[i][0] = random.uniform(1, 2)
            Product_in[i][1] = random.uniform(4, 6)
        for i in range(0,334):
            Parameters_fix[i][1] = random.uniform(7800/cf_dens, 7900/cf_dens)
            Parameters_fix[i][2] = random.uniform(450/cf_shc, 500/cf_shc)
            Parameters_var[i][0] = random.uniform(84000/cf_hc, 85000/cf_hc)
        for i in range(333,668):
            Parameters_fix[i][1] = random.uniform(2680/cf_dens, 2720/cf_dens)
            Parameters_fix[i][2] = random.uniform(890/cf_shc, 900/cf_shc)
            Parameters_var[i][0] = random.uniform(54000/cf_hc, 55000/cf_hc)
        for i in range(667,1000):
            Parameters_fix[i][1] = random.uniform(8700/cf_dens, 8760/cf_dens)
            Parameters_fix[i][2] = random.uniform(370/cf_shc, 385/cf_shc)
            Parameters_var[i][0] = random.uniform(73000000/cf_hc, 75000000/cf_hc)
        for i in range(setlength):
            Product_out[i][1] = Product_in[i][1] + (Parameters_var[i][0] *cf_hc * (((Parameters_fix[i][0] * Parameters_fix[i][1] * Parameters_fix[i][2] *cf_dens *cf_shc) ** (-1))))
    else:
        for i in range(setlength):
            Parameters_fix[i][0] = random.uniform(volume_down, volume_top)
            Parameters_fix[i][1] = random.uniform(density_down/cf_dens, density_top/cf_dens)
            Parameters_fix[i][2] = random.uniform(specific_heat_capacity_down/cf_shc, specific_heat_capacity_top/cf_shc)
            Product_in[i][1] = random.uniform(Product_in_temp_down, Product_in_temp_top)
            energy_input_top = (100 - Product_in[i][1]) * (Parameters_fix[i][0] * Parameters_fix[i][1] * Parameters_fix[i][2])
            Parameters_var[i][0] = random.uniform(energy_input_down/cf_hc, energy_input_top/cf_hc)
            Product_out[i][1] = Product_in[i][1] + (Parameters_var[i][0] *cf_hc * (((Parameters_fix[i][0] * Parameters_fix[i][1] * Parameters_fix[i][2] *cf_dens *cf_shc) ** (-1))))
    return Product_in, Product_out, Parameters_fix, Parameters_var

# Data Module
class DataModule(nn.Module):
    def __init__(self, hparam):
        super(DataModule, self).__init__()
        self.hparam = hparam

    def slicer(self):
        p_in, p_out, par_fix, par_var = data_generator(hparam["SET_LEN"], hparam["FEAT"], hparam["PAR_FIX"], hparam["PAR_VAR"])

        ds_in = np.concatenate([p_in, par_fix, par_var], axis=1)
        return ds_in, p_out

    def sampler(self):
        X, y = self.slicer()
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, shuffle=True)
        X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, shuffle=True)
        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_loader(self):
        X_train, _, _, y_train, _, _ = self.sampler()
        ds_train = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
        return DataLoader(ds_train, batch_size=self.hparam["BATCH_SIZE"], num_workers=4, shuffle=True, drop_last=True)

    def val_loader(self):
        _, X_val, _, _, y_val, _ = self.sampler()
        ds_train = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))
        return DataLoader(ds_train, batch_size=self.hparam["BATCH_SIZE"], num_workers=4, shuffle=True, drop_last=True)

    def test_loader(self):
        _, _, X_test, _, _, y_test = self.sampler()
        ds_train = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
        return DataLoader(ds_train, batch_size=self.hparam["BATCH_SIZE"], num_workers=4, shuffle=True, drop_last=True)

# Neural Net
class MLP(nn.Module):
    def __init__(self, hparam):
        super(MLP, self).__init__()
        self.hparam = hparam
        dim_in = hparam["FEAT"] + hparam["PAR_FIX"] + hparam["PAR_VAR"]
        dim_out = hparam["FEAT"]

        self.lin1 = nn.Linear(in_features=dim_in, out_features=dim_in * 1)
        self.lin2 = nn.Linear(in_features=dim_in * 1, out_features=dim_in * 1)
        self.lin3 = nn.Linear(in_features=dim_in * 1, out_features=dim_in * 1)
        self.lin4 = nn.Linear(in_features=dim_in * 1, out_features=dim_out)

        # self.act1  = nn.LeakyReLU(0.2, inplace=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.lin1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.lin2(x)
        x = self.act(x)
        x = self.drop(x)
        #x = self.lin3(x)
        #x = self.act(x)
        #x = self.drop(x)
        y = self.lin4(x)
        y = self.act(y)
        return y


# Net functionalities
def train_step(model, dl, optimizer):
    train_losses = []
    model.train()
    for x, y in dl:
        y_hat = model(x)
        loss = torch.mean((y - y_hat) ** 2)
        train_losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return sum(train_losses) / len(train_losses)

def val_step(model, dl):
    val_losses = []
    model.eval()
    with torch.no_grad():
        for x, y in dl:
            y_hat = model(x)
            loss = torch.mean((y - y_hat) ** 2)
            val_losses.append(loss.item())
    return sum(val_losses) / len(val_losses), y_hat, y

def test(model, dl):
    test_losses = []
    model.eval()
    with torch.no_grad():
        for x, y in dl:
            y_hat = model(x)
            loss = torch.mean((y - y_hat) ** 2)
            test_losses.append(loss.item())

            print(f"y - ground truth \n{y} \n\n y_hat - prediction \n{y_hat} \n\n delta {y-y_hat} \n\n\n")
    return test_losses, y_hat, y

def predict(model, x):
    model.eval()
    with torch.no_grad():
        y_hat = model(x)
    return y_hat

# apply
def run(hparam):
    random.seed(hparam["SEED"])
    np.random.seed(hparam["SEED"])
    torch.manual_seed(hparam["SEED"])

    dl_train = DataModule(hparam).train_loader()
    dl_val = DataModule(hparam).val_loader()
    dl_test = DataModule(hparam).test_loader()

    model = MLP(hparam)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparam["LEARNING_RATE"], weight_decay=hparam["WEIGHT_DECAY"])

    train_losses, val_losses = [], []


    with trange(hparam["EPOCHS"], desc="magic in process") as tep:
        for e in tep:
            train_loss = train_step(model, dl_train, optimizer)
            train_losses.append(train_loss)
            val_loss, y_hat, y = val_step(model, dl_val)
            val_losses.append(val_loss)

    test_loss, _, _ = test(model, dl_test)
    
    torch.save(model.state_dict(), PATH)
    return train_losses, val_losses, test_loss, y_hat, y


# Quickrun
if __name__ == "__main__":
    train_loss, val_loss, test_loss, y_hat, y = run(hparam)
    print(f"Results:\n\ttrain_loss: {train_loss[-1]}\n\tval_loss: {val_loss[-1]}\n\ttest_loss: {test_loss[-1]}\n")
