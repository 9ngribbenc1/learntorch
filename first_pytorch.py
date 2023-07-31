"""
This script is for me to mess around and create and train my first PyTorch
models. The data are synthetic ones that I randomly created.
This follows the tutorial at Machine Learning Mastery.com

by Neil Campbell
July 30, 2023
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from torch.nn import Module, MSELoss, BCELoss, Linear, Sigmoid, ReLU
from torch.nn.init import xavier_uniform_, xavier_normal_, kaiming_uniform_
from torch.optim import SGD, Adam
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix


class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, filename):
        data = pd.read_csv(filename)
        xcols = [f"col{i}" for i in range(8)]
        ycol = ["label"]
        # Convert the dataframe into torch tensors with types float32
        self.X = torch.tensor(data[xcols].values).to(torch.float32)
        self.y = torch.tensor(data[ycol].values).to(torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]


# Model Definition
class MLP(Module):
    # dfine model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        self.layer1 = Linear(n_inputs, 32)
        kaiming_uniform_(self.layer1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # layer 2
        self.layer2 = Linear(32, 128)
        kaiming_uniform_(self.layer2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # layer 3
        self.layer3 = Linear(128, 16)
        kaiming_uniform_(self.layer3.weight, nonlinearity='relu')
        self.act3 = ReLU()
         
        # layer 4
        self.layer4 = Linear(16, 1)
        xavier_uniform_(self.layer4.weight)
        self.act4 = Sigmoid()
            

    # forward propagation input
    def forward(self, X):
        # layer 1
        X = self.layer1(X)
        X = self.act1(X)
        # layer 2
        X = self.layer2(X)
        X = self.act2(X)
        # layer 3
        X = self.layer3(X)
        X = self.act3(X)
        # layer 4
        X = self.layer4(X)
        X = self.act4(X)
        return X


def main():


    # Create dataset
    dataset = CSVDataset("data.txt")
    print(len(dataset))

    # Setect rows from dataset
    train, test = random_split(dataset, [0.8, 0.2])
    # Create dataloaders for train adn test sets
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)

    # Create the model
    model = MLP(8)
    # define the optimization
    criterion = BCELoss()
    #optimizer = SGD(model.parameters(), lr = 0.01, momentum=0.9)
    optimizer = Adam(model.parameters(), lr = 0.007)


    
    # Train the model
    print("Started Training")
    for epoch in range(40):
        for i, (inputs, targets) in enumerate(train_dl):
            # Clear gradients
            optimizer.zero_grad()
            # Computer model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()

    print("Finished Training")

    for i, (inputs, targets) in enumerate(test_dl):
        yhat = model(inputs)
        yhat = yhat.detach().numpy()
        yhat = (yhat>0.4).astype(int)
        cm = confusion_matrix(targets.detach().numpy(), yhat)
        print(cm)

    print(yhat[:10])

    # Make a prediction
    row = [-3.1, -2.3, -1.9, 0.34, 1.34, 2.98, 3.04, 9.3]
    row = Variable(torch.tensor([row]).float())
    yhat = model(row)
    yhat = yhat.detach().numpy()
    print(yhat)



if __name__ == "__main__":
    main()
