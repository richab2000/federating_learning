import warnings
from collections import OrderedDict

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
import json
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
import flbenchmark.datasets

#NOTE: currently this client is only for HORIZONTAL cases that have BOTH 
#the training and the testing dataset

#our datasets: 
    #femnist
    #celeba 
    #reddit
    #sent140
    #shakespeare
    #synthetic

# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self) -> None:
        super(Net, self).__init__()
        input_size, output_size, hidden_layer_size = 784, 62, 128
        self.fc1 = nn.Linear(input_size, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, output_size)
        #self.conv1 = nn.Conv2d(3, 6, 5)
        #self.pool = nn.MaxPool2d(2, 2)
        #self.conv2 = nn.Conv2d(6, 16, 5)
        #self.fc1 = nn.Linear(16 * 5 * 5, 120)
        #self.fc2 = nn.Linear(120, 84)
        #self.fc3 = nn.Linear(84, 10)

    def forward(self, x): #: torch.Tensor) -> torch.Tensor:
        #print("X VALUE", x, type(x))
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        return out
        #x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        #x = x.view(-1, 16 * 5 * 5)
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #return self.fc3(x)


def train(net, trainloader, epochs):
    """Train the model on the training set."""
    print("TRAINING")
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for data in trainloader:
            labels = torch.as_tensor(data['y'])
            images = data['x']
            #images = np.asarray(data['x'])
            #images = torch.from_numpy([i.astype('long') for i in images])
            #images = torch.tensor(images)
            one_hot = torch.nn.functional.one_hot(labels.long(), num_classes= 62)

            #labels, images = one_hot.to(DEVICE), images.to(DEVICE)

            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

def test(net, testloader):
    """Validate the model on the test set."""
    print("TESTING")
    criterion = torch.nn.MSELoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for data in testloader:
            labels = torch.as_tensor(data['y'])
            images = data['x']
            images = torch.tensor(images)

            #images = np.asarray(data['x'])
            #print(images)
            #print("images", type(images), type(images[0]))
            #exit(0)
            #images = torch.from_numpy([i.astype('long') for i in images])

            one_hot = torch.nn.functional.one_hot(labels.long(), num_classes= 62)

            #labels, images = one_hot.to(DEVICE), images.to(DEVICE)

            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy


def load_data(folder_name, data_type, file_name):
    if "vertical" in folder_name: 
        print("Will not run this data becaus it is vertical")
        return 
    title_tr = "./data/" + folder_name + "/train"
    title_te = "./data/" + folder_name + "/test"
    transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = []
    with open(title_tr + '/' + file_name, 'r') as f:
        trainset.append(json.load(f))
        training_data = trainset[0]['records']
        training_data = np.array([np.array(lst) for lst in training_data])

    testset = []
    with open(title_te + '/' + file_name, 'r') as f:
        testset.append(json.load(f))
        testing_data = testset[0]['records']
        testing_data = np.array([np.array(lst) for lst in testing_data])

    train_lst = []
    d = {}
    for t in training_data:
        d['y'] = float(t[0])
        #print("DATA TYPE Y", type(t[0]))
        d['x'] = [float(x) for x in t[1:]]
        #print("DATA TYPE X", type(t[1]))
        train_lst.append(d)
        d = {}

    test_lst = []
    d = {}
    for t in testing_data:
        d['y'] = float(t[0])
        d['x'] = [float(x) for x in t[1:]] 
        test_lst.append(d)
        d = {}

    return DataLoader(train_lst, batch_size=32, shuffle=True), DataLoader(test_lst)


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)
net = Net().to(DEVICE)
trainloader, testloader = load_data('femnist','leaf', 'f0006_12.json')
#print("PRINTING TEST OUTSIDE")
#print(testloader)

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, testloader):
        self.trainloader = trainloader
        self.testloader = testloader

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, self.trainloader, epochs=1)
        return self.get_parameters(), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, self.testloader)
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client("[::]:8080", client=FlowerClient(trainloader, testloader))