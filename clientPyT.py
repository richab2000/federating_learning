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
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train(net, trainloader, epochs):
    """Train the model on the training set."""
    print("TRAINING")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    #for _ in range(epochs):
    #    for labels, images in trainloader:
    #        labels, images = labels.to(DEVICE), images.to(DEVICE)
    #        optimizer.zero_grad()
    #        loss = criterion(net(images), labels)
    #        loss.backward()
    #        optimizer.step()

    for _ in range(epochs):
        for instance in tqdm(trainloader):
            lst = instance[0]
            images = np.array(np.array([x[1:]] for x in lst))
            labels = np.array(np.array([x[0]] for x in lst))
            #print('image', img)
            #print('lab', label)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            #criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()
            print("success")

    print("FINISHED TRAINING")


def test(net, testloader):
    """Validate the model on the test set."""
    print("TESTING")
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        print("LEN TESTLOADER", len(testloader))
        for data in testloader:
            print("PRINTING DATA")
            print(data)
            print("PRINTING SINGULAR")
            #print(data[0].to(DEVICE),data[1].to(DEVICE))
            exit(0)
            #images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            #outputs = net(images)
            #loss += criterion(outputs, labels).item()
            #_, predicted = torch.max(outputs.data, 1)
            #total += labels.size(0)
            #correct += (predicted == labels).sum().item()
    #accuracy = correct / total
    #return loss, accuracy
    #with torch.no_grad():
    #    for images, labels in tqdm(testloader):
    #        outputs = net(images.to(DEVICE))
    #        labels = labels.to(DEVICE)
    #        loss += criterion(outputs, labels).item()
    #        total += labels.size(0)
    #        correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    #return loss / len(testloader.dataset), correct / total


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
        #print(trainset[0]['records'])
        #print(type(training_data))
        #print(len(trainset[0]['records']))
        #print(type(training_data[0]))

        #trainset = trainset.to_csv(file_name)
    #for file_name in listdir(title_tr):
    #    if isfile(join(title_tr, file_name)):
    #        if file_name != "_main.json":
    #            with open(title_tr + '/' + file_name, 'r') as f:
    #               trainset.append(json.load(f))

    testset = []
    with open(title_te + '/' + file_name, 'r') as f:
        testset.append(json.load(f))
        testing_data = testset[0]['records']
        testing_data = np.array([np.array(lst) for lst in testing_data])

    print(testing_data)
    exit(0)

    #for file_name in listdir(title_te):
    #    if isfile(join(title_te, file_name)):
    #        if file_name != "_main.json":
    #            with open(title_te + '/' + file_name, 'r') as f:
    #                testset.append(json.load(f))



    #update trainset
    #lst_train = []
    #for t in trainset:
    #    cols = t['column_name']
    #    for vals in t['records']:
    #        d = {}
    #        for i in range(len(cols)):
    #            d[cols[i]] = vals[i]
    #        lst_train.append(d)
    #trainset = lst_train

    #update testset     
    #lst_test = []
    #for t in testset:
    #    cols = t['column_name']
    #    for vals in t['records']:
    #        d = {}
    #        for i in range(len(cols)):
    #            d[cols[i]] = vals[i]
    #        lst_test.append(d)
    #testset = lst_test
    #print("Len tstset", len(testset))
            
    #print(testset)
    #exit(0)
    return DataLoader(training_data, batch_size=32, shuffle=False), DataLoader(testing_data)


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)
net = Net().to(DEVICE)
trainloader, testloader = load_data('femnist','leaf', 'f0003_42.json')
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