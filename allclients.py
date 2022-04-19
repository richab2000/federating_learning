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
import pandas as pd
from os import listdir
from os.path import isfile, join
import flbenchmark.datasets

from clientPyT import *

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

title_tr = "./data/" + 'sent140' + "/train"
net = Net().to(DEVICE)


for file_name in listdir(title_tr):
	if isfile(join(title_tr, file_name)):
		if file_name != "_main.json":
			trainloader, testloader = load_data('sent140','leaf', file_name)
			fl.client.start_numpy_client("[::]:8080", client=FlowerClient(trainloader, testloader))
