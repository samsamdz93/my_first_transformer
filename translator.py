#!/usr/bin/env python


import argparse
import os
import json
from time import gmtime, strftime, time
import numpy as np

import pandas as pd
from numpy import inf

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from neural_networks import *
from neural_networks.transformer import Transformer
from dataset_manager import *
from train_model import *


# Parsing arguments
parser = argparse.ArgumentParser(
        prog = "Translator",
        description = "Translator based on a transformer"
        )

parser.add_argument("--dataset_path", type = str, required = True, help = "path to the dataset file")
parser.add_argument("--model_path", type = str, default = '', help = "path to the model (if we want to load a model)")
parser.add_argument("--lr", type = float, default = 1e-3, help = "learning rate of the SGD")
parser.add_argument("--batch_size", type = int, default = 8, help = "batch size during the learning")
parser.add_argument("--momentum", type = float, default = 0, help = "momentum of the SGD")
parser.add_argument("--save_dir", type = str, default = "", help = "where to save the model, the logs and the configuration")
parser.add_argument("--nepochs", type = int, default = 20, help = "number of epochs to make")
parser.add_argument("--label_smoothing", type = float, default = 0.1, help = "label smoothing to prevent overfitting")
parser.add_argument("--dropout", type = float, default = 0.1, help = "dropout in the transformer")
args = parser.parse_args()


# Create the directory containing the model, the logs, etc.
dir_name = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
out_dir = os.path.join(args.save_dir, dir_name)
os.makedirs(out_dir)

path_model = os.path.join(out_dir, "model.pth")
path_config = os.path.join(out_dir, "config.json")
path_logs = os.path.join(out_dir, "logs.json")

# Store the configuration
with open(path_config, "w") as f:
    json.dump(vars(args), f)

########################
# Managing the dataset #
########################


# Using mps if available
device = torch.device("mps" if torch.mps.is_available() else "cpu")

# Getting the dataset
df = load_dataset(args.dataset_path)
print(df.head())

# Get vectorizer for french and english
vec_fr, vec_en = make_vectorizers(df)

# Get their analyzers
analyzer_fr, analyzer_en = make_analyzers(vec_fr, vec_en)
print(analyzer_fr('je mange des carrottes'))

# Vocabulary size of each language
vocabulary_size_fr = len(vec_fr.vocabulary_)
vocabulary_size_en = len(vec_en.vocabulary_)
print('vocabulary_size_fr :', vocabulary_size_fr)
print('vocabulary_size_en :', vocabulary_size_en)

# Length of the longest sequence in the dataset
max_len_fr, max_len_en = get_max_lenghts(df)
print('max_len_fr :', max_len_fr)
print('max_len_en :', max_len_en)


# Dictionaries to convert token_id into string
invert_vocabulary_fr, invert_vocabulary_en = invert_vocabularies(vec_fr, vec_en)

# Constant tokens
VOID_TOKEN_EN = vec_en.vocabulary_.get('vvvvv')
VOID_TOKEN_FR = vec_fr.vocabulary_.get('vvvvv')
END_TOKEN = vec_en.vocabulary_.get('eeeee')
START_TOKEN = vec_en.vocabulary_.get('sssss')
print('VOID_TOKEN_EN :', VOID_TOKEN_EN)
print('VOID_TOKEN_FR :', VOID_TOKEN_FR)
print('START_TOKEN :', START_TOKEN)
print('END_TOKEN :', END_TOKEN)

# Batch size
batch_size = args.batch_size
print('batch_size :', batch_size)

# Separating the train and test sets
train_df, test_df = train_test_split(
    df,
    test_size=0.15,
    random_state=42,
    shuffle=True
)

# Converting into a Pytorch Dataset
train_dataset = TranslationDataset(train_df, VOID_TOKEN_FR, VOID_TOKEN_EN)
test_dataset = TranslationDataset(test_df, VOID_TOKEN_FR, VOID_TOKEN_EN)
print('Train_size :', len(train_dataset))
print('Test_size :', len(test_dataset))

print('Train df')
print(train_df[['sample fr', 'sample en']][:batch_size])

# Making a loader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False)



####################
# Training a model #
####################
print('=== Start training ===', end = '\n\n')


# Device
device = torch.device("mps" if torch.mps.is_available() else "cpu")
print('Device :', device)

# Model
model = Transformer(vocabulary_size_fr, vocabulary_size_en, pad_fr = VOID_TOKEN_FR, pad_en = VOID_TOKEN_EN)
print(model)

# Load weights if specified
if args.model_path != '':
    model.load_state_dict(torch.load(args.model_path, weights_only=True))
model.to(device)

# Loss (while avoiding void token)
criterion = nn.CrossEntropyLoss(reduction = 'sum', label_smoothing = args.label_smoothing)

# Optimizer
optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum = args.momentum)

# Checking parameter space
print("Number of Parameters :", sum(p.numel() for p in model.parameters()))

# Number of epochs
nepochs = args.nepochs

# Train the model
train_model(
    model,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    nepochs,
    path_logs = path_logs,
    path_model = path_model,
    device = device,
    VOID_TOKEN = VOID_TOKEN_EN
    )

torch.save(model.state_dict(), path_model)





