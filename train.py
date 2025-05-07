import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from SegmentationDataset import *
from model import *

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import optim
from sklearn.cluster import KMeans

from tqdm import tqdm as tqdm

import numpy as np
import matplotlib.pyplot as plt

root_path = r'PATH'

data_dir = root_path

train_dir = os.path.join(data_dir, "train")

val_dir = os.path.join(data_dir, "val")

train_fns = os.listdir(train_dir)

val_fns = os.listdir(val_dir)

num_items = 1000

color_array = np.random.choice(range(256), 3*num_items).reshape(-1, 3)

num_classes = 10

label_model = KMeans(n_clusters = num_classes)
label_model.fit(color_array)

train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")


train_dataset = CityscapesDataset(image_dir=train_dir, label_model=label_model)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

val_dataset = CityscapesDataset(image_dir=val_dir, label_model=label_model)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

backbone = DilatedResNet50()
model = PSPN()

EPOCH = 30
batch_size = 16
LR = 1e-3
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

min_loss = float("inf")

for epoch in range(EPOCH):
    backbone.train()
    model.train()
    train_loss = 0
    for data in tqdm(train_loader, leave = False):
        inputs, labels = data["input"], data["label"]

        features = backbone(inputs)
        predict = model(features)

        loss = criterion(predict, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    with torch.no_grad():
        backbone.eval()
        model.eval()
        val_loss = 0
        for data_val in tqdm(val_loader, leave = False):
            inputs_val, labels_val = data_val["input"], data_val["label"]
            features_val = backbone(inputs_val)
            predict_val = model(features_val)

            loss_val = criterion(predict_val, labels_val)
            val_loss += loss_val.item()

    if val_loss < min_loss:
        min_loss = val_loss
        torch.save(backbone.state_dict(), r'backbone PATH')
        torch.save(model.state_dict(), r'model PATH')


    print(f"{epoch+1}/{EPOCH} Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f},")

