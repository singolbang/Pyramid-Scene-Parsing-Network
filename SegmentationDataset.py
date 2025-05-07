import os
import torch
import numpy as np
from torchvision import transforms
from sklearn.cluster import KMeans

from torch.utils.data import Dataset
from PIL import Image

class CityscapesDataset(Dataset):

    def __init__(self, image_dir, label_model):
        self.image_dir = image_dir
        self.image_fns = os.listdir(image_dir)
        self.label_model = label_model

    def __len__(self):
        return len(self.image_fns)

    def __getitem__(self, index):
        image_fn = self.image_fns[index]
        image_fp = os.path.join(self.image_dir, image_fn)
        image = Image.open(image_fp)
        image = np.array(image)
        cityscape, label = self.split_image(image)
        label_class = self.label_model.predict(label.reshape(-1, 3)).reshape(256, 256)
        label_class = torch.Tensor(label_class).long()
        data = {'input' : cityscape, 'label' : label_class}

        data['input'] = self.transform(data['input'])
        data['label'] = data['label'].long()

        return data

    def split_image(self, image):
        image = np.array(image)
        cityscape, label = image[:, :256, :], image[:, 256:, :]
        return cityscape, label

    def transform(self, image):
        transform_ops = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.56, 0.406), std=(0.229, 0.224, 0.225))
        ])
        return transform_ops(image)
