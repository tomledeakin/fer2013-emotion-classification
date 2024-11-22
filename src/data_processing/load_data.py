import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from src.data_processing.dataset import ImageDataset


# train_dir = '../../data/raw/train'
# test_dir = '../../data/raw/test'
#
# classes = os.listdir(test_dir)
#
# print(f'Classes: {classes}')
#
# label2idx = {cls:idx for idx, cls in enumerate(classes)}
# idx2label = {idx:cls for idx, cls in enumerate(classes)}
#
# print(label2idx)
# print(idx2label)
#
# test_image = '../../data/raw/train/angry/Training_10118481.jpg'
# img = cv2.imread(test_image)
# # cv2.imshow('img', img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
#
# img_height, img_weight = (128, 128)
# print(f'Image size: [height: {img_height}, width: {img_weight}]')
#
# batch_size = 256
#
# train_data = ImageDataset(train_dir, True, label2idx, split='train')
# train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
#
# val_dataset = ImageDataset(train_dir, True, label2idx, split='val')
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#
# test_dataset = ImageDataset(test_dir, True, label2idx, split='test')
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#
# image_batch, label_batch = next(iter(train_loader))

def get_data_loaders(train_dir, test_dir, label2idx, batch_size=256):
    train_data = ImageDataset(train_dir, True, label2idx, split='train')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    val_dataset = ImageDataset(train_dir, True, label2idx, split='val')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = ImageDataset(test_dir, True, label2idx, split='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader






# plt.figure(figsize=(10, 10))
# for i in range(9):
#   ax = plt.subplot(3, 3, i + 1)
#   minv = image_batch[i].numpy().min()
#   maxv = image_batch[i].numpy().max()
#   plt.imshow(np.squeeze(image_batch[i].numpy()), vmin=minv, vmax=maxv, cmap='gray')
#   label = label_batch[i]
#   plt.title(idx2label[label.item()])
#   plt.axis('off')