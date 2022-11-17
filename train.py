import os
import time, datetime
import numpy as np
import matplotlib.pyplot as plt

from dataset import Customdataset
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from torchvision import utils
from models.mobilenetv2 import mobnet
from models.efficientnet import effinet
from efficientnet_pytorch import EfficientNet
from torchvision.models import efficientnet_b0

from tqdm import tqdm
from utils import *

import wandb


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def Training(Config, model, model_name):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    
    files_list = os.listdir(Config.data_path)[4940:5060]

    cls_form = [file.split('_')[0] for file in files_list]
    x_train, x_re, y_train, y_re = train_test_split(files_list, cls_form, test_size=0.25, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_re, y_re, test_size=0.2, random_state=42)

    train_set = Customdataset(Config.data_path, x_train, y_train, n_class, Config.mode)
    train_dataloader = DataLoader(train_set, batch_size=Config.batch_size)

    val_set = Customdataset(Config.data_path, x_val, y_val, n_class, Config.mode)
    val_dataloader = DataLoader(val_set, batch_size=Config.batch_size)

    test_set = Customdataset(Config.data_path, x_test, y_test, n_class, Config.mode)
    test_dataloader = DataLoader(test_set, batch_size=Config.batch_size)

    wandb.watch(model)

    lowest_loss = 0.2
    for epoch in range(Config.epochs):
        print(f'{epoch} epoch start! : {datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")}')

        train_losses, train_accs, val_losses, val_accs, epoch_loss = train_one_epoch(train_dataloader, val_dataloader, model, criterion, optimizer, Config.device)

        train_loss = train_losses / len(train_dataloader)
        train_acc = train_accs / len(train_set)
        val_loss = val_losses / len(val_dataloader)
        val_acc = val_accs / len(val_set)
        print(f'Epoch {epoch}/{Config.epochs-1}')
        print(f"Train Loss : {train_loss:.4f} // Accuracy : {train_acc:.4f}")
        print(f'Val Loss : {val_loss:.4f} // Accuracy : {val_acc:.4f}')
        print(f'Total_loss : {epoch_loss}')
        scheduler.step(val_loss)

        wandb.log({
                "Train": {"loss":train_loss, "acc": train_acc},
                "Val": {"loss":val_loss, "acc": val_acc}})

        if epoch_loss < lowest_loss:
            lowest_loss = epoch_loss
            state = {'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 
                    'loss': train_loss}
            filename = Config.save_path + f'{model_name}_{epoch}&{lowest_loss:.3f}.pth'
            torch.save(state, filename)
            print(f'Save model_ [loss : {epoch_loss:.4f}, save_path : {filename}]\n')

        if lowest_loss < 0.00001:
                break
    
    evaluate(test_dataloader, model, Config.device)

    wandb.finish()

def train_one_epoch(train_loader, val_loader, model, criterion, optimizer, device):
    model.train()

    train_loss = 0.0
    train_acc = 0
    val_loss = 0.0
    val_acc = 0 
    losses = AverageMeter()

    for i, (images, strokes, labels) in enumerate(tqdm(train_loader)):

        images = images.to(device)
        # strokes = strokes.to(device)
        labels = labels.to(device)

        output = model(images)
        _, preds = torch.max(output, 1)

        t_loss = criterion(output, labels)

        labels = torch.argmax(labels, dim=1)

        optimizer.zero_grad()
        t_loss.backward()
        optimizer.step()

        train_loss += t_loss.item()
        train_acc += (labels == preds).sum()
        losses.update(t_loss.item(), images.size(0))
    
    del images, labels

    example_images = []
    model.eval()
    with torch.no_grad():
        for i, (images, strokes, labels) in enumerate(val_loader):

            images = images.to(device)
            # strokes = strokes.to(device)
            labels = labels.to(device)

            output = model(images)
            _, preds = torch.max(output, 1)

            v_loss = criterion(output, labels)

            labels = torch.argmax(labels, dim=1)

            val_loss += v_loss.item()
            val_acc += (labels == preds).sum()
            losses.update(v_loss.item(), images.size(0))

            example_images.append(wandb.Image(
                images[0], caption="Pred: {} Truth: {}".format(classes[preds[0].item()], classes[labels[0]])))
    
    wandb.log({"Examples": example_images})
    del images, labels

    return train_loss, train_acc, val_loss, val_acc, losses.avg


def evaluate(test_loader, model, device):
    model.eval()

    test_acc = 0
    total = 0

    with torch.no_grad():
        for i, (images, strokes, labels) in enumerate(test_loader):

            images = images.to(device)
            # strokes = strokes.to(device)
            labels = labels.to(device)

            output = model(images)
            _, preds = torch.max(output, 1)

            labels = torch.argmax(labels, dim=1)

            total += images.size(0)
            test_acc += (labels == preds).sum().item()
    accuracy = test_acc / total

    print(f"Accuracy of test set: {accuracy:.4f}")
    wandb.log({"Eval": accuracy})