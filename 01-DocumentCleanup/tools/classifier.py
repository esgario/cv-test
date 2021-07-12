#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import inspect
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torchvision
from torch.nn.utils.clip_grad import clip_grad_norm_

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

current_path = os.path.dirname(os.path.abspath(inspect.getframeinfo(inspect.currentframe()).filename))
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
sys.path.append(parent_path)

from net_models import ResNetUNet, UNetMini
from augmentations import (
    Compose, RandomHorizontallyFlip, RandomVerticallyFlip, Scale,
    AdjustContrast, AdjustBrightness, AdjustSaturation, FreeScale
)
from customdatasets import CustomDataset, custom_collate

def select_model(model_name, pretrained=True, num_classes=1):
    if model_name == "unet":
        model = ResNetUNet(pretrained=pretrained, num_classes=num_classes)

    if model_name == "unetmini":
        model = UNetMini(num_classes=num_classes)

    if torch.cuda.is_available():
        model.cuda()
    return model


def build_network(backend, pretrained, num_classes, snapshot_filename=None, use_snapshot=True):

    # Instantiate network model
    net = select_model(backend.lower(), pretrained, num_classes)

    if use_snapshot:
        # Load a pretrained network
        try:
            net.load_state_dict(torch.load(f"{parent_path}/net_weights/{snapshot_filename}.pth"))

        except Exception as err:
            print("A problem has occurred loading model.", err)
    
    # Parallel training
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    
    # Sending model to GPU
    if torch.cuda.is_available():
        net = net.cuda()

    return net


def adjust_learning_rate(optimizer, epoch, opt):
    if opt.optimizer == "sgd":
        lr_values = [0.01, 0.005, 0.001, 0.0005, 0.0001]
        step = round(opt.epochs / 5)

        idx = min(math.floor(epoch / step), len(lr_values))
        learning_rate = lr_values[idx]

        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate

    return optimizer


def data_augmentations(size=640):
    train_augmentations = Compose([
        Scale(size),
        RandomHorizontallyFlip(0.5),
        RandomVerticallyFlip(0.5),
        AdjustContrast(0.25),
        AdjustBrightness(0.25),
        AdjustSaturation(0.25)
    ])

    val_augmentations = Compose([
        Scale(size)
    ])

    return train_augmentations, val_augmentations


def data_loader(opt):
    dataset_dir = os.path.join(parent_path, opt.images_dir)

    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    train_augmentations, val_augmentations = data_augmentations()

    # Dataset
    train_dataset = CustomDataset(
        root_dir=parent_path + '/dataset/train',
        augmentations=train_augmentations
    )

    val_dataset = CustomDataset(
        root_dir=parent_path + '/dataset/val',
        augmentations=val_augmentations
    )

    test_dataset = CustomDataset(
        root_dir=parent_path + '/dataset/test',
        augmentations=val_augmentations,
        test=True
    )

    # Loader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=opt.batch_size,
        collate_fn=custom_collate,
        pin_memory=True,
        shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=opt.batch_size,
        collate_fn=custom_collate,
        pin_memory=True,
        shuffle=False
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=1,
        collate_fn=custom_collate,
        pin_memory=True,
        shuffle=False
    )

    return train_loader, val_loader, test_loader


class Classifier:
    def __init__(self, parser):
        if parser is not None:
            self.opt = parser.parse_args()
        else:
            self.opt = None

        if torch.cuda.is_available():
            self.device = torch.device("cuda:" + str(torch.cuda.current_device()))
        else:
            self.device = torch.device("cpu")


    def train(self, train_loader, model, criterion, optimizer):
        # tell to pytorch that we are training the model
        model.train()

        loss_arr = []

        for i, (images, targets) in enumerate(train_loader):

            # Clear gradients parameters
            model.zero_grad()

            for image, target in zip(images, targets):
                # Loading images on gpu
                if torch.cuda.is_available():
                    image, target = image.cuda(), target.cuda()

                # Pass images through the network
                output = model(image.unsqueeze(0))[0]

                # Compute error
                loss = criterion(output, target)
                loss_arr.append(loss.data.cpu())

                # Getting gradients
                loss.backward()

                # Clipping gradient
                clip_grad_norm_(model.parameters(), 5)

            # Updating parameters
            optimizer.step()

            ## Completed percentage
            p = (100.0 * (i + 1)) / len(train_loader)

            sys.stdout.write("\r[%s][%.2f%%][LOSS:%.4f]" %
                ("=" * round(p / 2) + "-" * (50 - round(p / 2)), p, np.mean(loss_arr),)
            )
            sys.stdout.flush()

        print("")

        avg_loss = np.mean(loss_arr)
        return avg_loss

    def validation(self, val_loader, model, criterion):
        # tell to pytorch that we are evaluating the model
        model.eval()

        loss_arr = []

        with torch.no_grad():
            for i, (images, targets) in enumerate(val_loader):
                
                for image, target in zip(images, targets):
                    # Loading images on gpu
                    if torch.cuda.is_available():
                        image, target = image.cuda(), target.cuda()

                    # Pass images through the network
                    output = model(image.unsqueeze(0))[0]

                    # Compute error
                    loss = criterion(output, target)
                    loss_arr.append(loss.data.cpu())


                # Completed percentage
                p = (100.0 * (i + 1)) / len(val_loader)

                sys.stdout.write(
                    "\r[%s][%.2f%%][LOSS:%.4f]"
                    % (
                        "=" * round(p / 2) + "-" * (50 - round(p / 2)),
                        p,
                        np.mean(loss_arr)
                    )
                )
                sys.stdout.flush()

        print("")

        avg_loss = np.mean(loss_arr)
        return avg_loss

    def print_info(self, **kwargs):
        data_type = kwargs.get("data_type")
        avg_loss = kwargs.get("avg_loss")
        epoch = kwargs.get("epoch")
        epochs = kwargs.get("epochs")

        print(
            "\r[Epoch:%3d/%3d][%s][LOSS: %4.4f]"
            % (epoch + 1, epochs, data_type, avg_loss)
        )

    def run_training(self):
        print("training: " + self.opt.filename)

        # Data
        train_loader, val_loader, _ = data_loader(self.opt)

        # Model
        model = build_network(
            self.opt.model,
            self.opt.pretrained,
            self.opt.num_classes,
            self.opt.filename,
            self.opt.use_snapshot
            )

        # Criterion
        criterion_train = nn.MSELoss()
        criterion_val = nn.MSELoss()

        # Optimizer
        if self.opt.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=0.01,
                momentum=0.9,
                weight_decay=self.opt.weight_decay,
            )
        else:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=0.001, weight_decay=self.opt.weight_decay
            )

        best_loss = 1000.0

        for epoch in range(self.opt.epochs):
            # Training
            avg_train_loss = self.train(
                train_loader,
                model,
                criterion_train,
                optimizer,
            )
            self.print_info(
                data_type="TRAIN",
                avg_loss=avg_train_loss,
                epoch=epoch,
                epochs=self.opt.epochs,
            )

            # Validation
            avg_val_loss = self.validation(val_loader, model, criterion_val)
            self.print_info(
                data_type="VAL",
                avg_loss=avg_val_loss,
                epoch=epoch,
                epochs=self.opt.epochs,
            )

            # Adjust learning rate
            optimizer = adjust_learning_rate(optimizer, epoch, self.opt)

            # Record best model
            curr_loss = avg_val_loss
            if (curr_loss < best_loss) and epoch >= 5:
                best_loss = curr_loss

                # Saving model
                torch.save(model.state_dict(), f"{parent_path}/net_weights/{self.opt.filename}.pth")
                print("model saved")


    def run_test(self):
        # Dataset
        _, _, test_loader = data_loader(self.opt)

        # Loading model
        model = build_network(
            self.opt.model,
            self.opt.pretrained,
            self.opt.num_classes,
            self.opt.filename
            )
        model.eval()

        with torch.no_grad():
            for i, (images, targets) in enumerate(test_loader):
                for image, target in zip(images, targets):
                    # Loading images on gpu
                    if torch.cuda.is_available():
                        image, target = image.cuda(), target.cuda()

                    # pass images through the network
                    output = model(image.unsqueeze(0))
                    print(output.shape)

                    plt.imshow(output[0][0].cpu(), cmap='gray', vmin=0, vmax=1)
                    plt.show()
