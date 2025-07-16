import copy
import math
import os

from torch.optim.lr_scheduler import ExponentialLR

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

torch.cuda.empty_cache()
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.optim as optim
from models.clientmodel_FE import UNET_FE
from models.clientmodel_BE import UNET_BE
from models.servermodel import UNET_server
from errorcorrection_main.adaptiveFL_utils_splitfed_shallow import (get_loaders, eval, get_loaders_test, test)
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from sklearn.metrics import jaccard_score
from options import args_parser
from agg.Fed_Avg import fedAvg
import numpy as np
import pandas as pd
import random
import time
from errorcorrection_main.cal_epoch import cal_epoch
from errorcorrection_main.cal_lr import cal_lr
from normali import min_max_scaling

# Hyperparameters
LEARNING_RATE = 0.0001
device = "cuda"
NUM_WORKERS = 1
SHUFFLE = False
NUM_CLASSES = 5
PIN_MEMORY = False

dataDir = "C:/Users/csj5/Projects/Data/BlastocystDATA/7Clients/Uniform/"
parentF="C:/Users/csj5/Projects/Adaptive_Splitfed-V2/"

# client 1
TRAIN_IMG_DIR_C1 = dataDir + "./client1/train_imgs/"
TRAIN_MASK_DIR_C1 = dataDir + "./client1/train_masks/"
VAL_IMG_DIR_C1 = dataDir + "./client1/val_imgs/"
VAL_MASK_DIR_C1 = dataDir + "./client1/val_masks/"

# client 2
TRAIN_IMG_DIR_C2 = dataDir + "./client2/train_imgs/"
TRAIN_MASK_DIR_C2 = dataDir + "./client2/train_masks/"
VAL_IMG_DIR_C2 = dataDir + "./client2/val_imgs/"
VAL_MASK_DIR_C2 = dataDir + "./client2/val_masks/"

# client 3
TRAIN_IMG_DIR_C3 = dataDir + "./client3/train_imgs/"
TRAIN_MASK_DIR_C3 = dataDir + "./client3/train_masks/"
VAL_IMG_DIR_C3 = dataDir + "./client3/val_imgs/"
VAL_MASK_DIR_C3 = dataDir + "./client3/val_masks/"

# client 4
TRAIN_IMG_DIR_C4 = dataDir + "./client4/train_imgs/"
TRAIN_MASK_DIR_C4 = dataDir + "./client4/train_masks/"
VAL_IMG_DIR_C4 = dataDir + "./client4/val_imgs/"
VAL_MASK_DIR_C4 = dataDir + "./client4/val_masks/"

# client 5
TRAIN_IMG_DIR_C5 = dataDir + "./client5/train_imgs/"
TRAIN_MASK_DIR_C5 = dataDir + "./client5/train_masks/"
VAL_IMG_DIR_C5 = dataDir + "./client5/val_imgs/"
VAL_MASK_DIR_C5 = dataDir + "./client5/val_masks/"

# client 6
TRAIN_IMG_DIR_C6 = dataDir + "./client6/train_imgs/"
TRAIN_MASK_DIR_C6 = dataDir + "./client6/train_masks/"
VAL_IMG_DIR_C6 = dataDir + "./client6/val_imgs/"
VAL_MASK_DIR_C6 = dataDir + "./client6/val_masks/"

# client 7
TRAIN_IMG_DIR_C7 = dataDir + "./client7/train_imgs/"
TRAIN_MASK_DIR_C7 = dataDir + "./client7/train_masks/"
VAL_IMG_DIR_C7 = dataDir + "./client7/val_imgs/"
VAL_MASK_DIR_C7 = dataDir + "./client7/val_masks/"

TEST_IMG_DIR = dataDir + "./test_imgs_new/"
TEST_MASK_DIR = dataDir + "./test_masks_new/"


# 1. Screen Train function
def train_screen(train_loader, local_model1, local_model2, local_model3, optimizer1, optimizer2, optimizer3, loss_fn,
                 PL):
    loop = tqdm(train_loader)
    train_running_loss = 0.0
    train_running_correct = 0.0
    train_iou_score = 0.0
    train_iou_score_class0 = 0.0
    train_iou_score_class1 = 0.0
    train_iou_score_class2 = 0.0
    train_iou_score_class3 = 0.0
    train_iou_score_class4 = 0.0
    PL_uplink1 = PL[0]
    PL_downlink1 = PL[1]
    PL_uplink2 = PL[2]
    PL_downlink2 = PL[3]
    reliablelst = []
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.type(torch.LongTensor).to(device)
        predictions1 = local_model1(data)
        lostpredictions1 = PL_uplink1 * predictions1
        predictions2 = local_model2(lostpredictions1)
        lostpredictions2 = PL_downlink1 * predictions2
        predictions3 = local_model3(lostpredictions2)
        are_equal = torch.equal(predictions1, lostpredictions1)
        if (are_equal):  # no zero rows
            isreliabledata = True  # NR
            quantified_PL = 0
        else:
            isreliabledata = False  # R
            # quantifying PL:
            mis_ls = torch.where((lostpredictions1[0][1] == predictions1[0][1]).all(dim=1))[0]
            quantified_PL = (256 - mis_ls.size()[0]) / 256
        print(isreliabledata)
        reliablelst.append(isreliabledata)
        loss = loss_fn(predictions3, targets)
        preds = torch.argmax(predictions3, dim=1)
        equals = preds == targets
        train_running_correct += torch.mean(equals.type(torch.FloatTensor)).item()
        train_running_loss += loss.item()
        train_iou_score += jaccard_score(targets.cpu().flatten(), preds.cpu().flatten(), average='micro')
        iou_sklearn = jaccard_score(targets.cpu().flatten(), preds.cpu().flatten(), average=None)
        train_iou_score_class0 += iou_sklearn[0]
        train_iou_score_class1 += iou_sklearn[1]
        train_iou_score_class2 += iou_sklearn[2]
        train_iou_score_class3 += iou_sklearn[3]
        train_iou_score_class4 += iou_sklearn[4]
        loss.backward(retain_graph=True)
        optimizer3.step()
        optimizer3.zero_grad()
        mygrad3 = grads3
        mygrad3lost = PL_uplink2 * mygrad3
        predictions2.backward(mygrad3lost, retain_graph=True)
        optimizer2.step()
        optimizer2.zero_grad()
        mygrad2 = grads2
        mygrad2lost = PL_downlink2 * mygrad2
        predictions1.backward(mygrad2lost)
        optimizer1.step()
        optimizer1.zero_grad()
        loop.set_postfix(loss=loss.item())
    epoch_loss = train_running_loss / len(train_loader.dataset)
    epoch_acc = 100. * (train_running_correct / len(train_loader.dataset))
    epoch_iou_class0 = (train_iou_score_class0 / len(train_loader.dataset))
    epoch_iou_class1 = (train_iou_score_class1 / len(train_loader.dataset))
    epoch_iou_class2 = (train_iou_score_class2 / len(train_loader.dataset))
    epoch_iou_class3 = (train_iou_score_class3 / len(train_loader.dataset))
    epoch_iou_class4 = (train_iou_score_class4 / len(train_loader.dataset))
    epoch_iou_withbackground = (
                                       epoch_iou_class0 + epoch_iou_class1 + epoch_iou_class2 + epoch_iou_class3 + epoch_iou_class4) / 5
    epoch_iou_nobackground = (epoch_iou_class1 + epoch_iou_class2 + epoch_iou_class3 + epoch_iou_class4) / 4
    if sum(reliablelst) == len(reliablelst):
        isreliableclient = True
    else:
        isreliableclient = False
    print("isreliableclient:", isreliableclient)

    return epoch_loss, epoch_acc, epoch_iou_withbackground, epoch_iou_nobackground, epoch_iou_class0, epoch_iou_class1, epoch_iou_class2, epoch_iou_class3, epoch_iou_class4, isreliableclient, quantified_PL


# 1. Train function
def train(train_loader, local_model1, local_model2, local_model3, optimizer1, optimizer2, optimizer3, scheduler1,
          scheduler2, scheduler3, loss_fn, PL):
    loop = tqdm(train_loader)
    train_running_loss = 0.0
    train_running_correct = 0.0
    train_iou_score = 0.0
    train_iou_score_class0 = 0.0
    train_iou_score_class1 = 0.0
    train_iou_score_class2 = 0.0
    train_iou_score_class3 = 0.0
    train_iou_score_class4 = 0.0
    PL_uplink1 = PL[0]
    PL_downlink1 = PL[1]
    PL_uplink2 = PL[2]
    PL_downlink2 = PL[3]
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.type(torch.LongTensor).to(device)
        predictions1 = local_model1(data)
        lostpredictions1 = PL_uplink1 * predictions1
        predictions2 = local_model2(lostpredictions1)
        lostpredictions2 = PL_downlink1 * predictions2
        predictions3 = local_model3(lostpredictions2)
        loss = loss_fn(predictions3, targets)
        preds = torch.argmax(predictions3, dim=1)
        equals = preds == targets
        train_running_correct += torch.mean(equals.type(torch.FloatTensor)).item()
        train_running_loss += loss.item()
        train_iou_score += jaccard_score(targets.cpu().flatten(), preds.cpu().flatten(), average='micro')
        iou_sklearn = jaccard_score(targets.cpu().flatten(), preds.cpu().flatten(), average=None)
        train_iou_score_class0 += iou_sklearn[0]
        train_iou_score_class1 += iou_sklearn[1]
        train_iou_score_class2 += iou_sklearn[2]
        train_iou_score_class3 += iou_sklearn[3]
        train_iou_score_class4 += iou_sklearn[4]
        loss.backward(retain_graph=True)
        optimizer3.step()
        optimizer3.zero_grad()
        mygrad3 = grads3
        mygrad3lost = PL_uplink2 * mygrad3
        predictions2.backward(mygrad3lost, retain_graph=True)
        optimizer2.step()
        optimizer2.zero_grad()
        mygrad2 = grads2
        mygrad2lost = PL_downlink2 * mygrad2
        predictions1.backward(mygrad2lost)
        optimizer1.step()
        optimizer1.zero_grad()
        loop.set_postfix(loss=loss.item())
    scheduler1.step()
    scheduler2.step()
    scheduler3.step()
    epoch_loss = train_running_loss / len(train_loader.dataset)
    epoch_acc = 100. * (train_running_correct / len(train_loader.dataset))
    epoch_iou_class0 = (train_iou_score_class0 / len(train_loader.dataset))
    epoch_iou_class1 = (train_iou_score_class1 / len(train_loader.dataset))
    epoch_iou_class2 = (train_iou_score_class2 / len(train_loader.dataset))
    epoch_iou_class3 = (train_iou_score_class3 / len(train_loader.dataset))
    epoch_iou_class4 = (train_iou_score_class4 / len(train_loader.dataset))
    epoch_iou_withbackground = (
                                       epoch_iou_class0 + epoch_iou_class1 + epoch_iou_class2 + epoch_iou_class3 + epoch_iou_class4) / 5
    epoch_iou_nobackground = (epoch_iou_class1 + epoch_iou_class2 + epoch_iou_class3 + epoch_iou_class4) / 4
    return epoch_loss, epoch_acc, epoch_iou_withbackground, epoch_iou_nobackground, epoch_iou_class0, epoch_iou_class1, epoch_iou_class2, epoch_iou_class3, epoch_iou_class4


# 2. Main function
def main():
    args = args_parser()
    start_time = time.time()
    train_transform = A.Compose(
        [
            A.Resize(height=args.image_height, width=args.image_width),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=args.image_height, width=args.image_width),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    train_loader_C1, val_loader_C1 = get_loaders(
        TRAIN_IMG_DIR_C1,
        TRAIN_MASK_DIR_C1,
        VAL_IMG_DIR_C1,
        VAL_MASK_DIR_C1,
        args.local_bs,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY

    )

    train_loader_C2, val_loader_C2 = get_loaders(
        TRAIN_IMG_DIR_C2,
        TRAIN_MASK_DIR_C2,
        VAL_IMG_DIR_C2,
        VAL_MASK_DIR_C2,
        args.local_bs,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY

    )

    train_loader_C3, val_loader_C3 = get_loaders(
        TRAIN_IMG_DIR_C3,
        TRAIN_MASK_DIR_C3,
        VAL_IMG_DIR_C3,
        VAL_MASK_DIR_C3,
        args.local_bs,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY
    )

    train_loader_C4, val_loader_C4 = get_loaders(
        TRAIN_IMG_DIR_C4,
        TRAIN_MASK_DIR_C4,
        VAL_IMG_DIR_C4,
        VAL_MASK_DIR_C4,
        args.local_bs,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY
    )

    train_loader_C5, val_loader_C5 = get_loaders(
        TRAIN_IMG_DIR_C5,
        TRAIN_MASK_DIR_C5,
        VAL_IMG_DIR_C5,
        VAL_MASK_DIR_C5,
        args.local_bs,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY
    )

    train_loader_C6, val_loader_C6 = get_loaders(
        TRAIN_IMG_DIR_C6,
        TRAIN_MASK_DIR_C6,
        VAL_IMG_DIR_C6,
        VAL_MASK_DIR_C6,
        args.local_bs,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY
    )

    train_loader_C7, val_loader_C7 = get_loaders(
        TRAIN_IMG_DIR_C7,
        TRAIN_MASK_DIR_C7,
        VAL_IMG_DIR_C7,
        VAL_MASK_DIR_C7,
        args.local_bs,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY
    )

    test_loader = get_loaders_test(
        TEST_IMG_DIR,
        TEST_MASK_DIR,
        train_transform
    )

    global_model1_fed = UNET_FE(in_channels=3).to(device)
    global_model2_fed = UNET_server(in_channels=32).to(device)
    global_model3_fed = UNET_BE(out_channels=NUM_CLASSES).to(device)

    # generating packet loss for the screening round - No retransmission -----------------------------------------------------------------------------------------------------
    SC1_PL, SC2_PL, SC3_PL, SC4_PL, SC5_PL, SC6_PL, SC7_PL, SC1_EPL, SC2_EPL, SC3_EPL, SC4_EPL, SC5_EPL, SC6_EPL, SC7_EPL = [], [], [], [], [], [], [], [], [], [], [], [], [], []
    imgsize = 256
    for c in range(1, 8):
        if (c == 1):  # C1------------------------------------------------
            c1ch_list1, c1ch_list2, c1ch_list3, c1ch_list4 = [], [], [], []
            for ch in range(32):
                rows1c1 = imgsize
                first = list(range(0, imgsize))
                random_tensor1c1 = torch.ones(imgsize, imgsize)
                num_zero_rows1c1 = math.floor(args.pack_prob_up_C1 * rows1c1)
                x1 = set(random.sample(list(first), num_zero_rows1c1))
                for i in range(imgsize):
                    if i in x1:
                        random_tensor1c1[i] = 0 * random_tensor1c1[i]
                output_tensor1c1 = random_tensor1c1
                c1ch_list1.append(output_tensor1c1)
            pack_loss_uplink1C1 = torch.stack(c1ch_list1, dim=0).to(device)

            for ch in range(32):
                rows2c1 = imgsize
                first = list(range(0, imgsize))
                random_tensor2c1 = torch.ones(imgsize, imgsize)
                num_zero_rows2c1 = math.floor(args.pack_prob_down_C1 * rows2c1)
                x2 = set(random.sample(list(first), num_zero_rows2c1))
                for i in range(imgsize):
                    if i in x2:
                        random_tensor2c1[i] = 0 * random_tensor2c1[i]
                output_tensor2c1 = random_tensor2c1
                c1ch_list2.append(output_tensor2c1)
            pack_loss_downlink1C1 = torch.stack(c1ch_list2, dim=0).to(device)

            for ch in range(32):
                rows3c1 = imgsize
                first = list(range(0, imgsize))
                random_tensor3c1 = torch.ones(imgsize, imgsize)
                num_zero_rows3c1 = math.floor(args.pack_prob_up_C1 * rows3c1)
                x3 = set(random.sample(list(first), num_zero_rows3c1))
                for i in range(imgsize):
                    if i in x3:
                        random_tensor3c1[i] = 0 * random_tensor3c1[i]
                output_tensor3c1 = random_tensor3c1
                c1ch_list3.append(output_tensor3c1)
            pack_loss_uplink2C1 = torch.stack(c1ch_list3, dim=0).to(device)

            for ch in range(32):
                rows4c1 = imgsize
                first = list(range(0, imgsize))
                random_tensor4c1 = torch.ones(imgsize, imgsize)
                num_zero_rows4c1 = math.floor(args.pack_prob_down_C1 * rows4c1)
                x4 = set(random.sample(list(first), num_zero_rows4c1))
                for i in range(imgsize):
                    if i in x4:
                        random_tensor4c1[i] = 0 * random_tensor4c1[i]
                output_tensor4c1 = random_tensor4c1
                c1ch_list4.append(output_tensor4c1)
            pack_loss_downlink2C1 = torch.stack(c1ch_list4, dim=0).to(device)

            SC1_PL.extend([pack_loss_uplink1C1, pack_loss_downlink1C1, pack_loss_uplink2C1, pack_loss_downlink2C1])
            SC1_EPL.extend([pack_loss_uplink1C1, pack_loss_downlink1C1])

        elif (c == 2):  # C2------------------------------------------------
            c2ch_list1, c2ch_list2, c2ch_list3, c2ch_list4 = [], [], [], []
            for ch in range(32):
                rows1c2 = imgsize
                first = list(range(0, imgsize))
                random_tensor1c2 = torch.ones(imgsize, imgsize)
                num_zero_rows1c2 = math.floor(args.pack_prob_up_C2 * rows1c2)
                x1 = set(random.sample(list(first), num_zero_rows1c2))
                for i in range(imgsize):
                    if i in x1:
                        random_tensor1c2[i] = 0 * random_tensor1c2[i]
                output_tensor1c2 = random_tensor1c2
                c2ch_list1.append(output_tensor1c2)
            pack_loss_uplink1C2 = torch.stack(c2ch_list1, dim=0).to(device)

            for ch in range(32):
                rows2c2 = imgsize
                first = list(range(0, imgsize))
                random_tensor2c2 = torch.ones(imgsize, imgsize)
                num_zero_rows2c2 = math.floor(args.pack_prob_down_C2 * rows2c2)
                x2 = set(random.sample(list(first), num_zero_rows2c2))

                for i in range(imgsize):
                    if i in x2:
                        random_tensor2c2[i] = 0 * random_tensor2c2[i]
                output_tensor2c2 = random_tensor2c2
                c2ch_list2.append(output_tensor2c2)
            pack_loss_downlink1C2 = torch.stack(c2ch_list2, dim=0).to(device)

            for ch in range(32):
                rows3c2 = imgsize
                first = list(range(0, imgsize))
                random_tensor3c2 = torch.ones(imgsize, imgsize)
                num_zero_rows3c2 = math.floor(args.pack_prob_up_C2 * rows3c2)
                x3 = set(random.sample(list(first), num_zero_rows3c2))
                for i in range(imgsize):
                    if i in x3:
                        random_tensor3c2[i] = 0 * random_tensor3c2[i]
                output_tensor3c2 = random_tensor3c2
                c2ch_list3.append(output_tensor3c2)
            pack_loss_uplink2C2 = torch.stack(c2ch_list3, dim=0).to(device)

            for ch in range(32):
                rows4c2 = imgsize
                first = list(range(0, imgsize))
                random_tensor4c2 = torch.ones(imgsize, imgsize)
                num_zero_rows4c2 = math.floor(args.pack_prob_down_C2 * rows4c2)
                x4 = set(random.sample(list(first), num_zero_rows4c2))
                for i in range(imgsize):
                    if i in x4:
                        random_tensor4c2[i] = 0 * random_tensor4c2[i]
                output_tensor4c2 = random_tensor4c2
                c2ch_list4.append(output_tensor4c2)
            pack_loss_downlink2C2 = torch.stack(c2ch_list4, dim=0).to(device)
            SC2_PL.extend(
                [pack_loss_uplink1C2, pack_loss_downlink1C2, pack_loss_uplink2C2, pack_loss_downlink2C2])
            SC2_EPL.extend([pack_loss_uplink1C2, pack_loss_downlink1C2])

        elif (c == 3):  # C3------------------------------------------------
            c3ch_list1, c3ch_list2, c3ch_list3, c3ch_list4 = [], [], [], []
            for ch in range(32):
                rows1c3 = imgsize
                first = list(range(0, imgsize))
                random_tensor1c3 = torch.ones(imgsize, imgsize)
                num_zero_rows1c3 = math.floor(args.pack_prob_up_C3 * rows1c3)
                x1 = set(random.sample(list(first), num_zero_rows1c3))
                for i in range(imgsize):
                    if i in x1:
                        random_tensor1c3[i] = 0 * random_tensor1c3[i]
                output_tensor1c3 = random_tensor1c3
                c3ch_list1.append(output_tensor1c3)
            pack_loss_uplink1C3 = torch.stack(c3ch_list1, dim=0).to(device)

            for ch in range(32):
                rows2c3 = imgsize
                first = list(range(0, imgsize))
                random_tensor2c3 = torch.ones(imgsize, imgsize)
                num_zero_rows2c3 = math.floor(args.pack_prob_down_C3 * rows2c3)
                x2 = set(random.sample(list(first), num_zero_rows2c3))
                for i in range(imgsize):
                    if i in x2:
                        random_tensor2c3[i] = 0 * random_tensor2c3[i]
                output_tensor2c3 = random_tensor2c3
                c3ch_list2.append(output_tensor2c3)
            pack_loss_downlink1C3 = torch.stack(c3ch_list2, dim=0).to(device)

            for ch in range(32):
                rows3c3 = imgsize
                first = list(range(0, imgsize))
                random_tensor3c3 = torch.ones(imgsize, imgsize)
                num_zero_rows3c3 = math.floor(args.pack_prob_up_C3 * rows3c3)
                x3 = set(random.sample(list(first), num_zero_rows3c3))
                for i in range(imgsize):
                    if i in x3:
                        random_tensor3c3[i] = 0 * random_tensor3c3[i]
                output_tensor3c3 = random_tensor3c3
                c3ch_list3.append(output_tensor3c3)
            pack_loss_uplink2C3 = torch.stack(c3ch_list3, dim=0).to(device)

            for ch in range(32):
                rows4c3 = imgsize
                first = list(range(0, imgsize))
                random_tensor4c3 = torch.ones(imgsize, imgsize)
                num_zero_rows4c3 = math.floor(args.pack_prob_down_C3 * rows4c3)
                x4 = set(random.sample(list(first), num_zero_rows4c3))
                for i in range(imgsize):
                    if i in x4:
                        random_tensor4c3[i] = 0 * random_tensor4c3[i]
                output_tensor4c3 = random_tensor4c3
                c3ch_list4.append(output_tensor4c3)
            pack_loss_downlink2C3 = torch.stack(c3ch_list4, dim=0).to(device)

            SC3_PL.extend([pack_loss_uplink1C3, pack_loss_downlink1C3, pack_loss_uplink2C3, pack_loss_downlink2C3])
            SC3_EPL.extend([pack_loss_uplink1C3, pack_loss_downlink1C3])

        elif (c == 4):  # C4------------------------------------------------
            c4ch_list1, c4ch_list2, c4ch_list3, c4ch_list4 = [], [], [], []
            for ch in range(32):
                rows1c4 = imgsize
                first = list(range(0, imgsize))
                random_tensor1c4 = torch.ones(imgsize, imgsize)
                num_zero_rows1c4 = math.floor(args.pack_prob_up_C4 * rows1c4)
                x1 = set(random.sample(list(first), num_zero_rows1c4))
                for i in range(imgsize):
                    if i in x1:
                        random_tensor1c4[i] = 0 * random_tensor1c4[i]
                output_tensor1c4 = random_tensor1c4
                c4ch_list1.append(output_tensor1c4)
            pack_loss_uplink1C4 = torch.stack(c4ch_list1, dim=0).to(device)

            for ch in range(32):
                rows2c4 = imgsize
                first = list(range(0, imgsize))
                random_tensor2c4 = torch.ones(imgsize, imgsize)
                num_zero_rows2c4 = math.floor(args.pack_prob_down_C4 * rows2c4)
                x2 = set(random.sample(list(first), num_zero_rows2c4))
                for i in range(imgsize):
                    if i in x2:
                        random_tensor2c4[i] = 0 * random_tensor2c4[i]
                output_tensor2c4 = random_tensor2c4
                c4ch_list2.append(output_tensor2c4)
            pack_loss_downlink1C4 = torch.stack(c4ch_list2, dim=0).to(device)

            for ch in range(32):
                rows3c4 = imgsize
                first = list(range(0, imgsize))
                random_tensor3c4 = torch.ones(imgsize, imgsize)
                num_zero_rows3c4 = math.floor(args.pack_prob_up_C4 * rows3c4)
                x3 = set(random.sample(list(first), num_zero_rows3c4))
                for i in range(imgsize):
                    if i in x3:
                        random_tensor3c4[i] = 0 * random_tensor3c4[i]
                output_tensor3c4 = random_tensor3c4
                c4ch_list3.append(output_tensor3c4)
            pack_loss_uplink2C4 = torch.stack(c4ch_list3, dim=0).to(device)

            for ch in range(32):
                rows4c4 = imgsize
                first = list(range(0, imgsize))
                random_tensor4c4 = torch.ones(imgsize, imgsize)
                num_zero_rows4c4 = math.floor(args.pack_prob_down_C4 * rows4c4)
                x4 = set(random.sample(list(first), num_zero_rows4c4))
                for i in range(imgsize):
                    if i in x4:
                        random_tensor4c4[i] = 0 * random_tensor4c4[i]
                output_tensor4c4 = random_tensor4c4
                c4ch_list4.append(output_tensor4c4)
            pack_loss_downlink2C4 = torch.stack(c4ch_list4, dim=0).to(device)

            SC4_PL.extend(
                [pack_loss_uplink1C4, pack_loss_downlink1C4, pack_loss_uplink2C4, pack_loss_downlink2C4])
            SC4_EPL.extend([pack_loss_uplink1C4, pack_loss_downlink1C4])

        elif (c == 5):  # C5------------------------------------------------
            c5ch_list1, c5ch_list2, c5ch_list3, c5ch_list4 = [], [], [], []
            for ch in range(32):
                rows1c5 = imgsize
                first = list(range(0, imgsize))
                random_tensor1c5 = torch.ones(imgsize, imgsize)
                num_zero_rows1c5 = math.floor(args.pack_prob_up_C5 * rows1c5)
                x1 = set(random.sample(list(first), num_zero_rows1c5))
                for i in range(imgsize):
                    if i in x1:
                        random_tensor1c5[i] = 0 * random_tensor1c5[i]
                output_tensor1c5 = random_tensor1c5
                c5ch_list1.append(output_tensor1c5)
            pack_loss_uplink1C5 = torch.stack(c5ch_list1, dim=0).to(device)

            for ch in range(32):
                rows2c5 = imgsize
                first = list(range(0, imgsize))
                random_tensor2c5 = torch.ones(imgsize, imgsize)
                num_zero_rows2c5 = math.floor(args.pack_prob_down_C5 * rows2c5)
                x2 = set(random.sample(list(first), num_zero_rows2c5))
                for i in range(imgsize):
                    if i in x2:
                        random_tensor2c5[i] = 0 * random_tensor2c5[i]
                output_tensor2c5 = random_tensor2c5
                c5ch_list2.append(output_tensor2c5)
            pack_loss_downlink1C5 = torch.stack(c5ch_list2, dim=0).to(device)

            for ch in range(32):
                rows3c5 = imgsize
                first = list(range(0, imgsize))
                random_tensor3c5 = torch.ones(imgsize, imgsize)
                num_zero_rows3c5 = math.floor(args.pack_prob_up_C5 * rows3c5)
                x3 = set(random.sample(list(first), num_zero_rows3c5))
                for i in range(imgsize):
                    if i in x3:
                        random_tensor3c5[i] = 0 * random_tensor3c5[i]
                output_tensor3c5 = random_tensor3c5
                c5ch_list3.append(output_tensor3c5)
            pack_loss_uplink2C5 = torch.stack(c5ch_list3, dim=0).to(device)

            for ch in range(32):
                rows4c5 = imgsize
                first = list(range(0, imgsize))
                random_tensor4c5 = torch.ones(imgsize, imgsize)
                num_zero_rows4c5 = math.floor(args.pack_prob_down_C5 * rows4c5)
                x4 = set(random.sample(list(first), num_zero_rows4c5))
                for i in range(imgsize):
                    if i in x4:
                        random_tensor4c5[i] = 0 * random_tensor4c5[i]
                output_tensor4c5 = random_tensor4c5
                c5ch_list4.append(output_tensor4c5)
            pack_loss_downlink2C5 = torch.stack(c5ch_list4, dim=0).to(device)

            SC5_PL.extend(
                [pack_loss_uplink1C5, pack_loss_downlink1C5, pack_loss_uplink2C5, pack_loss_downlink2C5])
            SC5_EPL.extend([pack_loss_uplink1C5, pack_loss_downlink1C5])



        elif (c == 6):  # C6------------------------------------------------
            c6ch_list1, c6ch_list2, c6ch_list3, c6ch_list4 = [], [], [], []
            for ch in range(32):
                rows1c6 = imgsize
                first = list(range(0, imgsize))
                random_tensor1c6 = torch.ones(imgsize, imgsize)
                num_zero_rows1c6 = math.floor(args.pack_prob_up_C6 * rows1c6)
                x1 = set(random.sample(list(first), num_zero_rows1c6))
                for i in range(imgsize):
                    if i in x1:
                        random_tensor1c6[i] = 0 * random_tensor1c6[i]
                output_tensor1c6 = random_tensor1c6
                c6ch_list1.append(output_tensor1c6)
            pack_loss_uplink1C6 = torch.stack(c6ch_list1, dim=0).to(device)

            for ch in range(32):
                rows2c6 = imgsize
                first = list(range(0, imgsize))
                random_tensor2c6 = torch.ones(imgsize, imgsize)
                num_zero_rows2c6 = math.floor(args.pack_prob_down_C6 * rows2c6)
                x2 = set(random.sample(list(first), num_zero_rows2c6))
                for i in range(imgsize):
                    if i in x2:
                        random_tensor2c6[i] = 0 * random_tensor2c6[i]
                output_tensor2c6 = random_tensor2c6
                c6ch_list2.append(output_tensor2c6)
            pack_loss_downlink1C6 = torch.stack(c6ch_list2, dim=0).to(device)

            for ch in range(32):
                rows3c6 = imgsize
                first = list(range(0, imgsize))
                random_tensor3c6 = torch.ones(imgsize, imgsize)
                num_zero_rows3c6 = math.floor(args.pack_prob_up_C6 * rows3c6)
                x3 = set(random.sample(list(first), num_zero_rows3c6))
                for i in range(imgsize):
                    if i in x3:
                        random_tensor3c6[i] = 0 * random_tensor3c6[i]
                output_tensor3c6 = random_tensor3c6
                c6ch_list3.append(output_tensor3c6)
            pack_loss_uplink2C6 = torch.stack(c6ch_list3, dim=0).to(device)

            for ch in range(32):
                rows4c6 = imgsize
                first = list(range(0, imgsize))
                random_tensor4c6 = torch.ones(imgsize, imgsize)
                num_zero_rows4c6 = math.floor(args.pack_prob_down_C6 * rows4c6)
                x4 = set(random.sample(list(first), num_zero_rows4c6))
                for i in range(imgsize):
                    if i in x4:
                        random_tensor4c6[i] = 0 * random_tensor4c6[i]
                output_tensor4c6 = random_tensor4c6
                c6ch_list4.append(output_tensor4c6)
            pack_loss_downlink2C6 = torch.stack(c6ch_list4, dim=0).to(device)

            SC6_PL.extend([pack_loss_uplink1C6, pack_loss_downlink1C6, pack_loss_uplink2C6, pack_loss_downlink2C6])
            SC6_EPL.extend([pack_loss_uplink1C6, pack_loss_downlink1C6])


        else:  # C7------------------------------------------------
            c7ch_list1, c7ch_list2, c7ch_list3, c7ch_list4 = [], [], [], []
            for ch in range(32):
                rows1c7 = imgsize
                first = list(range(0, imgsize))
                random_tensor1c7 = torch.ones(imgsize, imgsize)
                num_zero_rows1c7 = math.floor(args.pack_prob_up_C7 * rows1c7)
                x1 = set(random.sample(list(first), num_zero_rows1c7))
                for i in range(imgsize):
                    if i in x1:
                        random_tensor1c7[i] = 0 * random_tensor1c7[i]
                output_tensor1c7 = random_tensor1c7
                c7ch_list1.append(output_tensor1c7)
            pack_loss_uplink1C7 = torch.stack(c7ch_list1, dim=0).to(device)

            for ch in range(32):
                rows2c7 = imgsize
                first = list(range(0, imgsize))
                random_tensor2c7 = torch.ones(imgsize, imgsize)
                num_zero_rows2c7 = math.floor(args.pack_prob_down_C7 * rows2c7)
                x2 = set(random.sample(list(first), num_zero_rows2c7))
                for i in range(imgsize):
                    if i in x2:
                        random_tensor2c7[i] = 0 * random_tensor2c7[i]
                output_tensor2c7 = random_tensor2c7
                c7ch_list2.append(output_tensor2c7)
            pack_loss_downlink1C7 = torch.stack(c7ch_list2, dim=0).to(device)

            for ch in range(32):
                rows3c7 = imgsize
                first = list(range(0, imgsize))
                random_tensor3c7 = torch.ones(imgsize, imgsize)
                num_zero_rows3c7 = math.floor(args.pack_prob_up_C7 * rows3c7)
                x3 = set(random.sample(list(first), num_zero_rows3c7))
                for i in range(imgsize):
                    if i in x3:
                        random_tensor3c7[i] = 0 * random_tensor3c7[i]
                output_tensor3c7 = random_tensor3c7
                c7ch_list3.append(output_tensor3c7)
            pack_loss_uplink2C7 = torch.stack(c7ch_list3, dim=0).to(device)

            for ch in range(32):
                rows4c7 = imgsize
                first = list(range(0, imgsize))
                random_tensor4c7 = torch.ones(imgsize, imgsize)
                num_zero_rows4c7 = math.floor(args.pack_prob_down_C7 * rows4c7)
                x4 = set(random.sample(list(first), num_zero_rows4c7))
                for i in range(imgsize):
                    if i in x4:
                        random_tensor4c7[i] = 0 * random_tensor4c7[i]
                output_tensor4c7 = random_tensor4c7
                c7ch_list4.append(output_tensor4c7)
            pack_loss_downlink2C7 = torch.stack(c7ch_list4, dim=0).to(device)

            SC7_PL.extend([pack_loss_uplink1C7, pack_loss_downlink1C7, pack_loss_uplink2C7, pack_loss_downlink2C7])
            SC7_EPL.extend([pack_loss_uplink1C7, pack_loss_downlink1C7])

    # Overall data distribution
    tot_loader = len(train_loader_C1) + len(train_loader_C2) + len(train_loader_C3) + len(train_loader_C4) + len(train_loader_C5) + len(train_loader_C6) + len(train_loader_C7)
    D1 = len(train_loader_C1) / tot_loader;
    D2 = len(train_loader_C2) / tot_loader;
    D3 = len(train_loader_C3) / tot_loader;
    D4 = len(train_loader_C4) / tot_loader;
    D5 = len(train_loader_C5) / tot_loader;
    D6 = len(train_loader_C6) / tot_loader;
    D7 = len(train_loader_C7) / tot_loader;

    D = []
    D.extend([D1, D2, D3, D4, D5, D6, D7])
    Dmax = max(D)

    print("Data ratios")
    print(D1)
    print(D2)
    print(D3)
    print(D4)
    print(D5)
    print(D6)
    print(D7)

    # ====================================================SCREENING ROUND ===============================================
    print(f'\n |     SCREENING COM ROUND       |....')
    S1time, S2time, S3time, S4time, S5time, S6time, S7time = 0, 0, 0, 0, 0, 0, 0
    client1_train_acc, client1_train_loss, client1_train_withbackiou, client1_train_nobackiou, client1_val_acc, client1_val_loss, client1_val_withbackiou, client1_val_nobackiou, client1_g_val_acc, client1_g_val_loss, client1_g_val_iouwithback, client1_g_val_iounoback = [], [], [], [], [], [], [], [], [], [], [], []
    client2_train_acc, client2_train_loss, client2_train_withbackiou, client2_train_nobackiou, client2_val_acc, client2_val_loss, client2_val_withbackiou, client2_val_nobackiou, client2_g_val_acc, client2_g_val_loss, client2_g_val_iouwithback, client2_g_val_iounoback = [], [], [], [], [], [], [], [], [], [], [], []
    client3_train_acc, client3_train_loss, client3_train_withbackiou, client3_train_nobackiou, client3_val_acc, client3_val_loss, client3_val_withbackiou, client3_val_nobackiou, client3_g_val_acc, client3_g_val_loss, client3_g_val_iouwithback, client3_g_val_iounoback = [], [], [], [], [], [], [], [], [], [], [], []
    client4_train_acc, client4_train_loss, client4_train_withbackiou, client4_train_nobackiou, client4_val_acc, client4_val_loss, client4_val_withbackiou, client4_val_nobackiou, client4_g_val_acc, client4_g_val_loss, client4_g_val_iouwithback, client4_g_val_iounoback = [], [], [], [], [], [], [], [], [], [], [], []
    client5_train_acc, client5_train_loss, client5_train_withbackiou, client5_train_nobackiou, client5_val_acc, client5_val_loss, client5_val_withbackiou, client5_val_nobackiou, client5_g_val_acc, client5_g_val_loss, client5_g_val_iouwithback, client5_g_val_iounoback = [], [], [], [], [], [], [], [], [], [], [], []
    client6_train_acc, client6_train_loss, client6_train_withbackiou, client6_train_nobackiou, client6_val_acc, client6_val_loss, client6_val_withbackiou, client6_val_nobackiou, client6_g_val_acc, client6_g_val_loss, client6_g_val_iouwithback, client6_g_val_iounoback = [], [], [], [], [], [], [], [], [], [], [], []
    client7_train_acc, client7_train_loss, client7_train_withbackiou, client7_train_nobackiou, client7_val_acc, client7_val_loss, client7_val_withbackiou, client7_val_nobackiou, client7_g_val_acc, client7_g_val_loss, client7_g_val_iouwithback, client7_g_val_iounoback = [], [], [], [], [], [], [], [], [], [], [], []

    test_Acc, test_Iou_withback, test_Iou_noback, test_Loss = [], [], [], []
    global_weights1S, global_weights2S, global_weights3S = [], [], []
    least_lossS, least_lossg = 100000000, 100000000;
    least_lossC1S, least_lossC2S, least_lossC3S, least_lossC4S, least_lossC5S, least_lossC6S, least_lossC7S = 100000000, 100000000, 100000000, 100000000, 100000000, 100000000, 100000000;
    R_cls = []
    NR_cls = []
    for idx in range(7):
        localmodel1 = copy.deepcopy(global_model1_fed)
        localmodel2 = copy.deepcopy(global_model2_fed)
        localmodel3 = copy.deepcopy(global_model3_fed)

        loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
        screen_opt1 = optim.Adam(localmodel1.parameters(), lr=args.fixed_lr)
        screen_opt2 = optim.Adam(localmodel2.parameters(), lr=args.fixed_lr)
        screen_opt3 = optim.Adam(localmodel3.parameters(), lr=args.fixed_lr)

        # Backward hook for modelclientBE
        grads3 = 0
        def grad_hook1(model, grad_input, grad_output):
            global grads3
            grads3 = grad_input[0].clone().detach()
        localmodel3.decoder1.register_full_backward_hook(grad_hook1)

        # Backward hook for modelserver
        grads2 = 0
        def grad_hook2(model, grad_input, grad_output):
            global grads2
            grads2 = grad_input[0].clone().detach()
        localmodel2.encoder1.register_full_backward_hook(grad_hook2)

        cl_idx = idx + 1
        print("Selected client for screening com round:", cl_idx)
        if cl_idx == 1:
            train_loader = train_loader_C1
            val_loader = val_loader_C1
            folder = parentF + "./Outmain/screenout/Fed_Avg/Saved/local_models/client1"
        elif cl_idx == 2:
            train_loader = train_loader_C2
            val_loader = val_loader_C2
            folder = parentF + "./Outmain/screenout/Fed_Avg/Saved/local_models/client2"
        elif cl_idx == 3:
            train_loader = train_loader_C3
            val_loader = val_loader_C3
            folder = parentF + "./Outmain/screenout/Fed_Avg/Saved/local_models/client3"
        elif cl_idx == 4:
            train_loader = train_loader_C4
            val_loader = val_loader_C4
            folder = parentF + "./Outmain/screenout/Fed_Avg/Saved/local_models/client4"
        elif cl_idx == 5:
            train_loader = train_loader_C5
            val_loader = val_loader_C5
            folder = parentF + "./Outmain/screenout/Fed_Avg/Saved/local_models/client5"
        elif cl_idx == 6:
            train_loader = train_loader_C6
            val_loader = val_loader_C6
            folder = parentF + "./Outmain/screenout/Fed_Avg/Saved/local_models/client6"
        elif cl_idx == 7:
            train_loader = train_loader_C7
            val_loader = val_loader_C7
            folder = parentF + "./Outmain/screenout/Fed_Avg/Saved/local_models/client7"

            # local epoch
        for epoch in range(args.screen_ep):
            print(f"[INFO]: Epoch {epoch + 1} of {args.screen_ep}")
            print("Client", cl_idx, " training.........")
            if cl_idx == 1:  # C1---------------------------------------------------------------C1 screen training & validation--------------------------------------------------------------------------------------------------------------------
                start_times1 = time.time()
                S_train_epoch_loss, S_train_epoch_acc, S_trainepoch_iou_withbackground, S_trainepoch_iou_nobackground, S_trainepoch_iou_class0, S_trainepoch_iou_class1, S_trainepoch_iou_class2, S_trainepoch_iou_class3, S_trainepoch_iou_class4, isreliable1, quantified_PL = train_screen(
                    train_loader, localmodel1, localmodel2, localmodel3, screen_opt1, screen_opt2, screen_opt3, loss_fn,
                    SC1_PL)
                end_times1 = time.time()
                s1t = end_times1 - start_times1
                s1time = S1time + s1t
                print("Client", cl_idx, "local validating.........")
                S_val_epoch_loss, S_val_epoch_acc, S_valepoch_iou_withbackground, S_valepoch_iou_nobackground, S_valepoch_iou_class0, S_valepoch_iou_class1, S_valepoch_iou_class2, S_valepoch_iou_class3, S_valepoch_iou_class4 = eval(
                    val_loader, localmodel1, localmodel2, localmodel3, loss_fn, folder, SC1_EPL)
                client1_train_acc.append(S_train_epoch_acc)
                client1_train_loss.append(S_train_epoch_loss)
                client1_train_withbackiou.append(S_trainepoch_iou_withbackground)
                client1_train_nobackiou.append(S_trainepoch_iou_nobackground)
                client1_val_acc.append(S_val_epoch_acc)
                client1_val_loss.append(S_val_epoch_loss)
                client1_val_withbackiou.append(S_valepoch_iou_withbackground)
                client1_val_nobackiou.append(S_valepoch_iou_nobackground)
                if (isreliable1 == True):
                    R_cls.append(1)
                    pl_C1 = quantified_PL
                else:
                    NR_cls.append(1)
                    pl_C1 = quantified_PL
                if least_lossC1S > S_val_epoch_loss:
                    least_lossC1S = S_val_epoch_loss
                    torch.save(localmodel1.state_dict(),
                               parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C1M1_localcheckpointS.pth')
                    torch.save(localmodel2.state_dict(),
                               parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C1M2_localcheckpointS.pth')
                    torch.save(localmodel3.state_dict(),
                               parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C1M3_localcheckpointS.pth')
                    print('C1localmodel saved')
            if cl_idx == 2:  # C2--------------------------------------------------------------C2 screen training & validation--------------------------------------------------------------------------------------------------------------------
                start_times2 = time.time()
                S_train_epoch_loss, S_train_epoch_acc, S_trainepoch_iou_withbackground, S_trainepoch_iou_nobackground, S_trainepoch_iou_class0, S_trainepoch_iou_class1, S_trainepoch_iou_class2, S_trainepoch_iou_class3, S_trainepoch_iou_class4, isreliable1, quantified_PL = train_screen(
                    train_loader, localmodel1, localmodel2, localmodel3, screen_opt1, screen_opt2, screen_opt3, loss_fn,
                    SC2_PL)
                end_times2 = time.time()
                s2t = end_times2 - start_times2
                s2time = S2time + s2t
                print("Client", cl_idx, "local validating.........")
                S_val_epoch_loss, S_val_epoch_acc, S_valepoch_iou_withbackground, S_valepoch_iou_nobackground, S_valepoch_iou_class0, S_valepoch_iou_class1, S_valepoch_iou_class2, S_valepoch_iou_class3, S_valepoch_iou_class4 = eval(
                    val_loader, localmodel1, localmodel2, localmodel3, loss_fn, folder, SC2_EPL)
                client2_train_acc.append(S_train_epoch_acc)
                client2_train_loss.append(S_train_epoch_loss)
                client2_train_withbackiou.append(S_trainepoch_iou_withbackground)
                client2_train_nobackiou.append(S_trainepoch_iou_nobackground)
                client2_val_acc.append(S_val_epoch_acc)
                client2_val_loss.append(S_val_epoch_loss)
                client2_val_withbackiou.append(S_valepoch_iou_withbackground)
                client2_val_nobackiou.append(S_valepoch_iou_nobackground)
                if (isreliable1 == True):
                    R_cls.append(2)
                    pl_C2 = quantified_PL
                else:
                    NR_cls.append(2)
                    pl_C2 = quantified_PL
                if least_lossC2S > S_val_epoch_loss:
                    least_lossC2S = S_val_epoch_loss
                    torch.save(localmodel1.state_dict(),
                               parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C2M1_localcheckpointS.pth')
                    torch.save(localmodel2.state_dict(),
                               parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C2M2_localcheckpointS.pth')
                    torch.save(localmodel3.state_dict(),
                               parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C2M3_localcheckpointS.pth')
                    print('C2localmodel saved')
            if cl_idx == 3:  # C3--------------------------------------------------------------C3 screen training & validation-----------------------------------------------------------------------------------------------------------
                start_times3 = time.time()
                S_train_epoch_loss, S_train_epoch_acc, S_trainepoch_iou_withbackground, S_trainepoch_iou_nobackground, S_trainepoch_iou_class0, S_trainepoch_iou_class1, S_trainepoch_iou_class2, S_trainepoch_iou_class3, S_trainepoch_iou_class4, isreliable1, quantified_PL = train_screen(
                    train_loader, localmodel1, localmodel2, localmodel3, screen_opt1, screen_opt2, screen_opt3, loss_fn,
                    SC3_PL)
                end_times3 = time.time()
                s3t = end_times3 - start_times3
                s3time = S3time + s3t
                print("Client", cl_idx, "local validating.........")
                S_val_epoch_loss, S_val_epoch_acc, S_valepoch_iou_withbackground, S_valepoch_iou_nobackground, S_valepoch_iou_class0, S_valepoch_iou_class1, S_valepoch_iou_class2, S_valepoch_iou_class3, S_valepoch_iou_class4 = eval(
                    val_loader, localmodel1, localmodel2, localmodel3, loss_fn, folder, SC3_EPL)
                client3_train_acc.append(S_train_epoch_acc)
                client3_train_loss.append(S_train_epoch_loss)
                client3_train_withbackiou.append(S_trainepoch_iou_withbackground)
                client3_train_nobackiou.append(S_trainepoch_iou_nobackground)
                client3_val_acc.append(S_val_epoch_acc)
                client3_val_loss.append(S_val_epoch_loss)
                client3_val_withbackiou.append(S_valepoch_iou_withbackground)
                client3_val_nobackiou.append(S_valepoch_iou_nobackground)
                if (isreliable1 == True):
                    R_cls.append(3)
                    pl_C3 = quantified_PL
                else:
                    NR_cls.append(3)
                    pl_C3 = quantified_PL
                if least_lossC3S > S_val_epoch_loss:
                    least_lossC3S = S_val_epoch_loss
                    torch.save(localmodel1.state_dict(),
                               parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C3M1_localcheckpointS.pth')
                    torch.save(localmodel2.state_dict(),
                               parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C3M2_localcheckpointS.pth')
                    torch.save(localmodel3.state_dict(),
                               parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C3M3_localcheckpointS.pth')
                    print('C3localmodel saved')
            if cl_idx == 4:  # C4--------------------------------------------------------------C4 screen training & validation-----------------------------------------------------------------------------------------------------------
                start_times4 = time.time()
                S_train_epoch_loss, S_train_epoch_acc, S_trainepoch_iou_withbackground, S_trainepoch_iou_nobackground, S_trainepoch_iou_class0, S_trainepoch_iou_class1, S_trainepoch_iou_class2, S_trainepoch_iou_class3, S_trainepoch_iou_class4, isreliable1, quantified_PL = train_screen(
                    train_loader, localmodel1, localmodel2, localmodel3, screen_opt1, screen_opt2, screen_opt3, loss_fn,
                    SC4_PL)
                end_times4 = time.time()
                s4t = end_times4 - start_times4
                s4time = S4time + s4t
                print("Client", cl_idx, "local validating.........")
                S_val_epoch_loss, S_val_epoch_acc, S_valepoch_iou_withbackground, S_valepoch_iou_nobackground, S_valepoch_iou_class0, S_valepoch_iou_class1, S_valepoch_iou_class2, S_valepoch_iou_class3, S_valepoch_iou_class4 = eval(
                    val_loader, localmodel1, localmodel2, localmodel3, loss_fn, folder, SC4_EPL)
                client4_train_acc.append(S_train_epoch_acc)
                client4_train_loss.append(S_train_epoch_loss)
                client4_train_withbackiou.append(S_trainepoch_iou_withbackground)
                client4_train_nobackiou.append(S_trainepoch_iou_nobackground)
                client4_val_acc.append(S_val_epoch_acc)
                client4_val_loss.append(S_val_epoch_loss)
                client4_val_withbackiou.append(S_valepoch_iou_withbackground)
                client4_val_nobackiou.append(S_valepoch_iou_nobackground)
                if (isreliable1 == True):
                    R_cls.append(4)
                    pl_C4 = quantified_PL
                else:
                    NR_cls.append(4)
                    pl_C4 = quantified_PL
                if least_lossC4S > S_val_epoch_loss:
                    least_lossC4S = S_val_epoch_loss
                    torch.save(localmodel1.state_dict(),
                               parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C4M1_localcheckpointS.pth')
                    torch.save(localmodel2.state_dict(),
                               parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C4M2_localcheckpointS.pth')
                    torch.save(localmodel3.state_dict(),
                               parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C4M3_localcheckpointS.pth')
                    print('C4localmodel saved')
            if cl_idx == 5:  # C5--------------------------------------------------------------C5 screen training & validation-----------------------------------------------------------------------------------------------------------
                start_times5 = time.time()
                S_train_epoch_loss, S_train_epoch_acc, S_trainepoch_iou_withbackground, S_trainepoch_iou_nobackground, S_trainepoch_iou_class0, S_trainepoch_iou_class1, S_trainepoch_iou_class2, S_trainepoch_iou_class3, S_trainepoch_iou_class4, isreliable1, quantified_PL = train_screen(
                    train_loader, localmodel1, localmodel2, localmodel3, screen_opt1, screen_opt2, screen_opt3, loss_fn,
                    SC5_PL)
                end_times5 = time.time()
                s5t = end_times5 - start_times5
                s5time = S5time + s5t
                print("Client", cl_idx, "local validating.........")
                S_val_epoch_loss, S_val_epoch_acc, S_valepoch_iou_withbackground, S_valepoch_iou_nobackground, S_valepoch_iou_class0, S_valepoch_iou_class1, S_valepoch_iou_class2, S_valepoch_iou_class3, S_valepoch_iou_class4 = eval(
                    val_loader, localmodel1, localmodel2, localmodel3, loss_fn, folder, SC5_EPL)
                client5_train_acc.append(S_train_epoch_acc)
                client5_train_loss.append(S_train_epoch_loss)
                client5_train_withbackiou.append(S_trainepoch_iou_withbackground)
                client5_train_nobackiou.append(S_trainepoch_iou_nobackground)
                client5_val_acc.append(S_val_epoch_acc)
                client5_val_loss.append(S_val_epoch_loss)
                client5_val_withbackiou.append(S_valepoch_iou_withbackground)
                client5_val_nobackiou.append(S_valepoch_iou_nobackground)
                if (isreliable1 == True):
                    R_cls.append(5)
                    pl_C5 = quantified_PL
                else:
                    NR_cls.append(5)
                    pl_C5 = quantified_PL
                if least_lossC5S > S_val_epoch_loss:
                    least_lossC5S = S_val_epoch_loss
                    torch.save(localmodel1.state_dict(),
                               parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C5M1_localcheckpointS.pth')
                    torch.save(localmodel2.state_dict(),
                               parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C5M2_localcheckpointS.pth')
                    torch.save(localmodel3.state_dict(),
                               parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C5M3_localcheckpointS.pth')
                    print('C5localmodel saved')

            if cl_idx == 6:  # C6--------------------------------------------------------------C6 screen training & validation-----------------------------------------------------------------------------------------------------------
                start_times6 = time.time()
                S_train_epoch_loss, S_train_epoch_acc, S_trainepoch_iou_withbackground, S_trainepoch_iou_nobackground, S_trainepoch_iou_class0, S_trainepoch_iou_class1, S_trainepoch_iou_class2, S_trainepoch_iou_class3, S_trainepoch_iou_class4, isreliable1, quantified_PL = train_screen(
                    train_loader, localmodel1, localmodel2, localmodel3, screen_opt1, screen_opt2, screen_opt3, loss_fn,
                    SC6_PL)
                end_times6 = time.time()
                s6t = end_times6 - start_times6
                s6time = S6time + s6t
                print("Client", cl_idx, "local validating.........")
                S_val_epoch_loss, S_val_epoch_acc, S_valepoch_iou_withbackground, S_valepoch_iou_nobackground, S_valepoch_iou_class0, S_valepoch_iou_class1, S_valepoch_iou_class2, S_valepoch_iou_class3, S_valepoch_iou_class4 = eval(
                    val_loader, localmodel1, localmodel2, localmodel3, loss_fn, folder, SC6_EPL)
                client6_train_acc.append(S_train_epoch_acc)
                client6_train_loss.append(S_train_epoch_loss)
                client6_train_withbackiou.append(S_trainepoch_iou_withbackground)
                client6_train_nobackiou.append(S_trainepoch_iou_nobackground)
                client6_val_acc.append(S_val_epoch_acc)
                client6_val_loss.append(S_val_epoch_loss)
                client6_val_withbackiou.append(S_valepoch_iou_withbackground)
                client6_val_nobackiou.append(S_valepoch_iou_nobackground)
                if (isreliable1 == True):
                    R_cls.append(6)
                    pl_C6 = quantified_PL
                else:
                    NR_cls.append(6)
                    pl_C6 = quantified_PL
                if least_lossC6S > S_val_epoch_loss:
                    least_lossC6S = S_val_epoch_loss
                    torch.save(localmodel1.state_dict(),
                               parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C6M1_localcheckpointS.pth')
                    torch.save(localmodel2.state_dict(),
                               parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C6M2_localcheckpointS.pth')
                    torch.save(localmodel3.state_dict(),
                               parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C6M3_localcheckpointS.pth')
                    print('C6localmodel saved')

            if cl_idx == 7:  # C7--------------------------------------------------------------C7 screen training & validation-----------------------------------------------------------------------------------------------------------
                start_times7 = time.time()
                S_train_epoch_loss, S_train_epoch_acc, S_trainepoch_iou_withbackground, S_trainepoch_iou_nobackground, S_trainepoch_iou_class0, S_trainepoch_iou_class1, S_trainepoch_iou_class2, S_trainepoch_iou_class3, S_trainepoch_iou_class4, isreliable1, quantified_PL = train_screen(
                    train_loader, localmodel1, localmodel2, localmodel3, screen_opt1, screen_opt2, screen_opt3, loss_fn,
                    SC7_PL)
                end_times7 = time.time()
                s7t = end_times7 - start_times7
                s7time = S7time + s7t
                print("Client", cl_idx, "local validating.........")
                S_val_epoch_loss, S_val_epoch_acc, S_valepoch_iou_withbackground, S_valepoch_iou_nobackground, S_valepoch_iou_class0, S_valepoch_iou_class1, S_valepoch_iou_class2, S_valepoch_iou_class3, S_valepoch_iou_class4 = eval(
                    val_loader, localmodel1, localmodel2, localmodel3, loss_fn, folder, SC7_EPL)
                client7_train_acc.append(S_train_epoch_acc)
                client7_train_loss.append(S_train_epoch_loss)
                client7_train_withbackiou.append(S_trainepoch_iou_withbackground)
                client7_train_nobackiou.append(S_trainepoch_iou_nobackground)
                client7_val_acc.append(S_val_epoch_acc)
                client7_val_loss.append(S_val_epoch_loss)
                client7_val_withbackiou.append(S_valepoch_iou_withbackground)
                client7_val_nobackiou.append(S_valepoch_iou_nobackground)
                if (isreliable1 == True):
                    R_cls.append(7)
                    pl_C7 = quantified_PL
                else:
                    NR_cls.append(7)
                    pl_C7 = quantified_PL
                if least_lossC7S > S_val_epoch_loss:
                    least_lossC7S = S_val_epoch_loss
                    torch.save(localmodel1.state_dict(),
                               parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C7M1_localcheckpointS.pth')
                    torch.save(localmodel2.state_dict(),
                               parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C7M2_localcheckpointS.pth')
                    torch.save(localmodel3.state_dict(),
                               parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C7M3_localcheckpointS.pth')
                    print('C7localmodel saved')

            print(
                f"S_Training dice loss: {S_train_epoch_loss:.3f}, S_Training accuracy: {S_train_epoch_acc:.3f},S_Training iou Score with background: {S_trainepoch_iou_withbackground:.3f},Training iou Score without background: {S_trainepoch_iou_nobackground:.3f}")
            print("\n S_Training IoUs Client:", cl_idx)
            print("T: S_Background:", S_trainepoch_iou_class0)
            print("T: S_ZP:", S_trainepoch_iou_class1)
            print("T: S_TE:", S_trainepoch_iou_class2)
            print("T: S_ICM:", S_trainepoch_iou_class3)
            print("T: S_Blastocoel:", S_trainepoch_iou_class4)

            print(
                f"S_Validating dice loss: {S_val_epoch_loss:.3f}, Validating accuracy: {S_val_epoch_acc:.3f},Validating iou Score with background: {S_valepoch_iou_withbackground:.3f},Validating iou Score without background: {S_valepoch_iou_nobackground:.3f}")
            print("\n S_Validating IoUs Client:", cl_idx)
            print("V: S_Background:", S_valepoch_iou_class0)
            print("V: S_ZP:", S_valepoch_iou_class1)
            print("V: S_TE:", S_valepoch_iou_class2)
            print("V: S_ICM:", S_valepoch_iou_class3)
            print("V: S_Blastocoel:", S_valepoch_iou_class4)

    # Identify if the clients are experiencing packet loss.
    # Divide the clients in to 2 groups as reliable ((R)/strong) clients and unreliable ((NR)/weak) clients based on the packet loss.
    # Determine the local epochs, LR
    # global aggregation - Model updates of all the clients would be aggregated based on client’s global validation loss, data ratio and the packet loss probability

    C1M1 = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C1M1_localcheckpointS.pth')
    C1M2 = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C1M2_localcheckpointS.pth')
    C1M3 = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C1M3_localcheckpointS.pth')
    C2M1 = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C2M1_localcheckpointS.pth')
    C2M2 = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C2M2_localcheckpointS.pth')
    C2M3 = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C2M3_localcheckpointS.pth')
    C3M1 = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C3M1_localcheckpointS.pth')
    C3M2 = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C3M2_localcheckpointS.pth')
    C3M3 = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C3M3_localcheckpointS.pth')
    C4M1 = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C4M1_localcheckpointS.pth')
    C4M2 = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C4M2_localcheckpointS.pth')
    C4M3 = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C4M3_localcheckpointS.pth')
    C5M1 = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C5M1_localcheckpointS.pth')
    C5M2 = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C5M2_localcheckpointS.pth')
    C5M3 = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C5M3_localcheckpointS.pth')
    C6M1 = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C6M1_localcheckpointS.pth')
    C6M2 = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C6M2_localcheckpointS.pth')
    C6M3 = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C6M3_localcheckpointS.pth')
    C7M1 = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C7M1_localcheckpointS.pth')
    C7M2 = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C7M2_localcheckpointS.pth')
    C7M3 = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C7M3_localcheckpointS.pth')

    # Global aggregation after screening round
    print("Global aggregation starts")

    # updated parameters
    C1M1.update((x, y * D1) for x, y in C1M1.items())
    C1M2.update((x, y * D1) for x, y in C1M2.items())
    C1M3.update((x, y * D1) for x, y in C1M3.items())
    C2M1.update((x, y * D2) for x, y in C2M1.items())
    C2M2.update((x, y * D2) for x, y in C2M2.items())
    C2M3.update((x, y * D2) for x, y in C2M3.items())
    C3M1.update((x, y * D3) for x, y in C3M1.items())
    C3M2.update((x, y * D3) for x, y in C3M2.items())
    C3M3.update((x, y * D3) for x, y in C3M3.items())
    C4M1.update((x, y * D4) for x, y in C4M1.items())
    C4M2.update((x, y * D4) for x, y in C4M2.items())
    C4M3.update((x, y * D4) for x, y in C4M3.items())
    C5M1.update((x, y * D5) for x, y in C5M1.items())
    C5M2.update((x, y * D5) for x, y in C5M2.items())
    C5M3.update((x, y * D5) for x, y in C5M3.items())
    C6M1.update((x, y * D6) for x, y in C6M1.items())
    C6M2.update((x, y * D6) for x, y in C6M2.items())
    C6M3.update((x, y * D6) for x, y in C6M3.items())
    C7M1.update((x, y * D7) for x, y in C7M1.items())
    C7M2.update((x, y * D7) for x, y in C7M2.items())
    C7M3.update((x, y * D7) for x, y in C7M3.items())

    G1dict = [C1M1, C2M1, C3M1, C4M1, C5M1, C6M1, C7M1]
    G2dict = [C1M2, C2M2, C3M2, C4M2, C5M2, C6M2, C7M2]
    G3dict = [C1M3, C2M3, C3M3, C4M3, C5M3, C6M3, C7M3]
    global_weights1S.extend(G1dict)
    global_weights2S.extend(G2dict)
    global_weights3S.extend(G3dict)

    # averaging parameters
    global_fed_weights1 = fedAvg(global_weights1S)
    global_fed_weights2 = fedAvg(global_weights2S)
    global_fed_weights3 = fedAvg(global_weights3S)

    # load the new parameters - FedAvg
    global_model1_fed.load_state_dict(global_fed_weights1)
    global_model2_fed.load_state_dict(global_fed_weights2)
    global_model3_fed.load_state_dict(global_fed_weights3)

    print("New Global model formed")

    # Distributing the global model to R and NR Clients
    global_model1_fedR = copy.deepcopy(global_model1_fed)
    global_model2_fedR = copy.deepcopy(global_model2_fed)
    global_model3_fedR = copy.deepcopy(global_model3_fed)

    global_model1_fedNR = copy.deepcopy(global_model1_fed)
    global_model2_fedNR = copy.deepcopy(global_model2_fed)
    global_model3_fedNR = copy.deepcopy(global_model3_fed)

    print("R clients:", R_cls)
    print("NR clients:", NR_cls)
    print("C1 - PL", pl_C1)
    print("C2 - PL", pl_C2)
    print("C3 - PL", pl_C3)
    print("C4 - PL", pl_C4)
    print("C5 - PL", pl_C5)
    print("C4 - PL", pl_C6)
    print("C5 - PL", pl_C7)

    print("C1 screening time:", s1time)
    print("C2 screening time:", s2time)
    print("C3 creening time:", s3time)
    print("C4 screening time:", s4time)
    print("C5 screening time:", s5time)
    print("C6 screening time:", s6time)
    print("C7 screening time:", s7time)

    s1pertime = s1time / len(train_loader_C1)
    s2pertime = s2time / len(train_loader_C2)
    s3pertime = s3time / len(train_loader_C3)
    s4pertime = s4time / len(train_loader_C4)
    s5pertime = s5time / len(train_loader_C5)
    s6pertime = s6time / len(train_loader_C6)
    s7pertime = s7time / len(train_loader_C7)
    print("C1 perscreening time:", s1pertime)
    print("C2 perscreening time:", s2pertime)
    print("C3 perscreening time:", s3pertime)
    print("C4 perscreening time:", s4pertime)
    print("C5 perscreening time:", s5pertime)
    print("C6 perscreening time:", s6pertime)
    print("C7 perscreening time:", s7pertime)

    spertime = []
    spertime.extend([s1pertime, s2pertime, s3pertime, s4pertime, s5pertime, s6pertime, s7pertime])
    Tmax = max(spertime)

    # E=12
    # n= 0.0001
    N = 7
    # success probaility ratios

    S_C1 = (1 - pl_C1)
    S_C2 = (1 - pl_C2)
    S_C3 = (1 - pl_C3)
    S_C4 = (1 - pl_C4)
    S_C5 = (1 - pl_C5)
    S_C6 = (1 - pl_C6)
    S_C7 = (1 - pl_C7)

    # define the local updates per each client
    # Values for the 5 equations
    T1 = Tmax / s1pertime
    T2 = Tmax / s2pertime
    T3 = Tmax / s3pertime
    T4 = Tmax / s4pertime
    T5 = Tmax / s5pertime
    T6 = Tmax / s6pertime
    T7 = Tmax / s7pertime

    print("C1 T:", T1)
    print("C2 T:", T2)
    print("C3 T:", T3)
    print("C4 T:", T4)
    print("C5 T:", T5)
    print("C6 T:", T6)
    print("C7 T:", T7)
    arrT, arrT_scaled = [], []
    arrT.extend([T1, T2, T3, T4, T5, T6, T7])
    arrT_scaled = min_max_scaling(arrT)
    print("arrT_scaled:", arrT_scaled)

    D_1 = Dmax / D1
    D_2 = Dmax / D2
    D_3 = Dmax / D3
    D_4 = Dmax / D4
    D_5 = Dmax / D5
    D_6 = Dmax / D6
    D_7 = Dmax / D7

    print("C1 D:", D_1)
    print("C2 D:", D_2)
    print("C3 D:", D_3)
    print("C4 D:", D_4)
    print("C5 D:", D_5)
    print("C6 D:", D_6)
    print("C7 D:", D_7)
    arrD, arrD_scaled = [], []
    arrD.extend([D_1, D_2, D_3, D_4, D_5, D_6, D_7])
    arrD_scaled = min_max_scaling(arrD)
    print("arrD_scaled:", arrD_scaled)

    E = 1 / (N * N)
    print("E:",E)
    e11 = arrT_scaled[0] * E
    e12 = arrD_scaled[0] * E
    e13 = pl_C1 * E
    e21 = arrT_scaled[1] * E
    e22 = arrD_scaled[1] * E
    e23 = pl_C2 * E
    e31 = arrT_scaled[2] * E
    e32 = arrD_scaled[2] * E
    e33 = pl_C3 * E
    e41 = arrT_scaled[3] * E
    e42 = arrD_scaled[3] * E
    e43 = pl_C4 * E
    e51 = arrT_scaled[4] * E
    e52 = arrD_scaled[4] * E
    e53 = pl_C5 * E
    e61 = arrT_scaled[5] * E
    e62 = arrD_scaled[5] * E
    e63 = pl_C6 * E
    e71 = arrT_scaled[6] * E
    e72 = arrD_scaled[6] * E
    e73 = pl_C7 * E

    print("coefficients:", e11, e12, e13, e21, e22, e23, e31, e32, e33, e41, e42, e43, e51, e52, e53, e61, e62, e63,
          e71, e72, e73)
    # Using the least squares method to decide the epochs for each client
    E1, E2, E3, E4, E5, E6, E7 = cal_epoch(e11, e12, e13, e21, e22, e23, e31, e32, e33, e41, e42, e43, e51, e52, e53,
                                           e61, e62, e63, e71, e72, e73)

    local_ep1 = E1
    local_ep2 = E2
    local_ep3 = E3
    local_ep4 = E4
    local_ep5 = E5
    local_ep6 = E6
    local_ep7 = E7

    # define the learning rates per each client
    n11 = arrD_scaled[0] / arrT_scaled[0]
    n12 = S_C1
    n21 = arrD_scaled[1] / arrT_scaled[1]
    n22 = S_C2
    n31 = arrD_scaled[2] / arrT_scaled[2]
    n32 = S_C3
    n41 = arrD_scaled[3] / arrT_scaled[3]
    n42 = S_C4
    n51 = arrD_scaled[4] / arrT_scaled[4]
    n52 = S_C5
    n61 = arrD_scaled[5] / arrT_scaled[5]
    n62 = S_C6
    n71 = arrD_scaled[6] / arrT_scaled[6]
    n72 = S_C7

    print("coefficients:", n11, n12, n21, n22, n31, n32, n41, n42, n51, n52, n61, n62, n71, n72)
    n1, n2, n3, n4, n5, n6, n7 = cal_lr(n11, n12, n21, n22, n31, n32, n41, n42, n51, n52, n61, n62, n71, n72)

    n = N * N
    lr_C1 =  n1
    lr_C2 =  n2
    lr_C3 =  n3
    lr_C4 = n4
    lr_C5 = n5
    lr_C6 = n6
    lr_C7 =  n7

    print("lr_C1:", lr_C1)
    print("lr_C2:", lr_C2)
    print("lr_C3:", lr_C3)
    print("lr_C4:", lr_C4)
    print("lr_C5:", lr_C5)
    print("lr_C6:", lr_C6)
    print("lr_C7:", lr_C7)

    # values needed for weighted aggregation
    lenR, lenNR = [], []
    if 1 in R_cls:
        lenR.append(len(train_loader_C1))
    else:
        lenNR.append(len(train_loader_C1))
    if 2 in R_cls:
        lenR.append(len(train_loader_C2))
    else:
        lenNR.append(len(train_loader_C2))
    if 3 in R_cls:
        lenR.append(len(train_loader_C3))
    else:
        lenNR.append(len(train_loader_C3))
    if 4 in R_cls:
        lenR.append(len(train_loader_C4))
    else:
        lenNR.append(len(train_loader_C4))
    if 5 in R_cls:
        lenR.append(len(train_loader_C5))
    else:
        lenNR.append(len(train_loader_C5))
    if 6 in R_cls:
        lenR.append(len(train_loader_C6))
    else:
        lenNR.append(len(train_loader_C6))
    if 7 in R_cls:
        lenR.append(len(train_loader_C7))
    else:
        lenNR.append(len(train_loader_C7))

    D1R = len(train_loader_C1) / sum(lenR)
    D2R = len(train_loader_C2) / sum(lenR)
    D3R = len(train_loader_C3) / sum(lenR)
    D4R = len(train_loader_C4) / sum(lenR)
    D5R = len(train_loader_C5) / sum(lenR)
    D6R = len(train_loader_C6) / sum(lenR)
    D7R = len(train_loader_C7) / sum(lenR)


    print(f'\n | Screening Round ends....')

    # generating packet loss with retransmisison ------------------------------------------------------------------------------------------------------------------------------
    C1_PL, C2_PL, C3_PL, C4_PL, C5_PL, C6_PL, C7_PL, C1_EPL, C2_EPL, C3_EPL, C4_EPL, C5_EPL, C6_EPL, C7_EPL = [], [], [], [], [], [], [], [], [], [], [], [], [], []
    C1T, C2T, C3T, C4T, C5T, C6T, C7T = [], [], [], [], [], [], []
    max_retra = args.max_retra_shallow

    for c in range(1, 8):
        if (c == 1):  # C1------------------------------------------------
            c1ch_list1, c1ch_list2, c1ch_list3, c1ch_list4 = [], [], [], []
            for ch in range(32):
                rows1c1 = 256
                first = list(range(0, 256))
                for tr_rounds1c1 in range(max_retra):
                    random_tensor1c1 = torch.ones(256, 256)
                    num_zero_rows1c1 = math.floor(pl_C1 * rows1c1)
                    x1 = set(random.sample(list(first), num_zero_rows1c1))
                    rows1c1 = len(x1)
                    if (rows1c1 == 0):
                        output_tensor1c1 = random_tensor1c1
                        break
                    else:
                        first = x1
                        for i in range(256):
                            if i in x1:
                                random_tensor1c1[i] = 0 * random_tensor1c1[i]
                        output_tensor1c1 = random_tensor1c1
                c1ch_list1.append(output_tensor1c1)
            times1c1 = tr_rounds1c1 + 1
            pack_loss_uplink1C1 = torch.stack(c1ch_list1, dim=0).to(device)

            for ch in range(32):
                rows2c1 = 256
                first = list(range(0, 256))
                for tr_rounds2c1 in range(max_retra):
                    random_tensor2c1 = torch.ones(256, 256)
                    num_zero_rows2c1 = math.floor(pl_C1 * rows2c1)
                    x2 = set(random.sample(list(first), num_zero_rows2c1))
                    rows2c1 = len(x2)
                    if (rows2c1 == 0):
                        output_tensor2c1 = random_tensor2c1
                        break
                    else:
                        first = x2
                        for i in range(256):
                            if i in x2:
                                random_tensor2c1[i] = 0 * random_tensor2c1[i]
                        output_tensor2c1 = random_tensor2c1
                c1ch_list2.append(output_tensor2c1)
            times2c1 = tr_rounds2c1 + 1
            pack_loss_downlink1C1 = torch.stack(c1ch_list2, dim=0).to(device)

            for ch in range(32):
                rows3c1 = 256
                first = list(range(0, 256))
                for tr_rounds3c1 in range(max_retra):
                    random_tensor3c1 = torch.ones(256, 256)
                    num_zero_rows3c1 = math.floor(pl_C1 * rows3c1)
                    x3 = set(random.sample(list(first), num_zero_rows3c1))
                    rows3c1 = len(x3)
                    if (rows3c1 == 0):
                        output_tensor3c1 = random_tensor3c1
                        break
                    else:
                        first = x3
                        for i in range(256):
                            if i in x3:
                                random_tensor3c1[i] = 0 * random_tensor3c1[i]
                        output_tensor3c1 = random_tensor3c1
                c1ch_list3.append(output_tensor3c1)
            times3c1 = tr_rounds3c1 + 1
            pack_loss_uplink2C1 = torch.stack(c1ch_list3, dim=0).to(device)

            for ch in range(32):
                rows4c1 = 256
                first = list(range(0, 256))
                for tr_rounds4c1 in range(max_retra):
                    random_tensor4c1 = torch.ones(256, 256)
                    num_zero_rows4c1 = math.floor(pl_C1 * rows4c1)
                    x4 = set(random.sample(list(first), num_zero_rows4c1))
                    rows4c1 = len(x4)
                    if (rows4c1 == 0):
                        output_tensor4c1 = random_tensor4c1
                        break
                    else:
                        first = x4
                        for i in range(256):
                            if i in x4:
                                random_tensor4c1[i] = 0 * random_tensor4c1[i]
                        output_tensor4c1 = random_tensor4c1
                c1ch_list4.append(output_tensor4c1)
            times4c1 = tr_rounds4c1 + 1
            pack_loss_downlink2C1 = torch.stack(c1ch_list4, dim=0).to(device)

            C1_PL.extend([pack_loss_uplink1C1, pack_loss_downlink1C1, pack_loss_uplink2C1, pack_loss_downlink2C1])
            C1_EPL.extend([pack_loss_uplink1C1, pack_loss_downlink1C1])
            C1T.extend([times1c1, times2c1, times3c1, times4c1])

        elif (c == 2):  # C2------------------------------------------------
            c2ch_list1, c2ch_list2, c2ch_list3, c2ch_list4 = [], [], [], []
            for ch in range(32):
                rows1c2 = 256
                first = list(range(0, 256))
                for tr_rounds1c2 in range(max_retra):
                    random_tensor1c2 = torch.ones(256, 256)
                    num_zero_rows1c2 = math.floor(pl_C2 * rows1c2)
                    x1 = set(random.sample(list(first), num_zero_rows1c2))
                    rows1c2 = len(x1)
                    if (rows1c2 == 0):
                        output_tensor1c2 = random_tensor1c2
                        break
                    else:
                        first = x1
                        for i in range(256):
                            if i in x1:
                                random_tensor1c2[i] = 0 * random_tensor1c2[i]
                        output_tensor1c2 = random_tensor1c2
                c2ch_list1.append(output_tensor1c2)
            times1c2 = tr_rounds1c2 + 1
            pack_loss_uplink1C2 = torch.stack(c2ch_list1, dim=0).to(device)

            for ch in range(32):
                rows2c2 = 256
                first = list(range(0, 256))
                for tr_rounds2c2 in range(max_retra):
                    random_tensor2c2 = torch.ones(256, 256)
                    num_zero_rows2c2 = math.floor(pl_C2 * rows2c2)
                    x2 = set(random.sample(list(first), num_zero_rows2c2))
                    rows2c2 = len(x2)
                    if (rows2c2 == 0):
                        output_tensor2c2 = random_tensor2c2
                        break
                    else:
                        first = x2
                        for i in range(256):
                            if i in x2:
                                random_tensor2c2[i] = 0 * random_tensor2c2[i]
                        output_tensor2c2 = random_tensor2c2
                c2ch_list2.append(output_tensor2c2)
            times2c2 = tr_rounds2c2 + 1
            pack_loss_downlink1C2 = torch.stack(c2ch_list2, dim=0).to(device)

            for ch in range(32):
                rows3c2 = 256
                first = list(range(0, 256))
                for tr_rounds3c2 in range(max_retra):
                    random_tensor3c2 = torch.ones(256, 256)
                    num_zero_rows3c2 = math.floor(pl_C2 * rows3c2)
                    x3 = set(random.sample(list(first), num_zero_rows3c2))
                    rows3c2 = len(x3)
                    if (rows3c2 == 0):
                        output_tensor3c2 = random_tensor3c2
                        break
                    else:
                        first = x3
                        for i in range(256):
                            if i in x3:
                                random_tensor3c2[i] = 0 * random_tensor3c2[i]
                        output_tensor3c2 = random_tensor3c2
                c2ch_list3.append(output_tensor3c2)
            times3c2 = tr_rounds3c2 + 1
            pack_loss_uplink2C2 = torch.stack(c2ch_list3, dim=0).to(device)

            for ch in range(32):
                rows4c2 = 256
                first = list(range(0, 256))
                for tr_rounds4c2 in range(max_retra):
                    random_tensor4c2 = torch.ones(256, 256)
                    num_zero_rows4c2 = math.floor(pl_C2 * rows4c2)
                    x4 = set(random.sample(list(first), num_zero_rows4c2))
                    rows4c2 = len(x4)
                    if (rows4c2 == 0):
                        output_tensor4c2 = random_tensor4c2
                        break
                    else:
                        first = x4
                        for i in range(256):
                            if i in x4:
                                random_tensor4c2[i] = 0 * random_tensor4c2[i]
                        output_tensor4c2 = random_tensor4c2
                c2ch_list4.append(output_tensor4c2)
            times4c2 = tr_rounds4c2 + 1
            pack_loss_downlink2C2 = torch.stack(c2ch_list4, dim=0).to(device)
            C2_PL.extend(
                [pack_loss_uplink1C2, pack_loss_downlink1C2, pack_loss_uplink2C2, pack_loss_downlink2C2])
            C2_EPL.extend([pack_loss_uplink1C2, pack_loss_downlink1C2])
            C2T.extend([times1c2, times2c2, times3c2, times4c2])

        elif (c == 3):  # C3------------------------------------------------
            c3ch_list1, c3ch_list2, c3ch_list3, c3ch_list4 = [], [], [], []
            for ch in range(32):
                rows1c3 = 256
                first = list(range(0, 256))
                for tr_rounds1c3 in range(max_retra):
                    random_tensor1c3 = torch.ones(256, 256)
                    num_zero_rows1c3 = math.floor(pl_C3 * rows1c3)
                    x1 = set(random.sample(list(first), num_zero_rows1c3))
                    rows1c3 = len(x1)
                    if (rows1c3 == 0):
                        output_tensor1c3 = random_tensor1c3
                        break
                    else:
                        first = x1
                        for i in range(256):
                            if i in x1:
                                random_tensor1c3[i] = 0 * random_tensor1c3[i]
                        output_tensor1c3 = random_tensor1c3
                c3ch_list1.append(output_tensor1c3)
            times1c3 = tr_rounds1c3 + 1
            pack_loss_uplink1C3 = torch.stack(c3ch_list1, dim=0).to(device)

            for ch in range(32):
                rows2c3 = 256
                first = list(range(0, 256))
                for tr_rounds2c3 in range(max_retra):
                    random_tensor2c3 = torch.ones(256, 256)
                    num_zero_rows2c3 = math.floor(pl_C3 * rows2c3)
                    x2 = set(random.sample(list(first), num_zero_rows2c3))
                    rows2c3 = len(x2)
                    if (rows2c3 == 0):
                        output_tensor2c3 = random_tensor2c3
                        break
                    else:
                        first = x2
                        for i in range(256):
                            if i in x2:
                                random_tensor2c3[i] = 0 * random_tensor2c3[i]
                        output_tensor2c3 = random_tensor2c3
                c3ch_list2.append(output_tensor2c3)
            times2c3 = tr_rounds2c3 + 1
            pack_loss_downlink1C3 = torch.stack(c3ch_list2, dim=0).to(device)

            for ch in range(32):
                rows3c3 = 256
                first = list(range(0, 256))
                for tr_rounds3c3 in range(max_retra):
                    random_tensor3c3 = torch.ones(256, 256)
                    num_zero_rows3c3 = math.floor(pl_C3 * rows3c3)
                    x3 = set(random.sample(list(first), num_zero_rows3c3))
                    rows3c3 = len(x3)
                    if (rows3c3 == 0):
                        output_tensor3c3 = random_tensor3c3
                        break
                    else:
                        first = x3
                        for i in range(256):
                            if i in x3:
                                random_tensor3c3[i] = 0 * random_tensor3c3[i]
                        output_tensor3c3 = random_tensor3c3
                c3ch_list3.append(output_tensor3c3)
            times3c3 = tr_rounds3c3 + 1
            pack_loss_uplink2C3 = torch.stack(c3ch_list3, dim=0).to(device)

            for ch in range(32):
                rows4c3 = 256
                first = list(range(0, 256))
                for tr_rounds4c3 in range(max_retra):
                    random_tensor4c3 = torch.ones(256, 256)
                    num_zero_rows4c3 = math.floor(pl_C3 * rows4c3)
                    x4 = set(random.sample(list(first), num_zero_rows4c3))
                    rows4c3 = len(x4)
                    if (rows4c3 == 0):
                        output_tensor4c3 = random_tensor4c3
                        break
                    else:
                        first = x4
                        for i in range(256):
                            if i in x4:
                                random_tensor4c3[i] = 0 * random_tensor4c3[i]
                        output_tensor4c3 = random_tensor4c3
                c3ch_list4.append(output_tensor4c3)
            times4c3 = tr_rounds4c3 + 1
            pack_loss_downlink2C3 = torch.stack(c3ch_list4, dim=0).to(device)

            C3_PL.extend([pack_loss_uplink1C3, pack_loss_downlink1C3, pack_loss_uplink2C3, pack_loss_downlink2C3])
            C3_EPL.extend([pack_loss_uplink1C3, pack_loss_downlink1C3])
            C3T.extend([times1c3, times2c3, times3c3, times4c3])

        elif (c == 4):  # C4------------------------------------------------
            c4ch_list1, c4ch_list2, c4ch_list3, c4ch_list4 = [], [], [], []
            for ch in range(32):
                rows1c4 = 256
                first = list(range(0, 256))
                for tr_rounds1c4 in range(max_retra):
                    random_tensor1c4 = torch.ones(256, 256)
                    num_zero_rows1c4 = math.floor(pl_C4 * rows1c4)
                    x1 = set(random.sample(list(first), num_zero_rows1c4))
                    rows1c4 = len(x1)
                    if (rows1c4 == 0):
                        output_tensor1c4 = random_tensor1c4
                        break
                    else:
                        first = x1
                        for i in range(256):
                            if i in x1:
                                random_tensor1c4[i] = 0 * random_tensor1c4[i]
                        output_tensor1c4 = random_tensor1c4
                c4ch_list1.append(output_tensor1c4)
            times1c4 = tr_rounds1c4 + 1
            pack_loss_uplink1C4 = torch.stack(c4ch_list1, dim=0).to(device)

            for ch in range(32):
                rows2c4 = 256
                first = list(range(0, 256))
                for tr_rounds2c4 in range(max_retra):
                    random_tensor2c4 = torch.ones(256, 256)
                    num_zero_rows2c4 = math.floor(pl_C4 * rows2c4)
                    x2 = set(random.sample(list(first), num_zero_rows2c4))
                    rows2c4 = len(x2)
                    if (rows2c4 == 0):
                        output_tensor2c4 = random_tensor2c4
                        break
                    else:
                        first = x2
                        for i in range(256):
                            if i in x2:
                                random_tensor2c4[i] = 0 * random_tensor2c4[i]
                        output_tensor2c4 = random_tensor2c4
                c4ch_list2.append(output_tensor2c4)
            times2c4 = tr_rounds2c4 + 1
            pack_loss_downlink1C4 = torch.stack(c4ch_list2, dim=0).to(device)

            for ch in range(32):
                rows3c4 = 256
                first = list(range(0, 256))
                for tr_rounds3c4 in range(max_retra):
                    random_tensor3c4 = torch.ones(256, 256)
                    num_zero_rows3c4 = math.floor(pl_C4 * rows3c4)
                    x3 = set(random.sample(list(first), num_zero_rows3c4))
                    rows3c4 = len(x3)
                    if (rows3c4 == 0):
                        output_tensor3c4 = random_tensor3c4
                        break
                    else:
                        first = x3
                        for i in range(256):
                            if i in x3:
                                random_tensor3c4[i] = 0 * random_tensor3c4[i]
                        output_tensor3c4 = random_tensor3c4
                c4ch_list3.append(output_tensor3c4)
            times3c4 = tr_rounds3c4 + 1
            pack_loss_uplink2C4 = torch.stack(c4ch_list3, dim=0).to(device)

            for ch in range(32):
                rows4c4 = 256
                first = list(range(0, 256))
                for tr_rounds4c4 in range(max_retra):
                    random_tensor4c4 = torch.ones(256, 256)
                    num_zero_rows4c4 = math.floor(pl_C4 * rows4c4)
                    x4 = set(random.sample(list(first), num_zero_rows4c4))
                    rows4c4 = len(x4)
                    if (rows4c4 == 0):
                        output_tensor4c4 = random_tensor4c4
                        break
                    else:
                        first = x4
                        for i in range(256):
                            if i in x4:
                                random_tensor4c4[i] = 0 * random_tensor4c4[i]
                        output_tensor4c4 = random_tensor4c4
                c4ch_list4.append(output_tensor4c4)
            times4c4 = tr_rounds4c4 + 1
            pack_loss_downlink2C4 = torch.stack(c4ch_list4, dim=0).to(device)

            C4_PL.extend(
                [pack_loss_uplink1C4, pack_loss_downlink1C4, pack_loss_uplink2C4, pack_loss_downlink2C4])
            C4_EPL.extend([pack_loss_uplink1C4, pack_loss_downlink1C4])
            C4T.extend([times1c4, times2c4, times3c4, times4c4])

        elif (c == 5):# C5------------------------------------------------
            c5ch_list1, c5ch_list2, c5ch_list3, c5ch_list4 = [], [], [], []
            for ch in range(32):
                rows1c5 = 256
                first = list(range(0, 256))
                for tr_rounds1c5 in range(max_retra):
                    random_tensor1c5 = torch.ones(256, 256)
                    num_zero_rows1c5 = math.floor(pl_C5 * rows1c5)
                    x1 = set(random.sample(list(first), num_zero_rows1c5))
                    rows1c5 = len(x1)
                    if (rows1c5 == 0):
                        output_tensor1c5 = random_tensor1c5
                        break
                    else:
                        first = x1
                        for i in range(256):
                            if i in x1:
                                random_tensor1c5[i] = 0 * random_tensor1c5[i]
                        output_tensor1c5 = random_tensor1c5
                c5ch_list1.append(output_tensor1c5)
            times1c5 = tr_rounds1c5 + 1
            pack_loss_uplink1C5 = torch.stack(c5ch_list1, dim=0).to(device)

            for ch in range(32):
                rows2c5 = 256
                first = list(range(0, 256))
                for tr_rounds2c5 in range(max_retra):
                    random_tensor2c5 = torch.ones(256, 256)
                    num_zero_rows2c5 = math.floor(pl_C5 * rows2c5)
                    x2 = set(random.sample(list(first), num_zero_rows2c5))
                    rows2c5 = len(x2)
                    if (rows2c5 == 0):
                        output_tensor2c5 = random_tensor2c5
                        break
                    else:
                        first = x2
                        for i in range(256):
                            if i in x2:
                                random_tensor2c5[i] = 0 * random_tensor2c5[i]
                        output_tensor2c5 = random_tensor2c5
                c5ch_list2.append(output_tensor2c5)
            times2c5 = tr_rounds2c5 + 1
            pack_loss_downlink1C5 = torch.stack(c5ch_list2, dim=0).to(device)

            for ch in range(32):
                rows3c5 = 256
                first = list(range(0, 256))
                for tr_rounds3c5 in range(max_retra):
                    random_tensor3c5 = torch.ones(256, 256)
                    num_zero_rows3c5 = math.floor(pl_C5 * rows3c5)
                    x3 = set(random.sample(list(first), num_zero_rows3c5))
                    rows3c5 = len(x3)
                    if (rows3c5 == 0):
                        output_tensor3c5 = random_tensor3c5
                        break
                    else:
                        first = x3
                        for i in range(256):
                            if i in x3:
                                random_tensor3c5[i] = 0 * random_tensor3c5[i]
                        output_tensor3c5 = random_tensor3c5
                c5ch_list3.append(output_tensor3c5)
            times3c5 = tr_rounds3c5 + 1
            pack_loss_uplink2C5 = torch.stack(c5ch_list3, dim=0).to(device)

            for ch in range(32):
                rows4c5 = 256
                first = list(range(0, 256))
                for tr_rounds4c5 in range(max_retra):
                    random_tensor4c5 = torch.ones(256, 256)
                    num_zero_rows4c5 = math.floor(pl_C5 * rows4c5)
                    x4 = set(random.sample(list(first), num_zero_rows4c5))
                    rows4c5 = len(x4)
                    if (rows4c5 == 0):
                        output_tensor4c5 = random_tensor4c5
                        break
                    else:
                        first = x4
                        for i in range(256):
                            if i in x4:
                                random_tensor4c5[i] = 0 * random_tensor4c5[i]
                        output_tensor4c5 = random_tensor4c5
                c5ch_list4.append(output_tensor4c5)
            times4c5 = tr_rounds4c5 + 1
            pack_loss_downlink2C5 = torch.stack(c5ch_list4, dim=0).to(device)

            C5_PL.extend(
                [pack_loss_uplink1C5, pack_loss_downlink1C5, pack_loss_uplink2C5, pack_loss_downlink2C5])
            C5_EPL.extend([pack_loss_uplink1C5, pack_loss_downlink1C5])
            C5T.extend([times1c5, times2c5, times3c5, times4c5])

        elif (c == 6):  # C6------------------------------------------------
            c6ch_list1, c6ch_list2, c6ch_list3, c6ch_list4 = [], [], [], []
            for ch in range(32):
                rows1c6 = 256
                first = list(range(0, 256))
                for tr_rounds1c6 in range(max_retra):
                    random_tensor1c6 = torch.ones(256, 256)
                    num_zero_rows1c6 = math.floor(pl_C6 * rows1c6)
                    x1 = set(random.sample(list(first), num_zero_rows1c6))
                    rows1c6 = len(x1)
                    if (rows1c6 == 0):
                        output_tensor1c6 = random_tensor1c6
                        break
                    else:
                        first = x1
                        for i in range(256):
                            if i in x1:
                                random_tensor1c6[i] = 0 * random_tensor1c6[i]
                        output_tensor1c6 = random_tensor1c6
                c6ch_list1.append(output_tensor1c6)
            times1c6 = tr_rounds1c6 + 1
            pack_loss_uplink1C6 = torch.stack(c6ch_list1, dim=0).to(device)

            for ch in range(32):
                rows2c6 = 256
                first = list(range(0, 256))
                for tr_rounds2c6 in range(max_retra):
                    random_tensor2c6 = torch.ones(256, 256)
                    num_zero_rows2c6 = math.floor(pl_C6 * rows2c6)
                    x2 = set(random.sample(list(first), num_zero_rows2c6))
                    rows2c6 = len(x2)
                    if (rows2c6 == 0):
                        output_tensor2c6 = random_tensor2c6
                        break
                    else:
                        first = x2
                        for i in range(256):
                            if i in x2:
                                random_tensor2c6[i] = 0 * random_tensor2c6[i]
                        output_tensor2c6 = random_tensor2c6
                c6ch_list2.append(output_tensor2c6)
            times2c6 = tr_rounds2c6 + 1
            pack_loss_downlink1C6 = torch.stack(c6ch_list2, dim=0).to(device)

            for ch in range(32):
                rows3c6 = 256
                first = list(range(0, 256))
                for tr_rounds3c6 in range(max_retra):
                    random_tensor3c6 = torch.ones(256, 256)
                    num_zero_rows3c6 = math.floor(pl_C6 * rows3c6)
                    x3 = set(random.sample(list(first), num_zero_rows3c6))
                    rows3c6 = len(x3)
                    if (rows3c6 == 0):
                        output_tensor3c6 = random_tensor3c6
                        break
                    else:
                        first = x3
                        for i in range(256):
                            if i in x3:
                                random_tensor3c6[i] = 0 * random_tensor3c6[i]
                        output_tensor3c6 = random_tensor3c6
                c6ch_list3.append(output_tensor3c6)
            times3c6 = tr_rounds3c6 + 1
            pack_loss_uplink2C6 = torch.stack(c6ch_list3, dim=0).to(device)

            for ch in range(32):
                rows4c6 = 256
                first = list(range(0, 256))
                for tr_rounds4c6 in range(max_retra):
                    random_tensor4c6 = torch.ones(256, 256)
                    num_zero_rows4c6 = math.floor(pl_C6 * rows4c6)
                    x4 = set(random.sample(list(first), num_zero_rows4c6))
                    rows4c6 = len(x4)
                    if (rows4c6 == 0):
                        output_tensor4c6 = random_tensor4c6
                        break
                    else:
                        first = x4
                        for i in range(256):
                            if i in x4:
                                random_tensor4c6[i] = 0 * random_tensor4c6[i]
                        output_tensor4c6 = random_tensor4c6
                c6ch_list4.append(output_tensor4c6)
            times4c6 = tr_rounds4c6 + 1
            pack_loss_downlink2C6 = torch.stack(c6ch_list4, dim=0).to(device)

            C6_PL.extend(
                [pack_loss_uplink1C6, pack_loss_downlink1C6, pack_loss_uplink2C6, pack_loss_downlink2C6])
            C6_EPL.extend([pack_loss_uplink1C6, pack_loss_downlink1C6])
            C6T.extend([times1c6, times2c6, times3c6, times4c6])

        else:  # C7------------------------------------------------
            c7ch_list1, c7ch_list2, c7ch_list3, c7ch_list4 = [], [], [], []
            for ch in range(32):
                rows1c7 = 256
                first = list(range(0, 256))
                for tr_rounds1c7 in range(max_retra):
                    random_tensor1c7 = torch.ones(256, 256)
                    num_zero_rows1c7 = math.floor(pl_C7 * rows1c7)
                    x1 = set(random.sample(list(first), num_zero_rows1c7))
                    rows1c7 = len(x1)
                    if (rows1c7 == 0):
                        output_tensor1c7 = random_tensor1c7
                        break
                    else:
                        first = x1
                        for i in range(256):
                            if i in x1:
                                random_tensor1c7[i] = 0 * random_tensor1c7[i]
                        output_tensor1c7 = random_tensor1c7
                c7ch_list1.append(output_tensor1c7)
            times1c7 = tr_rounds1c7 + 1
            pack_loss_uplink1C7 = torch.stack(c7ch_list1, dim=0).to(device)

            for ch in range(32):
                rows2c7 = 256
                first = list(range(0, 256))
                for tr_rounds2c7 in range(max_retra):
                    random_tensor2c7 = torch.ones(256, 256)
                    num_zero_rows2c7 = math.floor(pl_C7 * rows2c7)
                    x2 = set(random.sample(list(first), num_zero_rows2c7))
                    rows2c7 = len(x2)
                    if (rows2c7 == 0):
                        output_tensor2c7 = random_tensor2c7
                        break
                    else:
                        first = x2
                        for i in range(256):
                            if i in x2:
                                random_tensor2c7[i] = 0 * random_tensor2c7[i]
                        output_tensor2c7 = random_tensor2c7
                c7ch_list2.append(output_tensor2c7)
            times2c7 = tr_rounds2c7 + 1
            pack_loss_downlink1C7 = torch.stack(c7ch_list2, dim=0).to(device)

            for ch in range(32):
                rows3c7 = 256
                first = list(range(0, 256))
                for tr_rounds3c7 in range(max_retra):
                    random_tensor3c7 = torch.ones(256, 256)
                    num_zero_rows3c7 = math.floor(pl_C7 * rows3c7)
                    x3 = set(random.sample(list(first), num_zero_rows3c7))
                    rows3c7 = len(x3)
                    if (rows3c7 == 0):
                        output_tensor3c7 = random_tensor3c7
                        break
                    else:
                        first = x3
                        for i in range(256):
                            if i in x3:
                                random_tensor3c7[i] = 0 * random_tensor3c7[i]
                        output_tensor3c7 = random_tensor3c7
                c7ch_list3.append(output_tensor3c7)
            times3c7 = tr_rounds3c7 + 1
            pack_loss_uplink2C7 = torch.stack(c7ch_list3, dim=0).to(device)

            for ch in range(32):
                rows4c7 = 256
                first = list(range(0, 256))
                for tr_rounds4c7 in range(max_retra):
                    random_tensor4c7 = torch.ones(256, 256)
                    num_zero_rows4c7 = math.floor(pl_C7 * rows4c6)
                    x4 = set(random.sample(list(first), num_zero_rows4c7))
                    rows4c7 = len(x4)
                    if (rows4c6 == 0):
                        output_tensor4c7 = random_tensor4c7
                        break
                    else:
                        first = x4
                        for i in range(256):
                            if i in x4:
                                random_tensor4c7[i] = 0 * random_tensor4c7[i]
                        output_tensor4c7 = random_tensor4c7
                c7ch_list4.append(output_tensor4c7)
            times4c7 = tr_rounds4c7 + 1
            pack_loss_downlink2C7 = torch.stack(c7ch_list4, dim=0).to(device)

            C7_PL.extend(
                [pack_loss_uplink1C7, pack_loss_downlink1C7, pack_loss_uplink2C7, pack_loss_downlink2C7])
            C7_EPL.extend([pack_loss_uplink1C7, pack_loss_downlink1C7])
            C7T.extend([times1c7, times2c7, times3c7, times4c7])

    loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
    # Startng communication rounds
    # ====================================================COMMUNICATION ROUNDS===============================================
    # global round
    C1time, C2time, C3time, C4time, C5time, C6time, C7time = 0, 0, 0, 0, 0, 0, 0
    client1_train_acc, client1_train_loss, client1_train_withbackiou, client1_train_nobackiou, client1_val_acc, client1_val_loss, client1_val_withbackiou, client1_val_nobackiou, client1_g_val_acc, client1_g_val_loss, client1_g_val_iouwithback, client1_g_val_iounoback = [], [], [], [], [], [], [], [], [], [], [], []
    client2_train_acc, client2_train_loss, client2_train_withbackiou, client2_train_nobackiou, client2_val_acc, client2_val_loss, client2_val_withbackiou, client2_val_nobackiou, client2_g_val_acc, client2_g_val_loss, client2_g_val_iouwithback, client2_g_val_iounoback = [], [], [], [], [], [], [], [], [], [], [], []
    client3_train_acc, client3_train_loss, client3_train_withbackiou, client3_train_nobackiou, client3_val_acc, client3_val_loss, client3_val_withbackiou, client3_val_nobackiou, client3_g_val_acc, client3_g_val_loss, client3_g_val_iouwithback, client3_g_val_iounoback = [], [], [], [], [], [], [], [], [], [], [], []
    client4_train_acc, client4_train_loss, client4_train_withbackiou, client4_train_nobackiou, client4_val_acc, client4_val_loss, client4_val_withbackiou, client4_val_nobackiou, client4_g_val_acc, client4_g_val_loss, client4_g_val_iouwithback, client4_g_val_iounoback = [], [], [], [], [], [], [], [], [], [], [], []
    client5_train_acc, client5_train_loss, client5_train_withbackiou, client5_train_nobackiou, client5_val_acc, client5_val_loss, client5_val_withbackiou, client5_val_nobackiou, client5_g_val_acc, client5_g_val_loss, client5_g_val_iouwithback, client5_g_val_iounoback = [], [], [], [], [], [], [], [], [], [], [], []
    client6_train_acc, client6_train_loss, client6_train_withbackiou, client6_train_nobackiou, client6_val_acc, client6_val_loss, client6_val_withbackiou, client6_val_nobackiou, client6_g_val_acc, client6_g_val_loss, client6_g_val_iouwithback, client6_g_val_iounoback = [], [], [], [], [], [], [], [], [], [], [], []
    client7_train_acc, client7_train_loss, client7_train_withbackiou, client7_train_nobackiou, client7_val_acc, client7_val_loss, client7_val_withbackiou, client7_val_nobackiou, client7_g_val_acc, client7_g_val_loss, client7_g_val_iouwithback, client7_g_val_iounoback = [], [], [], [], [], [], [], [], [], [], [], []
    test_Acc, test_Iou_withback, test_Iou_noback, test_Loss = [], [], [], []
    least_lossg = 100000000;

    for com_round in (range(args.comrounds)):
        global_weights1, global_weights2, global_weights3 = [], [], []
        global_weights1, global_weights2, global_weights3 = [], [], []
        round_idx = com_round + 1
        # --------------------------------------LOCAL TRAINING & VALIDATING---------------------------------------------------------------------------
        print(f'\n |       GLOBAL COMMUNICATION ROUND: {round_idx}       |\n')
        # Starting global training-R clients
        print("R clients global training")
        if (len(R_cls) == 0):
            print("No R clients")
            continue
        else:
            for R_round in (range(args.roundsR)):
                local_weights1R, local_weights2R, local_weights3R = [], [], []
                least_lossC1R, least_lossC2R, least_lossC3R, least_lossC4R, least_lossC5R, least_lossC6R, least_lossC7R = 100000000, 100000000, 100000000, 100000000, 100000000, 100000000, 100000000;
                round_idxR = R_round + 1
                # --------------------------------------LOCAL TRAINING & VALIDATING---------------------------------------------------------------------------
                print(f'\n | Global Training Round_R  : {round_idxR} |\n')
                for cl_idx in R_cls:
                    local_model1R = copy.deepcopy(global_model1_fedR)
                    local_model2R = copy.deepcopy(global_model2_fedR)
                    local_model3R = copy.deepcopy(global_model3_fedR)

                    # Backward hook for modelclientBE
                    grads3 = 0
                    def grad_hook1(model, grad_input, grad_output):
                        global grads3
                        grads3 = grad_input[0].clone().detach()
                    local_model3R.decoder1.register_full_backward_hook(grad_hook1)

                    # Backward hook for modelserver
                    grads2 = 0
                    def grad_hook2(model, grad_input, grad_output):
                        global grads2
                        grads2 = grad_input[0].clone().detach()
                    local_model2R.encoder1.register_full_backward_hook(grad_hook2)
                    print("Selected client:", cl_idx)
                    if cl_idx == 1:
                        train_loader = train_loader_C1
                        val_loader = val_loader_C1
                        folder = parentF + "./Outmain/noisycom/Fed_Avg/Saved/local_models/client1"
                        C1m1opR = optim.Adam(local_model1R.parameters(), lr=lr_C1)
                        C1m2opR = optim.Adam(local_model2R.parameters(), lr=lr_C1)
                        C1m3opR = optim.Adam(local_model3R.parameters(), lr=lr_C1)
                        scheduler1 = ExponentialLR(C1m1opR, gamma=1.0)
                        scheduler2 = ExponentialLR(C1m2opR, gamma=1.0)
                        scheduler3 = ExponentialLR(C1m3opR, gamma=1.0)

                    if cl_idx == 2:
                        train_loader = train_loader_C2
                        val_loader = val_loader_C2
                        folder = parentF + "./Outmain/noisycom/Fed_Avg/Saved/local_models/client2"
                        C2m1opR = optim.Adam(local_model1R.parameters(), lr=lr_C2)
                        C2m2opR = optim.Adam(local_model2R.parameters(), lr=lr_C2)
                        C2m3opR = optim.Adam(local_model3R.parameters(), lr=lr_C2)
                        scheduler1 = ExponentialLR(C2m1opR, gamma=1.0)
                        scheduler2 = ExponentialLR(C2m2opR, gamma=1.0)
                        scheduler3 = ExponentialLR(C2m3opR, gamma=1.0)

                    if cl_idx == 3:
                        train_loader = train_loader_C3
                        val_loader = val_loader_C3
                        folder = parentF + "./Outmain/noisycom/Fed_Avg/Saved/local_models/client3"
                        C3m1opR = optim.Adam(local_model1R.parameters(), lr=lr_C3)
                        C3m2opR = optim.Adam(local_model2R.parameters(), lr=lr_C3)
                        C3m3opR = optim.Adam(local_model3R.parameters(), lr=lr_C3)
                        scheduler1 = ExponentialLR(C3m1opR, gamma=1.0)
                        scheduler2 = ExponentialLR(C3m2opR, gamma=1.0)
                        scheduler3 = ExponentialLR(C3m3opR, gamma=1.0)

                    if cl_idx == 4:
                        train_loader = train_loader_C4
                        val_loader = val_loader_C4
                        folder = parentF + "./Outmain/noisycom/Fed_Avg/Saved/local_models/client4"
                        C4m1opR = optim.Adam(local_model1R.parameters(), lr=lr_C4)
                        C4m2opR = optim.Adam(local_model2R.parameters(), lr=lr_C4)
                        C4m3opR = optim.Adam(local_model3R.parameters(), lr=lr_C4)
                        scheduler1 = ExponentialLR(C4m1opR, gamma=1.0)
                        scheduler2 = ExponentialLR(C4m2opR, gamma=1.0)
                        scheduler3 = ExponentialLR(C4m3opR, gamma=1.0)

                    if cl_idx == 5:
                        train_loader = train_loader_C5
                        val_loader = val_loader_C5
                        folder = parentF + "./Outmain/noisycom/Fed_Avg/Saved/local_models/client5"
                        C5m1opR = optim.Adam(local_model1R.parameters(), lr=lr_C5)
                        C5m2opR = optim.Adam(local_model2R.parameters(), lr=lr_C5)
                        C5m3opR = optim.Adam(local_model3R.parameters(), lr=lr_C5)
                        scheduler1 = ExponentialLR(C5m1opR, gamma=1.0)
                        scheduler2 = ExponentialLR(C5m2opR, gamma=1.0)
                        scheduler3 = ExponentialLR(C5m3opR, gamma=1.0)

                    if cl_idx == 6:
                        train_loader = train_loader_C6
                        val_loader = val_loader_C6
                        folder = parentF + "./Outmain/noisycom/Fed_Avg/Saved/local_models/client6"
                        C6m1opR = optim.Adam(local_model1R.parameters(), lr=lr_C6)
                        C6m2opR = optim.Adam(local_model2R.parameters(), lr=lr_C6)
                        C6m3opR = optim.Adam(local_model3R.parameters(), lr=lr_C6)
                        scheduler1 = ExponentialLR(C6m1opR, gamma=1.0)
                        scheduler2 = ExponentialLR(C6m2opR, gamma=1.0)
                        scheduler3 = ExponentialLR(C6m3opR, gamma=1.0)

                    if cl_idx == 7:
                        train_loader = train_loader_C7
                        val_loader = val_loader_C7
                        folder = parentF + "./Outmain/noisycom/Fed_Avg/Saved/local_models/client7"
                        C7m1opR = optim.Adam(local_model1R.parameters(), lr=lr_C7)
                        C7m2opR = optim.Adam(local_model2R.parameters(), lr=lr_C7)
                        C7m3opR = optim.Adam(local_model3R.parameters(), lr=lr_C7)
                        scheduler1 = ExponentialLR(C7m1opR, gamma=1.0)
                        scheduler2 = ExponentialLR(C7m2opR, gamma=1.0)
                        scheduler3 = ExponentialLR(C7m3opR, gamma=1.0)


                    # R clients training
                    if cl_idx in R_cls:
                        # local epoch
                        if cl_idx == 1:  # C1---------------------------------------------------------------C1 local training & validation--------------------------------------------------------------------------------------------------------------------
                            for epoch in range(local_ep1):
                                print(f"[INFO]: Epoch {epoch + 1} of {local_ep1}")
                                print("Client", cl_idx, " training.........")
                                start_timec1 = time.time()
                                train_epoch_loss, train_epoch_acc, trainepoch_iou_withbackground, trainepoch_iou_nobackground, trainepoch_iou_class0, trainepoch_iou_class1, trainepoch_iou_class2, trainepoch_iou_class3, trainepoch_iou_class4 = train(
                                    train_loader, local_model1R, local_model2R, local_model3R, C1m1opR, C1m2opR, C1m3opR,
                                    scheduler1, scheduler2, scheduler3, loss_fn, C1_PL)
                                print("Client", cl_idx, "local validating.........")
                                val_epoch_loss, val_epoch_acc, valepoch_iou_withbackground, valepoch_iou_nobackground, valepoch_iou_class0, valepoch_iou_class1, valepoch_iou_class2, valepoch_iou_class3, valepoch_iou_class4 = eval(
                                    val_loader, local_model1R, local_model2R, local_model3R, loss_fn, folder, C1_EPL)
                                client1_train_acc.append(train_epoch_acc)
                                client1_train_loss.append(train_epoch_loss)
                                client1_train_withbackiou.append(trainepoch_iou_withbackground)
                                client1_train_nobackiou.append(trainepoch_iou_nobackground)
                                client1_val_acc.append(val_epoch_acc)
                                client1_val_loss.append(val_epoch_loss)
                                client1_val_withbackiou.append(valepoch_iou_withbackground)
                                client1_val_nobackiou.append(valepoch_iou_nobackground)
                                if least_lossC1R > val_epoch_loss:
                                    least_lossC1R = val_epoch_loss
                                    torch.save(local_model1R.state_dict(),
                                               parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C1M1_localcheckpointR.pth')
                                    torch.save(local_model2R.state_dict(),
                                               parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C1M2_localcheckpointR.pth')
                                    torch.save(local_model3R.state_dict(),
                                               parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C1M3_localcheckpointR.pth')
                                    print('C1localmodel saved')
                                end_timec1 = time.time()
                                c1t = end_timec1 - start_timec1
                                C1time = C1time + c1t
                                print("C1 cumulative time:", C1time)

                        print(
                            f"Training dice loss: {train_epoch_loss:.3f}, Training accuracy: {train_epoch_acc:.3f},Training iou Score with background: {trainepoch_iou_withbackground:.3f},Training iou Score without background: {trainepoch_iou_nobackground:.3f}")
                        print("\n Training IoUs Client:", cl_idx)
                        print("T: Background:", trainepoch_iou_class0)
                        print("T: ZP:", trainepoch_iou_class1)
                        print("T: TE:", trainepoch_iou_class2)
                        print("T: ICM:", trainepoch_iou_class3)
                        print("T: Blastocoel:", trainepoch_iou_class4)

                        print(
                            f"Validating dice loss: {val_epoch_loss:.3f}, Validating accuracy: {val_epoch_acc:.3f},Validating iou Score with background: {valepoch_iou_withbackground:.3f},Validating iou Score without background: {valepoch_iou_nobackground:.3f}")
                        print("\n Validating IoUs Client:", cl_idx)
                        print("V: Background:", valepoch_iou_class0)
                        print("V: ZP:", valepoch_iou_class1)
                        print("V: TE:", valepoch_iou_class2)
                        print("V: ICM:", valepoch_iou_class3)
                        print("V: Blastocoel:", valepoch_iou_class4)

                        if cl_idx == 2:  # C2---------------------------------------------------------------C2 local training & validation--------------------------------------------------------------------------------------------------------------------
                            for epoch in range(local_ep2):
                                print(f"[INFO]: Epoch {epoch + 1} of {local_ep2}")
                                print("Client", cl_idx, " training.........")
                                start_timec2 = time.time()
                                train_epoch_loss, train_epoch_acc, trainepoch_iou_withbackground, trainepoch_iou_nobackground, trainepoch_iou_class0, trainepoch_iou_class1, trainepoch_iou_class2, trainepoch_iou_class3, trainepoch_iou_class4 = train(
                                    train_loader, local_model1R, local_model2R, local_model3R, C2m1opR, C2m2opR, C2m3opR,
                                    scheduler1, scheduler2, scheduler3, loss_fn, C2_PL)
                                print("Client", cl_idx, "local validating.........")
                                val_epoch_loss, val_epoch_acc, valepoch_iou_withbackground, valepoch_iou_nobackground, valepoch_iou_class0, valepoch_iou_class1, valepoch_iou_class2, valepoch_iou_class3, valepoch_iou_class4 = eval(
                                    val_loader, local_model1R, local_model2R, local_model3R, loss_fn, folder, C2_EPL)
                                client2_train_acc.append(train_epoch_acc)
                                client2_train_loss.append(train_epoch_loss)
                                client2_train_withbackiou.append(trainepoch_iou_withbackground)
                                client2_train_nobackiou.append(trainepoch_iou_nobackground)
                                client2_val_acc.append(val_epoch_acc)
                                client2_val_loss.append(val_epoch_loss)
                                client2_val_withbackiou.append(valepoch_iou_withbackground)
                                client2_val_nobackiou.append(valepoch_iou_nobackground)
                                if least_lossC2R > val_epoch_loss:
                                    least_lossC2R = val_epoch_loss
                                    torch.save(local_model1R.state_dict(),
                                               parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C2M1_localcheckpointR.pth')
                                    torch.save(local_model2R.state_dict(),
                                               parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C2M2_localcheckpointR.pth')
                                    torch.save(local_model3R.state_dict(),
                                               parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C2M3_localcheckpointR.pth')
                                    print('C2localmodel saved')
                                end_timec2 = time.time()
                                c2t = end_timec2 - start_timec2
                                C2time = C2time + c2t
                                print("C2 cumulative time:", C2time)

                        print(
                            f"Training dice loss: {train_epoch_loss:.3f}, Training accuracy: {train_epoch_acc:.3f},Training iou Score with background: {trainepoch_iou_withbackground:.3f},Training iou Score without background: {trainepoch_iou_nobackground:.3f}")
                        print("\n Training IoUs Client:", cl_idx)
                        print("T: Background:", trainepoch_iou_class0)
                        print("T: ZP:", trainepoch_iou_class1)
                        print("T: TE:", trainepoch_iou_class2)
                        print("T: ICM:", trainepoch_iou_class3)
                        print("T: Blastocoel:", trainepoch_iou_class4)

                        print(
                            f"Validating dice loss: {val_epoch_loss:.3f}, Validating accuracy: {val_epoch_acc:.3f},Validating iou Score with background: {valepoch_iou_withbackground:.3f},Validating iou Score without background: {valepoch_iou_nobackground:.3f}")
                        print("\n Validating IoUs Client:", cl_idx)
                        print("V: Background:", valepoch_iou_class0)
                        print("V: ZP:", valepoch_iou_class1)
                        print("V: TE:", valepoch_iou_class2)
                        print("V: ICM:", valepoch_iou_class3)
                        print("V: Blastocoel:", valepoch_iou_class4)


                        if cl_idx == 3:  # C3---------------------------------------------------------------C3 local training & validation--------------------------------------------------------------------------------------------------------------------
                            for epoch in range(local_ep3):
                                print(f"[INFO]: Epoch {epoch + 1} of {local_ep3}")
                                print("Client", cl_idx, " training.........")
                                start_timec3 = time.time()
                                train_epoch_loss, train_epoch_acc, trainepoch_iou_withbackground, trainepoch_iou_nobackground, trainepoch_iou_class0, trainepoch_iou_class1, trainepoch_iou_class2, trainepoch_iou_class3, trainepoch_iou_class4 = train(
                                    train_loader, local_model1R, local_model2R, local_model3R, C3m1opR, C3m2opR, C3m3opR,
                                    scheduler1, scheduler2, scheduler3, loss_fn, C3_PL)
                                print("Client", cl_idx, "local validating.........")
                                val_epoch_loss, val_epoch_acc, valepoch_iou_withbackground, valepoch_iou_nobackground, valepoch_iou_class0, valepoch_iou_class1, valepoch_iou_class2, valepoch_iou_class3, valepoch_iou_class4 = eval(
                                    val_loader, local_model1R, local_model2R, local_model3R, loss_fn, folder, C3_EPL)
                                client3_train_acc.append(train_epoch_acc)
                                client3_train_loss.append(train_epoch_loss)
                                client3_train_withbackiou.append(trainepoch_iou_withbackground)
                                client3_train_nobackiou.append(trainepoch_iou_nobackground)
                                client3_val_acc.append(val_epoch_acc)
                                client3_val_loss.append(val_epoch_loss)
                                client3_val_withbackiou.append(valepoch_iou_withbackground)
                                client3_val_nobackiou.append(valepoch_iou_nobackground)
                                if least_lossC3R > val_epoch_loss:
                                    least_lossC3R = val_epoch_loss
                                    torch.save(local_model1R.state_dict(),
                                               parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C3M1_localcheckpointR.pth')
                                    torch.save(local_model2R.state_dict(),
                                               parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C3M2_localcheckpointR.pth')
                                    torch.save(local_model3R.state_dict(),
                                               parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C3M3_localcheckpointR.pth')
                                    print('C3localmodel saved')
                                end_timec3 = time.time()
                                c3t = end_timec3 - start_timec3
                                C3time = C3time + c3t
                                print("C3 cumulative time:", C3time)
                        print(
                            f"Training dice loss: {train_epoch_loss:.3f}, Training accuracy: {train_epoch_acc:.3f},Training iou Score with background: {trainepoch_iou_withbackground:.3f},Training iou Score without background: {trainepoch_iou_nobackground:.3f}")
                        print("\n Training IoUs Client:", cl_idx)
                        print("T: Background:", trainepoch_iou_class0)
                        print("T: ZP:", trainepoch_iou_class1)
                        print("T: TE:", trainepoch_iou_class2)
                        print("T: ICM:", trainepoch_iou_class3)
                        print("T: Blastocoel:", trainepoch_iou_class4)

                        print(
                            f"Validating dice loss: {val_epoch_loss:.3f}, Validating accuracy: {val_epoch_acc:.3f},Validating iou Score with background: {valepoch_iou_withbackground:.3f},Validating iou Score without background: {valepoch_iou_nobackground:.3f}")
                        print("\n Validating IoUs Client:", cl_idx)
                        print("V: Background:", valepoch_iou_class0)
                        print("V: ZP:", valepoch_iou_class1)
                        print("V: TE:", valepoch_iou_class2)
                        print("V: ICM:", valepoch_iou_class3)
                        print("V: Blastocoel:", valepoch_iou_class4)


                        if cl_idx == 4:  # C4---------------------------------------------------------------C4 local training & validation--------------------------------------------------------------------------------------------------------------------
                            for epoch in range(local_ep4):
                                print(f"[INFO]: Epoch {epoch + 1} of {local_ep4}")
                                print("Client", cl_idx, " training.........")
                                start_timec4 = time.time()
                                train_epoch_loss, train_epoch_acc, trainepoch_iou_withbackground, trainepoch_iou_nobackground, trainepoch_iou_class0, trainepoch_iou_class1, trainepoch_iou_class2, trainepoch_iou_class3, trainepoch_iou_class4 = train(
                                    train_loader, local_model1R, local_model2R, local_model3R, C4m1opR, C4m2opR, C4m3opR,
                                    scheduler1, scheduler2, scheduler3, loss_fn, C4_PL)
                                print("Client", cl_idx, "local validating.........")
                                val_epoch_loss, val_epoch_acc, valepoch_iou_withbackground, valepoch_iou_nobackground, valepoch_iou_class0, valepoch_iou_class1, valepoch_iou_class2, valepoch_iou_class3, valepoch_iou_class4 = eval(
                                    val_loader, local_model1R, local_model2R, local_model3R, loss_fn, folder, C4_EPL)
                                client4_train_acc.append(train_epoch_acc)
                                client4_train_loss.append(train_epoch_loss)
                                client4_train_withbackiou.append(trainepoch_iou_withbackground)
                                client4_train_nobackiou.append(trainepoch_iou_nobackground)
                                client4_val_acc.append(val_epoch_acc)
                                client4_val_loss.append(val_epoch_loss)
                                client4_val_withbackiou.append(valepoch_iou_withbackground)
                                client4_val_nobackiou.append(valepoch_iou_nobackground)
                                if least_lossC4R > val_epoch_loss:
                                    least_lossC4R = val_epoch_loss
                                    torch.save(local_model1R.state_dict(),
                                               parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C4M1_localcheckpointR.pth')
                                    torch.save(local_model2R.state_dict(),
                                               parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C4M2_localcheckpointR.pth')
                                    torch.save(local_model3R.state_dict(),
                                               parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C4M3_localcheckpointR.pth')
                                    print('C4localmodel saved')
                                end_timec4 = time.time()
                                c4t = end_timec4 - start_timec4
                                C4time = C4time + c4t
                                print("C4 cumulative time:", C4time)
                        print(
                            f"Training dice loss: {train_epoch_loss:.3f}, Training accuracy: {train_epoch_acc:.3f},Training iou Score with background: {trainepoch_iou_withbackground:.3f},Training iou Score without background: {trainepoch_iou_nobackground:.3f}")
                        print("\n Training IoUs Client:", cl_idx)
                        print("T: Background:", trainepoch_iou_class0)
                        print("T: ZP:", trainepoch_iou_class1)
                        print("T: TE:", trainepoch_iou_class2)
                        print("T: ICM:", trainepoch_iou_class3)
                        print("T: Blastocoel:", trainepoch_iou_class4)

                        print(
                            f"Validating dice loss: {val_epoch_loss:.3f}, Validating accuracy: {val_epoch_acc:.3f},Validating iou Score with background: {valepoch_iou_withbackground:.3f},Validating iou Score without background: {valepoch_iou_nobackground:.3f}")
                        print("\n Validating IoUs Client:", cl_idx)
                        print("V: Background:", valepoch_iou_class0)
                        print("V: ZP:", valepoch_iou_class1)
                        print("V: TE:", valepoch_iou_class2)
                        print("V: ICM:", valepoch_iou_class3)
                        print("V: Blastocoel:", valepoch_iou_class4)


                        if cl_idx == 5:  # C5---------------------------------------------------------------C5 local training & validation--------------------------------------------------------------------------------------------------------------------
                            for epoch in range(local_ep5):
                                print(f"[INFO]: Epoch {epoch + 1} of {local_ep5}")
                                print("Client", cl_idx, " training.........")
                                start_timec5 = time.time()
                                train_epoch_loss, train_epoch_acc, trainepoch_iou_withbackground, trainepoch_iou_nobackground, trainepoch_iou_class0, trainepoch_iou_class1, trainepoch_iou_class2, trainepoch_iou_class3, trainepoch_iou_class4 = train(
                                    train_loader, local_model1R, local_model2R, local_model3R, C5m1opR, C5m2opR, C5m3opR,
                                    scheduler1, scheduler2, scheduler3, loss_fn, C5_PL)
                                print("Client", cl_idx, "local validating.........")
                                val_epoch_loss, val_epoch_acc, valepoch_iou_withbackground, valepoch_iou_nobackground, valepoch_iou_class0, valepoch_iou_class1, valepoch_iou_class2, valepoch_iou_class3, valepoch_iou_class4 = eval(
                                    val_loader, local_model1R, local_model2R, local_model3R, loss_fn, folder, C5_EPL)
                                client5_train_acc.append(train_epoch_acc)
                                client5_train_loss.append(train_epoch_loss)
                                client5_train_withbackiou.append(trainepoch_iou_withbackground)
                                client5_train_nobackiou.append(trainepoch_iou_nobackground)
                                client5_val_acc.append(val_epoch_acc)
                                client5_val_loss.append(val_epoch_loss)
                                client5_val_withbackiou.append(valepoch_iou_withbackground)
                                client5_val_nobackiou.append(valepoch_iou_nobackground)
                                if least_lossC5R > val_epoch_loss:
                                    least_lossC5R = val_epoch_loss
                                    torch.save(local_model1R.state_dict(),
                                               parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C5M1_localcheckpointR.pth')
                                    torch.save(local_model2R.state_dict(),
                                               parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C5M2_localcheckpointR.pth')
                                    torch.save(local_model3R.state_dict(),
                                               parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C5M3_localcheckpointR.pth')
                                    print('C5localmodel saved')
                                end_timec5 = time.time()
                                c5t = end_timec5 - start_timec5
                                C5time = C5time + c5t
                                print("C5 cumulative time:", C5time)
                        print(
                            f"Training dice loss: {train_epoch_loss:.3f}, Training accuracy: {train_epoch_acc:.3f},Training iou Score with background: {trainepoch_iou_withbackground:.3f},Training iou Score without background: {trainepoch_iou_nobackground:.3f}")
                        print("\n Training IoUs Client:", cl_idx)
                        print("T: Background:", trainepoch_iou_class0)
                        print("T: ZP:", trainepoch_iou_class1)
                        print("T: TE:", trainepoch_iou_class2)
                        print("T: ICM:", trainepoch_iou_class3)
                        print("T: Blastocoel:", trainepoch_iou_class4)

                        print(
                            f"Validating dice loss: {val_epoch_loss:.3f}, Validating accuracy: {val_epoch_acc:.3f},Validating iou Score with background: {valepoch_iou_withbackground:.3f},Validating iou Score without background: {valepoch_iou_nobackground:.3f}")
                        print("\n Validating IoUs Client:", cl_idx)
                        print("V: Background:", valepoch_iou_class0)
                        print("V: ZP:", valepoch_iou_class1)
                        print("V: TE:", valepoch_iou_class2)
                        print("V: ICM:", valepoch_iou_class3)
                        print("V: Blastocoel:", valepoch_iou_class4)



                        if cl_idx == 6:  # C6---------------------------------------------------------------C6 local training & validation--------------------------------------------------------------------------------------------------------------------
                            for epoch in range(local_ep6):
                                print(f"[INFO]: Epoch {epoch + 1} of {local_ep6}")
                                print("Client", cl_idx, " training.........")
                                start_timec6 = time.time()
                                train_epoch_loss, train_epoch_acc, trainepoch_iou_withbackground, trainepoch_iou_nobackground, trainepoch_iou_class0, trainepoch_iou_class1, trainepoch_iou_class2, trainepoch_iou_class3, trainepoch_iou_class4 = train(
                                    train_loader, local_model1R, local_model2R, local_model3R, C6m1opR, C6m2opR, C6m3opR,
                                    scheduler1, scheduler2, scheduler3, loss_fn, C6_PL)
                                print("Client", cl_idx, "local validating.........")
                                val_epoch_loss, val_epoch_acc, valepoch_iou_withbackground, valepoch_iou_nobackground, valepoch_iou_class0, valepoch_iou_class1, valepoch_iou_class2, valepoch_iou_class3, valepoch_iou_class4 = eval(
                                    val_loader, local_model1R, local_model2R, local_model3R, loss_fn, folder, C6_EPL)
                                client6_train_acc.append(train_epoch_acc)
                                client6_train_loss.append(train_epoch_loss)
                                client6_train_withbackiou.append(trainepoch_iou_withbackground)
                                client6_train_nobackiou.append(trainepoch_iou_nobackground)
                                client6_val_acc.append(val_epoch_acc)
                                client6_val_loss.append(val_epoch_loss)
                                client6_val_withbackiou.append(valepoch_iou_withbackground)
                                client6_val_nobackiou.append(valepoch_iou_nobackground)
                                if least_lossC6R > val_epoch_loss:
                                    least_lossC6R = val_epoch_loss
                                    torch.save(local_model1R.state_dict(),
                                               parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C6M1_localcheckpointR.pth')
                                    torch.save(local_model2R.state_dict(),
                                               parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C6M2_localcheckpointR.pth')
                                    torch.save(local_model3R.state_dict(),
                                               parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C6M3_localcheckpointR.pth')
                                    print('C5localmodel saved')
                                end_timec6 = time.time()
                                c6t = end_timec6 - start_timec6
                                C6time = C6time + c6t
                                print("C6 cumulative time:", C6time)
                        print(
                            f"Training dice loss: {train_epoch_loss:.3f}, Training accuracy: {train_epoch_acc:.3f},Training iou Score with background: {trainepoch_iou_withbackground:.3f},Training iou Score without background: {trainepoch_iou_nobackground:.3f}")
                        print("\n Training IoUs Client:", cl_idx)
                        print("T: Background:", trainepoch_iou_class0)
                        print("T: ZP:", trainepoch_iou_class1)
                        print("T: TE:", trainepoch_iou_class2)
                        print("T: ICM:", trainepoch_iou_class3)
                        print("T: Blastocoel:", trainepoch_iou_class4)

                        print(
                            f"Validating dice loss: {val_epoch_loss:.3f}, Validating accuracy: {val_epoch_acc:.3f},Validating iou Score with background: {valepoch_iou_withbackground:.3f},Validating iou Score without background: {valepoch_iou_nobackground:.3f}")
                        print("\n Validating IoUs Client:", cl_idx)
                        print("V: Background:", valepoch_iou_class0)
                        print("V: ZP:", valepoch_iou_class1)
                        print("V: TE:", valepoch_iou_class2)
                        print("V: ICM:", valepoch_iou_class3)
                        print("V: Blastocoel:", valepoch_iou_class4)



                        if cl_idx == 7:  # C7--------------------------------------------------------------C7 local training & validation--------------------------------------------------------------------------------------------------------------------
                            for epoch in range(local_ep7):
                                print(f"[INFO]: Epoch {epoch + 1} of {local_ep7}")
                                print("Client", cl_idx, " training.........")
                                start_timec7 = time.time()
                                train_epoch_loss, train_epoch_acc, trainepoch_iou_withbackground, trainepoch_iou_nobackground, trainepoch_iou_class0, trainepoch_iou_class1, trainepoch_iou_class2, trainepoch_iou_class3, trainepoch_iou_class4 = train(
                                    train_loader, local_model1R, local_model2R, local_model3R, C7m1opR, C7m2opR, C7m3opR,
                                    scheduler1, scheduler2, scheduler3, loss_fn, C7_PL)
                                print("Client", cl_idx, "local validating.........")
                                val_epoch_loss, val_epoch_acc, valepoch_iou_withbackground, valepoch_iou_nobackground, valepoch_iou_class0, valepoch_iou_class1, valepoch_iou_class2, valepoch_iou_class3, valepoch_iou_class4 = eval(
                                    val_loader, local_model1R, local_model2R, local_model3R, loss_fn, folder, C7_EPL)
                                client7_train_acc.append(train_epoch_acc)
                                client7_train_loss.append(train_epoch_loss)
                                client7_train_withbackiou.append(trainepoch_iou_withbackground)
                                client7_train_nobackiou.append(trainepoch_iou_nobackground)
                                client7_val_acc.append(val_epoch_acc)
                                client7_val_loss.append(val_epoch_loss)
                                client7_val_withbackiou.append(valepoch_iou_withbackground)
                                client7_val_nobackiou.append(valepoch_iou_nobackground)
                                if least_lossC7R > val_epoch_loss:
                                    least_lossC7R = val_epoch_loss
                                    torch.save(local_model1R.state_dict(),
                                               parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C7M1_localcheckpointR.pth')
                                    torch.save(local_model2R.state_dict(),
                                               parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C7M2_localcheckpointR.pth')
                                    torch.save(local_model3R.state_dict(),
                                               parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C7M3_localcheckpointR.pth')
                                    print('C5localmodel saved')
                                end_timec7 = time.time()
                                c7t = end_timec7 - start_timec7
                                C7time = C7time + c7t
                                print("C7 cumulative time:", C7time)
                        print(
                            f"Training dice loss: {train_epoch_loss:.3f}, Training accuracy: {train_epoch_acc:.3f},Training iou Score with background: {trainepoch_iou_withbackground:.3f},Training iou Score without background: {trainepoch_iou_nobackground:.3f}")
                        print("\n Training IoUs Client:", cl_idx)
                        print("T: Background:", trainepoch_iou_class0)
                        print("T: ZP:", trainepoch_iou_class1)
                        print("T: TE:", trainepoch_iou_class2)
                        print("T: ICM:", trainepoch_iou_class3)
                        print("T: Blastocoel:", trainepoch_iou_class4)

                        print(
                            f"Validating dice loss: {val_epoch_loss:.3f}, Validating accuracy: {val_epoch_acc:.3f},Validating iou Score with background: {valepoch_iou_withbackground:.3f},Validating iou Score without background: {valepoch_iou_nobackground:.3f}")
                        print("\n Validating IoUs Client:", cl_idx)
                        print("V: Background:", valepoch_iou_class0)
                        print("V: ZP:", valepoch_iou_class1)
                        print("V: TE:", valepoch_iou_class2)
                        print("V: ICM:", valepoch_iou_class3)
                        print("V: Blastocoel:", valepoch_iou_class4)

                # ----------------------------R local training finished------------------------------------------

                if (round_idxR < args.roundsR):
                    C1M1_LBR = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C1M1_localcheckpointR.pth')
                    C1M2_LBR = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C1M2_localcheckpointR.pth')
                    C1M3_LBR = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C1M3_localcheckpointR.pth')
                    C2M1_LBR = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C2M1_localcheckpointR.pth')
                    C2M2_LBR = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C2M2_localcheckpointR.pth')
                    C2M3_LBR = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C2M3_localcheckpointR.pth')
                    C3M1_LBR = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C3M1_localcheckpointR.pth')
                    C3M2_LBR = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C3M2_localcheckpointR.pth')
                    C3M3_LBR = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C3M3_localcheckpointR.pth')
                    C4M1_LBR = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C4M1_localcheckpointR.pth')
                    C4M2_LBR = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C4M2_localcheckpointR.pth')
                    C4M3_LBR = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C4M3_localcheckpointR.pth')
                    C5M1_LBR = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C5M1_localcheckpointR.pth')
                    C5M2_LBR = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C5M2_localcheckpointR.pth')
                    C5M3_LBR = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C5M3_localcheckpointR.pth')
                    C6M1_LBR = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C6M1_localcheckpointR.pth')
                    C6M2_LBR = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C6M2_localcheckpointR.pth')
                    C6M3_LBR = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C6M3_localcheckpointR.pth')
                    C7M1_LBR = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C7M1_localcheckpointR.pth')
                    C7M2_LBR = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C7M2_localcheckpointR.pth')
                    C7M3_LBR = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C7M3_localcheckpointR.pth')

                    C1M1_LBR.update((x, y * D1R) for x, y in C1M1_LBR.items())
                    C1M2_LBR.update((x, y * D1R) for x, y in C1M2_LBR.items())
                    C1M3_LBR.update((x, y * D1R) for x, y in C1M3_LBR.items())
                    C2M1_LBR.update((x, y * D2R) for x, y in C2M1_LBR.items())
                    C2M2_LBR.update((x, y * D2R) for x, y in C2M2_LBR.items())
                    C2M3_LBR.update((x, y * D2R) for x, y in C2M3_LBR.items())
                    C3M1_LBR.update((x, y * D3R) for x, y in C3M1_LBR.items())
                    C3M2_LBR.update((x, y * D3R) for x, y in C3M2_LBR.items())
                    C3M3_LBR.update((x, y * D3R) for x, y in C3M3_LBR.items())
                    C4M1_LBR.update((x, y * D4R) for x, y in C4M1_LBR.items())
                    C4M2_LBR.update((x, y * D4R) for x, y in C4M2_LBR.items())
                    C4M3_LBR.update((x, y * D4R) for x, y in C4M3_LBR.items())
                    C5M1_LBR.update((x, y * D5R) for x, y in C5M1_LBR.items())
                    C5M2_LBR.update((x, y * D5R) for x, y in C5M2_LBR.items())
                    C5M3_LBR.update((x, y * D5R) for x, y in C5M3_LBR.items())
                    C6M1_LBR.update((x, y * D6R) for x, y in C6M1_LBR.items())
                    C6M2_LBR.update((x, y * D6R) for x, y in C6M2_LBR.items())
                    C6M3_LBR.update((x, y * D6R) for x, y in C6M3_LBR.items())
                    C7M1_LBR.update((x, y * D7R) for x, y in C7M1_LBR.items())
                    C7M2_LBR.update((x, y * D7R) for x, y in C7M2_LBR.items())
                    C7M3_LBR.update((x, y * D7R) for x, y in C7M3_LBR.items())

                    M1dictR = [C1M1_LBR, C2M1_LBR,C3M1_LBR,C4M1_LBR,C5M1_LBR,C6M1_LBR,C7M1_LBR]
                    M2dictR = [C1M2_LBR, C2M2_LBR,C3M2_LBR,C4M2_LBR,C5M2_LBR,C6M2_LBR,C7M2_LBR]
                    M3dictR = [C1M3_LBR, C2M3_LBR,C3M3_LBR,C4M3_LBR,C5M3_LBR,C6M3_LBR,C7M3_LBR]

                    # Model1Averaging
                    local_weights1R.extend(M1dictR)
                    local_weights2R.extend(M2dictR)
                    local_weights3R.extend(M3dictR)

                    # averaging parameters
                    global_fed_weights1R = fedAvg(local_weights1R)
                    global_fed_weights2R = fedAvg(local_weights2R)
                    global_fed_weights3R = fedAvg(local_weights3R)

                    # load the new parameters - FedAvg
                    global_model1_fedR.load_state_dict(global_fed_weights1R)
                    global_model2_fedR.load_state_dict(global_fed_weights2R)
                    global_model3_fedR.load_state_dict(global_fed_weights3R)

                    print("R Weights averaged, loaded new weights, new R models sent to R clients")

                if (round_idxR == args.roundsR):
                    print("R clients global training finished")
                    print("Final round updates sent to global aggregation")

        # ----------------------------------------------------------------------------------------------------------------------------------------------------
        # Starting global training-NR clients

        # Global aggregation
        print("Global aggregation starts")

        # Data Quality Measurement: Loss
        DQ1 = client1_val_loss[-1];
        DQ2 = client2_val_loss[-1];
        DQ3 = client3_val_loss[-1];
        DQ4 = client4_val_loss[-1];
        DQ5 = client5_val_loss[-1];
        DQ6 = client6_val_loss[-1];
        DQ7 = client7_val_loss[-1];

        intot_loss = (1 / DQ1 + 1 / DQ2 + 1 / DQ3 + 1 / DQ4 + 1 / DQ5 + 1 / DQ6 + 1 / DQ7)
        DQDis1 = (1 / DQ1) / intot_loss;
        DQDis2 = (1 / DQ2) / intot_loss;
        DQDis3 = (1 / DQ3) / intot_loss;
        DQDis4 = (1 / DQ4) / intot_loss;
        DQDis5 = (1 / DQ5) / intot_loss;
        DQDis6 = (1 / DQ6) / intot_loss;
        DQDis7 = (1 / DQ7) / intot_loss;

        C1M1 = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C1M1_localcheckpointR.pth')
        C1M2 = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C1M2_localcheckpointR.pth')
        C1M3 = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C1M3_localcheckpointR.pth')
        C2M1 = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C2M1_localcheckpointR.pth')
        C2M2 = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C2M2_localcheckpointR.pth')
        C2M3 = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C2M3_localcheckpointR.pth')
        C3M1 = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C3M1_localcheckpointR.pth')
        C3M2 = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C3M2_localcheckpointR.pth')
        C3M3 = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C3M3_localcheckpointR.pth')
        C4M1 = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C4M1_localcheckpointR.pth')
        C4M2 = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C4M2_localcheckpointR.pth')
        C4M3 = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C4M3_localcheckpointR.pth')
        C5M1 = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C5M1_localcheckpointR.pth')
        C5M2 = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C5M2_localcheckpointR.pth')
        C5M3 = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C5M3_localcheckpointR.pth')
        C6M1 = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C6M1_localcheckpointR.pth')
        C6M2 = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C6M2_localcheckpointR.pth')
        C6M3 = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C6M3_localcheckpointR.pth')
        C7M1 = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C7M1_localcheckpointR.pth')
        C7M2 = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C7M2_localcheckpointR.pth')
        C7M3 = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C7M3_localcheckpointR.pth')

        # updated parameters
        C1M1.update((x, y * DQDis1) for x, y in C1M1.items())
        C1M2.update((x, y * DQDis1) for x, y in C1M2.items())
        C1M3.update((x, y * DQDis1) for x, y in C1M3.items())
        C2M1.update((x, y * DQDis2) for x, y in C2M1.items())
        C2M2.update((x, y * DQDis2) for x, y in C2M2.items())
        C2M3.update((x, y * DQDis2) for x, y in C2M3.items())
        C3M1.update((x, y * DQDis3) for x, y in C3M1.items())
        C3M2.update((x, y * DQDis3) for x, y in C3M2.items())
        C3M3.update((x, y * DQDis3) for x, y in C3M3.items())
        C4M1.update((x, y * DQDis4) for x, y in C4M1.items())
        C4M2.update((x, y * DQDis4) for x, y in C4M2.items())
        C4M3.update((x, y * DQDis4) for x, y in C4M3.items())
        C5M1.update((x, y * DQDis5) for x, y in C5M1.items())
        C5M2.update((x, y * DQDis5) for x, y in C5M2.items())
        C5M3.update((x, y * DQDis5) for x, y in C5M3.items())
        C6M1.update((x, y * DQDis6) for x, y in C6M1.items())
        C6M2.update((x, y * DQDis6) for x, y in C6M2.items())
        C6M3.update((x, y * DQDis6) for x, y in C6M3.items())
        C7M1.update((x, y * DQDis7) for x, y in C7M1.items())
        C7M2.update((x, y * DQDis7) for x, y in C7M2.items())
        C7M3.update((x, y * DQDis7) for x, y in C7M3.items())

        G1dict = [C1M1, C2M1, C3M1, C4M1, C5M1, C6M1, C7M1]
        G2dict = [C1M2, C2M2, C3M2, C4M2, C5M2, C6M2, C7M2]
        G3dict = [C1M3, C2M3, C3M3, C4M3, C5M3, C6M3, C7M3]
        global_weights1.extend(G1dict)
        global_weights2.extend(G2dict)
        global_weights3.extend(G3dict)

        # averaging parameters
        global_fed_weights1 = fedAvg(global_weights1)
        global_fed_weights2 = fedAvg(global_weights2)
        global_fed_weights3 = fedAvg(global_weights3)

        # load the new parameters - FedAvg
        global_model1_fedR.load_state_dict(global_fed_weights1)
        global_model2_fedR.load_state_dict(global_fed_weights2)
        global_model3_fedR.load_state_dict(global_fed_weights3)

        global_model1_fedNR.load_state_dict(global_fed_weights1)
        global_model2_fedNR.load_state_dict(global_fed_weights2)
        global_model3_fedNR.load_state_dict(global_fed_weights3)

        print("New Global model formed")

        print("------------  GLOBAL VALIDATION -------------")

        # ------------------------------------------VALIDATING USING THE GLOBAL MODEL-----------------------------------------------------------------------
        # Validating using the global model
        m1 = max(int(args.frac * args.num_users), 1)
        idxs_users1 = np.random.choice(range(args.num_users), m1, replace=False)
        for idx in idxs_users1:

            cl_idx = idx + 1
            print("Selected client:", cl_idx)
            if cl_idx == 1:
                val_loader = val_loader_C1
                folder = parentF + "Outmain/noisycom/Fed_Avg/Saved/global_model/val/client1"
            elif cl_idx == 2:
                val_loader = val_loader_C2
                folder = parentF + "Outmain/noisycom/Fed_Avg/Saved/global_model/val/client2"
            elif cl_idx == 3:
                val_loader = val_loader_C3
                folder = parentF + "Outmain/noisycom/Fed_Avg/Saved/global_model/val/client3"
            elif cl_idx == 4:
                val_loader = val_loader_C4
                folder = parentF + "Outmain/noisycom/Fed_Avg/Saved/global_model/val/client4"
            elif cl_idx == 5:
                val_loader = val_loader_C5
                folder = parentF + "Outmain/noisycom/Fed_Avg/Saved/global_model/val/client5"
            elif cl_idx == 6:
                val_loader = val_loader_C6
                folder = parentF + "Outmain/noisycom/Fed_Avg/Saved/global_model/val/client6"
            elif cl_idx == 7:
                val_loader = val_loader_C7
                folder = parentF + "Outmain/noisycom/Fed_Avg/Saved/global_model/val/client7"

            best_epoch = 0
            for epoch in range(args.val_global_ep):
                print(f"[INFO]: Epoch {epoch + 1} of {args.val_global_ep}")
                print("Client", cl_idx, " validating.........")
                if cl_idx == 1:
                    g_val_epoch_loss, g_val_epoch_acc, g_val_epoch_iouwithback, g_val_epoch_iounoback, g_valepoch_iou_class0, g_valepoch_iou_class1, g_valepoch_iou_class2, g_valepoch_iou_class3, g_valepoch_iou_class4 = eval(
                        val_loader, global_model1_fedR, global_model2_fedR, global_model3_fedR, loss_fn, folder,
                        C1_EPL)
                    client1_g_val_acc.append(g_val_epoch_acc)
                    client1_g_val_loss.append(g_val_epoch_loss)
                    client1_g_val_iouwithback.append(g_val_epoch_iouwithback)
                    client1_g_val_iounoback.append(g_val_epoch_iounoback)
                if cl_idx == 2:
                    g_val_epoch_loss, g_val_epoch_acc, g_val_epoch_iouwithback, g_val_epoch_iounoback, g_valepoch_iou_class0, g_valepoch_iou_class1, g_valepoch_iou_class2, g_valepoch_iou_class3, g_valepoch_iou_class4 = eval(
                        val_loader, global_model1_fedR, global_model2_fedR, global_model3_fedR, loss_fn, folder,
                        C2_EPL)
                    client2_g_val_acc.append(g_val_epoch_acc)
                    client2_g_val_loss.append(g_val_epoch_loss)
                    client2_g_val_iouwithback.append(g_val_epoch_iouwithback)
                    client2_g_val_iounoback.append(g_val_epoch_iounoback)
                if cl_idx == 3:
                    g_val_epoch_loss, g_val_epoch_acc, g_val_epoch_iouwithback, g_val_epoch_iounoback, g_valepoch_iou_class0, g_valepoch_iou_class1, g_valepoch_iou_class2, g_valepoch_iou_class3, g_valepoch_iou_class4 = eval(
                        val_loader, global_model1_fedR, global_model2_fedR, global_model3_fedR, loss_fn, folder,
                        C3_EPL)
                    client3_g_val_acc.append(g_val_epoch_acc)
                    client3_g_val_loss.append(g_val_epoch_loss)
                    client3_g_val_iouwithback.append(g_val_epoch_iouwithback)
                    client3_g_val_iounoback.append(g_val_epoch_iounoback)
                if cl_idx == 4:
                    g_val_epoch_loss, g_val_epoch_acc, g_val_epoch_iouwithback, g_val_epoch_iounoback, g_valepoch_iou_class0, g_valepoch_iou_class1, g_valepoch_iou_class2, g_valepoch_iou_class3, g_valepoch_iou_class4 = eval(
                        val_loader, global_model1_fedR, global_model2_fedR, global_model3_fedR, loss_fn, folder,
                        C4_EPL)
                    client4_g_val_acc.append(g_val_epoch_acc)
                    client4_g_val_loss.append(g_val_epoch_loss)
                    client4_g_val_iouwithback.append(g_val_epoch_iouwithback)
                    client4_g_val_iounoback.append(g_val_epoch_iounoback)
                if cl_idx == 5:
                    g_val_epoch_loss, g_val_epoch_acc, g_val_epoch_iouwithback, g_val_epoch_iounoback, g_valepoch_iou_class0, g_valepoch_iou_class1, g_valepoch_iou_class2, g_valepoch_iou_class3, g_valepoch_iou_class4 = eval(
                        val_loader, global_model1_fedR, global_model2_fedR, global_model3_fedR, loss_fn, folder,
                        C5_EPL)
                    client5_g_val_acc.append(g_val_epoch_acc)
                    client5_g_val_loss.append(g_val_epoch_loss)
                    client5_g_val_iouwithback.append(g_val_epoch_iouwithback)
                    client5_g_val_iounoback.append(g_val_epoch_iounoback)
                if cl_idx == 6:
                    g_val_epoch_loss, g_val_epoch_acc, g_val_epoch_iouwithback, g_val_epoch_iounoback, g_valepoch_iou_class0, g_valepoch_iou_class1, g_valepoch_iou_class2, g_valepoch_iou_class3, g_valepoch_iou_class4 = eval(
                        val_loader, global_model1_fedR, global_model2_fedR, global_model3_fedR, loss_fn, folder,
                        C6_EPL)
                    client6_g_val_acc.append(g_val_epoch_acc)
                    client6_g_val_loss.append(g_val_epoch_loss)
                    client6_g_val_iouwithback.append(g_val_epoch_iouwithback)
                    client6_g_val_iounoback.append(g_val_epoch_iounoback)
                if cl_idx == 7:
                    g_val_epoch_loss, g_val_epoch_acc, g_val_epoch_iouwithback, g_val_epoch_iounoback, g_valepoch_iou_class0, g_valepoch_iou_class1, g_valepoch_iou_class2, g_valepoch_iou_class3, g_valepoch_iou_class4 = eval(
                        val_loader, global_model1_fedR, global_model2_fedR, global_model3_fedR, loss_fn, folder,
                        C7_EPL)
                    client7_g_val_acc.append(g_val_epoch_acc)
                    client7_g_val_loss.append(g_val_epoch_loss)
                    client7_g_val_iouwithback.append(g_val_epoch_iouwithback)
                    client7_g_val_iounoback.append(g_val_epoch_iounoback)

                print(
                    f"Global Validating dice loss: {g_val_epoch_loss:.3f}, Global Validating accuracy: {g_val_epoch_acc:.3f},Global Validating iou Score with background: {g_val_epoch_iouwithback:.3f},Global Validating iou Score without background: {g_val_epoch_iounoback:.3f}")
                print("\n Global Validating IoUs Client:", cl_idx)
                print("GV: Background:", g_valepoch_iou_class0)
                print("GV: ZP:", g_valepoch_iou_class1)
                print("GV: TE:", g_valepoch_iou_class2)
                print("GV: ICM:", g_valepoch_iou_class3)
                print("GV: Blastocoel:", g_valepoch_iou_class4)

        tot_gloss = client1_g_val_loss[-1] + client2_g_val_loss[-1] + client3_g_val_loss[-1] + client4_g_val_loss[-1] + \
                    client5_g_val_loss[-1] + client6_g_val_loss[-1] + client7_g_val_loss[-1]
        avg_g_val_loss = tot_gloss / 7;

        if least_lossg > avg_g_val_loss:
            least_lossg = avg_g_val_loss
            best_epoch = epoch
            torch.save(global_model1_fedR.state_dict(),
                       parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/M1_globalcheckpoint.pth')
            torch.save(global_model2_fedR.state_dict(),
                       parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/M2_globalcheckpoint.pth')
            torch.save(global_model3_fedR.state_dict(),
                       parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/M3_globalcheckpoint.pth')
            print('Global best model saved')
            print('-' * 50)

        # ------------------------------------------TESTING USING THE FINAL GLOBAL MODEL-----------------------------------------------------------------------

        test_folder = parentF + "Outmain/noisycom/Fed_Avg/testingsaved"
        M1_test = copy.deepcopy(global_model1_fed)
        M2_test = copy.deepcopy(global_model2_fed)
        M3_test = copy.deepcopy(global_model3_fed)

        M1_test.load_state_dict(torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/M1_globalcheckpoint.pth'))
        M2_test.load_state_dict(torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/M2_globalcheckpoint.pth'))
        M3_test.load_state_dict(torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/M3_globalcheckpoint.pth'))

        for epoch in range(args.val_global_ep):
            print("Global testing.........")
            test_epoch_loss, test_epoch_acc, test_epoch_accwithback, test_epoch_accnoback = test(test_loader,
                                                                                                 M1_test,
                                                                                                 M2_test,
                                                                                                 M3_test,
                                                                                                 loss_fn, test_folder)
            print('\n')
            print(
                f"Testing dice loss: {test_epoch_loss:.3f}, Testing accuracy: {test_epoch_acc:.3f},Testing iou Score with background: {test_epoch_accwithback:.3f},Testing iou Score without background: {test_epoch_accnoback:.3f}")
            test_Acc.append(test_epoch_acc)
            test_Iou_withback.append(test_epoch_accwithback)
            test_Iou_noback.append(test_epoch_accnoback)
            test_Loss.append(test_epoch_loss)

        # Training time per each client--------------------------------------
        print("Each client's cumulative training time")
        print("C1 cum time:", C1time)
        print("C2 cum time:", C2time)
        print("C3 cum time:", C3time)
        print("C4 cum time:", C4time)
        print("C5 cum time:", C5time)
        print("C6 cum time:", C6time)
        print("C7 cum time:", C7time)

        # -------------------------------------------------PLOTTING RESULTS-----------------------------------------------------------------------

        # local training accuracy plots
        plt.figure(figsize=(20, 5))
        plt.plot(client1_train_acc, color='green', linestyle='-', label='Client 1')
        plt.plot(client2_train_acc, color='blue', linestyle='-', label='Client 2')
        plt.plot(client3_train_acc, color='orange', linestyle='-', label='Client 3')
        plt.plot(client4_train_acc, color='red', linestyle='-', label='Client 4')
        plt.plot(client5_train_acc, color='black', linestyle='-', label='Client 5')
        plt.plot(client6_train_acc, color='red', linestyle='-', label='Client 6')
        plt.plot(client7_train_acc, color='black', linestyle='-', label='Client 7')
        plt.xlabel('Local epochs', fontsize=16, fontweight='bold')
        plt.ylabel('Train Accuracy', fontsize=16, fontweight='bold')
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.legend(fontsize=14)
        plt.grid(True)
        plt.savefig(parentF + './Outmain/noisycom/Fed_Avg/Outputs/train accuracy.jpg')

        plt.figure(figsize=(20, 5))
        plt.plot(client1_train_loss, color='green', linestyle='-', label='Client 1')
        plt.plot(client2_train_loss, color='blue', linestyle='-', label='Client 2')
        plt.plot(client3_train_loss, color='orange', linestyle='-', label='Client 3')
        plt.plot(client4_train_loss, color='red', linestyle='-', label='Client 4')
        plt.plot(client5_train_loss, color='black', linestyle='-', label='Client 5')
        plt.plot(client6_train_loss, color='red', linestyle='-', label='Client 6')
        plt.plot(client7_train_loss, color='black', linestyle='-', label='Client 7')
        plt.xlabel('Local epochs', fontsize=16, fontweight='bold')
        plt.ylabel('Train Dice loss', fontsize=16, fontweight='bold')
        plt.legend(fontsize=14)
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.grid(True)
        plt.savefig(parentF + './Outmain/noisycom/Fed_Avg/Outputs/train Dice loss.jpg')

        plt.figure(figsize=(20, 5))
        plt.plot(client1_train_withbackiou, color='green', linestyle='-', label='Client 1')
        plt.plot(client2_train_withbackiou, color='blue', linestyle='-', label='Client 2')
        plt.plot(client3_train_withbackiou, color='orange', linestyle='-', label='Client 3')
        plt.plot(client4_train_withbackiou, color='red', linestyle='-', label='Client 4')
        plt.plot(client5_train_withbackiou, color='black', linestyle='-', label='Client 5')
        plt.plot(client6_train_withbackiou, color='red', linestyle='-', label='Client 6')
        plt.plot(client7_train_withbackiou, color='black', linestyle='-', label='Client 7')
        plt.xlabel('Local epochs', fontsize=16, fontweight='bold')
        plt.ylabel('Train IoU: With Background', fontsize=16, fontweight='bold')
        plt.legend(fontsize=14)
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.grid(True)
        plt.savefig(parentF + './Outmain/noisycom/Fed_Avg/Outputs/train iou with background.jpg')

        plt.figure(figsize=(20, 5))
        plt.plot(client1_train_nobackiou, color='green', linestyle='-', label='Client 1')
        plt.plot(client2_train_nobackiou, color='blue', linestyle='-', label='Client 2')
        plt.plot(client3_train_nobackiou, color='orange', linestyle='-', label='Client 3')
        plt.plot(client4_train_nobackiou, color='red', linestyle='-', label='Client 4')
        plt.plot(client5_train_nobackiou, color='black', linestyle='-', label='Client 5')
        plt.plot(client6_train_nobackiou, color='red', linestyle='-', label='Client 6')
        plt.plot(client7_train_nobackiou, color='black', linestyle='-', label='Client 7')
        plt.xlabel('Local epochs', fontsize=16, fontweight='bold')
        plt.ylabel('Train IoU: Without Background', fontsize=16, fontweight='bold')
        plt.legend(fontsize=14)
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.grid(True)
        plt.savefig(parentF + './Outmain/noisycom/Fed_Avg/Outputs/train iou without background.jpg')

        # -----------------------------------------------------------------------------------------------------------------------
        # local validation accuracy plots
        plt.figure(figsize=(20, 5))
        plt.plot(client1_val_acc, color='green', linestyle='-', label='Client 1')
        plt.plot(client2_val_acc, color='blue', linestyle='-', label='Client 2')
        plt.plot(client3_val_acc, color='orange', linestyle='-', label='Client 3')
        plt.plot(client4_val_acc, color='red', linestyle='-', label='Client 4')
        plt.plot(client5_val_acc, color='black', linestyle='-', label='Client 5')
        plt.plot(client6_val_acc, color='red', linestyle='-', label='Client 6')
        plt.plot(client7_val_acc, color='black', linestyle='-', label='Client 7')
        plt.xlabel('Local epochs', fontsize=16, fontweight='bold')
        plt.ylabel('val Accuracy', fontsize=16, fontweight='bold')
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.legend(fontsize=14)
        plt.grid(True)
        plt.savefig(parentF + './Outmain/noisycom/Fed_Avg/Outputs/val accuracy.jpg')

        # global validation dice loss plots
        plt.figure(figsize=(20, 5))
        plt.plot(client1_val_loss, color='green', linestyle='-', label='Client 1')
        plt.plot(client2_val_loss, color='blue', linestyle='-', label='Client 2')
        plt.plot(client3_val_loss, color='orange', linestyle='-', label='Client 3')
        plt.plot(client4_val_loss, color='red', linestyle='-', label='Client 4')
        plt.plot(client5_val_loss, color='black', linestyle='-', label='Client 5')
        plt.plot(client6_val_loss, color='red', linestyle='-', label='Client 6')
        plt.plot(client7_val_loss, color='black', linestyle='-', label='Client 7')
        plt.xlabel('Local epochs', fontsize=16, fontweight='bold')
        plt.ylabel('val Dice loss', fontsize=16, fontweight='bold')
        plt.legend(fontsize=14)
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.grid(True)
        plt.savefig(parentF + './Outmain/noisycom/Fed_Avg/Outputs/val Dice loss.jpg')

        # global validation iou score plots
        plt.figure(figsize=(20, 5))
        plt.plot(client1_val_withbackiou, color='green', linestyle='-', label='Client 1')
        plt.plot(client2_val_withbackiou, color='blue', linestyle='-', label='Client 2')
        plt.plot(client3_val_withbackiou, color='orange', linestyle='-', label='Client 3')
        plt.plot(client4_val_withbackiou, color='red', linestyle='-', label='Client 4')
        plt.plot(client5_val_withbackiou, color='black', linestyle='-', label='Client 5')
        plt.plot(client6_val_withbackiou, color='red', linestyle='-', label='Client 6')
        plt.plot(client7_val_withbackiou, color='black', linestyle='-', label='Client 7')
        plt.xlabel('Local epochs', fontsize=16, fontweight='bold')
        plt.ylabel('Validation IoU: With Background', fontsize=16, fontweight='bold')
        plt.legend(fontsize=14)
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.grid(True)
        plt.savefig(parentF + './Outmain/noisycom/Fed_Avg/Outputs/val iou with background.jpg')

        plt.figure(figsize=(20, 5))
        plt.plot(client1_val_nobackiou, color='green', linestyle='-', label='Client 1')
        plt.plot(client2_val_nobackiou, color='blue', linestyle='-', label='Client 2')
        plt.plot(client3_val_nobackiou, color='orange', linestyle='-', label='Client 3')
        plt.plot(client4_val_nobackiou, color='red', linestyle='-', label='Client 4')
        plt.plot(client5_val_nobackiou, color='black', linestyle='-', label='Client 5')
        plt.plot(client6_val_nobackiou, color='red', linestyle='-', label='Client 6')
        plt.plot(client7_val_nobackiou, color='black', linestyle='-', label='Client 7')
        plt.xlabel('Local epochs', fontsize=16, fontweight='bold')
        plt.ylabel('Validation IoU: Without Background', fontsize=16, fontweight='bold')
        plt.legend(fontsize=14)
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.grid(True)
        plt.savefig(parentF + './Outmain/noisycom/Fed_Avg/Outputs/val iou without background.jpg')

        # -----------------------------------------------------------------------------------------------------------------------
        # global validation accuracy plots
        plt.figure(figsize=(20, 5))
        plt.plot(client1_g_val_acc, color='green', linestyle='-', label='Client 1')
        plt.plot(client2_g_val_acc, color='blue', linestyle='-', label='Client 2')
        plt.plot(client3_g_val_acc, color='orange', linestyle='-', label='Client 3')
        plt.plot(client4_g_val_acc, color='red', linestyle='-', label='Client 4')
        plt.plot(client5_g_val_acc, color='black', linestyle='-', label='Client 5')
        plt.plot(client6_g_val_acc, color='red', linestyle='-', label='Client 6')
        plt.plot(client7_g_val_acc, color='black', linestyle='-', label='Client 7')
        plt.xlabel('Global epochs', fontsize=16, fontweight='bold')
        plt.ylabel('Global val Accuracy', fontsize=16, fontweight='bold')
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.legend(fontsize=14)
        plt.grid(True)
        plt.savefig(parentF + './Outmain/noisycom/Fed_Avg/Outputs/global val accuracy.jpg')

        # global validation dice loss plots
        plt.figure(figsize=(20, 5))
        plt.plot(client1_g_val_loss, color='green', linestyle='-', label='Client 1')
        plt.plot(client2_g_val_loss, color='blue', linestyle='-', label='Client 2')
        plt.plot(client3_g_val_loss, color='orange', linestyle='-', label='Client 3')
        plt.plot(client4_g_val_loss, color='red', linestyle='-', label='Client 4')
        plt.plot(client5_g_val_loss, color='black', linestyle='-', label='Client 5')
        plt.plot(client6_g_val_loss, color='red', linestyle='-', label='Client 6')
        plt.plot(client7_g_val_loss, color='black', linestyle='-', label='Client 7')
        plt.xlabel('Global epochs', fontsize=16, fontweight='bold')
        plt.ylabel('Global val Dice loss', fontsize=16, fontweight='bold')
        plt.legend(fontsize=14)
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.grid(True)
        plt.savefig(parentF + './Outmain/noisycom/Fed_Avg/Outputs/global val Dice loss.jpg')

        # global validation iou score plots
        plt.figure(figsize=(20, 5))
        plt.plot(client1_g_val_iouwithback, color='green', linestyle='-', label='Client 1')
        plt.plot(client2_g_val_iouwithback, color='blue', linestyle='-', label='Client 2')
        plt.plot(client3_g_val_iouwithback, color='orange', linestyle='-', label='Client 3')
        plt.plot(client4_g_val_iouwithback, color='red', linestyle='-', label='Client 4')
        plt.plot(client5_g_val_iouwithback, color='black', linestyle='-', label='Client 5')
        plt.plot(client6_g_val_iouwithback, color='red', linestyle='-', label='Client 6')
        plt.plot(client7_g_val_iouwithback, color='black', linestyle='-', label='Client 7')
        plt.xlabel('Global epochs', fontsize=16, fontweight='bold')
        plt.ylabel('Global val IoU: With Background', fontsize=16, fontweight='bold')
        plt.legend(fontsize=14)
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.grid(True)
        plt.savefig(parentF + './Outmain/noisycom/Fed_Avg/Outputs/global val iou with background.jpg')

        # global validation iou score plots
        plt.figure(figsize=(20, 5))
        plt.plot(client1_g_val_iounoback, color='green', linestyle='-', label='Client 1')
        plt.plot(client2_g_val_iounoback, color='blue', linestyle='-', label='Client 2')
        plt.plot(client3_g_val_iounoback, color='orange', linestyle='-', label='Client 3')
        plt.plot(client4_g_val_iounoback, color='red', linestyle='-', label='Client 4')
        plt.plot(client5_g_val_iounoback, color='black', linestyle='-', label='Client 5')
        plt.plot(client6_g_val_iounoback, color='red', linestyle='-', label='Client 6')
        plt.plot(client7_g_val_iounoback, color='black', linestyle='-', label='Client 7')
        plt.xlabel('Global epochs', fontsize=16, fontweight='bold')
        plt.ylabel('Global val IoU: Without Background', fontsize=16, fontweight='bold')
        plt.legend(fontsize=14)
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.grid(True)
        plt.savefig(parentF + './Outmain/noisycom/Fed_Avg/Outputs/global val iou without background.jpg')

        # -------------------------------------------------------------------------------------

        # Global model testing performance
        plt.figure(figsize=(20, 5))
        plt.plot(test_Acc, color='black', linestyle='-', label='Global testing accuracy')
        plt.xlabel('Global epochs', fontsize=16, fontweight='bold')
        plt.ylabel('Testing Accuracy', fontsize=16, fontweight='bold')
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.legend(fontsize=14)
        plt.grid(True)
        plt.savefig(parentF + './Outmain/noisycom/Fed_Avg/Outputs/Testing accuracy.jpg')

        # Global model testing dice loss plots
        plt.figure(figsize=(20, 5))
        plt.plot(test_Loss, color='black', linestyle='-', label='Global testing loss')
        plt.xlabel('Global epochs', fontsize=16, fontweight='bold')
        plt.ylabel('Testing Dice loss', fontsize=16, fontweight='bold')
        plt.legend(fontsize=14)
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.grid(True)
        plt.savefig(parentF + './Outmain/noisycom/Fed_Avg/Outputs/Testing Dice loss.jpg')

        # Global model testing iou score plots
        plt.figure(figsize=(20, 5))
        plt.plot(test_Iou_withback, color='black', linestyle='-', label='Global testing IoU')
        plt.xlabel('Global epochs', fontsize=16, fontweight='bold')
        plt.ylabel('Testing IoU: With background', fontsize=16, fontweight='bold')
        plt.legend(fontsize=14)
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.grid(True)
        plt.savefig(parentF + './Outmain/noisycom/Fed_Avg/Outputs/Testing iou withbackground.jpg')

        # Global model testing iou score plots
        plt.figure(figsize=(20, 5))
        plt.plot(test_Iou_noback, color='black', linestyle='-', label='Global testing IoU')
        plt.xlabel('Global epochs', fontsize=16, fontweight='bold')
        plt.ylabel('Testing IoU: Without background', fontsize=16, fontweight='bold')
        plt.legend(fontsize=14)
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.grid(True)
        plt.savefig(parentF + './Outmain/noisycom/Fed_Avg/Outputs/Testing iou without background.jpg')

        alltest_acc, alltest_iouwithback, alltest_iounoback, alltest_loss = [], [], [], []
        alltest_acc.append(test_Acc)
        alltest_loss.append(test_Loss)
        alltest_iouwithback.append(test_Iou_withback)
        alltest_iounoback.append(test_Iou_noback)
        alltest_acc = pd.DataFrame(alltest_acc)
        alltest_loss = pd.DataFrame(alltest_loss)
        alltest_iouwithback = pd.DataFrame(alltest_iouwithback)
        alltest_iounoback = pd.DataFrame(alltest_iounoback)

        alltest_acc.to_csv(parentF + './Outmain/noisycom/Fed_Avg/Outputs/alltest_acc.csv')
        alltest_loss.to_csv(parentF + './Outmain/noisycom/Fed_Avg/Outputs/alltest_loss.csv')
        alltest_iouwithback.to_csv(parentF + './Outmain/noisycom/Fed_Avg/Outputs/alltest_iouwithback.csv')
        alltest_iounoback.to_csv(parentF + './Outmain/noisycom/Fed_Avg/Outputs/alltest_iouwithoutback.csv')

        # -------------------------------------------------------------------------------------

    print('TRAINING COMPLETE')
    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))

if __name__ == "__main__":
    main()
