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
from errorcorrection_main.cal_epoch_R_4 import cal_epoch
from errorcorrection_main.cal_lr_R_4 import cal_lr
from normali import min_max_scaling

# Hyperparameters
LEARNING_RATE = 0.0001
device = "cuda"
NUM_WORKERS = 1
SHUFFLE = False
NUM_CLASSES = 5
PIN_MEMORY = False


parentF = "C:/Users/csj5/Projects/Adaptive_Splitfed/"
Datadr = "C:/Users/csj5/Projects/Data/BlastocystDATA/datafed/uniform_dis/federated/"

# client 1
TRAIN_IMG_DIR_C1 = Datadr + "./client1/train_imgs/"
TRAIN_MASK_DIR_C1 = Datadr + "./client1/train_masks/"
VAL_IMG_DIR_C1 = Datadr + "./client1/val_imgs/"
VAL_MASK_DIR_C1 = Datadr + "./client1/val_masks/"
'''
# client 2
TRAIN_IMG_DIR_C2 = Datadr + "./client2/train_imgs/"
TRAIN_MASK_DIR_C2 = Datadr + "./client2/train_masks/"
VAL_IMG_DIR_C2 = Datadr + "./client2/val_imgs/"
VAL_MASK_DIR_C2 = Datadr + "./client2/val_masks/"

# client 3
TRAIN_IMG_DIR_C3 = Datadr + "./client3/train_imgs/"
TRAIN_MASK_DIR_C3 = Datadr + "./client3/train_masks/"
VAL_IMG_DIR_C3 = Datadr + "./client3/val_imgs/"
VAL_MASK_DIR_C3 = Datadr + "./client3/val_masks/"


# client 4
TRAIN_IMG_DIR_C4 = Datadr + "./client4/train_imgs/"
TRAIN_MASK_DIR_C4 = Datadr + "./client4/train_masks/"
VAL_IMG_DIR_C4 = Datadr + "./client4/val_imgs/"
VAL_MASK_DIR_C4 = Datadr + "./client4/val_masks/"

# client 5

TRAIN_IMG_DIR_C5 = parentF + "./datafed/data/client5/train_imgs/"
TRAIN_MASK_DIR_C5 = parentF + "./datafed/data/client5/train_masks/"
VAL_IMG_DIR_C5 = parentF + "./datafed/data/client5/val_imgs/"
VAL_MASK_DIR_C5 = parentF + "./datafed/data/client5/val_masks/"
'''

TEST_IMG_DIR = Datadr + "./test_imgs/"
TEST_MASK_DIR = Datadr + "./test_masks/"


# 1. Screen Train function
def train_screen(train_loader, local_model1, local_model2, local_model3, optimizer1, optimizer2, optimizer3, loss_fn, PL):
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
            #quantifying PL:
            mis_ls=torch.where((lostpredictions1[0][1] == predictions1[0][1]).all(dim=1))[0]
            quantified_PL =  (256-mis_ls.size()[0])/256
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
def train(train_loader, local_model1, local_model2, local_model3, optimizer1, optimizer2, optimizer3, scheduler1,scheduler2,scheduler3,loss_fn, PL):
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
    '''
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
    '''
    test_loader = get_loaders_test(
        TEST_IMG_DIR,
        TEST_MASK_DIR,
        train_transform
    )

    global_model1_fed = UNET_FE(in_channels=3).to(device)
    global_model2_fed = UNET_server(in_channels=32).to(device)
    global_model3_fed= UNET_BE(out_channels=NUM_CLASSES).to(device)


    # generating packet loss for the screening round - No retransmission -----------------------------------------------------------------------------------------------------
    SC1_PL, SC2_PL, SC3_PL, SC4_PL, SC5_PL, SC1_EPL, SC2_EPL, SC3_EPL, SC4_EPL, SC5_EPL = [], [], [], [], [], [], [], [], [], []
    imgsize = 256
    for c in range(1, 6):
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


    # Overall data distribution
    tot_loader = len(train_loader_C1)
    D1 = len(train_loader_C1) / tot_loader;

    D=[]
    D.extend([D1])
    Dmax = max(D)

    print("Data ratios")
    print(D1)

    # ====================================================SCREENING ROUND ===============================================
    print(f'\n |     SCREENING COM ROUND       |....')
    S1time, S2time, S3time, S4time, S5time = 0, 0, 0, 0, 0
    client1_train_acc, client1_train_loss, client1_train_withbackiou,client1_train_nobackiou, client1_val_acc, client1_val_loss, client1_val_withbackiou,client1_val_nobackiou,client1_g_val_acc, client1_g_val_loss, client1_g_val_iouwithback,client1_g_val_iounoback = [], [], [], [], [], [], [], [], [],[], [],[]
    test_Acc,test_Iou_withback,test_Iou_noback,test_Loss =[],[],[],[]
    global_weights1S, global_weights2S, global_weights3S = [], [], []
    least_lossS, least_lossg = 100000000,100000000 ;
    least_lossC1S, least_lossC2S, least_lossC3S, least_lossC4S, least_lossC5S = 100000000, 100000000, 100000000, 100000000, 100000000;
    R_cls = []
    NR_cls = []
    for idx in range(5):
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


            # local epoch
        for epoch in range(args.screen_ep):
            print(f"[INFO]: Epoch {epoch + 1} of {args.screen_ep}")
            print("Client", cl_idx, " training.........")
            if cl_idx == 1:  # C1---------------------------------------------------------------C1 screen training & validation--------------------------------------------------------------------------------------------------------------------
                start_times1 =time.time()
                S_train_epoch_loss, S_train_epoch_acc, S_trainepoch_iou_withbackground, S_trainepoch_iou_nobackground, S_trainepoch_iou_class0, S_trainepoch_iou_class1, S_trainepoch_iou_class2, S_trainepoch_iou_class3, S_trainepoch_iou_class4, isreliable1, quantified_PL = train_screen(
                    train_loader, localmodel1, localmodel2, localmodel3, screen_opt1, screen_opt2, screen_opt3,loss_fn, SC1_PL)
                end_times1 = time.time()
                s1t = end_times1 - start_times1
                s1time =S1time + s1t
                print("Client", cl_idx, "local validating.........")
                S_val_epoch_loss, S_val_epoch_acc, S_valepoch_iou_withbackground, S_valepoch_iou_nobackground, S_valepoch_iou_class0, S_valepoch_iou_class1, S_valepoch_iou_class2, S_valepoch_iou_class3, S_valepoch_iou_class4 = eval(
                    val_loader,  localmodel1, localmodel2, localmodel3, loss_fn, folder, SC1_EPL)
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
                    torch.save(localmodel1.state_dict(),parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C1M1_localcheckpointS.pth')
                    torch.save(localmodel2.state_dict(),parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C1M2_localcheckpointS.pth')
                    torch.save(localmodel3.state_dict(),parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C1M3_localcheckpointS.pth')
                    print('C1localmodel saved')

            print(f"S_Training dice loss: {S_train_epoch_loss:.3f}, S_Training accuracy: {S_train_epoch_acc:.3f},S_Training iou Score with background: {S_trainepoch_iou_withbackground:.3f},Training iou Score without background: {S_trainepoch_iou_nobackground:.3f}")
            print("\n S_Training IoUs Client:", cl_idx)
            print("T: S_Background:", S_trainepoch_iou_class0)
            print("T: S_ZP:", S_trainepoch_iou_class1)
            print("T: S_TE:", S_trainepoch_iou_class2)
            print("T: S_ICM:", S_trainepoch_iou_class3)
            print("T: S_Blastocoel:", S_trainepoch_iou_class4)

            print(f"S_Validating dice loss: {S_val_epoch_loss:.3f}, Validating accuracy: {S_val_epoch_acc:.3f},Validating iou Score with background: {S_valepoch_iou_withbackground:.3f},Validating iou Score without background: {S_valepoch_iou_nobackground:.3f}")
            print("\n S_Validating IoUs Client:", cl_idx)
            print("V: S_Background:", S_valepoch_iou_class0)
            print("V: S_ZP:", S_valepoch_iou_class1)
            print("V: S_TE:", S_valepoch_iou_class2)
            print("V: S_ICM:", S_valepoch_iou_class3)
            print("V: S_Blastocoel:", S_valepoch_iou_class4)

    #Identify if the clients are experiencing packet loss.
    #Divide the clients in to 2 groups as reliable ((R)/strong) clients and unreliable ((NR)/weak) clients based on the packet loss.
    #Determine the local epochs, LR
    #global aggregation - Model updates of all the clients would be aggregated based on client’s global validation loss, data ratio and the packet loss probability


    C1M1 = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C1M1_localcheckpointS.pth')
    C1M2 = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C1M2_localcheckpointS.pth')
    C1M3 = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C1M3_localcheckpointS.pth')


    #Global aggregation after screening round
    print("Global aggregation starts")

    #updated parameters
    C1M1.update((x,y*D1) for x,y in C1M1.items())
    C1M2.update((x,y*D1) for x,y in C1M2.items())
    C1M3.update((x,y*D1) for x,y in C1M3.items())


    G1dict = [C1M1]
    G2dict = [C1M2]
    G3dict = [C1M3]
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

    #Distributing the global model to R and NR Clients
    global_model1_fedR = copy.deepcopy(global_model1_fed)
    global_model2_fedR = copy.deepcopy(global_model2_fed)
    global_model3_fedR = copy.deepcopy(global_model3_fed)

    global_model1_fedNR = copy.deepcopy(global_model1_fed)
    global_model2_fedNR = copy.deepcopy(global_model2_fed)
    global_model3_fedNR = copy.deepcopy(global_model3_fed)


    print("R clients:", R_cls)
    print("NR clients:", NR_cls)
    print("C1 - PL", pl_C1)


    print("C1 screening time:", s1time)


    s1pertime = s1time/len(train_loader_C1)


    print("C1 perscreening time:", s1pertime)


    spertime=[]
    spertime.extend([s1pertime])
    Tmax = max(spertime)

    E=12
    n= 0.0001
    # success probaility ratios

    S_C1 = (1 -pl_C1)


    #define the local updates per each client
    #Values for the 5 equations
    T1 = Tmax / s1pertime


    print("C1 T:", T1)


    arrT,arrT_scaled = [],[]
    arrT.extend([T1])
    arrT_scaled = min_max_scaling(arrT)
    print("arrT_scaled:",arrT_scaled)

    D_1 = Dmax/D1


    print("C1 D:", D_1)


    arrD, arrD_scaled = [],[]
    arrD.extend([D_1])
    arrD_scaled = min_max_scaling(arrD)
    print("arrD_scaled:",arrD_scaled)

    e11 = arrT_scaled[0]*E
    e12 =arrD_scaled[0]*E
    e13 = pl_C1*E


    print ("coefficients:", e11, e12, e13)
    #Using the least squares method to decide the epochs for each client
    E1= cal_epoch(e11, e12, e13)

    local_ep1 = E1


    #define the learning rates per each client
    n11 = arrD_scaled[0]/ arrT_scaled[0]
    n12 =S_C1


    print ("coefficients:", n11, n12)
    n1 = cal_lr(n11, n12)

    lr_C1 = n*n1


    print("lr_C1:", lr_C1)



    #values needed for weighted aggregation
    lenR, lenNR = [], []
    if 1 in R_cls:
        lenR.append(len(train_loader_C1))
    else:
        lenNR.append(len(train_loader_C1))



    D1R = len(train_loader_C1) / sum(lenR)

    print(f'\n | Screening Round ends....')


    # generating packet loss with retransmisison ------------------------------------------------------------------------------------------------------------------------------
    C1_PL, C2_PL, C3_PL, C4_PL, C5_PL, C1_EPL, C2_EPL, C3_EPL, C4_EPL, C5_EPL = [], [], [], [], [], [], [], [], [], []
    C1T, C2T, C3T, C4T, C5T = [], [], [], [], []
    max_retra = args.max_retra_shallow

    for c in range(1, 4):
        if (c == 1):  # C1------------------------------------------------
            c1ch_list1, c1ch_list2, c1ch_list3, c1ch_list4 = [], [], [], []
            for ch in range(32):
                rows1c1 = 256
                first = list(range(0, 256))
                for tr_rounds1c1 in range(max_retra):
                    random_tensor1c1 = torch.ones(256, 256)
                    num_zero_rows1c1 = math.floor(pl_C1* rows1c1)
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



    loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
    # Startng communication rounds
    # ====================================================COMMUNICATION ROUNDS===============================================
    # global round
    C1time, C2time, C3time, C4time, C5time = 0, 0, 0, 0, 0
    client1_train_acc, client1_train_loss, client1_train_withbackiou, client1_train_nobackiou, client1_val_acc, client1_val_loss, client1_val_withbackiou, client1_val_nobackiou, client1_g_val_acc, client1_g_val_loss, client1_g_val_iouwithback, client1_g_val_iounoback = [], [], [], [], [], [], [], [], [], [], [], []
    client2_train_acc, client2_train_loss, client2_train_withbackiou, client2_train_nobackiou, client2_val_acc, client2_val_loss, client2_val_withbackiou, client2_val_nobackiou, client2_g_val_acc, client2_g_val_loss, client2_g_val_iouwithback, client2_g_val_iounoback = [], [], [], [], [], [], [], [], [], [], [], []
    client3_train_acc, client3_train_loss, client3_train_withbackiou, client3_train_nobackiou, client3_val_acc, client3_val_loss, client3_val_withbackiou, client3_val_nobackiou, client3_g_val_acc, client3_g_val_loss, client3_g_val_iouwithback, client3_g_val_iounoback = [], [], [], [], [], [], [], [], [], [], [], []
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
                least_lossC1R, least_lossC2R, least_lossC3R, least_lossC4R, least_lossC5R = 100000000, 100000000, 100000000, 100000000, 100000000;
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


                    # R clients training
                    if cl_idx in R_cls:
                        # local epoch
                        if cl_idx == 1:  # C1---------------------------------------------------------------C1 local training & validation--------------------------------------------------------------------------------------------------------------------
                            for epoch in range(local_ep1):
                                print(f"[INFO]: Epoch {epoch + 1} of {local_ep1}")
                                print("Client", cl_idx, " training.........")
                                start_timec1 = time.time()
                                train_epoch_loss, train_epoch_acc, trainepoch_iou_withbackground, trainepoch_iou_nobackground, trainepoch_iou_class0, trainepoch_iou_class1, trainepoch_iou_class2, trainepoch_iou_class3, trainepoch_iou_class4 = train(
                                    train_loader, local_model1R, local_model2R, local_model3R, C1m1opR, C1m2opR,C1m3opR,scheduler1,scheduler2,scheduler3, loss_fn, C1_PL)
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

                # ----------------------------R local training finished------------------------------------------

                if (round_idxR < args.roundsR):
                    C1M1_LBR = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C1M1_localcheckpointR.pth')
                    C1M2_LBR = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C1M2_localcheckpointR.pth')
                    C1M3_LBR = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C1M3_localcheckpointR.pth')

                    C1M1_LBR.update((x, y * D1R) for x, y in C1M1_LBR.items())
                    C1M2_LBR.update((x, y * D1R) for x, y in C1M2_LBR.items())
                    C1M3_LBR.update((x, y * D1R) for x, y in C1M3_LBR.items())


                    M1dictR = [C1M1_LBR]
                    M2dictR = [C1M2_LBR]
                    M3dictR = [C1M3_LBR]

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

        # Data Quality Measurement: Loss
        DQ1 = client1_val_loss[-1];


        intot_loss = (1 / DQ1 )
        DQDis1 = (1 / DQ1) / intot_loss;


        C1M1 = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C1M1_localcheckpointR.pth')
        C1M2 = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C1M2_localcheckpointR.pth')
        C1M3 = torch.load(parentF + './Outmain/noisycom/Fed_Avg/Checkpoints/C1M3_localcheckpointR.pth')



        # updated parameters
        C1M1.update((x, y * DQDis1) for x, y in C1M1.items())
        C1M2.update((x, y * DQDis1) for x, y in C1M2.items())
        C1M3.update((x, y * DQDis1) for x, y in C1M3.items())



        G1dict = [C1M1]
        G2dict = [C1M2]
        G3dict = [C1M3]
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



                print(
                    f"Global Validating dice loss: {g_val_epoch_loss:.3f}, Global Validating accuracy: {g_val_epoch_acc:.3f},Global Validating iou Score with background: {g_val_epoch_iouwithback:.3f},Global Validating iou Score without background: {g_val_epoch_iounoback:.3f}")
                print("\n Global Validating IoUs Client:", cl_idx)
                print("GV: Background:", g_valepoch_iou_class0)
                print("GV: ZP:", g_valepoch_iou_class1)
                print("GV: TE:", g_valepoch_iou_class2)
                print("GV: ICM:", g_valepoch_iou_class3)
                print("GV: Blastocoel:", g_valepoch_iou_class4)

        tot_gloss = client1_g_val_loss[0]
        avg_g_val_loss = tot_gloss ;

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

        for epoch in range(args.val_global_ep):
            print("Global testing.........")
            test_epoch_loss, test_epoch_acc, test_epoch_accwithback, test_epoch_accnoback = test(test_loader,
                                                                                                 global_model1_fedR,
                                                                                                 global_model2_fedR,
                                                                                                 global_model3_fedR,
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

        # -------------------------------------------------PLOTTING RESULTS-----------------------------------------------------------------------

        # local training accuracy plots
        plt.figure(figsize=(20, 5))
        plt.plot(client1_train_acc, color='green', linestyle='-', label='Client 1')
        plt.xlabel('Local epochs', fontsize=16, fontweight='bold')
        plt.ylabel('Train Accuracy', fontsize=16, fontweight='bold')
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.legend(fontsize=14)
        plt.grid(True)
        plt.savefig(parentF + './Outmain/noisycom/Fed_Avg/Outputs/train accuracy.jpg')

        plt.figure(figsize=(20, 5))
        plt.plot(client1_train_loss, color='green', linestyle='-', label='Client 1')
        plt.xlabel('Local epochs', fontsize=16, fontweight='bold')
        plt.ylabel('Train Dice loss', fontsize=16, fontweight='bold')
        plt.legend(fontsize=14)
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.grid(True)
        plt.savefig(parentF + './Outmain/noisycom/Fed_Avg/Outputs/train Dice loss.jpg')

        plt.figure(figsize=(20, 5))
        plt.plot(client1_train_withbackiou, color='green', linestyle='-', label='Client 1')
        plt.xlabel('Local epochs', fontsize=16, fontweight='bold')
        plt.ylabel('Train IoU: With Background', fontsize=16, fontweight='bold')
        plt.legend(fontsize=14)
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.grid(True)
        plt.savefig(parentF + './Outmain/noisycom/Fed_Avg/Outputs/train iou with background.jpg')

        plt.figure(figsize=(20, 5))
        plt.plot(client1_train_nobackiou, color='green', linestyle='-', label='Client 1')
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
        plt.xlabel('Local epochs', fontsize=16, fontweight='bold')
        plt.ylabel('Validation IoU: With Background', fontsize=16, fontweight='bold')
        plt.legend(fontsize=14)
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.grid(True)
        plt.savefig(parentF + './Outmain/noisycom/Fed_Avg/Outputs/val iou with background.jpg')

        plt.figure(figsize=(20, 5))
        plt.plot(client1_val_nobackiou, color='green', linestyle='-', label='Client 1')
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
