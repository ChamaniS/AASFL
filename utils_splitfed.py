import torch
import torchvision
from dataset import EmbryoDataset
from torch.utils.data import DataLoader

DEVICE = "cuda"
from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import random
import time

def get_loaders(
        train_dir,
        train_maskdir,
        val_dir,
        val_maskdir,
        batch_size,
        train_transform,
        val_transform,
        num_workers,
        pin_memory=False

):
    train_ds = EmbryoDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_ds = EmbryoDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return train_loader, val_loader

def get_loaders_test(
        test_dir,
        test_maskdir,
        test_transform
):
    test_ds = EmbryoDataset(
        image_dir=test_dir,
        mask_dir=test_maskdir,
        transform=test_transform
    )

    test_loader = DataLoader(
        test_ds,
    )
    return test_loader

def eval(loader, local_model1,local_model2,local_model3,loss_fn,folder,EPL, T):
    local_model1.eval()
    local_model2.eval()
    local_model3.eval()
    val_running_loss = 0.0
    valid_running_correct = 0.0
    valid_iou_score_class0 = 0.0
    valid_iou_score_class1 = 0.0
    valid_iou_score_class2 = 0.0
    valid_iou_score_class3 = 0.0
    valid_iou_score_class4 = 0.0
    valid_accuracy = 0.0
    valid_f1_score = 0.0
    PL_uplink1 = EPL[0]
    PL_downlink1 = EPL[1]
    with torch.no_grad():
        for idx,(x, y) in enumerate(loader):
            x = x.to(DEVICE)
            y = y.type(torch.LongTensor).to(device=DEVICE)
            predictions1 = local_model1(x)
            lostpredictions1 = PL_uplink1 * predictions1
            predictions2 = local_model2(lostpredictions1)
            lostpredictions2 = PL_downlink1 * predictions2
            predictions3 = local_model3(lostpredictions2)
            loss = loss_fn(predictions3, y)
            preds = torch.argmax(predictions3, dim=1)
            equals = preds == y
            valid_running_correct += torch.mean(equals.type(torch.FloatTensor)).item()
            val_running_loss += loss.item()
            valid_f1_score += f1_score(y.cpu().flatten(), preds.cpu().flatten(), average='micro')
            valid_accuracy += accuracy_score(y.cpu().flatten(), preds.cpu().flatten())
            iou_sklearn = jaccard_score(y.cpu().flatten(), preds.cpu().flatten(), average=None)
            valid_iou_score_class0 +=  iou_sklearn[0]
            valid_iou_score_class1 += iou_sklearn[1]
            valid_iou_score_class2 +=  iou_sklearn[2]
            valid_iou_score_class3 += iou_sklearn[3]
            valid_iou_score_class4 += iou_sklearn[4]
            #torchvision.utils.save_image(preds.float(), f"{folder}/pred_{idx}.BMP", padding=0,scale_each=True,normalize=True)
    epoch_loss = val_running_loss / len(loader.dataset)
    epoch_acc = 100. * (valid_running_correct / len(loader.dataset))
    epoch_iou_class0 = (valid_iou_score_class0 / len(loader.dataset))
    epoch_iou_class1 = (valid_iou_score_class1 / len(loader.dataset))
    epoch_iou_class2 = (valid_iou_score_class2 / len(loader.dataset))
    epoch_iou_class3 = (valid_iou_score_class3 / len(loader.dataset))
    epoch_iou_class4 = (valid_iou_score_class4 / len(loader.dataset))
    epoch_iou_withbackground = (epoch_iou_class0 + epoch_iou_class1 + epoch_iou_class2 + epoch_iou_class3 + epoch_iou_class4) / 5
    epoch_iou_nobackground = (epoch_iou_class1 + epoch_iou_class2 + epoch_iou_class3 + epoch_iou_class4) / 4
    return epoch_loss, epoch_acc, epoch_iou_withbackground, epoch_iou_nobackground, epoch_iou_class0, epoch_iou_class1, epoch_iou_class2, epoch_iou_class3, epoch_iou_class4
    model.train()



def test(loader, modelclientFE,modelserver,modelclientBE,loss_fn,folder):
    modelclientFE.eval()
    modelserver.eval()
    modelclientBE.eval()
    val_running_loss = 0.0
    valid_running_correct = 0.0
    valid_iou_score_class0 = 0.0
    valid_iou_score_class1 = 0.0
    valid_iou_score_class2 = 0.0
    valid_iou_score_class3 = 0.0
    valid_iou_score_class4 = 0.0
    valid_accuracy = 0.0
    valid_f1_score = 0.0
    with torch.no_grad():
        for idx,(x, y) in enumerate(loader):
            x = x.to(DEVICE)
            y = y.type(torch.LongTensor).to(device=DEVICE)
            predictions1 = modelclientFE(x)
            predictions2 = modelserver(predictions1)
            predictions3 = modelclientBE(predictions2)
            loss = loss_fn(predictions3, y)

            # calculate the testing accuracy
            preds = torch.argmax(predictions3, dim=1)
            equals = preds == y
            valid_running_correct += torch.mean(equals.type(torch.FloatTensor)).item()

            #  Validation loss
            val_running_loss += loss.item()

            # iou score
            valid_f1_score += f1_score(y.cpu().flatten(), preds.cpu().flatten(), average='micro')
            valid_accuracy += accuracy_score(y.cpu().flatten(), preds.cpu().flatten())
            iou_sklearn = jaccard_score(y.cpu().flatten(), preds.cpu().flatten(), average=None)
            valid_iou_score_class0 +=  iou_sklearn[0]
            valid_iou_score_class1 += iou_sklearn[1]
            valid_iou_score_class2 +=  iou_sklearn[2]
            valid_iou_score_class3 += iou_sklearn[3]
            valid_iou_score_class4 += iou_sklearn[4]
            torchvision.utils.save_image(preds.float(), f"{folder}/pred_{idx}.BMP", padding=0,scale_each=True,normalize=True)

    #print(confusion_matrix)
    epoch_loss = val_running_loss / len(loader.dataset)
    epoch_acc = 100. * (valid_running_correct / len(loader.dataset))
    epoch_iou_class0 = (valid_iou_score_class0 / len(loader.dataset))
    epoch_iou_class1 = (valid_iou_score_class1 / len(loader.dataset))
    epoch_iou_class2 = (valid_iou_score_class2 / len(loader.dataset))
    epoch_iou_class3 = (valid_iou_score_class3 / len(loader.dataset))
    epoch_iou_class4 = (valid_iou_score_class4 / len(loader.dataset))
    epoch_iou_withbackground = (epoch_iou_class0+epoch_iou_class1+epoch_iou_class2+epoch_iou_class3+epoch_iou_class4)/5
    epoch_iou_nobackground = (epoch_iou_class1 + epoch_iou_class2 + epoch_iou_class3 + epoch_iou_class4) / 4
    print("Testing accuracy score:",epoch_acc)
    print("Testing mean IoU withbackground:",epoch_iou_withbackground)
    print("Testing mean IoU withoutbackground:",epoch_iou_nobackground)
    print ("IoU of Background:", epoch_iou_class0)
    print ("IoU of ZP:", epoch_iou_class1)
    print ("IoU of TE:", epoch_iou_class2)
    print ("IoU of ICM:", epoch_iou_class3)
    print ("IoU of Blastocoel:", epoch_iou_class4)
    return epoch_loss, epoch_acc, epoch_iou_withbackground,epoch_iou_nobackground
    model.train()
