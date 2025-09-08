import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np

def train_model(model,train_loader,optimzer,criterion,device,clip=5):
    """
    Train the model for one epoch"""
    model.train()
    train_losses=[]
    train_acc=0.0
    for inputs,labels in tqdm(train_loader,desc="Training",leave=False):
        inputs,labels=inputs.to(device),labels.to(device)
        h = model.init_hidden(inputs.size(0), device) 
        if isinstance(h,tuple):
            h=tuple([each.data for each in h])
        else:
            h=h.data
        model.zero_grad()
        output,h=model(inputs,h)
        loss=criterion(output.squeeze(),labels.float())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),clip)
        optimzer.step()
        train_losses.append(loss.item())
        train_acc+=acc(output,labels)
    epoch_loss=np.mean(train_losses)
    epoch_acc=train_acc/len(train_loader.dataset)
    return epoch_loss,epoch_acc,h

def eval_model(model,val_loader,criterion,device):
    """Evaluate the model on validation set"""
    model.eval()
    val_losses=[]
    val_acc=0.0
    with torch.no_grad():
        for inputs,lables in tqdm(val_loader,desc="Validation",leave=False):
            inputs,labels=inputs.to(device),lables.to(device)
            h = model.init_hidden(inputs.size(0), device) 
            if isinstance(h,tuple):
                h=tuple([each.data for each in h])
            else:
                h=h.data
            output,h=model(inputs,h)
            loss=criterion(output.squeeze(),labels.float())
            val_losses.append(loss.item())
            val_acc+=acc(output,labels)
    epoch_loss=np.mean(val_losses)
    epoch_acc=val_acc/len(val_loader.dataset)
    return epoch_loss,epoch_acc,h

def acc(pred,labels):
    """Calculate accuracy
    """
    pred=torch.round(pred)
    correct=torch.sum(pred==labels).item()
    return correct
