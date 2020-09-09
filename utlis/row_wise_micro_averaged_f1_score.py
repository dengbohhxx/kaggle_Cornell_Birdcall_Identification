# -*- coding: utf-8 -*-
import torch
from sklearn.metrics import f1_score
import numpy as np

def row_wise_micro_averaged_f1_score(pred,label,threshold=0.5):
    batch_size=len(label)
    F1_score=0.
    pred=torch.nn.functional.sigmoid(pred)
    pred=pred.detach().cpu().numpy()
    pred=pred>=threshold
    pred=pred+0
    label=label.detach().cpu().numpy()
    for pred_row,label_row in zip(pred,label):
        F1_score+=f1_score(label_row,pred_row,average="binary")/batch_size
    return F1_score    

if __name__=='__main__':    
    pre=torch.tensor([[0.9,-0.1,-0.1,-0.1,-0.1,0.9,0.9,0.9],[0.9,-0.1,-0.1,-0.1,-0.1,0.9,0.9,0.9]])
    label=torch.tensor([[1,1,0,1,1,0,1,0],[1,1,0,1,1,0,1,0]])   
    f1=row_wise_micro_averaged_f1_score(pre,label,0.5)
    print(f1)
    TP=2
    FP=2
    FN=3
    TN=1
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    f1_score=2*precision*recall/(precision+recall)
    print(f1_score)