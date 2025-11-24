# heavily borrowed from: https://github.com/JingbiaoMei/RGCL/blob/main/src/utils/metrics.py

import torch
import torch.nn as nn
import numpy as np
import torchmetrics
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn import metrics

def compute_metrics_retrieval(logging_dict, labels, topk=0, use_sim=False):
    list_majority_voted = []
    list_majority_voted_prob = []
    weight = np.arange(1, topk+1)[::-1]

    if not use_sim:
        for key, value in logging_dict.items():
            retrieved_labels = value["retrieved_label"]
            length = len(retrieved_labels)
            
            # taking the :length since we have both topk and threshold to decide the length
            list_majority_voted.append(np.sum(np.array(retrieved_labels)*weight[:length])/np.sum(weight[:length]))
    else:
        for key, value in logging_dict.items():
            retrieved_labels = value["retrieved_label"]
            retrieved_sims = value["retrieved_scores"]
            retrieved_sims = np.array([sim.item() for sim in retrieved_sims])
            retrieved_labels_map = np.array(retrieved_labels)*2-1
            retrieved_labels_map = retrieved_labels_map*retrieved_sims
            length = len(retrieved_labels_map)

            # taking the :length since we have both topk and threshold to decide the length
            list_majority_voted.append(np.sum(np.array(retrieved_labels_map)*weight[:length])/np.sum(weight[:length])) 

    try:
        labels = labels.detach().cpu().numpy()
    except:
        labels = labels

    if not use_sim:
        list_majority_voted_round = (np.array(list_majority_voted)>=0.5)*1
    else:
        def sigmoid(z):
            return 1/(1 + np.exp(-z))
        list_majority_voted_round = (sigmoid(np.array(list_majority_voted))>=0.5)*1

    acc = np.mean(list_majority_voted_round == labels)
    roc = roc_auc_score(labels, list_majority_voted)
    pre = precision_score(labels, list_majority_voted_round)
    recall = recall_score(labels, list_majority_voted_round)
    f1 = f1_score(labels, list_majority_voted_round)    
    
    return acc, roc, pre, recall, f1, list_majority_voted, labels

def iterate_dl(device, dl, classifier):
    with torch.no_grad():
        ids = []
        for step, batch in enumerate(dl):
            ids.extend(batch["ids"])
            if step == 0:
                labels = batch["labels"].detach().cpu()
                predicted, embed = classifier(batch["image_feats"].to(device),batch["text_feats"].to(device),batch["face_feats"].to(device),return_embed=True)
                predicted = predicted.detach().cpu()
                embed = embed.detach().cpu()
            else:
                labels = torch.cat((labels, batch["labels"].detach().cpu()), dim=0)
                new_pred, new_embed = classifier(batch["image_feats"].to(device),batch["text_feats"].to(device),batch["face_feats"].to(device),return_embed=True)
                predicted = torch.cat((predicted, new_pred.detach().cpu()), dim=0)
                embed = torch.cat((embed, new_embed.detach().cpu()), dim=0)
    return ids, labels, predicted, embed

def eval_metrics(labels, predicted, name, epoch, compute_loss=True):
    if len(labels.shape) == 1:
        labels = labels.unsqueeze(1)
    
    preds_proxy = torch.sigmoid(predicted)
    preds = (preds_proxy >= 0.5).long()

    acc = torchmetrics.Accuracy(task='binary')(preds, labels)
    roc = torchmetrics.AUROC(task='binary')(preds_proxy, labels)
    pre = torchmetrics.Precision(task='binary')(preds, labels)
    recall = torchmetrics.Recall(task='binary')(preds, labels)
    f1 = torchmetrics.F1Score(task='binary')(preds, labels)
    
    if compute_loss:
        loss = nn.BCEWithLogitsLoss()(predicted, labels.float())
        print("{}  Epoch {} acc: {:.4f} roc: {:.4f} pre: {:.4f} recall: {:.4f} f1: {:.4f} loss: {:.4f} ".format(name, epoch, acc, roc, pre, recall, f1, loss.item()))    
    else:
        loss = None
        print("{} Epoch {} acc: {:.4f} roc: {:.4f} pre: {:.4f} recall: {:.4f} f1: {:.4f}".format(name, epoch, acc, roc, pre, recall, f1))  
    return acc, roc, pre, recall, f1, loss

def eval_and_save_epoch_end(device, train_dl, dev_dl, test_dl, classifier, epoch):
    classifier.eval()
        
    dev_ids, dev_labels, dev_predicted, dev_embed = iterate_dl(device, dev_dl, classifier)
    dev_acc, dev_roc, dev_pre, dev_recall, dev_f1, loss = eval_metrics(dev_labels, dev_predicted, "dev", epoch, compute_loss=True)
    test_ids, test_labels, test_predicted, _ = iterate_dl(device, test_dl, classifier)
    test_acc, test_roc, test_pre, test_recall, test_f1, _ = eval_metrics(test_labels, test_predicted, "test", epoch, compute_loss=False) 
        
    return (dev_acc, dev_roc, dev_pre, dev_recall, dev_f1, loss), (test_acc, test_roc, test_pre, test_recall, test_f1)  