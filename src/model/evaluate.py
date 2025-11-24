# heavily borrowed from: https://github.com/JingbiaoMei/RGCL/blob/main/src/model/evaluate_rac.py

import sys
sys.path.append("../")

import torch
import torch.nn as nn
import faiss
import numpy as np
from easydict import EasyDict
from utils.metrics import compute_metrics_retrieval

def retrieve_evaluate_RAC(train_dl, evaluate_dl, model, epoch, largest_retrieval=100, threshold=0.5, args=None, eval_name=None):
    model.eval()

    # train set
    train_ids = []
    train_dl_is_list = False
    if type(train_dl) == list:
        train_dl_rest = train_dl[1:]
        train_dl = train_dl[0]
        train_dl_is_list = True

    for i, batch in enumerate(train_dl):
        train_ids.extend(batch["ids"])
        out, all_feats = model(batch["image_feats"].to('cuda'), batch["text_feats"].to('cuda'), return_embed=True)
        if i == 0:
            train_feats = all_feats
            train_labels = batch["labels"]
            train_out = out
        else:
            train_feats = torch.cat((train_feats, all_feats), dim=0)
            train_labels = torch.cat((train_labels, batch["labels"]), dim=0)
            train_out = torch.cat((train_out, out), dim=0)

    if train_dl_is_list:
        for train_dl_ in train_dl_rest:
            for batch in train_dl_:
                train_ids.extend(batch["ids"])
                out, all_feats = model(batch["image_feats"].to('cuda'), batch["text_feats"].to('cuda'), return_embed=True)
                train_feats = torch.cat((train_feats, all_feats), dim=0)
                train_labels = torch.cat((train_labels, batch["labels"]), dim=0)
                train_out = torch.cat((train_out, out), dim=0)

    # evaluation set
    evaluate_ids = []
    evaluate_feats = np.array([[]])
    evaluate_labels = np.array([[]])
    for i, batch in enumerate(evaluate_dl):
        evaluate_ids.extend(batch["ids"])
        out, all_feats = model(batch["image_feats"].to('cuda'), batch["text_feats"].to('cuda'), return_embed=True)
        if i == 0:
            evaluate_feats = all_feats
            evaluate_labels = batch["labels"]
            eval_out = out
        else:
            evaluate_feats = torch.cat((evaluate_feats, all_feats), dim=0)
            evaluate_labels = torch.cat((evaluate_labels, batch["labels"]), dim=0)
            eval_out = torch.cat((eval_out, out), dim=0)

    # Initialize the index
    dim = all_feats.shape[1]
    index = faiss.IndexFlatIP(dim)
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index)
    train_feats = torch.nn.functional.normalize(train_feats, p=2, dim=1)
    evaluate_feats = torch.nn.functional.normalize(evaluate_feats, p=2, dim=1)
    index.add(train_feats)
    D, I = index.search(evaluate_feats, largest_retrieval)

    # create logging dict
    logging_dict = EasyDict()
    for i, row in enumerate(D):
        retrieved_ids, retrieved_scores, retrieved_label, retrieved_out = [], [], [], []
        for j, value in enumerate(row):
            if j == 0:
                retrieved_ids.append(train_ids[I[i, j]])
                retrieved_scores.append(value)
                retrieved_label.append(train_labels[I[i, j]].item())
                retrieved_out.append(train_out[I[i, j]].cpu().detach())
            else:
                if value < threshold or threshold == -1:
                    # for the temp list, we use the image ids rather than the ordered number
                    retrieved_ids.append(train_ids[I[i, j]])
                    retrieved_scores.append(value)
                    retrieved_label.append(train_labels[I[i, j]].item())
                    retrieved_out.append(train_out[I[i, j]].cpu().detach())
                else:
                    break
        # Record the number of images retrieved for each query
        no_retrieved = len(retrieved_ids)
        logging_dict[evaluate_ids[i]] = {
            "no_retrieved": no_retrieved,
            "retrieved_ids": retrieved_ids,
            "retrieved_scores": retrieved_scores,
            "retrieved_label": retrieved_label,
            "retrieved_out": torch.cat(retrieved_out),
            "eval_out": eval_out[i].cpu().detach(),
        }

    return logging_dict, evaluate_labels