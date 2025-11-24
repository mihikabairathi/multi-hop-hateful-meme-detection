# heavily borrowed from: https://github.com/JingbiaoMei/RGCL/blob/main/src/utils/retrieval.py

import torch
import faiss
import faiss.contrib.torch_utils
import numpy as np

def dense_retrieve_hard_negatives_pseudo_positive(train_dl, query_feats, query_labels, model, args, train_feats=None, train_labels=None):
    Faiss_GPU = args.Faiss_GPU
    device = args.device
    hard_negatives_multiple = args.hard_negatives_multiple
    no_hard_negatives = args.no_hard_negatives
    no_pseudo_gold_positives = args.no_pseudo_gold_positives
    largest_retrieval = 1

    model.eval()
    batch_size = query_feats.shape[0]
    
    if not Faiss_GPU:
        query_feats = query_feats.cpu().detach().numpy().astype("float32")

    # If we set the train_feats and train_labels to None in upper level, reindex the search index with updated training data
    if train_feats == None or train_labels == None:
        for i, batch in enumerate(train_dl):
            image_feats = batch["image_feats"].to(device)
            text_feats = batch["text_feats"].to(device)
            face_feats = batch["face_feats"].to(device)

            # Image+Text+Face features after modality fusion
            _, all_feats = model(image_feats, text_feats, face_feats, return_embed=True)
            if i == 0:
                if Faiss_GPU:
                    train_feats = all_feats
                    train_labels = batch["labels"]
                else:
                    train_feats = all_feats.cpu().detach().numpy().astype("float32")
                    train_labels = batch["labels"].cpu().detach().numpy().astype("int")
            else:
                if Faiss_GPU:
                    train_feats = torch.cat((train_feats, all_feats), dim=0)
                    train_labels = torch.cat((train_labels, batch["labels"]), dim=0)
                else:
                    train_feats = np.concatenate((train_feats, all_feats.cpu().detach().numpy().astype("float32")))
                    train_labels = np.concatenate((train_labels, batch["labels"].cpu().detach().numpy().astype("int")))

    dim = train_feats.shape[1]
    index = faiss.IndexFlatIP(dim)

    if Faiss_GPU:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
        train_feats_normalized = torch.nn.functional.normalize(train_feats, p=2, dim=1)
        query_feats_normalized = torch.nn.functional.normalize(query_feats, p=2, dim=1)
    else:
        train_feats_normalized = train_feats
        query_feats_normalized = query_feats
        faiss.normalize_L2(train_feats_normalized)
        faiss.normalize_L2(query_feats_normalized)
    index.add(train_feats_normalized)

    # Search at most hardest_negatives_multiple of the largest retrieval no.
    D, I = index.search(query_feats_normalized, largest_retrieval*hard_negatives_multiple)

    hard_negative_features = torch.zeros(batch_size, no_hard_negatives, dim, device="cuda")
    pseudo_positive_features = torch.zeros(batch_size, no_pseudo_gold_positives, dim, device="cuda")
    hard_negative_scores = torch.zeros(batch_size, largest_retrieval, device="cuda")
    pseudo_positive_scores = torch.zeros(batch_size, no_pseudo_gold_positives, device="cuda")

    for i, row in enumerate(D):
        # Initialize the counter for the number of hard negatives, pseudo gold positives
        j, k = 0, 0
        for iter, value in enumerate(row):
            if train_labels[I[i, iter]].item() != query_labels[i].item() and j < no_hard_negatives:
                if Faiss_GPU:
                    hard_negative_features[i][j] = train_feats[I[i, iter]]
                    hard_negative_scores[i][j] = value
                else:
                    hard_negative_features[i][j] = torch.from_numpy(train_feats[I[i, iter]]).float().to("cuda")
                    hard_negative_scores[i][j] = torch.from_numpy(np.asarray(value)).float().to("cuda")
                j += 1
            elif train_labels[I[i, iter]].item() == query_labels[i].item() and k < no_pseudo_gold_positives:
                if Faiss_GPU:
                    pseudo_positive_features[i][k] = train_feats[I[i, iter]]
                    pseudo_positive_scores[i][k] = value
                else:
                    pseudo_positive_features[i][k] = torch.from_numpy(train_feats[I[i, iter]]).float().to("cuda")
                    pseudo_positive_scores[i][k] = torch.from_numpy(np.asarray(value)).float().to("cuda")
                k += 1

            # Only if both the number of hard negatives and pseudo gold positives are found, then break
            if j == largest_retrieval and k == no_pseudo_gold_positives:
                break
        
    return hard_negative_features, hard_negative_scores, pseudo_positive_features, pseudo_positive_scores, train_feats, train_labels