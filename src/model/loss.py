# heavily borrowed from: https://github.com/JingbiaoMei/RGCL/blob/main/src/model/loss.py
# TODO: see if we want to introduce a new kind of loss as well

import torch
import torch.nn as nn
from utils.retrieval import dense_retrieve_hard_negatives_pseudo_positive

def compute_loss(batch, train_dl, model, args, train_set=None, train_feats=None, train_labels=None):
    # read args params once (careful that args gets passed down the line to a helper method)
    device = args.device
    no_hard_negatives = args.no_hard_negatives
    triplet_margin = 0.1
    ce_weight = 0.5
    ids = batch["ids"]
    batch_size = len(ids)
    image_feats = batch["image_feats"].to(device)
    text_feats = batch["text_feats"].to(device)
    face_feats = batch["face_feats"].to(device)
    labels = batch["labels"].to(device)

    model.train()
    output, feats = model(image_feats, text_feats, face_feats, return_embed=True)

    # We construct a matrix for label coincidences (Mask matrix for later loss computation)
    # 1 if the labels are the same (positive), 0 otherwise (negative)
    # The dimension would be batch_size x batch_size
    # This is used for the in-batch positive/negative mining
    # We construct it by stacking rows of the labels then for ith row with label 0, we flip the label bit for whole row.
    # We can do this since, if the original label is 0 and the target label is 0, then we have in-batch positive (1);
    # if the target label is 1, then we have in-batch negative (0) - so we flip the label for 0.

    # We first construct the inverse label, i.e., binary NOT operator on the label
    labels = labels.bool()
    labels_inverse = ~labels
    label_matrix = torch.stack([labels if labels[i] == True else labels_inverse for i in range(batch_size)], axis=0)
    label_matrix_negative = (~label_matrix).int()

    # We then compute the number of in-batch positives and negatives per sample in the batch vectors of sizes batch_size
    # Since the matrix is symmetric, use which dimension does not matter
    in_batch_positives_no = torch.sum(label_matrix, dim=1) - 1
    in_batch_negative_no = batch_size - in_batch_positives_no - 1

    # We expand the feature matrix to a 3D tensor for vectorized computation
    # feats_expand Dimension: batch_size x feature_size x batch_size
    feats_expanded = feats.unsqueeze(2).expand(batch_size, -1, batch_size)

    # We compute the cosine similarity between each pair of features
    cos = nn.CosineSimilarity(dim=1, eps=1e-8)
    sim_matrix = cos(feats_expanded, feats_expanded.transpose(0, 2))
    sim_matrix.fill_diagonal_(0)

    # We compute the loss matrix by multiplying the similarity matrix
    in_batch_negative_loss = sim_matrix * label_matrix_negative

    # We set it to a matrix of zeros to make sure the contrastive loss can still use the same code
    in_batch_positives_loss = torch.zeros(batch_size, batch_size).to(device)

    # We compute the loss by summing over the loss matrix
    # Pick out the non-zero terms (gives 1), mask out the zero terms (gives 0)
    neg_mask = in_batch_negative_loss != 0
    neg_zero_count = (neg_mask == 0).sum(dim=1)

    # However, if all the terms are zero, we will get nan due to zero division,
    # We will form a further mask to only operate on the sample with at least one non-zero term
    neg_zero_count_zero_mask = torch.zeros(batch_size, device=device) != in_batch_negative_no

    in_batch_negative_loss_sum = torch.zeros(batch_size, device=device)
    in_batch_negative_loss_sum[neg_zero_count_zero_mask] = torch.sum(
        in_batch_negative_loss[neg_zero_count_zero_mask], dim=1) / neg_mask.sum(dim=1)[neg_zero_count_zero_mask]
    in_batch_loss = in_batch_negative_loss_sum

    # In default we will consider hard negative, which is key to the good performance. 
    # But if we want to test without hard negative, this is also fine
    # We can just ignore the hard negative features and scores
    (
        hard_negative_features, hard_negative_scores, pseudo_positive_features, pseudo_positive_scores, train_feats, train_labels,
    ) = dense_retrieve_hard_negatives_pseudo_positive(
        train_dl, feats, labels, model, args, train_feats=train_feats, train_labels=train_labels
    )

    # Now we have the hard negatives features, we compute the loss, similarity matrix
    # The dimension of hard_negative_features is batch_size x no_hard_negatives x dim
    # The dimension of original feats is batch_size x dim
    # We thus need to expand the original feats to batch_size x no_hard_negatives x embed_dim/hidden_dim
    feats_expanded = feats.unsqueeze(1).expand(batch_size, no_hard_negatives, -1)

    # For simplicity, we only check if the first dimension is zero in the feature embedding
    # The mask is batch_size x no_hard_negatives, 1 if embedding non zero, 0 if embedding zero,
    # Thus we can multiply the mask with the loss.
    zeroLoss_mask = torch.sum(hard_negative_features, dim=2) != 0

    # Compute loss as batch_size x no_hard_negatives
    cos_hard = nn.CosineSimilarity(dim=2, eps=1e-8)
    hard_loss = zeroLoss_mask * cos_hard(feats_expanded, hard_negative_features)
    hard_loss = torch.sum(hard_loss, dim=1)

    # Now we have the pseudo gold positive features, we compute the loss
    feats_expanded = feats.unsqueeze(1).expand(batch_size, 1, -1)

    # Compute loss as batch_size x no_pseudo_gold_positives (1)
    cos_pseudo_gold = nn.CosineSimilarity(dim=2, eps=1e-8)
    pseudo_gold_loss = cos_pseudo_gold(feats_expanded, pseudo_positive_features)
    pseudo_gold_loss = torch.mean(pseudo_gold_loss, dim=1)

    total_loss = torch.mean(torch.relu(in_batch_loss + hard_loss - pseudo_gold_loss + triplet_margin))
    lossFn_classifier = nn.BCEWithLogitsLoss()
    loss_classifier = lossFn_classifier(output, labels.float().reshape(-1, 1))
    total_loss = total_loss * (1-ce_weight) + loss_classifier * ce_weight

    return total_loss, torch.mean(in_batch_loss), torch.mean(hard_loss), torch.mean(pseudo_gold_loss), loss_classifier, train_feats, train_labels