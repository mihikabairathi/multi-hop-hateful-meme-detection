# heavily borrowed from: https://github.com/JingbiaoMei/RGCL/blob/main/src/run_rac.py

import torch.nn as nn
import torch
import argparse
import os
from model.evaluate import retrieve_evaluate_RAC
from model.classifier import FirstHopHateClipper
from model.loss import compute_loss
from utils.dataset import load_feats_from_CLIP, CLIP2Dataloader
from utils.metrics import eval_and_save_epoch_end, compute_metrics_retrieval
from tqdm import tqdm

def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--path", type=str, default="./data/")
    arg_parser.add_argument("--output_path", type=str, default="./logging/")
    arg_parser.add_argument("--model", type=str, default="openai_clip-vit-large-patch14-336_HF")
    arg_parser.add_argument("--similarity_threshold", type=float, default=-1.0)
    arg_parser.add_argument("--topk", type=int, default=20, help="Retrieve at most k pairs for validation")
    arg_parser.add_argument('--num_layers', type=int, default=3)
    arg_parser.add_argument('--num_layers_face', type=int, default=3)
    arg_parser.add_argument('--proj_dim', type=int, default=1024)
    arg_parser.add_argument('--map_dim', type=int, default=1024)
    arg_parser.add_argument('--dropout', type=float, nargs=3, default=[0.2, 0.4, 0.1])
    arg_parser.add_argument('--dropout_face', type=float, nargs=3, default=[0.2, 0.4, 0.1])
    arg_parser.add_argument("--epochs", type=int, default=30)
    arg_parser.add_argument("--batch_size", type=int, default=64)
    arg_parser.add_argument("--lr", type=float, default=0.0001)
    arg_parser.add_argument("--grad_clip", type=float, default=0.1, help="Gradient clipping")
    arg_parser.add_argument("--no_pseudo_gold_positives", type=int, default=1)
    arg_parser.add_argument("--in_batch_loss", type=lambda x: (str(x).lower() == "true"), default=True) 
    arg_parser.add_argument("--hard_negatives_loss", type=lambda x: (str(x).lower() == "true"), default=True)
    arg_parser.add_argument("--no_hard_negatives", type=int, default=1)
    arg_parser.add_argument("--no_hard_positives", type=int, default=0)
    arg_parser.add_argument("--hard_negatives_multiple", type=int, default=12)
    arg_parser.add_argument("--Faiss_GPU", type=lambda x: (str(x).lower() == "true"), default=True)
    arg_parser.add_argument("--seed", type=int, default=0)
    arg_parser.add_argument("--device", type=str, default="cuda")    
    args = arg_parser.parse_args()
    return args

def model_pass(train_dl, evaluate_dl, test_seen_dl, model, epochs, log_interval=10, args=None, train_set=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_acc = 0.0
    best_roc = 0.0
    best_epoch_path = None

    # After every epoch, we reindex the dense vector embeddings
    for epoch in tqdm(range(epochs)):
        # run training
        train_feats, train_labels = None, None # force the system to reindex the dense vector embeddings
        for step, batch in enumerate(train_dl):
            (total_loss, in_batch_loss, hard_loss, pseudo_gold_loss, cross_entropy, train_feats, train_labels) = compute_loss(
                batch, train_dl, model, args, train_set=train_set, train_feats=train_feats, train_labels=train_labels
            )
            train_feats = train_feats.detach()
            train_labels = train_labels.detach()
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            if step % log_interval == 0:
                print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch, step, len(train_dl), 100.0 * step / len(train_dl), total_loss.item()
                ))
                hard_loss_val = hard_loss.item() if args.hard_negatives_loss else 0
                in_batch_loss_val = in_batch_loss.item() if type(in_batch_loss) != int else in_batch_loss
                pseudo_gold_loss_val = pseudo_gold_loss.item()

        # run eval on dev set
        logging_dict, evaluate_labels = retrieve_evaluate_RAC(
            train_dl, evaluate_dl, model, epoch, largest_retrieval=args.topk, threshold=args.similarity_threshold,
            args=args, eval_name="dev"
        )
        acc, roc, pre, recall, f1, prediction, labels = compute_metrics_retrieval(
            logging_dict, evaluate_labels, topk=args.topk, use_sim=True
        )

        # run eval on seen test set
        logging_dict_test, test_labels = retrieve_evaluate_RAC(
            train_dl, test_seen_dl, model, epoch, largest_retrieval=args.topk, threshold=args.similarity_threshold,
            args=args, eval_name="test"
        )
        acc_test, roc_test, pre_test, recall_test, f1_test, prediction, labels = compute_metrics_retrieval(
            logging_dict_test, test_labels, topk=args.topk, use_sim=True
        )
            
        # logging at the end of each epoch
        (acc_, roc_, pre_, recall_, f1_, eval_loss_), _ = eval_and_save_epoch_end(
            args.device, train_dl, evaluate_dl, test_seen_dl, model, epoch
        )

        # Print out the summary of the epoch
        print("Val_Retrieval Epoch {} acc: {:.4f} roc: {:.4f} pre: {:.4f} recall: {:.4f} f1: {:.4f}".format(
            epoch, acc, roc, pre, recall, f1)
        )
        print("Test_Retrieval Epoch {} acc: {:.4f} roc: {:.4f} pre: {:.4f} recall: {:.4f} f1: {:.4f}\n".format(
            epoch, acc_test, roc_test, pre_test, recall_test, f1_test)
        )

        # Save the model if the val criterion is the best so far
        if acc_ > best_acc:
            print("Current Epoch Acc: ", acc_, "Best model so far, saving...")
            best_acc = acc_
            best_epoch_path = args.output_path + "/ckpt/best_model_{}_{}.pt".format(epoch, str(acc_))
            torch.save(model.state_dict(), best_epoch_path)
        if epoch == args.epochs - 1:
            print("Last Epoch, saving...")
            torch.save(model.state_dict(), args.output_path + "/ckpt/last_model_{}_{}.pt".format(epoch, acc))

    return model, best_epoch_path

def main(args):
    # Create the logging directory
    hard_negative_name = "_hard_negative_{}".format(args.no_hard_negatives)
    if args.no_pseudo_gold_positives!=0 and args.no_hard_positives !=0:
        positive_name = f"_PseudoGold_positive_{args.no_pseudo_gold_positives}_hard_positive_{args.no_hard_positives}"
    elif args.no_pseudo_gold_positives!=0:
        positive_name = f"_PseudoGold_positive_{args.no_pseudo_gold_positives}"
    elif args.no_hard_positives !=0:
        positive_name = f"_hard_positive_{args.no_hard_positives}"
    else:
        positive_name = "inbatch_positive"
 
    exp_name = "RAC_lr{}_Bz{}_Ep{}_cosSim_triplet_drop{}_topK{}_{}{}_seed{}_hybrid_loss".format(
        args.lr, args.batch_size, args.epochs, args.dropout, args.topk, positive_name, hard_negative_name, args.seed
    )

    args.output_path = os.path.join(args.output_path, "Retrieval", "FB", "RAC", exp_name)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        os.makedirs(args.output_path + "/ckpt/")

    # Load precomputed CLIP embeddings
    train, dev, test_seen, test_unseen = load_feats_from_CLIP(os.path.join(args.path, "CLIP_Embedding"), args.model)
    (train_dl, dev_dl, test_seen_dl), (train_set, _, _) = CLIP2Dataloader(train, dev, test_seen, batch_size=args.batch_size)

    # Construct the model
    image_feat_dim = list(enumerate(train_dl))[0][1]["image_feats"].shape[1]
    text_feat_dim = list(enumerate(train_dl))[0][1]["text_feats"].shape[1]
    face_feat_dim = list(enumerate(train_dl))[0][1]["face_feats"].shape[1]
    model = FirstHopHateClipper(image_feat_dim, text_feat_dim, face_feat_dim, args.num_layers, args.proj_dim, args.map_dim, args.dropout, 
        args.num_layers_face, args.dropout_face)
    model.to(args.device)
    print(model)

    # Train the model
    model, best_epoch_path = model_pass(train_dl, dev_dl, test_seen_dl, model, args.epochs, args=args, train_set=train_set)

if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    main(args)