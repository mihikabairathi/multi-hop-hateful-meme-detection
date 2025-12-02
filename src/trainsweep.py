# heavily borrowed from: https://github.com/JingbiaoMei/RGCL/blob/main/src/run_rac.py

import torch.nn as nn
import torch
import argparse
import os
import numpy as np

from model.evaluate import retrieve_evaluate_RAC
from model.classifier import MultiHopMemeClassifier
from model.loss import compute_loss
from utils.dataset import load_feats_from_CLIP, CLIP2Dataloader
from utils.metrics import eval_and_save_epoch_end, compute_metrics_retrieval
from tqdm import tqdm


def parse_args():
    arg_parser = argparse.ArgumentParser()
    # paths / model
    arg_parser.add_argument("--path", type=str, default="./data/")
    arg_parser.add_argument("--output_path", type=str, default="./logging/")
    arg_parser.add_argument("--model", type=str, default="openai_clip-vit-large-patch14-336_HF")

    # retrieval settings
    arg_parser.add_argument("--similarity_threshold", type=float, default=-1.0)
    arg_parser.add_argument("--topk", type=int, default=20, help="Retrieve at most k pairs for validation")

    # classifier architecture
    arg_parser.add_argument('--num_layers', type=int, default=3)
    arg_parser.add_argument('--num_layers_face', type=int, default=3)
    arg_parser.add_argument('--proj_dim', type=int, default=1024)
    arg_parser.add_argument('--map_dim', type=int, default=1024)
    arg_parser.add_argument('--dropout', type=float, nargs=3, default=[0.2, 0.4, 0.1])
    arg_parser.add_argument('--dropout_face', type=float, nargs=3, default=[0.2, 0.4, 0.1])

    # training hyperparams
    arg_parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    arg_parser.add_argument("--batch_size", type=int, default=64, help="Mini-batch size")
    arg_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    arg_parser.add_argument("--grad_clip", type=float, default=0.1, help="Gradient clipping")
    arg_parser.add_argument("--weight_decay", type=float, default=1e-2, help="AdamW weight decay")

    # retrieval / hybrid loss controls (same semantics as original)
    arg_parser.add_argument("--no_pseudo_gold_positives", type=int, default=1)
    arg_parser.add_argument("--in_batch_loss", type=lambda x: (str(x).lower() == "true"), default=True)
    arg_parser.add_argument("--hard_negatives_loss", type=lambda x: (str(x).lower() == "true"), default=True)
    arg_parser.add_argument("--no_hard_negatives", type=int, default=1)
    arg_parser.add_argument("--no_hard_positives", type=int, default=0)
    arg_parser.add_argument("--hard_negatives_multiple", type=int, default=12)
    arg_parser.add_argument("--Faiss_GPU", type=lambda x: (str(x).lower() == "true"), default=True)

    # fixed loss weights (these will *always* be used)
    arg_parser.add_argument("--w_inbatch", type=float, default=0.5, help="Weight for in-batch retrieval loss")
    arg_parser.add_argument("--w_hard", type=float, default=0.5, help="Weight for hard-negative loss")
    arg_parser.add_argument("--w_pseudo", type=float, default=0.5, help="Weight for pseudo-gold loss")
    arg_parser.add_argument("--w_ce", type=float, default=2.0, help="Weight for cross-entropy (classification) loss")

    # threshold sweep settings
    arg_parser.add_argument("--threshold_min", type=float, default=-1.0)
    arg_parser.add_argument("--threshold_max", type=float, default=1.0)
    arg_parser.add_argument("--threshold_steps", type=int, default=41)

    # misc
    arg_parser.add_argument("--seed", type=int, default=0)
    arg_parser.add_argument("--device", type=str, default="cuda")

    return arg_parser.parse_args()


def sweep_retrieval_thresholds(train_dl, eval_dl, model, args, split_name="dev"):
    """
    Sweep similarity_threshold over [threshold_min, threshold_max] and
    pick the threshold that maximizes F1 on the given split.
    """
    thresholds = np.linspace(args.threshold_min, args.threshold_max, args.threshold_steps)
    best_thr = None
    best_f1 = -1.0
    best_metrics = None

    print(f"\n[Threshold sweep] split={split_name} range=({args.threshold_min}, {args.threshold_max}) "
          f"steps={args.threshold_steps}")

    for thr in thresholds:
        logging_dict, labels = retrieve_evaluate_RAC(
            train_dl,
            eval_dl,
            model,
            epoch=-1,
            largest_retrieval=args.topk,
            threshold=float(thr),
            args=args,
            eval_name=f"{split_name}_thr_{thr:.3f}",
        )
        acc, roc, pre, recall, f1, _, _ = compute_metrics_retrieval(
            logging_dict, labels, topk=args.topk, use_sim=True
        )
        print(f"  thr={thr:.4f} acc={acc:.4f} roc={roc:.4f} pre={pre:.4f} "
              f"recall={recall:.4f} f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
            best_metrics = (acc, roc, pre, recall, f1)

    if best_thr is not None:
        acc, roc, pre, recall, f1 = best_metrics
        print(f"\n[Best threshold on {split_name}] thr={best_thr:.4f} "
              f"acc={acc:.4f} roc={roc:.4f} pre={pre:.4f} recall={recall:.4f} f1={f1:.4f}")
    else:
        print(f"\n[Threshold sweep] No valid thresholds evaluated on {split_name}.")

    return best_thr, best_metrics


def eval_at_threshold(train_dl, eval_dl, model, args, thr, split_name="test"):
    """
    Evaluate retrieval metrics at a fixed threshold on a given split.
    """
    logging_dict, labels = retrieve_evaluate_RAC(
        train_dl,
        eval_dl,
        model,
        epoch=-1,
        largest_retrieval=args.topk,
        threshold=float(thr),
        args=args,
        eval_name=f"{split_name}_thr_{thr:.3f}",
    )
    acc, roc, pre, recall, f1, _, _ = compute_metrics_retrieval(
        logging_dict, labels, topk=args.topk, use_sim=True
    )
    print(f"[{split_name} @ best dev thr={thr:.4f}] "
          f"acc={acc:.4f} roc={roc:.4f} pre={pre:.4f} recall={recall:.4f} f1={f1:.4f}")
    return acc, roc, pre, recall, f1


def model_pass(train_dl, evaluate_dl, test_seen_dl, model, epochs, log_interval=10, args=None, train_set=None):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    best_acc = 0.0
    best_roc = 0.0
    best_epoch_path = None

    # CSV metrics logging
    metrics_path = os.path.join(args.output_path, "metrics")
    os.makedirs(metrics_path, exist_ok=True)
    csv_path = os.path.join(metrics_path, "epoch_metrics.csv")
    with open(csv_path, "w") as f:
        f.write(",".join([
            "epoch",
            "train_total",
            "train_inbatch",
            "train_hardneg",
            "train_pseudogold",
            "train_ce",
            "dev_acc", "dev_roc", "dev_pre", "dev_recall", "dev_f1",
            "test_acc", "test_roc", "test_pre", "test_recall", "test_f1",
            "eval_loss"
        ]) + "\n")

    for epoch in tqdm(range(epochs)):
        # ----- training -----
        model.train()
        train_feats, train_labels = None, None

        train_total_accum = 0.0
        train_inbatch_accum = 0.0
        train_hard_accum = 0.0
        train_pseudo_accum = 0.0
        train_ce_accum = 0.0
        batches = 0

        for step, batch in enumerate(train_dl):
            (
                base_total_loss,
                in_batch_loss,
                hard_loss,
                pseudo_gold_loss,
                cross_entropy,
                train_feats,
                train_labels,
            ) = compute_loss(
                batch,
                train_dl,
                model,
                args,
                train_set=train_set,
                train_feats=train_feats,
                train_labels=train_labels,
            )

            # --- ALWAYS REBALANCE LOSS HERE ---
            loss_terms = []

            if args.in_batch_loss and not isinstance(in_batch_loss, int):
                loss_terms.append(args.w_inbatch * in_batch_loss)

            if args.hard_negatives_loss and not isinstance(hard_loss, int):
                loss_terms.append(args.w_hard * hard_loss)

            if args.no_pseudo_gold_positives != 0 and pseudo_gold_loss is not None and not isinstance(pseudo_gold_loss, int):
                loss_terms.append(args.w_pseudo * pseudo_gold_loss)

            if not isinstance(cross_entropy, int):
                loss_terms.append(args.w_ce * cross_entropy)

            if len(loss_terms) > 0:
                total_loss = sum(loss_terms)
            else:
                total_loss = base_total_loss  # fallback
            # -----------------------------------

            train_feats = train_feats.detach()
            train_labels = train_labels.detach()

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            # accumulate training losses (for logging)
            train_total_accum += float(total_loss.item())
            if not isinstance(in_batch_loss, int):
                train_inbatch_accum += float(in_batch_loss.item())
            if args.hard_negatives_loss and not isinstance(hard_loss, int):
                train_hard_accum += float(hard_loss.item())
            if pseudo_gold_loss is not None and not isinstance(pseudo_gold_loss, int):
                train_pseudo_accum += float(pseudo_gold_loss.item())
            if not isinstance(cross_entropy, int):
                train_ce_accum += float(cross_entropy.item())
            batches += 1

            if step % log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch, step, len(train_dl), 100.0 * step / len(train_dl), total_loss.item()
                    )
                )
                hard_loss_val = hard_loss.item() if args.hard_negatives_loss and not isinstance(hard_loss, int) else 0
                in_batch_loss_val = in_batch_loss.item() if not isinstance(in_batch_loss, int) else float(in_batch_loss)
                pseudo_gold_loss_val = (
                    pseudo_gold_loss.item()
                    if pseudo_gold_loss is not None and not isinstance(pseudo_gold_loss, int)
                    else 0
                )
                ce_val = cross_entropy.item() if not isinstance(cross_entropy, int) else float(cross_entropy)
                print(
                    "  in-batch loss: {:.6f}, hard-neg loss: {:.6f}, pseudo-gold loss: {:.6f}, CE: {:.6f}".format(
                        float(in_batch_loss_val),
                        float(hard_loss_val),
                        float(pseudo_gold_loss_val),
                        float(ce_val),
                    )
                )

        # epoch-averaged train losses
        if batches > 0:
            train_total_avg = train_total_accum / batches
            train_inbatch_avg = train_inbatch_accum / batches
            train_hard_avg = train_hard_accum / batches
            train_pseudo_avg = train_pseudo_accum / batches
            train_ce_avg = train_ce_accum / batches
        else:
            train_total_avg = train_inbatch_avg = train_hard_avg = train_pseudo_avg = train_ce_avg = 0.0

        # ----- evaluation -----
        model.eval()
        # retrieval-style dev eval (using current args.similarity_threshold)
        logging_dict, evaluate_labels = retrieve_evaluate_RAC(
            train_dl,
            evaluate_dl,
            model,
            epoch,
            largest_retrieval=args.topk,
            threshold=args.similarity_threshold,
            args=args,
            eval_name="dev",
        )
        acc, roc, pre, recall, f1, _, _ = compute_metrics_retrieval(
            logging_dict, evaluate_labels, topk=args.topk, use_sim=True
        )

        # retrieval-style test eval
        logging_dict_test, test_labels = retrieve_evaluate_RAC(
            train_dl,
            test_seen_dl,
            model,
            epoch,
            largest_retrieval=args.topk,
            threshold=args.similarity_threshold,
            args=args,
            eval_name="test",
        )
        acc_test, roc_test, pre_test, recall_test, f1_test, _, _ = compute_metrics_retrieval(
            logging_dict_test, test_labels, topk=args.topk, use_sim=True
        )

        # classification-style eval (whatever eval_and_save_epoch_end does)
        (acc_, roc_, pre_, recall_, f1_, eval_loss_), _ = eval_and_save_epoch_end(
            args.device, train_dl, evaluate_dl, test_seen_dl, model, epoch
        )

        print(
            "Val_Retrieval Epoch {} acc: {:.4f} roc: {:.4f} pre: {:.4f} recall: {:.4f} f1: {:.4f}".format(
                epoch, acc, roc, pre, recall, f1
            )
        )
        print(
            "Test_Retrieval Epoch {} acc: {:.4f} roc: {:.4f} pre: {:.4f} recall: {:.4f} f1: {:.4f}".format(
                epoch, acc_test, roc_test, pre_test, recall_test, f1_test
            )
        )

        # log to CSV
        with open(csv_path, "a") as f:
            f.write(
                ",".join(
                    [
                        str(epoch),
                        f"{train_total_avg:.6f}",
                        f"{train_inbatch_avg:.6f}",
                        f"{train_hard_avg:.6f}",
                        f"{train_pseudo_avg:.6f}",
                        f"{train_ce_avg:.6f}",
                        f"{acc:.6f}",
                        f"{roc:.6f}",
                        f"{pre:.6f}",
                        f"{recall:.6f}",
                        f"{f1:.6f}",
                        f"{acc_test:.6f}",
                        f"{roc_test:.6f}",
                        f"{pre_test:.6f}",
                        f"{recall_test:.6f}",
                        f"{f1_test:.6f}",
                        f"{eval_loss_:.6f}",
                    ]
                )
                + "\n"
            )

        # save best model by dev classification acc
        if acc_ > best_acc:
            print("Current Epoch Acc: ", acc_, "Best model so far, saving...")
            best_acc = acc_
            best_epoch_path = os.path.join(
                args.output_path, "ckpt", f"best_model_{epoch}_{acc_}.pt"
            )
            torch.save(model.state_dict(), best_epoch_path)
        if roc_ > best_roc:
            best_roc = roc_

        if epoch == args.epochs - 1:
            print("Last Epoch, saving...")
            last_path = os.path.join(args.output_path, "ckpt", f"last_model_{epoch}_{acc}.pt")
            torch.save(model.state_dict(), last_path)

    # ----- threshold sweep after training -----
    best_thr, dev_best_metrics = sweep_retrieval_thresholds(train_dl, evaluate_dl, model, args, split_name="dev")
    if best_thr is not None:
        eval_at_threshold(train_dl, test_seen_dl, model, args, best_thr, split_name="test")

    return model, best_epoch_path


def main(args):
    # experiment name and output dirs
    hard_negative_name = "_hard_negative_{}".format(args.no_hard_negatives)
    if args.no_pseudo_gold_positives != 0 and args.no_hard_positives != 0:
        positive_name = (
            f"_PseudoGold_positive_{args.no_pseudo_gold_positives}_hard_positive_{args.no_hard_positives}"
        )
    elif args.no_pseudo_gold_positives != 0:
        positive_name = f"_PseudoGold_positive_{args.no_pseudo_gold_positives}"
    elif args.no_hard_positives != 0:
        positive_name = f"_hard_positive_{args.no_hard_positives}"
    else:
        positive_name = "inbatch_positive"

    exp_name = "RAC_lr{}_Bz{}_Ep{}_drop{}_topK{}_{}{}_seed{}_hybrid_loss".format(
        args.lr,
        args.batch_size,
        args.epochs,
        args.dropout,
        args.topk,
        positive_name,
        hard_negative_name,
        args.seed,
    )

    args.output_path = os.path.join(args.output_path, "Retrieval", "FB", "RAC", exp_name)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        os.makedirs(os.path.join(args.output_path, "ckpt"), exist_ok=True)

    # load embeddings
    train, dev, test_seen, test_unseen = load_feats_from_CLIP(
        os.path.join(args.path, "CLIP_Embedding"), args.model
    )
    (train_dl, dev_dl, test_seen_dl), (train_set, _, _) = CLIP2Dataloader(
        train, dev, test_seen, batch_size=args.batch_size
    )

    # dims
    first_batch = next(iter(train_dl))
    image_feat_dim = first_batch["image_feats"].shape[1]
    text_feat_dim = first_batch["text_feats"].shape[1]
    face_feat_dim = first_batch["face_feats"].shape[1]

    # model
    model = MultiHopMemeClassifier(
        image_dim=image_feat_dim,
        text_dim=text_feat_dim,
        face_dim=face_feat_dim,
        num_layers=args.num_layers,
        proj_dim=args.proj_dim,
        map_dim=args.map_dim,
        dropout=args.dropout,
        num_layers_face=args.num_layers_face,
        dropout_face=args.dropout_face,
    )
    model.to(args.device)
    print(model)

    # train
    model, best_epoch_path = model_pass(
        train_dl, dev_dl, test_seen_dl, model, args.epochs, args=args, train_set=train_set
    )


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    main(args)
