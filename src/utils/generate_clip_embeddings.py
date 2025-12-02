# heavily borrowed from: https://github.com/JingbiaoMei/RGCL/blob/main/src/utils/generate_CLIP_embedding_HF.py
# TODO: fix to add "face_feats" to the tf files

import argparse
import sys
import os
import torch
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPVisionModel, CLIPTokenizer, CLIPTextModel
from dataset import get_dataloader

# This script generates CLIP CLS embeddings and the last hidden state of the model
# Last hidden state represents the token embedding for the texts and the patch embedding for the images

def parse_args_sys(args_list=None):
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument("--EXP_FOLDER", type=str, default="./data/CLIP_Embedding", help="The path to save results.")
    arg_parser.add_argument("--model", type=str, default="openai/clip-vit-large-patch14-336", help="The clip model to use")
    return arg_parser.parse_args()

# This function extract both the last hidden state and the CLS token from the CLIP model
def extract_clip_features_HF(dataloader, device, vision_model, text_model, preprocess, tokenizer, args=None):
    all_image_features = [torch.zeros(1), torch.zeros(1)]
    all_text_features = torch.empty(3,3)
    pooler_image_features, all_labels, pooler_text_features, all_ids = [], [], [], []
    with torch.no_grad():
        for images, texts, labels, ids in tqdm(dataloader):
            texts = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            features = vision_model(**images)
            text_features = text_model(**texts.to(device))
            pooler_image_features.append(features.pooler_output.detach().cpu())
            pooler_text_features.append(text_features.pooler_output.detach().cpu())
            all_labels.append(labels)
            all_ids.append(ids)

    return (
        torch.cat(all_image_features), all_text_features, torch.cat(pooler_image_features),
        torch.cat(pooler_text_features), torch.cat(all_labels), all_ids
    )


def main(args):
    dataset = "FB"
    if os.path.exists("{}/{}".format(args.EXP_FOLDER, dataset)) == False:
        os.makedirs("{}/{}".format(args.EXP_FOLDER, dataset))
    
    Vision_model = CLIPVisionModel.from_pretrained(args.model)
    Text_model = CLIPTextModel.from_pretrained(args.model)
    preprocess = CLIPProcessor.from_pretrained(args.model)
    tokenizer = CLIPTokenizer.from_pretrained(args.model)
    if device == "cuda":
        Vision_model.cuda().eval()
        Text_model.cuda().eval()
    else:
        Vision_model.eval()
        Text_model.eval()

    train, dev_seen, test_seen, test_unseen = get_dataloader(preprocess, batch_size=32, num_workers=24, train_batch_size=32, image_size=336)
    loader_list = [train, dev_seen, test_seen, test_unseen]
    name_list = ["train", "dev_seen", "test_seen", "test_unseen"]

    for loader, name in zip(loader_list, name_list):
        (all_img_feats, all_text_feats, pooler_img_feats, pooler_text_feats, labels, ids) = \
            extract_clip_features_HF(loader, device, Vision_model, Text_model, preprocess, tokenizer)
        torch.save(
            {"ids": ids, "img_feats": pooler_img_feats, "text_feats": pooler_text_feats, "labels": labels},
            "{}/{}/{}_{}_HF.pt".format(args.EXP_FOLDER, dataset, name, str(args.model).replace("/", "_")),
        )

if __name__ == "__main__":
    sys.path.append('./src')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    main(parse_args_sys())