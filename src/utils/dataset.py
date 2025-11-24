# heavily borrowed from: https://github.com/JingbiaoMei/RGCL/blob/main/src/data_loader/dataset.py

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import torch
import pandas as pd

def to_device(data, device):
    if isinstance(data, (str)):
        return data
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    return data.to(device)

class DeviceDataLoader:
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)

class ImageTextDataset(Dataset):
    def __init__(self, img_data, preprocess, device="cuda", image_size=224):
        list_image_path, list_text, list_label, list_ids = img_data
        self.image_path = list_image_path
        self.text = list_text
        self.label = list_label
        self.list_ids = list_ids
        self.preprocess = preprocess
        self.device = device
        self.image_size = image_size

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        image = self.preprocess(images=Image.open(self.image_path[idx]).convert('RGB').resize((self.image_size, self.image_size)), return_tensors="pt")
        image["pixel_values"] = image["pixel_values"].squeeze()
        text = self.text[idx]
        label = self.label[idx]
        label = torch.tensor(label)
        return image, text, label, self.list_ids[idx] 

class RACDataset(Dataset):
    def __init__(self, feats, ids, labels):
        self.image_feats = feats[0]
        self.text_feats = feats[1]
        self.face_feats = feats[2]
        self.ids = ids
        self.labels = labels

    def __getitem__(self, index):
        return {"ids": self.ids[index], "image_feats": self.image_feats[index], "text_feats": self.text_feats[index], "face_feats": self.face_feats[index], "labels": self.labels[index]}

    def __len__(self):
        return len(self.ids)

def CLIP2Dataloader(*datasets, batch_size=128):
    dataloader_list = []
    dataset_list = []
    for index, dataset in enumerate(datasets):
        ids,  img_feats, text_feats, face_feats, labels = dataset
        feats = (img_feats.float(), text_feats.float(), face_feats.float())
        dataset = RACDataset(feats, ids, labels)
        dataset_list.append(dataset)
        if index == 0:
            # For training set, shuffle the data
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        else:
            # For validation set, do not shuffle the data, batch size is 4 times larger than training set
            dataloader = DataLoader(dataset, batch_size=batch_size*4, shuffle=False, num_workers=0)
        dataloader_list.append(dataloader)
    return dataloader_list, dataset_list

# Called from generate_clip_embeddings
def get_values_from_gt(split):
    gt_df = pd.read_json(f"./data/gt/FB/{split}.jsonl", lines=True, dtype=False)
    list_ids = gt_df["id"].values
    list_image_path = [f"./data/image/FB/All/{img_id}.png" for img_id in list_ids] 
    
    return list_image_path, gt_df["text"].to_list(), gt_df["label"].to_list(), list_ids

def get_dataloader(preprocess, batch_size=128, num_workers=4, train_batch_size=32, device="cuda", image_size=224):
    imgtxt_dataset = ImageTextDataset(get_values_from_gt("train"), preprocess)
    train = DeviceDataLoader(DataLoader(imgtxt_dataset, batch_size=train_batch_size, num_workers=num_workers), device)

    imgtxt_dataset = ImageTextDataset(get_values_from_gt("dev_seen"), preprocess, image_size=image_size)
    dev_seen = DeviceDataLoader(DataLoader(imgtxt_dataset, batch_size=batch_size, num_workers=num_workers), device)

    imgtxt_dataset = ImageTextDataset(get_values_from_gt("test_seen"), preprocess, image_size=image_size)
    test_seen = DeviceDataLoader(DataLoader(imgtxt_dataset, batch_size=batch_size, num_workers=num_workers), device)

    imgtxt_dataset = ImageTextDataset(get_values_from_gt("test_unseen"), preprocess, image_size=image_size)
    test_unseen = DeviceDataLoader(DataLoader(imgtxt_dataset, batch_size=batch_size, num_workers=num_workers), device)

    return train, dev_seen, test_seen, test_unseen

# Called from train: return the pre-extracted features from CLIP model
def load_feats_split(path, dataset=None):
    dict = torch.load(path)
    ids = dict["ids"]
    ids = [item for sublist in ids for item in sublist]
    img_feats = dict["img_feats"]
    text_feats = dict["text_feats"]
    face_feats = dict["face_feats"]
    labels = dict["labels"]

    return [ids, img_feats, text_feats, face_feats, labels]

def load_feats_from_CLIP(path, model):
    train = load_feats_split("{}/FB/train_{}.pt".format(path, model))
    dev = load_feats_split("{}/FB/dev_seen_{}.pt".format(path, model))
    test_seen = load_feats_split("{}/FB/test_seen_{}.pt".format(path, model))
    test_unseen = load_feats_split("{}/FB/test_unseen_{}.pt".format(path, model))

    return train, dev, test_seen, test_unseen