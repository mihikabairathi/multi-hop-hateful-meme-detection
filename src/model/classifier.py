# heavily borrowed from: https://github.com/JingbiaoMei/RGCL/blob/main/src/model/classifier.py
# TODO: mess around with the model architecture, hyperparameters, and how we combine embeddings (ex: concat v/s mult)

import torch.nn as nn
import torch

class MultiHopMemeClassifier(nn.Module):
    def __init__(self, image_dim, text_dim, face_dim, num_layers, proj_dim, map_dim, dropout, num_layers_face, dropout_face):
        super().__init__()
        
        # Projection layers prior to modality fusion
        self.img_proj = nn.Sequential(nn.Linear(image_dim, map_dim), nn.Dropout(dropout[0]))
        self.text_proj = nn.Sequential(nn.Linear(text_dim, map_dim), nn.Dropout(dropout[0]))
        self.face_proj = nn.Sequential(nn.Linear(face_dim, map_dim), nn.Dropout(dropout_face[0]))

        # first-hop
        input_shape = map_dim
        layers = [nn.Dropout(dropout[1])]
        for _ in range(num_layers):
            layers.append(nn.Linear(input_shape, proj_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout[2]))
            input_shape = proj_dim
        self.first_hop = nn.Sequential(*layers)

        # second-hop
        input_shape = map_dim
        layers = [nn.Dropout(dropout_face[1])]
        for _ in range(num_layers_face):
            layers.append(nn.Linear(input_shape, proj_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_face[2]))
            input_shape = proj_dim
        self.second_hop = nn.Sequential(*layers)

        self.output_layer = nn.Linear(proj_dim, 1)

    def forward(self, img_feats, text_feats, face_feats, return_embed=False):
        img_feats = nn.functional.normalize(self.img_proj(img_feats), p=2, dim=1)
        text_feats = nn.functional.normalize(self.text_proj(text_feats), p=2, dim=1)
        face_feats = nn.functional.normalize(self.face_proj(face_feats), p=2, dim=1)
        clip_input = torch.mul(img_feats, text_feats)

        # For embedding, we don't need the relu and dropout
        clip_embed = self.first_hop[:-2](clip_input)
        face_embed = self.second_hop[:-2](face_feats)
        embed = torch.mul(clip_embed, face_embed)

        # Call the models
        output = self.output_layer(torch.mul(self.first_hop(clip_input), self.second_hop(face_feats)))

        if return_embed:
            return output, embed
        else:
            return output