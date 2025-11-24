# heavily borrowed from: https://github.com/JingbiaoMei/RGCL/blob/main/src/model/classifier.py

import torch.nn as nn
import torch

class FirstHopHateClipper(nn.Module):
    def __init__(self, image_dim, text_dim, num_layers, proj_dim, map_dim, dropout):
        super().__init__()
        
        # Projection layers prior to modality fusion
        self.img_proj = nn.Sequential(nn.Linear(image_dim, map_dim), nn.Dropout(dropout[0]))
        self.text_proj = nn.Sequential(nn.Linear(text_dim, map_dim), nn.Dropout(dropout[0]))

        # Modality fusion
        input_shape = map_dim
        layers = [nn.Dropout(dropout[1])]
        for _ in range(num_layers):
            layers.append(nn.Linear(input_shape, proj_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout[2]))
            input_shape = proj_dim
                
        self.mlp = nn.Sequential(*layers)
        self.output_layer = nn.Linear(proj_dim, 1)

    def forward(self,img_feats, text_feats, return_embed=False):
        img_feats = nn.functional.normalize(self.img_proj(img_feats), p=2, dim=1)
        text_feats = nn.functional.normalize(self.text_proj(text_feats), p=2, dim=1)
        
        # For embedding, we don't need the relu and dropout
        x = torch.mul(img_feats, text_feats)
        embed = self.mlp[:-2](x)
        output = self.output_layer(self.mlp(x))

        if return_embed:
            return output, embed
        else:
            return output