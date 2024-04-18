'''
* @name: almt.py
* @description: Implementation of ALMT
'''

import torch
from torch import nn
from .Transformer_Layers import Transformer, CrossTransformer
from .bert import BertTextEncoder


class KMSA(nn.Module):
    def __init__(self, dataset, bert_pretrained='bert-base-uncased'):
        super(KMSA, self).__init__()
        self.bertmodel = BertTextEncoder(use_finetune=True, pretrained=bert_pretrained)

        # Input Dimension Align
        if dataset == 'mosi':
            self.dim_align_t = nn.Linear(768, 128)
            self.dim_align_a = nn.Linear(5, 128)
            self.dim_align_v = nn.Linear(20, 128)
        elif dataset == 'mosei':
            self.dim_align_t = nn.Linear(768, 128)
            self.dim_align_a = nn.Linear(74, 128)
            self.dim_align_v = nn.Linear(35, 128)
        elif dataset == 'sims':
            self.dim_align_t = nn.Linear(768, 128)
            self.dim_align_a = nn.Linear(33, 128)
            self.dim_align_v = nn.Linear(709, 128)
        else:
            assert False, "DatasetName must be mosi, mosei or sims."

        # Length Align (Learning Context Information)
        self.len_align_a = nn.Linear(300, 50, bias=True)
        self.len_align_v = nn.Linear(300, 50, bias=True)

        # All Encoders of Each Modality
        self.proj_a = Transformer(num_frames=300, dim=128, depth=1, heads=8, mlp_dim=128)
        self.proj_v = Transformer(num_frames=300, dim=128, depth=1, heads=8, mlp_dim=128)

        # Multimodal Fusion and Classfication Module
        # self.fusion_layer = CrossTransformer(source_num_frames=8, tgt_num_frames=8, dim=128, depth=fusion_layer_depth, heads=8, mlp_dim=128)
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 * 3, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 1, bias=True)
        )

    def forward(self, x_visual, x_audio, x_text):
        x_text = self.bertmodel(x_text)

        # Dimension Align
        x_visual = self.dim_align_v(x_visual)
        x_audio = self.dim_align_a(x_audio)
        x_text = self.dim_align_t(x_text)

        # Length Align
        # h_t = self.len_align_t(x_text.permute(0, 2, 1)).permute(0, 2, 1)
        h_a = self.len_align_a(x_audio.permute(0, 2, 1)).permute(0, 2, 1)
        h_v = self.len_align_v(x_visual.permute(0, 2, 1)).permute(0, 2, 1)

        # Encoder Part
        h_v = self.proj_v(x_visual)
        h_a = self.proj_a(x_audio)
        # h_t = self.proj_t(x_text)

        # Fusion and Classficiation
        h_v_global = torch.mean(h_v, dim=1)
        h_a_global = torch.mean(h_a, dim=1)
        h_t_global = torch.mean(x_text, dim=1)
        fusion_features = torch.cat((h_v_global, h_a_global, h_t_global), dim=-1)
        output = self.fusion_layer(fusion_features)

        return output


def build_model(opt):
    if opt.datasetName == 'sims':
        l_pretrained = './bert-base-chinese'
    else:
        l_pretrained = './bert-base-uncased'

    model = KMSA(dataset=opt.datasetName, bert_pretrained=l_pretrained)

    return model
