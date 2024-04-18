import torch
from torch import nn


class KMSA(nn.Module):
    def __init__(self, dataset, bert_pretrained='bert-base-uncased'):
        super(KMSA, self).__init__()
        # Multimodal Fusion and Classfication Module
        # self.fusion_layer = CrossTransformer(source_num_frames=8, tgt_num_frames=8, dim=128, depth=fusion_layer_depth, heads=8, mlp_dim=128)
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 * 3, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 1, bias=True)
        )

    def forward(self, hidden_video, hidden_acoustic, hidden_text):
        # Fusion and Classficiation
        h_v_global = torch.mean(hidden_video, dim=1)
        h_a_global = torch.mean(hidden_acoustic, dim=1)
        h_t_global = torch.mean(hidden_text, dim=1)

        fusion_features = torch.cat((h_v_global, h_a_global, h_t_global), dim=-1)
        output = self.fusion_layer(fusion_features)

        return output
