import torch
from torch import nn
from models.Transformer_Layers import Transformer, BertTextEncoder


class MixDomainAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()


class UnimodalAdapter(nn.Module):
    def __init__(self, opt, dataset, bert_pretrained='bert-base-uncased'):
        super(UnimodalAdapter, self).__init__()
        self.bertmodel = BertTextEncoder(use_finetune=True, pretrained=bert_pretrained)

        # Input Dimension Align
        if dataset == 'mosi':
            self.dim_align_t = nn.Linear(768, 128)
            self.dim_align_v = nn.Linear(20, 128)
            self.dim_align_a = nn.Linear(5, 128)
        elif dataset == 'mosei':
            self.dim_align_t = nn.Linear(768, 128)
            self.dim_align_v = nn.Linear(35, 128)
            self.dim_align_a = nn.Linear(74, 128)
        elif dataset == 'sims':
            self.dim_align_t = nn.Linear(768, 128)
            self.dim_align_v = nn.Linear(709, 128)
            self.dim_align_a = nn.Linear(33, 128)
        else:
            assert False, "DatasetName must be mosi, mosei or sims."

        # Length Align (Learning Context Information)
        self.len_align_v = nn.Linear(300, 50, bias=True)
        self.len_align_a = nn.Linear(300, 50, bias=True)

        # All Encoders of Each Modality
        self.proj_v = Transformer(num_frames=300, dim=128, depth=1, heads=8, mlp_dim=128)
        self.proj_a = Transformer(num_frames=300, dim=128, depth=1, heads=8, mlp_dim=128)

    def forward(self, text, visual, acoustic):
        hidden_t = self.bertmodel(text)

        # Dimension Align
        hidden_t = self.dim_align_t(hidden_t)
        visual = self.dim_align_v(visual)
        acoustic = self.dim_align_a(acoustic)

        # Length Align
        visual = self.len_align_v(visual.permute(0, 2, 1)).permute(0, 2, 1)
        acoustic = self.len_align_a(acoustic.permute(0, 2, 1)).permute(0, 2, 1)

        # Encoder Part
        hidden_v = self.proj_v(visual)
        hidden_a = self.proj_a(acoustic)

        return [hidden_t, hidden_v, hidden_a], []
