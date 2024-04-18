import torch
from torch import nn
from .Transformer_Layers import Transformer, CrossTransformer
from .bert import BertTextEncoder


class KMSA(nn.Module):
    def __init__(self, dataset, bert_pretrained='bert-base-uncased'):
        super(KMSA, self).__init__()


    def forward(self, x_visual, x_audio, x_text):

        return prediction


def build_model(opt):
    if opt.datasetName == 'sims':
        l_pretrained = './bert-base-chinese'
    else:
        l_pretrained = './bert-base-uncased'

    model = KMSA(dataset=opt.datasetName, bert_pretrained=l_pretrained)

    return model
