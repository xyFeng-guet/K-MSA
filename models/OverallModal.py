import torch
from torch import nn
from models.Encoder_KIAdapter import UnimodalEncoder
from models.DyRoutFusion_CLS import DyRoutTrans, SentiCLS


class KMSA(nn.Module):
    def __init__(self, opt, dataset, bert_pretrained='bert-base-uncased'):
        super(KMSA, self).__init__()
        # Unimodal Encoder & Knowledge Inject Adapter
        self.UniEncKI = UnimodalEncoder(opt, bert_pretrained)

        # Multimodal Fusion
        # self.DyMultiFus = DyRoutTrans(opt)
        # self.DyMultiFus = DynRT(text_encoder, vision_encoder, acoustic_encoder, opt)

        # Output Classification for Sentiment Analysis
        self.CLS = SentiCLS(opt)

    def forward(self, inputs_data_mask):
        # Unimodal Encoder & Knowledge Inject // Unimodal Sentiment Prediction
        unimodal_features, _ = self.UniEncKI(inputs_data_mask)

        # Dynamic Multimodal Fusion using Dynamic Route Transformer with Unimodal Sentiment Prediction
        # multimodal_features = self.DyMultiFus(unimodal_features, _)

        # Sentiment Classification
        prediction = self.CLS(unimodal_features[0], unimodal_features[1], unimodal_features[2])

        return prediction


def build_model(opt):
    if opt.datasetName == 'sims':
        l_pretrained = './BERT/bert-base-chinese'
    else:
        l_pretrained = './BERT/bert-base-uncased'

    model = KMSA(opt, dataset=opt.datasetName, bert_pretrained=l_pretrained)

    return model
