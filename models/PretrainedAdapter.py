import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import BertConfig, BertModel, BertTokenizer


class FeatureProjector(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=3, drop_out=0.1):
        super(FeatureProjector, self).__init__()
        self.feed_foward_size = int(output_dim / 2)
        self.project_size = output_dim - self.feed_foward_size
        self.proj1 = nn.Linear(input_dim, self.feed_foward_size, bias=True)

        self.proj2 = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.proj2.append(nn.Linear(input_dim, self.project_size, bias=False))
            else:
                self.proj2.append(nn.Linear(self.project_size, self.project_size, bias=False))
            self.proj2.append(nn.GELU())

        self.layernorm_ff = nn.LayerNorm(self.feed_foward_size)
        self.layernorm = nn.LayerNorm(self.project_size)
        self.MLP = nn.Sequential(*self.proj2)
        self.drop = nn.Dropout(p=drop_out)

    def forward(self, batch):
        # input: list of data samples with different seq length
        dropped = self.drop(batch)
        ff = self.proj1(dropped)
        x = self.MLP(dropped)
        x = torch.cat([self.layernorm(x), self.layernorm_ff(ff)], dim=-1)
        # return x.transpose(0, 1)  # return shape: [seq,batch,fea]
        return x


class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, drop_out):
        super(Classifier, self).__init__()
        ModuleList = []
        for i, h in enumerate(hidden_size):
            if i == 0:
                ModuleList.append(nn.Linear(input_size, h))
                ModuleList.append(nn.GELU())
            else:
                ModuleList.append(nn.Linear(hidden_size[i - 1], h))
                ModuleList.append(nn.GELU())
        ModuleList.append(nn.Linear(hidden_size[-1], output_size))

        self.MLP = nn.Sequential(*ModuleList)
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        x = self.dropout(x)
        x = x[:, 0, :]
        output = self.MLP(x)
        return output


class PositionEncodingTraining(nn.Module):
    """Construct the CLS token, position and patch embeddings.
    """
    def __init__(self, num_patches, fea_size, tf_hidden_dim, drop_out):
        super().__init__()
        self.cls_token = nn.Parameter(torch.ones(1, 1, tf_hidden_dim))
        self.proj = nn.Linear(fea_size, tf_hidden_dim)
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, tf_hidden_dim))
        self.dropout = nn.Dropout(drop_out)

    def forward(self, embeddings):
        batch_size = embeddings.shape[0]
        embeddings = self.proj(embeddings)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class TfEncoder(nn.Module):
    def __init__(self, fea_size, nhead, dim_feedforward, num_layers, dropout=0.2, activation='gelu'):
        super(TfEncoder, self).__init__()
        self.src_mask = None
        self.pos_encoder = PositionEncodingTraining(
            num_patches=50,
            fea_size=fea_size,
            tf_hidden_dim=dim_feedforward,
            drop_out=0.5
        )

        encoder_layers = TransformerEncoderLayer(dim_feedforward, nhead, dim_feedforward, dropout, activation=activation)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, has_mask=True, src_key_padding_mask=None):
        src = self.pos_encoder(src)

        src = src.transpose(0, 1)
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        output = self.transformer_encoder(src, mask=self.src_mask, src_key_padding_mask=src_key_padding_mask)

        return output.transpose(0, 1)


class VisionEncoder(nn.Module):
    def __init__(self, fea_size, nhead, dim_feedforward, num_layers, drop_out):
        super(VisionEncoder, self).__init__()
        self.tfencoder = TfEncoder(
            fea_size=fea_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
            dropout=drop_out,
            activation='gelu'
        )
        self.layernorm = nn.LayerNorm(dim_feedforward)

    def forward(self, vision, key_padding_mask):
        x = self.tfencoder(vision, has_mask=False, src_key_padding_mask=key_padding_mask)
        x = self.layernorm(x)
        x = torch.mean(x, dim=-2, keepdim=True)
        return x


class VisionPretrain(nn.Module):
    def __init__(self, proj_fea_dim, drop_out):
        super(VisionPretrain, self).__init__()
        self.encoder = VisionEncoder(
            fea_size=709,
            nhead=8,
            dim_feedforward=1024,
            num_layers=4,
            drop_out=0.5
        )
        self.classifier = Classifier(
            input_size=proj_fea_dim,
            hidden_size=[int(proj_fea_dim / 2), int(proj_fea_dim / 4), int(proj_fea_dim / 8)],
            output_size=1,
            drop_out=drop_out
        )

    def forward(self, img, audio, text):
        x = self.encoder(img, None)
        pred = self.classifier(x).squeeze()
        return pred, x


class AudioEncoder(nn.Module):
    def __init__(self, fea_size, encoder_fea_dim, nhead, dim_feedforward, num_layers, drop_out=0.5):
        super(AudioEncoder, self).__init__()
        self.fc = nn.Linear(fea_size, encoder_fea_dim)
        self.cls_embedding = nn.Parameter()
        self.encoder = TfEncoder(
            d_model=encoder_fea_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
            dropout=drop_out,
            activation='gelu'
        )
        self.layernorm = nn.LayerNorm(encoder_fea_dim)

    def forward(self, audio, key_padding_mask):
        x = self.encoder(audio, has_mask=False, src_key_padding_mask=key_padding_mask)
        x = self.layernorm(x)
        x = torch.mean(x, dim=-2, keepdim=True)
        return x


class AudioPretrain(nn.Module):
    def __init__(self, encoder_fea_dim, drop_out):
        super(AudioPretrain, self).__init__()
        self.encoder = AudioEncoder()
        self.classifier = Classifier(
            input_size=encoder_fea_dim,
            hidden_size=[int(encoder_fea_dim / 2), int(encoder_fea_dim / 4), int(encoder_fea_dim / 8)],
            output_size=1,
            drop_out=drop_out
        )

    def forward(self, audio, key_padding_mask):
        x = self.encoder(audio, key_padding_mask)
        pred = self.classifier(x).squeeze()
        return pred, x


class TextEncoder(nn.Module):
    def __init__(self, pretrained, fea_size=None, proj_fea_dim=None):
        super(TextEncoder, self).__init__()
        self.model_config = BertConfig.from_pretrained(pretrained, output_hidden_states=True)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained, do_lower_case=True)
        self.model = BertModel.from_pretrained(pretrained, config=self.model_config)
        # self.projector = FeatureProjector(fea_size, proj_fea_dim)

    def forward(self, text):
        input_ids, input_mask, segment_ids = text[:, 0, :].long(), text[:, 1, :].float(), text[:, 2, :].long()    # 更换原始文本，使用tokenizer
        # x = self.model(**x)['pooler_output']

        last_hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=segment_ids
        )  # type: ignore # Models outputs are now tuples
        hidden_text = last_hidden_states[0]
        # hidden_text = self.projector(hidden_text)
        return hidden_text


class TextPretrain(nn.Module):
    def __init__(self, proj_fea_dim, drop_out):
        super(TextPretrain, self).__init__()
        self.encoder = TextEncoder(
            pretrained='./BERT/bert-base-chinese',
            fea_size=768,
            proj_fea_dim=proj_fea_dim
        )
        self.classifier = Classifier(
            input_size=proj_fea_dim,
            hidden_size=[int(proj_fea_dim / 2), int(proj_fea_dim / 4), int(proj_fea_dim / 8)],
            output_size=1,
            drop_out=drop_out
        )

    def forward(self, img, audio, text):
        bert_output = self.encoder(text)
        pred = self.classifier(bert_output)

        return pred


def build_pretrained_model(modality):
    if modality == 'T':
        pretrained_model = TextPretrain(
            proj_fea_dim=768,
            drop_out=0.1
        )
    elif modality == 'V':
        pretrained_model = VisionPretrain(
            proj_fea_dim=1024,
            drop_out=0.1
        )
    # elif modality == 'A':
    #     pretrained_model = AudioPretrain(
    #         proj_fea_dim=512,
    #         drop_out=0.1
    #     )
    else:
        raise ValueError("modality must be in t, v, and a")

    return pretrained_model
