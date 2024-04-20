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


class BaseClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, drop_out):
        super(BaseClassifier, self).__init__()
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
        x = self.MLP(x)
        return x


class PositionEncodingTraining(nn.Module):
    """Construct the CLS token, position and patch embeddings.
    """
    def __init__(self, num_patches, fea_size=None, tf_hidden_dim=None, drop_out=None, config=default_config):
        super().__init__()
        if fea_size is None:
            fea_size = config.SIMS.downStream.vision_fea_dim
        if tf_hidden_dim is None:
            tf_hidden_dim = config.SIMS.downStream.encoder_fea_dim
        if drop_out is None:
            drop_out = config.SIMS.downStream.vision_drop_out

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
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, dropout=0.2, activation='gelu'):
        super(TfEncoder, self).__init__()
        self.src_mask = None
        self.pos_encoder = PositionEncodingTraining()

        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation=activation)
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
    def __init__(self, name=None, fea_size=None, encoder_fea_dim=None, nhead=None, dim_feedforward=None,
                 num_layers=None,
                 drop_out=0.5, config=default_config):
        super(VisionEncoder, self).__init__()
        self.name = name
        if fea_size is None:
            fea_size = config.SIMS.downStream.vision_fea_dim
        if encoder_fea_dim is None:
            encoder_fea_dim = config.SIMS.downStream.encoder_fea_dim
        if nhead is None:
            nhead = config.SIMS.downStream.vision_nhead
        if drop_out is None:
            drop_out = config.SIMS.downStream.vision_drop_out
        if dim_feedforward is None:
            dim_feedforward = config.SIMS.downStream.encoder_fea_dim
        if num_layers is None:
            num_layers = config.SIMS.downStream.vision_tf_num_layers

        self.fc = nn.Linear(fea_size, encoder_fea_dim)
        self.encoder = TfEncoder(d_model=encoder_fea_dim, nhead=nhead, dim_feedforward=dim_feedforward,
                                 num_layers=num_layers,
                                 dropout=drop_out, activation='gelu',
                                 config=config)

        self.device = config.DEVICE
        self.encoder.device = self.device
        self.activation = nn.Tanh()
        self.cls_embedding = nn.Parameter()
        self.layernorm = nn.LayerNorm(encoder_fea_dim)
        self.dense = nn.Linear(encoder_fea_dim, encoder_fea_dim)

    def forward(self, vision, key_padding_mask, device=None):
        if device is None:
            device = self.device

        x = self.encoder(vision, has_mask=False, src_key_padding_mask=key_padding_mask)
        x = self.layernorm(x)
        x = torch.mean(x, dim=-2, keepdim=True)

        return x


class VisionPretrain(nn.Module):
    def __init__(self, name=None, encoder_fea_dim=None, drop_out=None, config=default_config):
        super(VisionPretrain, self).__init__()
        if encoder_fea_dim is None:
            encoder_fea_dim = config.SIMS.downStream.encoder_fea_dim
        if drop_out is None:
            drop_out = config.SIMS.downStream.text_drop_out
        self.encoder = VisionEncoder(name=name)
        self.classifier = BaseClassifier(input_size=encoder_fea_dim,
                                         hidden_size=[int(encoder_fea_dim / 2), int(encoder_fea_dim / 4),
                                                      int(encoder_fea_dim / 8)],
                                         output_size=1, drop_out=drop_out, name='VisionRegClassifier', )
        self.device = config.DEVICE
        self.criterion = torch.nn.MSELoss()
        self.config = config

    def forward(self, vision, label, key_padding_mask, return_loss=True, device=None):
        if device is None:
            device = self.device
        x = self.encoder(vision, key_padding_mask, device=device)
        pred = self.classifier(x).squeeze()

        if return_loss:
            loss = self.criterion(pred.squeeze(), label.squeeze())
            return pred, x, loss
        else:
            return pred, x


class AudioEncoder(nn.Module):
    def __init__(self, name=None, fea_size=None, encoder_fea_dim=None, nhead=None, dim_feedforward=None,
                 num_layers=None,
                 drop_out=0.5, config=default_config):
        super(AudioEncoder, self).__init__()
        self.name = name
        if fea_size is None:
            fea_size = config.SIMS.downStream.audio_fea_dim
        if encoder_fea_dim is None:
            encoder_fea_dim = config.SIMS.downStream.encoder_fea_dim
        if nhead is None:
            nhead = config.SIMS.downStream.audio_nhead
        if drop_out is None:
            drop_out = config.SIMS.downStream.audio_drop_out
        if dim_feedforward is None:
            dim_feedforward = config.SIMS.downStream.encoder_fea_dim
        if num_layers is None:
            num_layers = config.SIMS.downStream.audio_tf_num_layers

        self.fc = nn.Linear(fea_size, encoder_fea_dim)
        self.encoder = TfEncoder(d_model=encoder_fea_dim, nhead=nhead, dim_feedforward=dim_feedforward,
                                 num_layers=num_layers,
                                 dropout=drop_out, activation='gelu',
                                 config=config)

        self.device = config.DEVICE
        self.encoder.device = self.device
        self.activation = nn.Tanh()
        self.cls_embedding = nn.Parameter()
        self.layernorm = nn.LayerNorm(encoder_fea_dim)
        self.dense = nn.Linear(encoder_fea_dim, encoder_fea_dim)
        # self.fc = nn.Linear(709,768)

    def forward(self, audio, key_padding_mask, device=None):
        if device is None:
            device = self.device

        x = self.encoder(audio, has_mask=False, src_key_padding_mask=key_padding_mask)
        x = self.layernorm(x)
        x = torch.mean(x, dim=-2, keepdim=True)

        return x


class AudioPretrain(nn.Module):
    def __init__(self, name=None, encoder_fea_dim=None, drop_out=None, config=default_config):
        super(AudioPretrain, self).__init__()
        if encoder_fea_dim is None:
            encoder_fea_dim = config.SIMS.downStream.encoder_fea_dim
        if drop_out is None:
            drop_out = config.SIMS.downStream.text_drop_out
        self.encoder = AudioEncoder(name=name)
        self.classifier = BaseClassifier(input_size=encoder_fea_dim,
                                         hidden_size=[int(encoder_fea_dim / 2), int(encoder_fea_dim / 4),
                                                      int(encoder_fea_dim / 8)],
                                         output_size=1, drop_out=drop_out, name='AudioRegClassifier', )
        self.device = config.DEVICE
        self.criterion = torch.nn.MSELoss()
        self.config = config

    def forward(self, audio, label, key_padding_mask, return_loss=True, device=None):
        if device is None:
            device = self.device
        x = self.encoder(audio, key_padding_mask, device=device)
        pred = self.classifier(x).squeeze()

        if return_loss:
            loss = self.criterion(pred.squeeze(), label.squeeze())
            return pred, x, loss
        else:
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
        self.classifier = BaseClassifier(
            input_size=proj_fea_dim,
            hidden_size=[int(proj_fea_dim / 2), int(proj_fea_dim / 4), int(proj_fea_dim / 8)],
            output_size=1,
            drop_out=drop_out
        )

    def forward(self, text):
        bert_output = self.encoder(text)
        pred = self.classifier(bert_output)

        return pred, bert_output


def build_pretrained_model(modality):
    if modality == 't':
        pretrained_model = TextPretrain(
            proj_fea_dim=768,
            drop_out=0.1
        )
    elif modality == 'v':
        pretrained_model = VisionPretrain()
    elif modality == 'a':
        pretrained_model = AudioPretrain()
    else:
        raise ValueError("modality must be in t, v, and a")

    return pretrained_model
