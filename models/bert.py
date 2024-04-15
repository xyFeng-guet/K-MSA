import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig


class BertTextEncoder(nn.Module):
    def __init__(self, use_finetune=False, pretrained='bert-base-uncased'):
        super().__init__()
        self.model_config = BertConfig.from_pretrained(pretrained, output_hidden_states=True)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained, do_lower_case=True)
        self.model = BertModel.from_pretrained(pretrained, config=self.model_config)
        self.use_finetune = use_finetune

    def get_tokenizer(self):
        return self.tokenizer

    def forward(self, text):
        """
        text: (batch_size, 3, seq_len)
        3: input_ids, input_mask, segment_ids
        input_ids: input_ids,
        input_mask: attention_mask,
        segment_ids: token_type_ids
        """
        input_ids, input_mask, segment_ids = text[:, 0, :].long(), text[:, 1, :].float(), text[:, 2, :].long()    # 更换原始文本，使用tokenizer
        if self.use_finetune:
            last_hidden_states = self.model(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=segment_ids
            )  # type: ignore # Models outputs are now tuples
            last_hidden_states = last_hidden_states[0]
        else:
            with torch.no_grad():
                last_hidden_states = self.model(
                    input_ids=input_ids,
                    attention_mask=input_mask,
                    token_type_ids=segment_ids
                )  # type: ignore # Models outputs are now tuples
                last_hidden_states = last_hidden_states[0]
        return last_hidden_states
