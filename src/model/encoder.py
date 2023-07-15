import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


class trigger_tagger(nn.Module):
    def __init__(self, encoder, hidden_size, vocab_size,
                 padding_idx=0, label_padding_idx=0,
                 dropout=0.1):
        super(trigger_tagger, self).__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.label_padding_idx = label_padding_idx
        self.encoder = encoder
        self.emb2tag = nn.Linear(self.hidden_size, self.vocab_size)
        self.dropout_out = nn.Dropout(dropout)
        self.loss_function = nn.CrossEntropyLoss(ignore_index=-100)
        self.loss_function_entity = nn.BCEWithLogitsLoss(reduction='none')
        self.sigmoid = nn.Sigmoid()
        self.init_embed()

    def init_embedding(self, input_embedding):
        """
        Initialize embedding
        """
        bias = np.sqrt(3.0 / input_embedding.size(1))
        nn.init.uniform_(input_embedding, -bias, bias)

    def init_embed(self):
        nn.init.uniform_(self.emb2tag.weight, -0.1, 0.1)
        nn.init.constant_(self.emb2tag.bias, 0.)

    def forward(self, inputs):
        sentence_ids = inputs['sentence_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
        bert_outs = self.encoder(sentence_ids, attention_mask=attention_mask,
                                 token_type_ids=token_type_ids)
        lstm_out = bert_outs[0]
        return lstm_out

    def loss(self, inputs, label_ids):
        lstm_logits = self.forward(inputs)
        lstm_out = self.emb2tag(
            self.dropout_out(lstm_logits))
        loss = self.loss_function(lstm_out.view(-1, self.vocab_size),
                                  label_ids.view(-1))
        return loss, lstm_logits

    def decode(self, inputs):
        lstm_logits = self.forward(inputs)
        lstm_out = self.emb2tag(self.dropout_out(lstm_logits)).transpose(1, 2)
        lstm_out = torch.nn.functional.softmax(lstm_out, dim=1)
        scores_max, preds = torch.max(lstm_out, dim=1)
        return preds, lstm_logits
