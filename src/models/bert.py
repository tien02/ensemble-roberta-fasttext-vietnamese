import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig

class PhoBertFeedForward_base(nn.Module):
    def __init__(self, from_pretrained:bool=True, freeze_backbone:bool=False, drop_out:float=0.1, out_channels:int=3):
        super(PhoBertFeedForward_base, self).__init__()
        phobert_config = RobertaConfig.from_pretrained("vinai/phobert-base-v2")
        self.bert = RobertaModel(config=phobert_config)
        if from_pretrained:
          self.bert = RobertaModel.from_pretrained("vinai/phobert-base-v2")
        self.classifier = nn.Sequential(
            nn.Linear(768, 768),
            nn.Dropout(drop_out),
            nn.Linear(768, out_channels))
        
        if freeze_backbone:
            for param in self.bert.parameters():
                param.require_grad = False
    
    def forward(self, input_ids, attn_mask):
        bert_feature = self.bert(input_ids=input_ids, attention_mask=attn_mask)
        last_hidden_cls = bert_feature[0][:, 0, :]
        logits = self.classifier(last_hidden_cls)
        return logits


class PhoBertFeedForward_large(nn.Module):
    def __init__(self, from_pretrained:bool=True, freeze_backbone:bool=False, drop_out:float=0.1, out_channels:int=3):
        super(PhoBertFeedForward_large, self).__init__()
        phobert_config = RobertaConfig.from_pretrained("vinai/phobert-large")
        self.bert = RobertaModel(config=phobert_config)
        if from_pretrained:
          self.bert = RobertaModel.from_pretrained("vinai/phobert-large")
        self.classifier = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.Dropout(drop_out),
            nn.Linear(1024, out_channels))
        
        if freeze_backbone:
            for param in self.bert.parameters():
                param.require_grad = False
    
    def forward(self, input_ids, attn_mask):
        bert_feature = self.bert(input_ids=input_ids, attention_mask=attn_mask)
        last_hidden_cls = bert_feature[0][:, 0, :]
        logits = self.classifier(last_hidden_cls)
        return logits


class PhoBERTLSTM_base(nn.Module):
  def __init__(self, from_pretrained:bool=True, freeze_backbone:bool=False, drop_out:float=0.1, out_channels:int=3):
    super(PhoBERTLSTM_base, self).__init__()
    phobert_config = RobertaConfig.from_pretrained("vinai/phobert-base-v2")
    self.bert = RobertaModel(config=phobert_config)
    if from_pretrained:
        self.bert = RobertaModel.from_pretrained("vinai/phobert-base-v2")

    self.lstm = nn.LSTM(input_size=768, hidden_size=768, 
                        batch_first=True, bidirectional=True, num_layers=1)
    self.dropout = nn.Dropout(drop_out)
    self.linear = nn.Linear(768 * 2, out_channels)
    self.act = nn.LogSoftmax(dim=1)

    if freeze_backbone:
      for param in self.bert.parameters():
          param.require_grad = False
  
  def forward(self, input_ids, attn_mask):
    out = self.bert(input_ids=input_ids, attention_mask=attn_mask)[0]
    out, (h, c) = self.lstm(out)
    hidden = torch.cat((h[0], h[1]), dim = 1)
    out = self.linear(self.dropout(hidden))
    return out

class PhoBERTLSTM_large(nn.Module):
  def __init__(self, from_pretrained:bool=True, freeze_backbone:bool=False, drop_out:float=0.1, out_channels:int=3):
    super(PhoBERTLSTM_large, self).__init__()
    phobert_config = RobertaConfig.from_pretrained("vinai/phobert-large")
    self.bert = RobertaModel(config=phobert_config)
    if from_pretrained:
        self.bert = RobertaModel.from_pretrained("vinai/phobert-large")

    self.lstm = nn.LSTM(input_size=1024, 
                        hidden_size=1024, 
                        batch_first=True, bidirectional=True, num_layers=1)
    self.dropout = nn.Dropout(drop_out)
    self.linear = nn.Linear(1024 * 2, out_channels)

    if freeze_backbone:
      for param in self.bert.parameters():
          param.require_grad = False
  
  def forward(self, input_ids, attn_mask):
    out = self.bert(input_ids=input_ids, attention_mask=attn_mask)[0]
    out, (h, c) = self.lstm(out)
    hidden = torch.cat((h[0], h[1]), dim = 1)
    out = self.linear(self.dropout(hidden))
    return out