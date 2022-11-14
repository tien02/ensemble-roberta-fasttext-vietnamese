from .config import config
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig

class PhoBertFeedForward_base(nn.Module):
    def __init__(self, from_pretrained=True, freeze_backbone=False):
        super(PhoBertFeedForward_base, self).__init__()
        phobert_config = RobertaConfig.from_pretrained("vinai/phobert-base")
        self.bert = RobertaModel(config=phobert_config)
        if from_pretrained:
          self.bert = RobertaModel.from_pretrained("vinai/phobert-base")
        self.classifier = nn.Sequential(
            nn.Linear(768, 768),
            nn.Dropout(config.DROP_OUT),
            nn.Linear(768, config.NUM_CLASSES))
        
        if freeze_backbone:
            for param in self.bert.parameters():
                param.require_grad = False
    
    def forward(self, input_ids, attn_mask):
        bert_feature = self.bert(input_ids=input_ids, attention_mask=attn_mask)
        last_hidden_cls = bert_feature[0][:, 0, :]
        logits = self.classifier(last_hidden_cls)
        return logits


class PhoBertFeedForward_large(nn.Module):
    def __init__(self, from_pretrained=True, freeze_backbone=False):
        super(PhoBertFeedForward_large, self).__init__()
        phobert_config = RobertaConfig.from_pretrained("vinai/phobert-large")
        self.bert = RobertaModel(config=phobert_config)
        if from_pretrained:
          self.bert = RobertaModel.from_pretrained("vinai/phobert-large")
        self.classifier = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.Dropout(config.DROP_OUT),
            nn.Linear(1024, config.NUM_CLASSES))
        
        if freeze_backbone:
            for param in self.bert.parameters():
                param.require_grad = False
    
    def forward(self, input_ids, attn_mask):
        bert_feature = self.bert(input_ids=input_ids, attention_mask=attn_mask)
        last_hidden_cls = bert_feature[0][:, 0, :]
        logits = self.classifier(last_hidden_cls)
        return logits


class PhoBERTLSTM_base(nn.Module):
  def __init__(self, from_pretrained=True, freeze_backbone=False):
    super(PhoBERTLSTM_base, self).__init__()
    phobert_config = RobertaConfig.from_pretrained("vinai/phobert-base")
    self.bert = RobertaModel(config=phobert_config)
    if from_pretrained:
        self.bert = RobertaModel.from_pretrained("vinai/phobert-base")

    self.lstm = nn.LSTM(input_size=768, hidden_size=768, 
                        batch_first=True, bidirectional=True, num_layers=1)
    self.dropout = nn.Dropout(config.DROP_OUT)
    self.linear = nn.Linear(768 * 2, config.NUM_CLASSES)
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
  def __init__(self, from_pretrained=True, freeze_backbone=False):
    super(PhoBERTLSTM_large, self).__init__()
    phobert_config = RobertaConfig.from_pretrained("vinai/phobert-large")
    self.bert = RobertaModel(config=phobert_config)
    if from_pretrained:
        self.bert = RobertaModel.from_pretrained("vinai/phobert-large")

    self.lstm = nn.LSTM(input_size=1024, 
                        hidden_size=1024, 
                        batch_first=True, bidirectional=True, num_layers=1)
    self.dropout = nn.Dropout(config.DROP_OUT)
    self.linear = nn.Linear(1024 * 2, config.NUM_CLASSES)

    if freeze_backbone:
      for param in self.bert.parameters():
          param.require_grad = False
  
  def forward(self, input_ids, attn_mask):
    out = self.bert(input_ids=input_ids, attention_mask=attn_mask)[0]
    out, (h, c) = self.lstm(out)
    hidden = torch.cat((h[0], h[1]), dim = 1)
    out = self.linear(self.dropout(hidden))
    return out