import os
import re
import unicodedata
import py_vncorenlp

import torch
from torch import nn

class WordSegmenter:
    vncorenlp_data_dir = 'vncorenlp'
    if not os.path.exists('vncorenlp'):
        os.makedirs('vncorenlp')
    absolute_vncorenlp_path = os.path.abspath('vncorenlp')

    if len(os.listdir(absolute_vncorenlp_path)) == 0:
        py_vncorenlp.download_model(save_dir=absolute_vncorenlp_path)
    
    rdrsegmenter = None

    @staticmethod
    def getSegmenter():
        if __class__.rdrsegmenter is None:
            __class__.rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir= __class__.absolute_vncorenlp_path)
        return __class__.rdrsegmenter



def bert_collate_fn(batch):
    '''
    Create collate function for batch of data when feeding to BERT base model
    '''
    input_ids_list, attn_mask_list, label_list = [], [], []

    for input_ids, attn_mask, label in batch:
        input_ids_list.append(input_ids)
        attn_mask_list.append(attn_mask)
        label_list.append(label)
    label_list = torch.tensor(label_list)
    
    input_ids_list = nn.utils.rnn.pad_sequence(input_ids_list)
    attn_mask_list = nn.utils.rnn.pad_sequence(attn_mask_list)

    input_ids_list = torch.permute(input_ids_list, (1, 0))
    attn_mask_list = torch.permute(attn_mask_list, (1, 0))
    
    return input_ids_list, attn_mask_list, label_list


def lstm_collate_fn(batch):
    '''
    Create collate function for batch of data when feeding to LSTM base model
    '''
    vectors_list, label_list = [], []
    for text, label in batch:
        vectors_list.append(text)
        label_list.append(label)

    label_list = torch.tensor(label_list, dtype=torch.int64)
    vectors_list = nn.utils.rnn.pad_sequence(vectors_list)

    return vectors_list, label_list


def preprocess_fn(text):
    '''
    Preprocessing text
    '''
    text = unicodedata.normalize('NFKC', str(text))
    text = re.sub('\s+', ' ', text)
    text = text.strip()
    text = WordSegmenter.getSegmenter().word_segment(text)
    return text[0]