import os
import pickle
import torch
from torch.utils.data.dataset import Dataset
from utils.data_utils import load_datas, bert_tokenize


class ACDDataset(Dataset):
    def __init__(self, args, tokenizer, flag='train'):
        super(ACDDataset, self).__init__()
        self.args = args
        self.flag = flag
        # tmp_cache_path = os.path.join(args.data_path, args.data_cache, flag + '.pkl')
        # if not os.path.exists(args.data_cache):
        self.inputs, self.label_ids, _, self.label_dict = load_datas(args)
        self.tokenizer = tokenizer
        _, \
        self.all_token_ids, \
        self.all_mask_ids, \
        self.all_segment_ids = bert_tokenize(self.tokenizer, self.inputs)
        self.all_input_text = []  # Initialize all_input_text
        self.all_labels = []  # Initialize all_labels
        # self.all_attention_mask = [0] * len(self.all_token_ids)   # Initialize all_attention_mask
        self.all_attention_mask = []

    def __len__(self):
        return len(self.all_token_ids)

    def __getitem__(self, item):
        input_ids = self.all_token_ids[item]
        input_mask = self.all_mask_ids[item]
        segment_ids = self.all_segment_ids[item]
        label_id = self.label_ids[item]

        tmp = {
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids,
            'label_ids': label_id
        }
        return tmp

    def collate_fn(self, samples, pad_token=0):
        batch_input_ids = [s['input_ids'] for s in samples]
        batch_input_mask = [s['input_mask'] for s in samples]
        batch_segment_ids = [s['segment_ids'] for s in samples]
        batch_label_ids = [s['label_ids'] for s in samples]

        max_len = max([len(s) for s in batch_input_ids])
        p_input_ids, p_input_mask, p_segment_ids = [], [], []
        for i in range(len(batch_input_ids)):
            tmp_input_ids = batch_input_ids[i] + [pad_token] * (max_len - len(batch_input_ids[i]))
            tmp_input_mask = batch_input_mask[i] + [pad_token] * (max_len - len(batch_input_ids[i]))
            tmp_segment_ids = batch_segment_ids[i] + [pad_token] * (max_len - len(batch_input_ids[i]))
            p_input_ids.append(tmp_input_ids)
            p_input_mask.append(tmp_input_mask)
            p_segment_ids.append(tmp_segment_ids)

        p_input_ids = torch.tensor(p_input_ids, dtype=torch.long)
        p_input_mask = torch.tensor(p_input_mask, dtype=torch.long)
        p_segment_ids = torch.tensor(p_segment_ids, dtype=torch.long)
        batch_label_ids = torch.tensor(batch_label_ids, dtype=torch.long)

        batch_tmp = {
            'input_ids': p_input_ids,
            'input_mask': p_input_mask,
            'segment_ids': p_segment_ids,
            'label_ids': batch_label_ids
        }

        return batch_tmp

    def truncate(self, max_length):
        for idx in range(len(self)):
            input_ids = self.all_token_ids[idx]
            # attention_mask = self.all_attention_mask[idx]
            if len(input_ids) > max_length:
                self.all_token_ids[idx] = input_ids[:max_length]
                # self.all_attention_mask[idx] = self.all_attention_mask[idx][:max_length]

        return input_ids
