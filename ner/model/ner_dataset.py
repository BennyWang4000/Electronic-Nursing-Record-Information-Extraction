from torch.utils.data import Dataset
import json
import torch
import numpy as np


class NerCorpusDataset(Dataset):
    def __init__(self, tokenizer, data_path, label_dict, max_len=120):
        self.data = []
        with open(data_path) as f:
            for line in f:
                self.data.append(json.loads(line))
        self.len = len(self.data)
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        character = self.data[index]['character']
        character_label = self.data[index]['character_label']
        character_label_id = [self.label_dict[label]
                              for label in character_label]

        '''@return dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'offset_mapping'])'''
        encoding = self.tokenizer(' '.join(character),
                                  return_offsets_mapping=True,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_len)

        item = {key: torch.as_tensor(val) for key, val in encoding.items()}

        character_label_id.insert(0, -100)
        if len(character_label_id) > self.max_len:
            character_label_id = character_label_id[:self.max_len]
        else:
            character_label_id += [-100] * \
                (self.max_len - len(character_label_id))

        item['labels'] = torch.as_tensor(character_label_id)

        return item

    def __len__(self):
        return self.len
