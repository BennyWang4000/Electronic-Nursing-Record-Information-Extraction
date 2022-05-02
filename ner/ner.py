# %%
import os
import glob
from ltp import LTP
import sys
if r'D:\CodeRepositories\aiot2022\Electronic-Nursing-Record-Information-Extraction\ner' not in sys.path:
    sys.path.append(
        r"D:\CodeRepositories\aiot2022\Electronic-Nursing-Record-Information-Extraction\ner")
from model.utils import ids_dict, label_dict
import torch
from transformers import BertTokenizerFast


class HealthNER:
    def __init__(self, model_path, is_cpu=True):
        '''
        params
            model_path: str, path of ner model
            is_cpu: bool, use cpu or not
        '''
        self.tokenizer = BertTokenizerFast.from_pretrained(
            'bert-base-chinese')

        if is_cpu:
            self.model = torch.load(
                model_path, map_location=torch.device('cpu'))
        else:
            self.model = torch.load(model_path)

        self.model.eval()

    def _get_model_output(self, sentence):
        '''get output of model after decode
        params
            sentence: str
        returns
            list<str>
        '''
        sent_len = len(sentence)
        encoding = self.tokenizer(sentence,
                                  return_offsets_mapping=True,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=128,)
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}

        outputs = self.model(item['input_ids'].unsqueeze(
            0), attention_mask=item['attention_mask'].unsqueeze(0))
        logits = outputs[0]

        active_logits = logits.view(-1, self.model.num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1)
        flt_pre_np = flattened_predictions.cpu().numpy()
        labels = [ids_dict[label] for label in flt_pre_np]
        labels = labels[1:sent_len + 1]
        return labels

    def get_decoding(self, sentence: str) -> list:
        encoding = self.tokenizer.encode(sentence, return_offsets_mapping=True,
                                         padding='max_length',
                                         truncation=True,
                                         max_length=128,)
        return self.tokenizer.decode(encoding[1:encoding.index(102)]).split(' ')

    def get_ne(self, sentence):
        '''get named entity
        params
            sentence: str
        return 
            tuple<str, str, tuple<int, int>>
        '''
        entities = []
        labels = self._get_model_output(sentence)
        decoding = self.get_decoding(sentence)

        isEntity = False
        for i in range(len(labels)):
            if labels[i][0] == 'B':
                begin = i
                isEntity = True
            elif labels[i][0] == 'I':
                end = i
            elif isEntity:
                entities.append(
                    (''.join(decoding[begin:end + 1]), labels[begin][2:], (begin, end)))
                isEntity = False

        return entities

        # # %%
        # n = HealthNER(
        #     model_path='D:\\CodeRepositories\\aiot2022\\ie\\data\\models\\model_ner_adam_1e-06_2.pt')
        # # %%
        # print(n.get_ne(doc1))
