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
        '''get list of setence chunk. Devivde by sigle character and numbers
        there's a [UNK] prob occured while encoding and decoding 

        params
            setence: str, sentence 
        return
            list<str>
        '''
        # decoding = []
        # isNumber = False
        # chunk = ''
        # for i in range(len(sentence)):
        #     if sentence[i].isnumeric():
        #         isNumber = True
        #         chunk += sentence[i]
        #     elif sentence[i] in ['.', ',']:
        #         isNumber = False
        #         chunk += sentence[i]
        #         decoding.append(chunk)
        #         chunk = ''
        #     else:
        #         if isNumber:
        #             isNumber = False
        #             decoding.append(chunk)
        #             chunk = ''
        #         decoding.append(sentence[i])

        # return decoding
        encoding = self.tokenizer.encode(sentence, return_offsets_mapping=True,
                                         padding='max_length',
                                         truncation=True,
                                         max_length=128,)

        unk_lst = [i for i, x in enumerate(encoding) if x == 100]

        decoding = self.tokenizer.decode(
            encoding[1:encoding.index(102)]).split(' ')

        for unk_idx in unk_lst:
            decoding[unk_idx] = encoding[unk_idx]

        return decoding

        # count = 0
        # for idx_words, words in enumerate(decoding):
        #     for idx_word, word in enumerate(words):
        #         c += 1
        #         if word== '[':
        #             if len(words)>= idx_word+ 3:
        #                 if '[UNK]' in words[idx_words][idx_word:idx_word+3]:

    def get_ne(self, sentence):
        '''get named entity
        params
            sentence: str
        return 
            list<dict<>>, {'word': str, 'type': str, 'pos': (int, int)}
        '''
        entities = []
        labels = self._get_model_output(sentence)
        decoding = self.get_decoding(sentence)

        isEntity = False
        for i in range(len(labels)):
            if labels[i][0] == 'B':
                if isEntity:
                    end = i - 1
                    entities.append(
                        {'word': ''.join(
                            decoding[begin:end + 1]), 'type': labels[begin][2:], 'pos': (begin, end)}
                    )
                begin = i
                isEntity = True
            elif labels[i][0] == 'I':
                end = i
            elif isEntity:
                entities.append(
                    {'word': ''.join(
                        decoding[begin:end + 1]), 'type': labels[begin][2:], 'pos': (begin, end)}
                )
                isEntity = False
        return entities

        # # %%
        # n = HealthNER(
        #     model_path='D:\\CodeRepositories\\aiot2022\\ie\\data\\models\\model_ner_adam_1e-06_2.pt')
        # # %%
        # print(n.get_ne(doc1))
# %%
# hner = HealthNER(
#     r'D:\CodeRepositories\aiot2022\data\models\model_ner_adam_1e-06_2.pt')
# # %%
# sentence = '我媽媽查出有心臟病，還有早搏，醫生給他開了穩心顆粒和鹽酸美西律片，吃了以後就噁心，嘔吐，頭暈，手腳無力，還顫動，是怎麼回事，已經兩個多小時了，有危險嗎？，'
# sentence = '眼底病變：當微細動脈硬化會導致動脈內腔變細，動脈內壁變厚，使微細動脈出血，視神經乳頭浮腫，造成患者視力逐漸減低。但患者大多是再出現視力模糊後，接受眼科醫師檢查時，才發現罹患高血壓疾病。'
# # sentence = '懷孕53.天，有2.5次自然流產是不是正常'
# sentence= '（五） 返家後若有發燒、腹痛厲害或嚴重嘔吐、腹瀉應立即返診。'
# # print(len(sentence))
# # encoding = hner.tokenizer.encode(sentence, return_offsets_mapping=True,
# #                                  padding='max_length',
# #                                  truncation=True,
# #                                  max_length=128,)
# # print(encoding)
# # print(hner.get_decoding(sentence))
# # print(hner._get_model_output(sentence))
# for symp in hner.get_ne(sentence):
#     print(symp)
# %%
