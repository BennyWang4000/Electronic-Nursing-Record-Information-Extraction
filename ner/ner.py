# %%
import os
import glob
from ltp import LTP
import sys
if '../../ner' not in sys.path:
    sys.path.append('../../ner')

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
        encoding = self.tokenizer.encode(sentence, return_offsets_mapping=True,
                                         padding='max_length',
                                         truncation=True,
                                         max_length=128,)

        # unk_lst = [i for i, x in enumerate(encoding) if x == 100]

        decoding = self.tokenizer.decode(
            encoding[1:encoding.index(102)]).split(' ')

        count = 0
        for idx_words, words in enumerate(decoding):
            if words == '[UNK]':
                decoding[idx_words] = sentence[count]
                count += 1
                continue
            count += len(str(words))

        return decoding

    def get_ne(self, sentence, type=None):
        '''get named entity
        params
            sentence: str
            type: list<str> label that you want
        return 
            list<dict<>>, {'word': str, 'type': str, 'pos': (int, int)}

        known issue
            if named entity is at the last or the first of sentence, it will retur empty string
            temporary solution: add 。 in sentence :(
        '''
        entities = []
        labels = self._get_model_output(sentence)
        decoding = self.get_decoding(sentence)

        # isEntity = False

        begin_lst = [i for i, x in enumerate(labels) if x[0] == 'B']

        for begin in begin_lst:
            end = begin
            while labels[end + 1 if end + 1 < len(labels) else -1][0] == 'I':
                end += 1
            entities.append(
                {'word': ''.join(
                    decoding[begin:end + 1 if end + 1 < len(labels) else -1]), 'type': labels[begin][2:], 'pos': (begin, end + 1 if end + 1 < len(labels) else -1)}
            )

        if type:
            for ne in entities:
                if ne['type'] not in type:
                    entities.remove(ne)

        return entities

    def get_ne_idx(self, sent_lst, ne_lst, ignore=None):
        pos_lst = []
        ne_idx = 0
        for idx, word in enumerate(sent_lst):
            if ne_lst[ne_idx]['word'] in word:
                pos_lst.append(idx)
                ne_idx += 1
                if ne_idx >= len(ne_lst):
                    break

        return pos_lst

    def ne_seg(self, sentence, ltp: LTP):
        '''segment entity in ltp for dependency tree. required ltp
        params
            setence: str
            ltp: ltp.LTP()
        return
            seg: list<str>
            hidden: ?
            ignore: list<ne>

        known issue: 


        '''
        nes = self.get_ne(sentence)
        seg, _ = ltp.seg([sentence])

        for ne in nes:
            # * ne_word = sentence[ne['pos'][0]:ne['pos'][1]]
            # if ne['word'] not in seg[0]:
            isSet0 = False
            isSet1 = False
            count = 0
            for idx_w, word in enumerate(seg[0]):

                isSet0 = False
                for idx_c, word in enumerate(word):
                    if (word == '@' and word[-1 if idx_c + 1 >= len(word) else idx_c + 1] == '@') or (word == '@' and word[idx_c - 1] == '@'):
                        continue
                    if count == ne['pos'][0] or count == ne['pos'][1]:
                        if isSet0:
                            seg[0][idx_w] = seg[0][idx_w][:idx_c + 2] + \
                                '@@' + seg[0][idx_w][idx_c + 2:]
                            isSet1 = True
                        else:
                            seg[0][idx_w] = seg[0][idx_w][:idx_c] + \
                                '@@' + seg[0][idx_w][idx_c:]
                            isSet0 = True
                    count += 1
                    if isSet1:
                        break
                isSet0 = False

        seg_preseg = '@@'.join(seg[0])
        seg_preseg = seg_preseg.replace('@@@@@@@@@@@@', '@@')
        seg_preseg = seg_preseg.replace('@@@@@@@@@@', '@@')
        seg_preseg = seg_preseg.replace('@@@@@@@@', '@@')
        seg_preseg = seg_preseg.replace('@@@@@@', '@@')
        seg_preseg = seg_preseg.replace('@@@@', '@@')
        seg, hidden = ltp.seg([seg_preseg.split('@@')], is_preseged=True)

        ignore = []
        for ne in nes:
            if ne['word'] not in seg[0]:
                ignore.append(ne)

        return seg, hidden, ignore


# # %%
# hner = HealthNER(
#     r'D:\CodeRepositories\aiot2022\data\models\model_ner_adam_1e-06_2.pt')
# # %%
# with open(r'D:\CodeRepositories\aiot2022\data\vghks\pdftxt_clean\7A10023.txt', 'r', encoding='utf-8') as file:
#     for para in file.readlines():
#         para= para.replace(' ', '').replace('.', '')

#         print(para)
# #%%
    # ltp = LTP()
    # # %%
    # content = '我痛在背部，背部很痛。'

    # ne_lst = []

    # nes = hner.get_ne(content, type=['BODY', 'DISE', 'SYMP'])
    # seg, hidden = ltp.seg([content])
    # print(seg)
    # seg, hidden, _ = hner.ne_seg(content, ltp)
    # # print(seg)
    # print(seg)
    # nes = hner.get_ne(content)
    # for ne in nes:
    #     print(ne)

    # ne_pos_lst = hner.get_ne_idx(seg[0], nes)

    # dep = ltp.dep(hidden)
    # for i in range(len(seg[0])):
    #     print(seg[0][i], dep[0][i])
    # print()

    # %%
    # # sentence = '我媽媽查出有心臟病，還有早搏，醫生給他開了穩心顆粒和鹽酸美西律片，吃了以後就噁心，嘔吐，頭暈，手腳無力，還顫動，是怎麼回事，已經兩個多小時了，有危險嗎？，'
    # # sentence = '眼底病變：當微細動脈硬化會導致動脈內腔變細，動脈內壁變厚，使微細動脈出血，視神經乳頭浮腫，造成患者視力逐漸減低。但患者大多是再出現視力模糊後，接受眼科醫師檢查時，才發現罹患高血壓疾病。'
    # # sentence = '懷孕53天，有25次自然流產是不是正常，有嘔吐，肚子痛'
    # sentence= ' （五） 返家後若有發燒、腹痛厲害或嚴重嘔吐、腹瀉應立即返診。'
    # # # print(len(sentence))
    # # # encoding = hner.tokenizer.encode(sentence, return_offsets_mapping=True,
    # # #                                  padding='max_length',
    # # #                                  truncation=True,
    # # #                                  max_length=128,)
    # # # # print(encoding)
    # # print(hner.get_decoding(sentence))
    # # print(hner._get_model_output(sentence))
    # for symp in hner.get_ne(sentence):
    #     print(symp)
    # %%
