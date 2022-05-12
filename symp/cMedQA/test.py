# %%
from ltp import LTP
import pandas as pd
import os
from config import *
from IPython.display import display
import sys
sys.path.append(r"..\..")
if True:
    from ner.ner import HealthNER
# %%
df = pd.read_csv(question_path)
# %%
display(df.head())
# %%
hner = HealthNER(model_path=nermodel_path)
# %%
ltp = LTP()
#%%

with open(r'7630002.txt', 'r', encoding='utf-8') as file:
    for para in file.readlines():
        seg, hidden, ignore= hner.ne_seg(para, ltp)
        print(seg)
        print(ignore, '\n')
# %% check position in decode
count = 2
isStart = True
with open(r'7630002.txt', 'r', encoding='utf-8') as file:
    for para in file.readlines():
        if isStart:
            isStart = False
            continue

        para = para.replace(' ', '')
#
        # para = para.replace(' ', '')
        # count-= 1
        # if count== 0:
        nes = hner.get_ne(para)
        seg, hidden = ltp.seg([para])

        # dep = ltp.dep(hidden)

        for ne in nes:

            # if
            # * ne_word = para[ne['pos'][0]:ne['pos'][1]]
            ne_word = ne['word']
            # print(ne_word, '\t', '' if ne_word in seg[0] else 'NONE')
            # if ne_word not in seg[0]:
            #     para= para[:ne['pos'][0]]+ '/'+ para[:ne['pos'][0]]
            #     para= para[:ne['pos'][1]]+ '/'+ para[:ne['pos'][1]]

        # print(ltp.seg([para])[0])

            if ne_word not in seg[0]:
                isSet0 = False
                isSet1 = False
                count = 0
                for idx_w, word in enumerate(seg[0]):
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

        # print(seg[0])
        seg_preseg = '@@'.join(seg[0])
        seg_preseg = seg_preseg.replace('@@@@', '@@')

        seg, hidden = ltp.seg([seg_preseg.split('@@')], is_preseged=True)
        for ne in nes:
            if ne['word'] not in seg[0]:
                print(ne)

        print(seg[0])
        # dep = ltp.dep(hidden)

        # for i in range(len(seg[0])):
        #     print(seg[0][i], '\t', dep[0][i])
        # break
        # %%


def qoo(sentence):
    # ltp.
    pass


# %%
df.loc['content'].apply(lambda x: x)
# %%
for i, s in enumerate('12345678'):
    print(i, s)
# %%
