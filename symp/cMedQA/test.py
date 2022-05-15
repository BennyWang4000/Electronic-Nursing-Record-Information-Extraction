# %%
from ltp import LTP
import pandas as pd
import os
from config import *
from IPython.display import display
import sys

sys.path.append(r"..\..")
sys.path.append(r'..\..\ner')
if True:
    from ner.ner import HealthNER
# # %%
# df = pd.read_csv(os.path.join(cmedqa_dir, cmedqa_s2t_dct['q']))
# # %%
# display(df.head())
# print(df.iloc[1].to_dict()['content'])
# %%
hner = HealthNER(model_path=nermodel_path)
# %%
ltp = LTP()
# %%
class SentenceUnit():


class WordUnit():
    pass


# %%
for i in range(1):
    # content = df.iloc[i].to_dict()['content']
    content = '我痛在背部，背部很痛。'

    ne_lst = []

    nes = hner.get_ne(content, type=['BODY', 'DISE', 'SYMP'])
    seg, hidden = ltp.seg([content])
    print(seg)
    seg, hidden, _ = hner.ne_seg(content, ltp)
    # print(seg)
    print(seg)
    nes = hner.get_ne(content)
    for ne in nes:
        print(ne)

    dep = ltp.dep(hidden)
    for i in range(len(seg[0])):
        print(seg[0][i], dep[0][i])

    ne_pos_lst = hner.get_ne_idx(seg[0], nes)
    for idx, ne_pos in enumerate(ne_pos_lst):
        print(nes[idx]['word'], nes[idx]['type'])
        print(seg[0][dep[0][dep[0][ne_pos][1]-1][0]-1])
    '''
    用ne的位置找到他依存的對象
    '''
    print('===========')
# %%
# %%
# # body= set()
# # isSymp= False
# with open(r'7630002.txt', 'r', encoding='utf-8') as file:
#     for para in file.readlines():
#         isSymp = False
#         para = para.replace(' ', '')
#         seg, hidden, ignore = hner.ne_seg(para, ltp)
#         nes = hner.get_ne(para)
#         dep = ltp.dep(hidden)
#         # # for ne in nes:

#         # print(nes)
#         for i in range(len(seg)):

#             print(seg[i])
#             print(dep[0][i])

#         # for ne in nes:
#         #     if ne['type'] in ['SYMP', 'DISE']:
#         #         isSymp= True
#         # if isSymp:
#         #     for ne in nes:
#         #         if ne['type'] in ['BODY']:
#         #             body.add(ne['word'])
#         # print(ignore, '\n')
#         # dep = ltp.dep(hidden)
#         # for d in dep:
#         #     print(d)
# # print(body)
# # %%


# # %%
# df.loc['content'].apply(lambda x: x)  # %%

# %%
