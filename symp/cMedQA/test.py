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
# %%
df = pd.read_csv(question_path)
# %%
display(df.head())
# %%
hner = HealthNER(model_path=nermodel_path)
# %%
ltp = LTP()
# %%
# body= set()
# isSymp= False
with open(r'7630002.txt', 'r', encoding='utf-8') as file:
    for para in file.readlines():
        isSymp = False
        para = para.replace(' ', '')
        seg, hidden, ignore = hner.ne_seg(para, ltp)
        nes = hner.get_ne(para)
        dep = ltp.dep(hidden)

        print(nes)
        for i in range(len(seg)):

            print(seg[i])
            print(dep[0][i])

        # for ne in nes:
        #     if ne['type'] in ['SYMP', 'DISE']:
        #         isSymp= True
        # if isSymp:
        #     for ne in nes:
        #         if ne['type'] in ['BODY']:
        #             body.add(ne['word'])
        # print(ignore, '\n')
        # dep = ltp.dep(hidden)
        # for d in dep:
        #     print(d)
# print(body)
# %%


# %%
df.loc['content'].apply(lambda x: x)  # %%
