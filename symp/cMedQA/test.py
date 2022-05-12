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
# %% check position in decode
count = 2
with open(r'7630002.txt', 'r', encoding='utf-8') as file:
    for para in file.readlines():
        para = para.replace(' ', '')
        # count-= 1
        # if count== 0:
        if True:
            nes = hner.get_ne(para)
            seg, hidden = ltp.seg([para])
            dep = ltp.dep(hidden)
            for ne in nes:
                # if
                ne_word = para[ne['pos'][0]:ne['pos'][1]]
                print(ne_word, '\t', '' if ne_word in seg[0] else 'NONE')
            # break

# %%
a = [0, 1, 2, 3, 4]
