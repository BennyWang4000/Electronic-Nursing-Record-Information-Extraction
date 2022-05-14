# %%
import os
from IPython.display import display
import pandas as pd
from config import *
import sys
if '../..' not in sys.path:
    sys.path.append('../..')
    sys.path.append('../../ner')
from ner.ner import HealthNER

# %%
df = pd.read_csv(os.path.join(cmedqa_dir, cmedqa_s2t_lst[0]))
# %%
hner = HealthNER(nermodel_path)

# %%
body = set()
symp = set()
dise = set()
# %%


def add_to_set(para):
    nes = hner.get_ne(para)
    for ne in nes:
        if ne['word'] == ['SYMP']:
            symp.add(ne['word'])
        if ne['word'] == ['BODY']:
            body.add(ne['word'])
        if ne['word'] == ['DISE']:
            dise.add(ne['word'])


# %%
for filename in cmedqa_s2t_lst:
    df = pd.read_csv(os.path.join(cmedqa_dir, filename))
    df['content'].apply(lambda x: add_to_set(x))
    break
#%%
print(symp)

# %%
with open(os.path.join(saving_dir, 'SYMP'), 'w+') as symp, open(os.path.join(saving_dir, 'BODY'), 'w+') as body, open(os.path.join(saving_dir, 'DISE'), 'w+') as dise:

    # %%
