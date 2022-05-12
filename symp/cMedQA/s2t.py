# %%
import pandas as pd
import os
from config import *
from opencc import OpenCC
import sys
from IPython.display import display
sys.path.append(r"..\..")
if True:
    from ner.ner import HealthNER

# %% s2t
opencc_cvt = OpenCC('s2t')
df = pd.read_csv(os.path.join(cmedqa_dir, cmedqa_dct['q']))
df['content'] = df['content'].apply(lambda x: opencc_cvt.convert(x))
df.to_csv(os.path.join(saving_dir, 'question_s2t.csv'))
# %%
pd.read_csv(question_path)
# %%
hner = HealthNER(model_path=nermodel_path)
# %% check position in decode
doc = '約束部位(雙手)皮膚完整且無受損情形，肢體末梢溫暖、膚色粉紅，無拔管意外事件發生。'
doc_e = hner._get_model_output(doc)
print(len(doc))
print(doc_e)
print(len(doc_e))
for i in range(len(doc)):
    print(doc[i], doc_e[i])
# %%
