# %%
import sys
import pandas as pd
# %%
if '..' not in sys.path:
    sys.path.append('..')
from ner.ner import HealthNER
from ltp import LTP
import os
import glob
# %%
hner = HealthNER(
    model_path=r'D:\CodeRepositories\aiot2022\Electronic-Nursing-Record-Information-Extraction\data\models\model_ner_adam_1e-06_2.pt')
# %%
doc = '受侵犯的神經節及其支配的皮膚會出現整片狀紅疹及水疱'

print(hner.get_ne(doc))
# %%

ltp = LTP()
# %%

doc = ['受侵犯的神', '經節及其支配的皮膚會出現整片狀紅疹及水疱']
seg, hidden = ltp.seg([doc], is_preseged=True)
dep = ltp.dep(hidden)
for i in range(len(seg[0])):
    print(seg[0][i], '\t', dep[0][i])
# %%
seg, hidden = ltp.seg([doc])
print(hner._get_decoding(doc))
print(seg)
# %%
seg, hidden = ltp.seg([doc])
dep = ltp.dep(hidden)
for i in range(len(seg[0])):
    print(dep[0][i], seg[0][i])
# %% clean \n
# TODO: strip, remove \n, get
pdftxt_dir = r'D:\CodeRepositories\aiot2022\Electronic-Nursing-Record-Information-Extraction\data\vghks\pdftxt'
saving_dir = r'D:\CodeRepositories\aiot2022\Electronic-Nursing-Record-Information-Extraction\data\vghks\pdftxt_clean'
char_lst = ['。', '；', '：', ':', '?', '？']
for filename in os.listdir(pdftxt_dir):
    last_para = ''
    with open(os.path.join(pdftxt_dir, filename), 'r') as pdftxt:
        with open(os.path.join(saving_dir, filename), 'a+') as txt:
            for para in pdftxt:
                para = para.strip()
                if len(para) >= 1:
                    if para[-1] in char_lst:
                        txt.write(last_para + para + '\n')
                        last_para = ''
                    else:
                        last_para += para
            txt.write(last_para)

# %% find ne and coo
pdftxt_dir = r'D:\CodeRepositories\aiot2022\Electronic-Nursing-Record-Information-Extraction\data\vghks\pdftxt_clean'
saving_dir = r'D:\CodeRepositories\aiot2022\Electronic-Nursing-Record-Information-Extraction\data\vghks\symp'

# df= pd.DataFrame()
# df.columns= ['sent', 'symp', 'coo']
count = 2

for filename in os.listdir(pdftxt_dir):
    count -= 1
    if count > 0:
        continue
    with open(os.path.join(pdftxt_dir, filename), 'r') as pdftxt, open(os.path.join(saving_dir, 'symp.csv'), 'a+') as csv:
        for para in pdftxt:
            isSymp = False
            entities = hner.get_ne(para)
            if entities:
                for idx, entity in enumerate(entities):
                    # if entity['word'] not in seg[0]:
                    #     print(entity['word'], '\t', 'p')
                    # else:
                    #     print(entity['word'])
                    if entity['type'] == 'SYMP':
                        isSymp = True

                if isSymp:
                    seg, hidden = ltp.seg([para])
                    for idx, entity in enumerate(entities):
                        if entity['word'] not in seg[0]:
                            print(entity, '\t', 'p')
                        else:
                            print(entity['word'])
                    print(seg[0], '\n')

    break


# %%
