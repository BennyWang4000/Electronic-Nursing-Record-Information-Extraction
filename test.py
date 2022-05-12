# %%
import jieba
import os
from ltp import LTP
from config import *
import utils.word_segment as ws
import pandas as pd
from tqdm import tqdm
from IPython.display import display
from ner.ner import HealthNER
from opencc import OpenCC
# %%
from harvesttext import HarvestText
ht = HarvestText()

# %%
output = ws.word_segment(doc1, stopwords_path)
print(output)


# %%

ltp = LTP()
# %%
# seg
print(' '.join(list(jieba.cut('聽診有腸蠕動音'))))
print(' '.join(ltp.seg(['聽診有腸蠕動音'])[0][0]))
# %%
# %%
count = 0
crit_lst = ['HED', 'SBV', 'VOB', 'POB']
with open('D:\\CodeRepositories\\aiot2022\\ie\\data\\dep.csv', 'a+') as saving_csv:
    with open(os.path.join(alltext_dir, alltext_dict['e']), 'r') as file:
        for line in file:
            count += 1
            if count % 50 == 0:
                print(count)
            for sent in ltp.sent_split([line]):

                seg, hidden = ltp.seg([sent])
                pos = ltp.pos(hidden)
                dep = ltp.dep(hidden)
                lst = []
                for i in range(len(seg[0])):
                    lst.append((seg[0][i], pos[0][i], dep[0][i]))

                saving_csv.write('"' + sent + '","' + str(lst) + '",\n')
                # print(lst)
                # df = df.append({'text': lst, 'crit': ''}, ignore_index=True)
                # pd.concat([df, pd.DataFrame({'text': lst})], ignore_index=True)
                # dict= {'text':

                # }

                # ? show crit_lst
                # for i in range(len(seg[0])):
                #     word = dep[0][i][2]
                #     if word in crit_lst:
                #         print(word, ':', seg[0][i])
# %%
'''
'''
# ** ner
hner = HealthNER(model_path=hnermodel_path)
# %%


# %%
print(doc4)
print(hner._get_model_output(doc4))
# %%
'''
蒐集症狀 entity
常見問題
'''
# ** print crit
crit_lst = ['HED', 'SBV', 'VOB', 'POB']
sents = '肺炎主要以臨床診斷為主。一般說來，如果病人有臨床症狀如發 燒、倦=怠、寒顫、肌肉痛、呼吸快速、胸痛、.咳嗽（乾咳或有痰的 咳嗽）、痰量增加倂顏色改變且變得較黏稠、或心智改變。加上胸部X 光片上呈現異常陰影，就可診斷肺炎。如再能排除肺外感染源時， 肺炎診斷的可信度就更高。'
for sent in ltp.sent_split([sents]):
    # print('\n', sent)
    seg, hidden = ltp.seg([sent])
    print('\n', seg)
    pos = ltp.pos(hidden)
    print(pos)
    dep = ltp.dep(hidden)
    print(dep)
    for i in range(len(seg[0])):
        word = dep[0][i][2]
        if word in crit_lst:
            print(word, ':', seg[0][i])


# %%
dep_csv = pd.read_csv('D:\\CodeRepositories\\aiot2022\\ie\\data\\dep.csv')

display(dep_csv.head())
# %%
count = 0
with open('D:\\CodeRepositories\\aiot2022\\ie\\data\\nusing_prob.csv', 'a+') as saving_csv:
    with open(os.path.join(alltext_dir, alltext_dict['e']), 'r') as file:
        for line in file:
            count += 1
            if count % 50 == 0:
                print(count)

            if '護理問題' in line:
                line = line[5:]
                line = line.split('。')[0]
                saving_csv.write(line + ',\n')
# %%
df = pd.read_csv('D:\\CodeRepositories\\aiot2022\\ie\\data\\nusing_prob.csv')
display(df.drop_duplicates())
df.drop_duplicates().to_csv(
    'D:\\CodeRepositories\\aiot2022\\ie\\data\\nusing_prob_drop.csv')
# %%
