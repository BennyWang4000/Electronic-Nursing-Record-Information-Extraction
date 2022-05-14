# %%
import json
import os
from tqdm import tqdm
from time import sleep
# * ==============

corpus_dir = r'D:\CodeRepositories\aiot2022\data\Chinese-HealthNER-Corpus'
corpus_lst = ['test.json', 'train.json']
saving_dir = r'D:\CodeRepositories\aiot2022\data\Chinese-HealthNER-Corpus'

# * ==============

# %%
with open(os.path.join(saving_dir, 'SYMP.txt'), 'a+') as symp:
    for i in range(10):
        symp.write(str(i) + '\n')
# %%
bar = tqdm(range(5))
for line in bar:
    sleep(1)
# %%
with open(os.path.join(saving_dir, 'SYMP.txt'), 'r') as symp:

    sentences = tqdm(symp.readlines(), unit='MB')
    for line in sentences:
        sleep(1)
    # bar= tqdm(range(len(symp.readlines())))
    # for line in symp.readlines():
    #     print(line)
    #     bar.update()
    #     sleep(1)
    # print(len(symp.readlines()))
    # for line in tqdm(symp.readlines()):
    #     print(line)
    #     sleep(100)
# %% # * write all type to txt
with open(os.path.join(saving_dir, 'SYMP.txt'), 'a+') as symp, open(os.path.join(saving_dir, 'BODY.txt'), 'a+') as body, open(os.path.join(saving_dir, 'DISE.txt'), 'a+') as dise:
    for corpus_name in corpus_lst:
        with open(os.path.join(corpus_dir, corpus_name), 'r') as corpus:
            sentences = tqdm(corpus.readlines(), unit='MB')
            for json_line in sentences:
                json_dct = json.loads(json_line)
                for idx, word_l in enumerate(json_dct['word_label']):
                    if word_l not in ['O']:
                        if word_l == 'BODY':
                            body.write(json_dct['word'][idx] + '\n')
                        elif word_l == 'DISE':
                            dise.write(json_dct['word'][idx] + '\n')
                        elif word_l == 'SYMP':
                            symp.write(json_dct['word'][idx] + '\n')
# %%
s = set((1, 2, 3, 4))
print(list(s))
# %%
type_lst = ['SYMP', 'DISE', 'BODY']
for type_name in type_lst:
    with open(os.path.join(saving_dir, type_name + '.txt'), 'r') as ori_txt, open(os.path.join(saving_dir, type_name + '_set.txt'), 'a+') as set_txt:
        word_set = set(line.strip() for line in ori_txt)
        for word in list(word_set):
            set_txt.write(word+ '\n')