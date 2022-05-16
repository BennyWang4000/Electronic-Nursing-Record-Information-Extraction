# %%
from glob import glob
from ltp import LTP
import pandas as pd
import os
from config import *
from IPython.display import display
import sys
from sentence_unit import SentenceUnit

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
# # %%


# class get_dep():
#     pass


# %%
for i in range(1):
    # content = df.iloc[i].to_dict()['content']
    # content = '痛在背部，背部很痛。'
    # content = '算盤珠移位，骨裂，導致腰部，臀部疼痛，怎麼辦？'
    # ! content= '一發病就頭暈，全身出虛汗，臉色蒼白，心裏作嘔，說話吃力，身體站不穩。'
    # ? content = '出現自拔管路行為、混亂行為可能自傷或傷人，持續保護性身體約束中'  # * return 身體 約束
    # content = '口腔有異味'
    # content= '我前天早上暈倒了。在那站着，過了一會感覺疲乏無力，接着嘔吐，但什麼也麼吐出來，還頭疼，肚子也有點疼，耳垂特別燙，還出汗。一陣熱一陣冷的'
    # content= '全身發熱，但沒發燒，手腳冰冷，全身都沒有力氣，是不是食物中毒' #* return null
    # content= '意識欠清、躁動不安，預防自拔管路行為，持續保護性身體約束中。'
    # content= '喉嚨疼痛，無法進食'
    # ? content= '背臀區皮膚完整，嘴唇因插管過程導致黏膜破損存，右腰一處血泡未破存'
    # content= '口臭，嘴裏苦，小便黃嘴裏總是沒味，有時候很苦，口臭很大'
    # content= '從上禮拜背上就有起疹子很癢很痛，無法入睡'
    # content= '發燒至三十七度，全身發熱，頭暈無食慾'
    # content = '指甲周圍變黑，看起來很噁'
    # ? content= '皮膚完整且無受損情形，肢體末梢水腫、膚色粉紅'
    # content= '昨晚開始耳嗚，頭暈，今天早上左耳聽到的聲音很小想知道是不是中耳炎'
    # content= '腹部柔軟，聽診腸蠕動音正常，灌食過程中暫無肺吸入情形發生'
    content = '從昨天開始頭就很暈，沒有發燒，右腳和左手手腕的關節痛'

    content = '。' + content + '。'

    nes = hner.get_ne(content, type=['BODY', 'DISE', 'SYMP'])

    seg, hidden = ltp.seg([content])
    print(seg)
    seg, hidden, ignore = hner.ne_seg(content, ltp)
    print(seg)
    print('ignore:', ignore)

    pos = ltp.pos(hidden)
    dep = ltp.dep(hidden)
    sdp = ltp.sdp(hidden)
    # for i in range(len(seg[0])):
    #     print(seg[0][i], '\t', pos[0][i], '\t', dep[0][i])

    ne_idx_lst = hner.get_ne_idx(seg[0], nes)
    print('===========')
    srl = ltp.srl(hidden, keep_empty=False)
    for s in srl[0]:
        s_lst = [(arg[0], seg[0][arg[1]:arg[2]+1])for arg in s[1]]
        print(
            seg[0][s[0] - 1],
            s_lst
        )

    sentence = SentenceUnit(
        seg_lst=seg[0],
        pos_lst=pos[0],
        dep_lst=dep[0],
        sdp_lst=sdp[0],
        ne_idx_lst=ne_idx_lst,
        ne_dct=nes,
    )
    print('===========')
    sentence.print_words()

    print('===========')

    body_sdp_lst = []
    symp_sdp_lst = []
    dise_sdp_lst = []
    body_dep_lst = []
    symp_dep_lst = []
    dise_dep_lst = []

    body_set = set()
    symp_set = set()
    dise_set = set()

    dep_used_idx_lst = []

    for idx, ne_idx in enumerate(ne_idx_lst):
        sdp_tup = ()
        dep_tup = ()
        # * SDP
        ne_word = sentence.get_word_by_idx(ne_idx)
        # if ne_word.type == 'BODY':
        #     body_set.add(ne_word['word'])
        # elif ne_word.type == 'DISE':
        #     dise_set.add(ne_word['word'])
        # elif ne_word.type == 'SYMP':
        #     symp_set.add(ne_word['word'])

        ne_sdp_idx = ne_word.sdp_idx
        ne_sdp_word = sentence.get_word_by_idx(ne_word.sdp_idx)

        ne_dep_idx = ne_word.dep_idx
        ne_dep_word = sentence.get_word_by_idx(ne_word.dep_idx)

        if ne_word.type== 'BODY':
            sdp_tup += (ne_word.word, ne_sdp_word.word,)
            dep_used_idx_lst.append(ne_dep_word.idx)
            
            dep_tup += (ne_word.word,)
            while ne_dep_word.type== 'BODY':
                # if ne_dep_word.idx not in dep_used_idx_lst:
                dep_tup += (ne_dep_word.word,)
                dep_used_idx_lst.append(ne_dep_word.idx)
                ne_dep_word = sentence.get_word_by_idx(ne_dep_word.dep_idx)
            dep_tup += (ne_dep_word.word,)

        sdp_on_ne_lst = sentence.get_depend(ne_idx, 'sdp')
        for word in sdp_on_ne_lst:
            if word.sdp_type == 'mNEG':
                sdp_tup += (word.word,)

        if ne_dep_word.dep_type == 'CMP':
            print('CMP:')
            ne_dep_idx = ne_dep_word.dep_idx
        if ne_dep_word.dep_type == 'HED' and ne_dep_word.pos == 'v':
            for word in sentence.words:
                if word.dep_idx == ne_dep_word.idx and word.dep_type == 'VOB':
                    ne_dep_idx = word.idx
                    break
        if ne_dep_word.dep_type == 'HED' and ne_dep_word.pos == 'p':
            for word in sentence.words:
                if word.dep_idx == ne_dep_word.idx and word.dep_type == 'SBV':
                    ne_dep_idx = word.idx
                    break

        ne_sdp_word = sentence.get_word_by_idx(ne_sdp_idx)
        ne_dep_word = sentence.get_word_by_idx(ne_dep_idx)

        print(ne_word.word + '\t' + ne_word.type, '\t--->\t',
              ne_sdp_word.word + '\t' + ne_sdp_word.type)

        if ne_word.type == 'SYMP':
            sdp_tup += (ne_word.word,)
            dep_tup += (ne_word.word,)
            symp_sdp_lst.append(sdp_tup)
            symp_dep_lst.append(dep_tup)
        if ne_word.type == 'DISE':
            sdp_tup += (ne_word.word,)
            dep_tup += (ne_word.word,)
            dise_sdp_lst.append(sdp_tup)
            dise_dep_lst.append(dep_tup)
        if ne_word.type == 'BODY':
            body_sdp_lst.append(sdp_tup)
            body_dep_lst.append(dep_tup)
        # health_lst.append(health_tup)
    print('===========')

    print('BODY')
    for body in body_sdp_lst:
        print('\t', body)
    print()
    for body in body_dep_lst:
        print('\t', body)
    print('SYMP')
    for symp in symp_sdp_lst:
        print('\t', symp)
    print()
    for symp in symp_dep_lst:
        print('\t', symp)
    print('DISE')
    for dise in dise_sdp_lst:
        print('\t', dise)
    print()
    for dise in dise_dep_lst:
        print('\t', dise)
# %% #* body search
body_dir = r'G:\共用雲端硬碟\Devcup&ElderGOGO\衛教\entity_ver2\BODY'

edu_lst = []

for filepath in glob(body_dir, '*.txt'):
    bodies = set(line.strip() for line in open(filepath, 'r'))
    body_union = ''

# %% old D:
    # for idx, ne_idx in enumerate(ne_idx_lst):
    #     ne_dep_idx = dep[0][ne_idx][1] - 1
    #     # ne_dep = dep[0][ne_dep_idx][0]
    #     # ne_dep_word = seg[0][ne_dep_idx]
    #     ne_dep = dep[0][ne_dep_idx]
    #     if ne_dep[2] == 'CMP':
    #         print('CMP:')
    #         ne_dep_idx = dep[0][ne_dep_idx][1]
    #     if ne_dep[2] == 'HED':
    #         print('HED:')
    #         for i in range(0):
    #             pass

    #     ne_dep_word = seg[0][ne_dep_idx]
    #     print(nes[idx]['word'], nes[idx]['type'], ne_dep_word,
    #           'NE' if ne_dep_idx in ne_idx_lst else '')
    # print('===========')

    # **** DEP
    # ne_word = sentence.get_word_by_idx(ne_idx)
    # ne_dep_idx = ne_word.dep_idx
    # ne_dep_word = sentence.get_word_by_idx(ne_word.dep_idx)

    # if ne_dep_word.dep_type == 'CMP':
    #     print('CMP:')
    #     ne_dep_idx = ne_dep_word.dep_idx
    # if ne_dep_word.dep_type == 'HED' and ne_dep_word.pos == 'v':
    #     for word in sentence.words:
    #         if word.dep_idx == ne_dep_word.idx and word.dep_type == 'VOB':
    #             ne_dep_idx = word.idx
    #             break
    # if ne_dep_word.dep_type == 'HED' and ne_dep_word.pos == 'p':
    #     for word in sentence.words:
    #         if word.dep_idx == ne_dep_word.idx and word.dep_type == 'SBV':
    #             ne_dep_idx = word.idx
    #             break
    # ne_dep_word = sentence.get_word_by_idx(ne_dep_idx)
    # print('\t', ne_word.word + '\t' + ne_word.type, '\t--->\t',
    #       ne_dep_word.word + '\t' + ne_dep_word.type)


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
