# %%
import sys
if '..' not in sys.path:
    sys.path.append('..')
from ner.ner import HealthNER
from ltp import LTP
import os
import glob
hner = HealthNER(
    model_path=r'D:\CodeRepositories\aiot2022\Electronic-Nursing-Record-Information-Extraction\data\models\model_ner_adam_1e-06_2.pt')
# %%
doc = '受侵犯的神經節及其支配的皮膚會出現整片狀紅疹及水疱'

print(hner.get_ne(doc))
# %%

ltp = LTP()


# %%
seg, hidden = ltp.seg([doc])
print(hner._get_decoding(doc))
print(seg)
# %%
seg, hidden = ltp.seg([doc])
dep = ltp.dep(hidden)
for i in range(len(seg[0])):
    print(dep[0][i], seg[0][i])
# %%
# TODO: strip, remove \n, get
pdftxt_dir = r'G:\共用雲端硬碟\Devcup&ElderGOGO\pdftxt'
saving_dir = r'D:\CodeRepositories\aiot2022\Electronic-Nursing-Record-Information-Extraction\data\symp\pdftxt'
for i in glob.glob(os.path.join(pdftxt_dir, '*.txt')):
    pass
    # %%
