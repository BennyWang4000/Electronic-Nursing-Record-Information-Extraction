# %%
from .model.utils import ids_dict, label_dict
import torch
from transformers import BertTokenizerFast
import sys
sys.path.append("ner")
# %%


class HealthNER:
    def __init__(self, model_path, is_cpu=True):
        '''
        params
            model_path: str, path of ner model
            is_cpu: bool, use cpu or not
        '''
        self.tokenizer = BertTokenizerFast.from_pretrained(
            'bert-base-chinese')

        if is_cpu:
            self.model = torch.load(
                model_path, map_location=torch.device('cpu'))
        else:
            self.model = torch.load(model_path)

        self.model.eval()

    def _get_model_output(self, sentence):
        '''get output of model after decode
        params
            sentence: str
        returns
            list<str>
        '''
        sent_len = len(sentence)
        encoding = self.tokenizer(sentence,
                                  return_offsets_mapping=True,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=128,)
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}

        outputs = self.model(item['input_ids'].unsqueeze(
            0), attention_mask=item['attention_mask'].unsqueeze(0))
        logits = outputs[0]

        active_logits = logits.view(-1, self.model.num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1)
        flt_pre_np = flattened_predictions.cpu().numpy()
        labels = [ids_dict[label] for label in flt_pre_np]
        labels = labels[1:sent_len]
        return labels

    def get_ne(self, sentence):
        '''get named entity
        params
        '''
        labels = self._get_model_output(sentence)
        for label in labels:
            if label:
                pass

            # # %%
            # n = HealthNER(
            #     model_path='D:\\CodeRepositories\\aiot2022\\ie\\data\\models\\model_ner_adam_1e-06_2.pt')
            # # %%
            # print(n.get_ne(doc1))
