from matplotlib import pyplot as plt
import torch
from sklearn.metrics import accuracy_score
import os


label_dict= {'O': 0,
             'B-BODY': 1, 'I-BODY': 2, 
             'B-SYMP': 3, 'I-SYMP': 4, 
             'B-INST': 5, 'I-INST': 6, 
             'B-EXAM': 7, 'I-EXAM': 8, 
             'B-CHEM': 9, 'I-CHEM': 10, 
             'B-DISE': 11, 'I-DISE': 12, 
             'B-DRUG': 13, 'I-DRUG': 14,
             'B-SUPP': 15, 'I-SUPP': 16,
             'B-TREAT': 17, 'I-TREAT': 18,
             'B-TIME': 19, 'I-TIME': 20, 
             }
             
ids_dict= {label_dict[k]: k for idx, k in enumerate(label_dict)}


def show_n_save_plt(name, data, path):
    plt.figure()
    plt.plot(data)
    plt.title(name)
    plt.show
    plt.savefig(os.path.join(path, name + '.jpg'))

def save_model(state, path, name):
    torch.save(state, os.path.join(path, name))
    
def cal_acc(labels, tr_logits, num_labels):
    # compute training accuracy
    flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
    active_logits = tr_logits.view(-1, num_labels) # shape (batch_size * seq_len, num_labels)
    flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
    active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
    #active_labels = torch.where(active_accuracy, labels.view(-1), torch.tensor(-100).type_as(labels))
    labels = torch.masked_select(flattened_targets, active_accuracy)
    predictions = torch.masked_select(flattened_predictions, active_accuracy)
#     tr_labels.extend(labels)
#     tr_preds.extend(predictions)
    tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
    
    return tmp_tr_accuracy
