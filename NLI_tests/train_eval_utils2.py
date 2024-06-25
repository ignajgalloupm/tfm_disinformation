import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.cuda.amp import autocast

NUM_MODELS = 8

@autocast()
def nli_step(input_batch, emb_gen, nli, outputs, loss_fn2, device):
    similar_embeds = [v for v in input_batch['similar_embs']]
    similar_embeds = torch.tensor(np.array(similar_embeds), dtype=torch.float32).to(device)
    outputs = torch.stack(outputs).to(device)
    outputs = outputs.unsqueeze(1)

    # concat the output of the embedding generator
    nli_inputs = torch.cat([outputs, similar_embeds], dim=1)

    nli_outputs = [n(nli_inputs) for n in nli]
    
    preds = [[1 if i > 0 else 0 for i in nli_out] for nli_out in nli_outputs]
    targets = torch.tensor(input_batch['dynamic_nli_targets'], dtype=torch.float32).unsqueeze(1).to(device)

    # Convert lists of tensors to tensors
    loss2 = [loss_fn2(nli_outs, targets) for nli_outs in nli_outputs]

    return preds, loss2

@torch.no_grad()
def get_metrics(results):
    # unfold de list of dictionaries
    unfolded_preds = [[] for _ in range(NUM_MODELS)]
    unfolded_original_labels = []
    unfolded_dynamic_labels = []
    unfolded_percentage_retrieved = []
    unfolded_loss2 = [[] for _ in range(NUM_MODELS)]
    for r in results:
        for i in range(NUM_MODELS):
            unfolded_preds[i].extend(r['preds'][i])
            unfolded_loss2[i].append(r['loss2'][i])
        unfolded_original_labels.extend(r['original_labels'])
        unfolded_dynamic_labels.extend(r['dynamic_labels'])
        unfolded_percentage_retrieved.extend(r['percentage_retrieved'])
        

    # to numpy
    unfolded_original_labels = np.array(unfolded_original_labels)
    unfolded_dynamic_labels = np.array(unfolded_dynamic_labels)
    unfolded_preds = np.array(unfolded_preds)
    unfolded_percentage_retrieved = np.array(unfolded_percentage_retrieved)
    
    # calculate the metrics
    nli_f1 = [f1_score(unfolded_dynamic_labels, unfolded_preds[i], average='macro') for i in range(len(unfolded_preds))]

    # sum the total difference between the original labels and the dynamic labels
    average_enough_retrieved = 1 - np.sum(unfolded_original_labels - unfolded_dynamic_labels) / len(unfolded_original_labels) 
    average_total_retrieved = np.sum(unfolded_percentage_retrieved) / len(unfolded_percentage_retrieved)
    average_loss2 = np.sum(unfolded_loss2, axis=-1) / len(unfolded_loss2[0])

    return { 'nli_f1': nli_f1, 
            'average_enough_retrieved': average_enough_retrieved.tolist(),
            'average_total_retrieved': average_total_retrieved.tolist(), 
            'average_loss2': average_loss2.tolist()}