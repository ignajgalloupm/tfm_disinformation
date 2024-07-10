import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.cuda.amp import autocast


PAGES_FOR_EVIDENCE = 5
MAX_EVIDENCES = 2


def cap_max_evidences(evidence_texts, max_evidences):
    # if the sum of the lengths of the evidence texts is less than max_evidences, return all the texts
    if sum([len(e) for e in evidence_texts]) <= max_evidences * len(evidence_texts):
        return evidence_texts
    else:
        # select random max_evidences from each evidence set
        capped_evidence_texts = []
        for e in evidence_texts:
            if len(e) > max_evidences:
                capped_evidence_texts.append(list(np.random.choice(e, max_evidences, replace=False)))
            else:
                capped_evidence_texts.append(e)
        return capped_evidence_texts



@autocast()
def emb_gen_step(input_batch, emb_gen, loss_fn1, device):
    batch_size = len(input_batch['claims'])

    # # get embeddings of the claims
    outputs = emb_gen(input_batch['claims'])

    evidence_texts = cap_max_evidences(input_batch['evidence_texts'], max_evidences=MAX_EVIDENCES)
    negative_examples = [dt[:len(et)] for dt, et in zip(input_batch['dissimilar_texts'], evidence_texts)]

    # check if there is at least one element in the batch with some evidence
    if all([len(e) == 0 for e in input_batch['evidence_texts']]):
        return outputs, torch.tensor(0.0).to(device)

    # combine all the batches
    unfolded_outputs = []
    unfolded_combined_texts = []
    unfolded_labels = []
    for i in range(batch_size):
        unfolded_outputs.extend([outputs[i]] * (len(evidence_texts[i]) + len(negative_examples[i])))
        unfolded_combined_texts.extend(evidence_texts[i] + negative_examples[i])
        unfolded_labels.extend([1] * len(evidence_texts[i]) + [-1] * len(negative_examples[i]))
        # unfolded_labels.extend([1] * len(evidence_texts[i]) + [0] * len(negative_examples[i]))
    
    # encode the combined texts in batches
    combined_embeddings = []
    for i in range(0, len(unfolded_combined_texts), batch_size):
        combined_embeddings.extend(emb_gen(unfolded_combined_texts[i:i+batch_size]))

    combined_embeddings = torch.stack(combined_embeddings)
    unfolded_outputs = torch.stack(unfolded_outputs)
    unfolded_labels = torch.tensor(np.array(unfolded_labels), dtype=torch.float32, requires_grad=False).to(device)

    
    loss1 = loss_fn1(combined_embeddings, unfolded_outputs, unfolded_labels) #* 500


    # scores = torch.mm(unfolded_outputs, combined_embeddings.transpose(0, 1)) * 20 # code from mpnet fine-tuning on huggingface
    # # keep only the diagonal elements
    # scores = torch.diag(scores)
    # print(scores)
    # print(unfolded_labels)
    # loss1 = loss_fn1(scores, unfolded_labels)

    return outputs, loss1


@autocast()
def nli_step(input_batch, emb_gen, nli, outputs, loss_fn2, device):    
    #with torch.no_grad():
    similar_embeds = [emb_gen(s) for s in input_batch['similar_texts']]
    similar_embeds = torch.stack(similar_embeds)

    outputs = outputs.unsqueeze(1)

    # concat the output of the embedding generator
    nli_inputs = torch.cat([outputs, similar_embeds], dim=1)

    nli_outputs = nli(nli_inputs)
    
    preds = [1 if i > 0 else 0 for i in nli_outputs]
    targets = torch.tensor(input_batch['dynamic_nli_targets'], dtype=torch.float32, requires_grad=False).unsqueeze(1).to(device)

    # Convert lists of tensors to tensors
    loss2 = loss_fn2(nli_outputs, targets)

    return preds, loss2

@torch.no_grad()
def get_metrics(results):
    # unfold de list of dictionaries
    unfolded_preds = []
    unfolded_original_labels = []
    unfolded_dynamic_labels = []
    unfolded_percentage_retrieved = []
    unfolded_loss1 = []
    unfolded_loss2 = []
    unfolded_total_loss = []
    for r in results:
        unfolded_preds.extend(r['preds'])
        unfolded_original_labels.extend(r['original_labels'])
        unfolded_dynamic_labels.extend(r['dynamic_labels'])
        unfolded_percentage_retrieved.extend(r['percentage_retrieved'])
        unfolded_loss1.append(r['loss1'])
        unfolded_loss2.append(r['loss2'])
        unfolded_total_loss.append(r['total_loss'])

    # to numpy
    unfolded_original_labels = np.array(unfolded_original_labels)
    unfolded_dynamic_labels = np.array(unfolded_dynamic_labels)
    unfolded_preds = np.array(unfolded_preds)
    unfolded_percentage_retrieved = np.array(unfolded_percentage_retrieved)
    
    # calculate the metrics
    overall_accuracy = accuracy_score(unfolded_original_labels, unfolded_preds)
    overall_f1 = f1_score(unfolded_original_labels, unfolded_preds, average='macro')

    nli_accuracy = accuracy_score(unfolded_dynamic_labels, unfolded_preds)
    nli_f1 = f1_score(unfolded_dynamic_labels, unfolded_preds, average='macro')

    changes = unfolded_original_labels - unfolded_dynamic_labels
    conditional_preds = np.where(changes == 0, unfolded_preds, 0)
    conditional_accuracy = accuracy_score(unfolded_original_labels, conditional_preds)
    conditional_f1 = f1_score(unfolded_original_labels, conditional_preds, average='macro')

    # sum the total difference between the original labels and the dynamic labels
    average_enough_retrieved = 1 - sum(changes) / len(unfolded_original_labels) 
    average_total_retrieved = sum(unfolded_percentage_retrieved) / len(unfolded_percentage_retrieved)
    average_loss1 = sum(unfolded_loss1) / len(unfolded_loss1)
    average_loss2 = sum(unfolded_loss2) / len(unfolded_loss2)
    average_total_loss = sum(unfolded_total_loss) / len(unfolded_total_loss)

    return {'nli_accuracy': nli_accuracy, 
            'nli_f1': nli_f1, 
            'overall_accuracy': overall_accuracy, 
            'overall_f1': overall_f1, 
            'conditional_accuracy': conditional_accuracy,
            'conditional_f1': conditional_f1,
            'average_enough_retrieved': average_enough_retrieved,
            'average_total_retrieved': average_total_retrieved,
            'average_loss1': average_loss1, 
            'average_loss2': average_loss2, 
            'average_total_loss': average_total_loss}