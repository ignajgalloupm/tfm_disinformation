import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch


PAGES_RETRIEVED = 50
PAGES_FOR_EVIDENCE = 10
EMBEDDING_SIZE = 768


# check if the evidence is enough
def get_target_changes(evidence, similar_ids, pages_for_evidence=10):
    target_changes = []
    precentage_retrieved = []
    for i, r in enumerate(evidence):
        enough_evidence = False
        for evidence_set in r['unique_evidence']:
            if evidence_set is None or evidence_set.issubset(set(similar_ids[i][:pages_for_evidence])):
                enough_evidence = True
                break
        target_changes.append(enough_evidence)

        if r['all_evidence'] == [None]:
            pr = 1.0
        else:
            pr = len(set(similar_ids[i][:pages_for_evidence]).intersection(set(r['all_evidence']))) / len(r['all_evidence'])
        precentage_retrieved.append(pr)
    return target_changes, precentage_retrieved


# get the negative examples
def get_negative_examples(similar_texts, similar_ids, all_evidence):
    negative_examples = []
    # iterate similar_texts to get the last len(all_evidence) negative examples
    for i, s in enumerate(zip(similar_texts, similar_ids)):
        text_id_reversed = zip(s[0][::-1], s[1][::-1])
        n = [t[0] for t in text_id_reversed if t[1] not in all_evidence[i]]
        negative_examples.append(n[:len(all_evidence[i])])
    return negative_examples


def emb_gen_step(input_batch, vdb, emb_gen, loss_fn1, device):
    batch_size = len(input_batch['claims'])

    all_evidence = [r['all_evidence'] if r['all_evidence'] != [None] else [] for r in input_batch['evidence']]
    evidence_pages = [vdb.search_ids(all_evidence[i]) for i in range(batch_size)]
    evidence_texts = [[t.payload['text'] for t in s] for s in evidence_pages]

    # get embeddings of the claims
    outputs = emb_gen(input_batch['claims'])
 
    # search for similar pages
    similar_pages = vdb.search_similar(outputs, PAGES_RETRIEVED, with_payload=True)

    similar_texts, similar_ids = [], []
    for s in similar_pages:
        similar_texts.append([t.payload['text'] for t in s])
        similar_ids.append([t.payload['id'] for t in s])

    
    # pick as negative examples the texts of the last len(evidence_texts) of the 50 retrieved pages
    negative_examples = get_negative_examples(similar_texts, similar_ids, all_evidence)

    target_changes, percentage_retrieved = get_target_changes(input_batch['evidence'], similar_ids, PAGES_FOR_EVIDENCE)
    original_nli_targets = [int(v == 'VERIFIABLE') for v in input_batch['verifiable']]
    dynamic_nli_targets = [int(t and tc) for t, tc in zip(original_nli_targets, target_changes)]

    # check if there is at least one element in the batch with some evidence
    if all([len(e) == 0 for e in all_evidence]):
        return outputs, dynamic_nli_targets, original_nli_targets, percentage_retrieved, torch.tensor(0.0).to(device)

    # combine all the batches
    unfolded_outputs = []
    unfolded_combined_texts = []
    unfolded_labels = []
    for i in range(batch_size):
        unfolded_outputs.extend([outputs[i]] * len(evidence_texts[i] + negative_examples[i]))
        unfolded_combined_texts.extend(evidence_texts[i] + negative_examples[i])
        unfolded_labels.extend([1] * len(evidence_texts[i]) + [0] * len(negative_examples[i]))
    
    # encode the combined texts in batches
    combined_embeddings = []
    for i in range(0, len(unfolded_combined_texts), batch_size):
        combined_embeddings.extend(emb_gen(unfolded_combined_texts[i:i+batch_size]))

    combined_embeddings = torch.cat(combined_embeddings, dim=0).view([-1, EMBEDDING_SIZE])
    unfolded_outputs = torch.cat(unfolded_outputs, dim=0).view([-1, EMBEDDING_SIZE])
    unfolded_labels = torch.tensor(np.array(unfolded_labels)).to(device)

    loss1 = loss_fn1(combined_embeddings, unfolded_outputs, unfolded_labels)
    return outputs, dynamic_nli_targets, original_nli_targets, percentage_retrieved, loss1


def nli_step(vdb, nli, outputs, dynamic_nli_targets, loss_fn2, device):
    similar_embeddings = vdb.search_similar(outputs, PAGES_FOR_EVIDENCE, with_vector=True)
    # input for the NLI model
    similar_embeds = [[t.vector for t in s] for s in similar_embeddings]
    similar_embeds = torch.tensor(np.array(similar_embeds), dtype=torch.float32).to(device)
    outputs = outputs.unsqueeze(1)

    # concat the output of the embedding generator
    nli_inputs = torch.cat([outputs, similar_embeds], dim=1)

    nli_outputs = nli(nli_inputs)
    
    preds = torch.argmax(nli_outputs, dim=1).cpu().numpy()
    targets = torch.tensor(dynamic_nli_targets, dtype=torch.float32).to(device)

    # nli_outputs is 32,2 keep only the 1
    nli_outputs = nli_outputs[:, 1]

    # Convert lists of tensors to tensors
    loss2 = loss_fn2(nli_outputs, targets)

    return preds, loss2


def get_metrics(results):
    # unfold de list of dictionaries
    unfolded_preds = []
    unfolded_original_labels = []
    unfolded_dynamic_labels = []
    unfolded_percentage_retrieved = []
    unfolded_loss1 = []
    unfolded_loss2 = []
    for r in results:
        unfolded_preds.extend(r['preds'])
        unfolded_original_labels.extend(r['original_labels'])
        unfolded_dynamic_labels.extend(r['dynamic_labels'])
        unfolded_percentage_retrieved.extend(r['percentage_retrieved'])
        unfolded_loss1.append(r['loss1'])
        unfolded_loss2.append(r['loss2'])

    # to numpy
    unfolded_original_labels = np.array(unfolded_original_labels)
    unfolded_dynamic_labels = np.array(unfolded_dynamic_labels)
    unfolded_preds = np.array(unfolded_preds)
    unfolded_percentage_retrieved = np.array(unfolded_percentage_retrieved)
    
    # calculate the metrics
    overall_accuracy = accuracy_score(unfolded_original_labels, unfolded_preds)
    overall_f1 = f1_score(unfolded_original_labels, unfolded_preds)

    nli_accuracy = accuracy_score(unfolded_dynamic_labels, unfolded_preds)
    nli_f1 = f1_score(unfolded_dynamic_labels, unfolded_preds)

    # sum the total difference between the original labels and the dynamic labels
    average_enough_retrieved = 1 - sum(unfolded_original_labels - unfolded_dynamic_labels) / len(unfolded_original_labels) 
    average_total_retrieved = sum(unfolded_percentage_retrieved) / len(unfolded_percentage_retrieved)
    average_loss1 = sum(unfolded_loss1) / len(unfolded_loss1)
    average_loss2 = sum(unfolded_loss2) / len(unfolded_loss2)

    return {'nli_accuracy': nli_accuracy, 
            'nli_f1': nli_f1, 
            'overall_accuracy': overall_accuracy, 
            'overall_f1': overall_f1, 
            'average_enough_retrieved': average_enough_retrieved,
            'average_total_retrieved': average_total_retrieved,
            'average_loss1': average_loss1, 
            'average_loss2': average_loss2}