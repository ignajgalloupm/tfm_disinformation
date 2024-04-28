import numpy as np
from sklearn.metrics import accuracy_score, f1_score


# check if the evidence is enough
def get_target_changes(input_batch, similar_ids, pages_for_evidence=10):
    target_changes = []
    precentage_retrieved = []
    for i, r in enumerate(input_batch['evidence']):
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