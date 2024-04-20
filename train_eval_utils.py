

# check if the evidence is enough
def get_target_changes(input_batch, similar_ids):
    target_changes = []
    precentage_retrieved = []
    for i, r in enumerate(input_batch['evidence']):
        enough_evidence = False
        for evidence_set in r['unique_evidence']:
            if evidence_set is None or evidence_set.issubset(set(similar_ids[i])):
                enough_evidence = True
                break
        target_changes.append(enough_evidence)

        pr = len(set(similar_ids[i]).intersection(set(r['all_evidence']))) / len(r['all_evidence'])
        precentage_retrieved.append(pr)
    return target_changes, precentage_retrieved


# get the negative examples
def get_negative_examples(similar_texts, similar_ids, all_evidence):
    negative_examples = []
    for i, s in enumerate(similar_texts):
        n = [t for t in s if t not in all_evidence[i]]
        negative_examples.append(n[:len(all_evidence[i])])
    return negative_examples