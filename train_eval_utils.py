

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