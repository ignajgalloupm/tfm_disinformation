from torch.utils.data import Dataset
import random

PAGES_FOR_RETRIEVAL = 100
PAGES_FOR_EVIDENCE = 5
MAX_EVIDENCES = 2


class Sub_Dataset(Dataset):
    def __init__(self, data, vdb, set_type='train', evdb=None):
        self.dataset = data
        self.vdb = vdb
        self.evdb = evdb
        self.set_type = set_type

    def __len__(self):            
        return len(self.dataset['claims'])

    def __getitem__(self, index):

        similar_pages = self.vdb.search_similar(self.dataset['claim_embs'][index].unsqueeze(0), PAGES_FOR_RETRIEVAL, with_payload=True, with_vector=False)[0]
        original_nli_target = int(self.dataset['label'][index] == 'SUPPORTS')

        if self.set_type == 'train':
            if original_nli_target: 
                if random.random() < 0.4:
                    # randomly select 5 indices from the top 10 retrieved pages
                    random_indices = random.sample(range(10), PAGES_FOR_EVIDENCE)
                    selected_pages = [similar_pages[i] for i in random_indices]
                    similar_texts = [t.payload['text'] for t in selected_pages]
                    similar_ids = [t.payload['id'] for t in selected_pages]
                    enough_evidence = self.check_enough_evidence(self.dataset['evidence'][index], similar_ids)
                    if not enough_evidence:
                        original_nli_target = 0
                else:
                    shuffled_pages = random.sample(similar_pages[:PAGES_FOR_EVIDENCE], PAGES_FOR_EVIDENCE)
                    similar_texts = [t.payload['text'] for t in shuffled_pages]
                    similar_ids = [t.payload['id'] for t in shuffled_pages]
                    enough_evidence = self.check_enough_evidence(self.dataset['evidence'][index], similar_ids)
            else:
                similar_evidences = self.evdb.search_similar(self.dataset['claim_embs'][index].unsqueeze(0), 5, with_payload=True, with_vector=False)[0]
                combined_pages = similar_pages[:PAGES_FOR_EVIDENCE] + similar_evidences
                selected_pages = random.sample(combined_pages, PAGES_FOR_EVIDENCE)
                similar_texts = [t.payload['text'] for t in selected_pages]
                similar_ids = [t.payload['id'] for t in selected_pages]
                enough_evidence = False
        else:
            similar_texts = [t.payload['text'] for t in similar_pages[:PAGES_FOR_EVIDENCE]]
            similar_ids = [t.payload['id'] for t in similar_pages[:PAGES_FOR_EVIDENCE]]
            enough_evidence = self.check_enough_evidence(self.dataset['evidence'][index], similar_ids)

        all_evidence = self.dataset['evidence'][index]['all_evidence'] if self.dataset['evidence'][index]['all_evidence'] != [None] else []
        evidence_pages = self.vdb.search_ids(all_evidence)
        evidence_texts = [t.payload['text'] for t in evidence_pages]



        easy_difficulty = self.vdb.search_dissimilar(10)
        easy_texts = [t['text'] for t in easy_difficulty]
        easy_ids = [t['id'] for t in easy_difficulty]
        filtered_easy_texts = self.filter_evidence(all_evidence, easy_ids, easy_texts)
        filtered_easy_texts = random.sample(filtered_easy_texts, min(MAX_EVIDENCES, len(filtered_easy_texts)))

        medium_difficulty = random.sample(similar_pages[-10:], 5)
        medium_texts = [t.payload['text'] for t in medium_difficulty]
        medium_ids = [t.payload['id'] for t in medium_difficulty]
        filtered_medium_texts = self.filter_evidence(all_evidence, medium_ids, medium_texts)
        filtered_medium_texts = random.sample(filtered_medium_texts, min(MAX_EVIDENCES, len(filtered_medium_texts)))

        hard_difficulty = random.sample(similar_pages[:10], 5)
        hard_texts = [t.payload['text'] for t in hard_difficulty]
        hard_ids = [t.payload['id'] for t in hard_difficulty]
        filtered_hard_texts = self.filter_evidence(all_evidence, hard_ids, hard_texts)
        filtered_hard_texts = random.sample(filtered_hard_texts, min(MAX_EVIDENCES, len(filtered_hard_texts)))

        dissimilar_texts = filtered_easy_texts + filtered_medium_texts + filtered_hard_texts
        random.shuffle(dissimilar_texts)


        # dissimilar_pages = random.sample(similar_pages, 2*MAX_EVIDENCES)
        # dissimilar_pages = random.sample(similar_pages[-4*MAX_EVIDENCES:], 2*MAX_EVIDENCES)
        # dissimilar_texts = [t.payload['text'] for t in dissimilar_pages]
        # dissimilar_ids = [t.payload['id'] for t in dissimilar_pages]
        # dissimilar_pages = self.vdb.search_dissimilar(2*MAX_EVIDENCES)
        # dissimilar_texts = [t['text'] for t in dissimilar_pages]
        # dissimilar_ids = [t['id'] for t in dissimilar_pages]


        
        dynamic_nli_target = int(original_nli_target and enough_evidence)
        percentage_retrieved = len(set(similar_ids).intersection(set(all_evidence))) / len(all_evidence) if all_evidence != [] else 1.0

        return {'claim': self.dataset['claims'][index],
                'similar_texts': similar_texts,
                'dissimilar_texts': dissimilar_texts,
                'evidence_texts': evidence_texts,
                'original_nli_target': original_nli_target,
                'dynamic_nli_target': dynamic_nli_target,
                'percentage_retrieved': percentage_retrieved}

        
    def check_enough_evidence(self, evidence, similar_ids):
        for evidence_set in evidence['unique_evidence']:
            if evidence_set is None or evidence_set.issubset(set(similar_ids)):
                return True
        return False
    
    def filter_evidence(self, all_evidence, dissimilar_ids, dissimilar_texts):
        filtered_texts = []
        for t, id in zip(dissimilar_texts, dissimilar_ids):
            if id not in all_evidence:
                filtered_texts.append(t)
        return filtered_texts
    
    

class Sub_Collator:
    def __init__(self,):
        pass

    def __call__(self, batch):
        claims = [item['claim'] for item in batch]
        similar_texts = [item['similar_texts'] for item in batch]
        dissimilar_texts = [item['dissimilar_texts'] for item in batch]
        evidence_texts = [item['evidence_texts'] for item in batch]
        original_nli_targets = [item['original_nli_target'] for item in batch]
        dynamic_nli_targets = [item['dynamic_nli_target'] for item in batch]
        percentage_retrieved = [item['percentage_retrieved'] for item in batch]

        return {'claims': claims,
                'similar_texts': similar_texts,
                'dissimilar_texts': dissimilar_texts,
                'evidence_texts': evidence_texts,
                'original_nli_targets': original_nli_targets,
                'dynamic_nli_targets': dynamic_nli_targets,
                'percentage_retrieved': percentage_retrieved}