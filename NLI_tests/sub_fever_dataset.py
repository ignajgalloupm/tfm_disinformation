from torch.utils.data import Dataset

PAGES_RETRIEVED = 10


class Sub_Dataset(Dataset):
    def __init__(self, data, vdb, set_type='train'):
        self.dataset = data
        self.vdb = vdb
        self.set_type = set_type

    def __len__(self):            
        return len(self.dataset['claims'])

    def __getitem__(self, index):
        similar_pages = self.vdb.search_similar(self.dataset['claim_embs'][index].unsqueeze(0), PAGES_RETRIEVED, with_payload=True, with_vector=True)[0]#
        similar_embs = [t.vector for t in similar_pages]
        similar_ids = [t.payload['id'] for t in similar_pages]


        all_evidence = self.dataset['evidence'][index]['all_evidence'] if self.dataset['evidence'][index]['all_evidence'] != [None] else []
        
        enough_evidence = self.check_enough_evidence(self.dataset['evidence'][index], similar_ids)
        original_nli_target = int(self.dataset['label'][index] == 'SUPPORTS')
        dynamic_nli_target = int(original_nli_target and enough_evidence)
        percentage_retrieved = len(set(similar_ids).intersection(set(all_evidence))) / len(all_evidence) if all_evidence != [] else 1.0

        return {'claim_emb': self.dataset['claim_embs'][index],
                'similar_embs': similar_embs,
                'original_nli_target': original_nli_target,
                'dynamic_nli_target': dynamic_nli_target,
                'percentage_retrieved': percentage_retrieved}

        
    def check_enough_evidence(self, evidence, similar_ids):
        for evidence_set in evidence['unique_evidence']:
            if evidence_set is None or evidence_set.issubset(set(similar_ids)):
                return True
        return False
    
    

class Sub_Collator:
    def __init__(self,):
        pass

    def __call__(self, batch):
        claim_embs = [item['claim_emb'] for item in batch]
        similar_embs = [item['similar_embs'] for item in batch]
        original_nli_targets = [item['original_nli_target'] for item in batch]
        dynamic_nli_targets = [item['dynamic_nli_target'] for item in batch]
        percentage_retrieved = [item['percentage_retrieved'] for item in batch]

        return {'claim_embs': claim_embs,
                'similar_embs': similar_embs,
                'original_nli_targets': original_nli_targets,
                'dynamic_nli_targets': dynamic_nli_targets,
                'percentage_retrieved': percentage_retrieved}