import jsonlist
from torch.utils.data import Dataset, default_collate


class FeverDataset(Dataset):
    def __init__(self, type):
        self.type = type
        # from json file to dictionary
        with open(f'fever/{self.type}.jsonl', 'r') as f:
            data = jsonlist.load(f)
        self.dataset = self.__simplify__(data)

    def __simplify__(self, data):
        for statement in data:
            evidence_sets = []
            for evidences in statement['evidence']:
                set2 = set()
                for evidence in evidences:
                    set2.add(evidence[2])
                evidence_sets.append(set2)
            statement['evidence'] = {'all_evidence': self.__all_evidence__(evidence_sets), 
                                     'unique_evidence': self.__unique_non_supersets__(evidence_sets)}
        return data
            
    def __unique_non_supersets__(self, sets):
        unique_sets = []
        for s1 in sets:
            is_superset = False
            for s2 in sets:
                if s1 != s2 and s1.issuperset(s2):
                    is_superset = True
                    break
            if not is_superset:
                unique_sets.append(s1)
        return unique_sets
    
    def __all_evidence__(self, sets):
        all_evidence = []
        for s in sets:
            all_evidence.extend(list(s))
        return all_evidence
        

    def __len__(self):            
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]
    

class FeverCollator:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # call default collate_fn
        ids = [item['id'] for item in batch]
        verifiable = [item['verifiable'] for item in batch]
        label = [item['label'] for item in batch]
        evidence = [item['evidence'] for item in batch]
        #claims = self.tokenizer(list(item['claim'] for item in batch), return_tensors='pt', truncation=True, padding='longest')
        claims = [item['claim'] for item in batch]

        return {'ids': ids, 'verifiable': verifiable, 'label': label, 'claims': claims, 'evidence': evidence}
        