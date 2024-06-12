import jsonlist
from torch.utils.data import Dataset
import unicodedata
import random


class FeverDataset(Dataset):
    def __init__(self, type):
        if type not in ['train', 'eval', 'test']:
            raise ValueError('type must be one of "train", "dev", or "test"')
        if type in ['train', 'eval']:
            # from json file to dictionary
            with open(f'fever/train.jsonl', 'r') as f:
                data = jsonlist.load(f)
            self.dataset = self.__simplify__(data)
            # random shuffle with seed
            random.seed(42)
            random.shuffle(self.dataset)
            # perform a random split 80/20 for train and dev
            split = int(0.9 * len(self.dataset))
            self.dataset = self.dataset[:split] if type == 'train' else self.dataset[split:]
        elif type == 'test':
            # from json file to dictionary
            with open(f'fever/dev.jsonl', 'r') as f:
                data = jsonlist.load(f)
            self.dataset = self.__simplify__(data)


    def __simplify__(self, data):
        for statement in data:
            evidence_sets = []
            for evidences in statement['evidence']:
                set2 = set()
                for evidence in evidences:
                    set2.add(unicodedata.normalize('NFC', evidence[2]) if evidence[2] is not None else None)
                evidence_sets.append(set2)
            statement['evidence'] = {'all_evidence': self.__all_evidence__(evidence_sets), 
                                     'unique_evidence': self.__unique_non_supersets__(evidence_sets)}
        return data
    
    def __unique_non_supersets__(self, sets):
        unique_sets = []
        for i, s1 in enumerate(sets):
            if s1 == {None}:
                unique_sets.append(set())
                continue
            is_superset = False
            for s2 in sets[:i]:
                if s1.issuperset(s2):
                    is_superset = True
                    break
            if not is_superset:
                for s2 in sets[i+1:]:
                    if s1.issuperset(s2) and len(s1) > len(s2):
                        is_superset = True
                        break
            if not is_superset:
                unique_sets.append(s1)
        return unique_sets
    
    def __all_evidence__(self, sets):
        all_evidence = set()
        for s in sets:
            all_evidence = all_evidence.union(s)
        return list(all_evidence)
        

    def __len__(self):            
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]
    

class FeverCollator:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        ids = [item['id'] for item in batch]
        verifiable = [item['verifiable'] for item in batch]
        label = [item['label'] for item in batch]
        evidence = [item['evidence'] for item in batch]
        claims = [item['claim'] for item in batch]

        return {'ids': ids, 'verifiable': verifiable, 'label': label, 'claims': claims, 'evidence': evidence}
        