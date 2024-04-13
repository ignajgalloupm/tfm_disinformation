import jsonlist
from torch.utils.data import Dataset, default_collate


class FeverDataset(Dataset):
    def __init__(self, type):
        self.type = type
        # from json file to dictionary
        with open(f'fever/{self.type}.jsonl', 'r') as f:
            self.dataset = jsonlist.load(f)
        
    def __len__(self):            
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]
    

class FeverCollator:
    def __init__(self, tokenizer):
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
        