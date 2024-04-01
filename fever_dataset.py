import jsonlist
from torch.utils.data import Dataset


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