
import jsonlist
from torch.utils.data import Dataset
import glob
from fever_dataset import FeverDataset
import unicodedata

class WikiDataset(Dataset):
    def __init__(self, type='train', reduced=True):
        # open all json files in directory 'wiki-pages' and create a dictionary
        self.dataset = []
        for file in glob.glob('wiki-pages/*.jsonl'):
            with open(file, 'r') as f:
                self.dataset.append(jsonlist.load(f))
        self.dataset = [item for sublist in self.dataset for item in sublist]
        if reduced:
            wiki_dict = {}
            fever = FeverDataset(type)
            for statement in fever:
                for evidences in statement['evidence']:
                    for evidence in evidences:
                        if evidence[2] is not None:
                            wiki_dict[unicodedata.normalize('NFC', evidence[2])] = wiki_dict.get(unicodedata.normalize('NFC', evidence[2]), []) + [statement['id']]
            # keep in self.dataset only the pages that are in wiki_dict.keys()
            self.dataset = [page for page in self.dataset if page['id'] in wiki_dict]

            # free memory deleting fever and wiki_dict
            del fever
            del wiki_dict


    def __len__(self):            
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]