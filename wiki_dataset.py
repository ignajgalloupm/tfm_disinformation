
import jsonlist
from torch.utils.data import Dataset
import glob
from fever_dataset import FeverDataset
import unicodedata
import os
import ast
import json

TOTAL_WIKI_PAGES = 5416537
PAGES_PER_FILE = 50000


class WikiDataset(Dataset):
    def __init__(self, type='train', reduced=True):
        if type not in ['train', 'dev', 'test']:
            raise ValueError('type must be one of: train, dev, test')
        self.dataset = []
        if reduced:
            # chech if file fever/reduced_indices.txt exists
            if not os.path.isfile(f'fever/{type}_reduced_indices.txt'):
                # if it does not exist, create it
                self.__create_reduced_indices__(type)
            with open(f'fever/{type}_reduced_indices.txt') as f:
                indices = f.read()
                # keep in self.dataset only the pages that are in indices
                self.dataset = ast.literal_eval(indices)

        else:
            self.dataset = [range(TOTAL_WIKI_PAGES)]


    def __create_reduced_indices__(self, type):
        wiki_dict = self.__wiki_dict__(type)
        counter = 0
        indices = []
        for file in glob.glob('wiki-pages/*.jsonl'):
            with open(file, 'r') as f:
                curr_file = jsonlist.load(f)
                for page in curr_file:
                    if page['id'] in wiki_dict:
                        indices.append(counter)
                    counter += 1
        with open(f'fever/{type}_reduced_indices.txt', mode='w') as f:
            f.write(str(indices))
    

    def __wiki_dict__(self, type):
        wiki_dict = {}
        fever = FeverDataset(type)
        for statement in fever:
            for evidences in statement['evidence']:
                for evidence in evidences:
                    if evidence[2] is not None:
                        wiki_dict[unicodedata.normalize('NFC', evidence[2])] = wiki_dict.get(unicodedata.normalize('NFC', evidence[2]), []) + [statement['id']]
        return wiki_dict


    def __len__(self):            
        return len(self.dataset)

    def __getitem__(self, index):
        index = self.dataset[index]
        file_subfix, subindex = index//PAGES_PER_FILE, index%PAGES_PER_FILE
        # file_subfix has to be three digits
        file_subfix = str(file_subfix+1).zfill(3)

        # find the page in the file (avoid loading the whole file)
        with open(f'wiki-pages/wiki-{file_subfix}.jsonl', 'r') as f:
            for i, line in enumerate(f):
                if i == subindex:
                    return json.loads(line)