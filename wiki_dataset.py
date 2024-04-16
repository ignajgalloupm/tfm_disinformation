
import jsonlist
from torch.utils.data import Dataset
import glob
from fever_dataset import FeverDataset
import unicodedata
import os
import ast
import json
import random
import multiprocessing

TOTAL_WIKI_PAGES = 5416537
PAGES_PER_FILE = 50000


class WikiDataset(Dataset):
    def __init__(self, in_mem=False, reduced=True, type='train', num_extra_pages=0, seed=None):
        self.in_mem = in_mem

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
            if num_extra_pages > 0:
                ## add random extra pages using seed (must not already be in dataset)
                random.seed(seed)
                for _ in range(num_extra_pages):
                    page = random.randint(0, TOTAL_WIKI_PAGES-1)
                    while page in self.dataset:
                        page = random.randint(0, TOTAL_WIKI_PAGES-1)
                    self.dataset.append(page)

        else:
            self.dataset = [range(TOTAL_WIKI_PAGES)]
        
        if in_mem:
            self.dataset = self.__to_mem__(self.dataset)


    def __create_reduced_indices__(self, type):
        wiki_dict = self.__wiki_dict__(type)
        counter = 0
        indices = []
        for file in sorted(glob.glob('wiki-pages/*.jsonl')):
            with open(file, 'r') as f:
                curr_file = jsonlist.load(f)
                for page in curr_file:
                    if unicodedata.normalize('NFC', page['id']) in wiki_dict:
                        indices.append(counter)
                    counter += 1
        with open(f'fever/{type}_reduced_indices.txt', mode='w') as f:
            f.write(str(indices))
    

    def __wiki_dict__(self, type):
        wiki_dict = {}
        fever = FeverDataset(type)
        for statement in fever:
            for evidence in statement['evidence']['all_evidence']:
                if evidence is not None:
                    wiki_dict[unicodedata.normalize('NFC', evidence)] = wiki_dict.get(unicodedata.normalize('NFC', evidence), []) + [statement['id']]
        return wiki_dict
    

    def __to_mem__(self, indices):
        ## for each file, get the list of pages we need
        pages = {}
        for index in indices:
            file_subfix, subindex = index//PAGES_PER_FILE, index%PAGES_PER_FILE
            # file_subfix has to be three digits
            file_subfix = str(file_subfix+1).zfill(3)
            pages[file_subfix] = pages.get(file_subfix, []) + [subindex]
        
        ## for each file, get the pages
        data = []
        with multiprocessing.Pool(10) as pool:
            data = pool.starmap(self.__getbulk_fom_disk__, pages.items())
        return [page for sublist in data for page in sublist]
         
            
    def __getbulk_fom_disk__(self, file_subfix, subindices):
        data = []
        with open(f'wiki-pages/wiki-{file_subfix}.jsonl', 'r') as f:
            for i, line in enumerate(f):
                if i in subindices:
                    data.append(json.loads(line))
        return data

    def __getitem_fom_disk__(self, index):
        file_subfix, subindex = index//PAGES_PER_FILE, index%PAGES_PER_FILE
        # file_subfix has to be three digits
        file_subfix = str(file_subfix+1).zfill(3)

        # find the page in the file (avoid loading the whole file)
        with open(f'wiki-pages/wiki-{file_subfix}.jsonl', 'r') as f:
            for i, line in enumerate(f):
                if i == subindex:
                    return json.loads(line)


    def __len__(self):            
        return len(self.dataset)

    def __getitem__(self, index):
        index = self.dataset[index]
        if self.in_mem:
            return index
        else: 
            return self.__getitem_fom_disk__(index)