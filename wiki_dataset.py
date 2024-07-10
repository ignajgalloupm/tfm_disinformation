
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
    def __init__(self, in_mem=False, reduced=True, evidence=False, num_extra_pages=0, seed=None):
        self.in_mem = in_mem
        self.reduced = reduced
        self.evidence = evidence
        self.num_extra_pages = num_extra_pages
        self.seed = seed
        self.current_block = {'subfix': '000', 'data': None}
        self.__initialization__()

    def __initialization__(self):
        # rebuild the dataset with different random extra pages
        self.dataset = []
        if self.reduced:
            # chech if file fever/reduced_indices.txt exists
            if not os.path.isfile(f'fever/reduced_indices_all.txt'):
                # if it does not exist, create it
                self.__create_reduced_indices__()
            with open(f'fever/reduced_indices_all.txt') as f:
                indices = f.read()
                # keep in self.dataset only the pages that are in indices
                self.dataset = ast.literal_eval(indices)
            if self.num_extra_pages > 0:
                ## add random extra pages using seed (must not already be in dataset)
                random.seed(self.seed)
                all_pages = set(range(TOTAL_WIKI_PAGES))
                rest = all_pages - set(self.dataset)
                self.dataset = self.dataset + random.sample(rest, self.num_extra_pages)
                self.dataset.sort()

        elif self.evidence:
            # chech if file fever/reduced_indices_evidence.txt exists
            if not os.path.isfile(f'fever/reduced_indices_evidence.txt'):
                # if it does not exist, create it
                self.__create_reduced_indices__()
            with open(f'fever/reduced_indices_evidence.txt') as f:
                indices = f.read()
                # keep in self.dataset only the pages that are in indices
                self.dataset = ast.literal_eval(indices)

        else:
            self.dataset = [i for i in range(TOTAL_WIKI_PAGES)]

        if self.in_mem:
            self.dataset = self.__to_mem__(self.dataset)


    def __create_reduced_indices__(self):
        wiki_dict = self.__wiki_dict__()
        counter = 0
        indices = []
        for file in sorted(glob.glob('wiki-pages/*.jsonl')):
            with open(file, 'r') as f:
                curr_file = jsonlist.load(f)
                for page in curr_file:
                    if unicodedata.normalize('NFC', page['id']) in wiki_dict:
                        indices.append(counter)
                    counter += 1
        file_subfix = 'all' if self.reduced else 'evidence'
        with open(f'fever/reduced_indices_{file_subfix}.txt', mode='w') as f:
            f.write(str(indices))
    

    def __wiki_dict__(self):
        wiki_dict = {}
        if self.reduced:
            sets = ['train', 'eval', 'test']
        if self.evidence:
            sets = ['train']
        for set in sets:
            fever = FeverDataset(set)
            for statement in fever:
                for evidence in statement['evidence']['all_evidence']:
                    if evidence is not None:
                        wiki_dict[evidence] = wiki_dict.get(evidence, []) + [statement['id']]
        return wiki_dict
    

    def __to_mem__(self, indices):
        if self.reduced or self.evidence:
            ## form each file, get the list of pages we need
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
        
        else:
            ## get all the pages
            pages = [i for i in range(1, TOTAL_WIKI_PAGES//PAGES_PER_FILE + 2)]
            data = []
            with multiprocessing.Pool(10) as pool:
                data = pool.map(self.__get_all_from_disk__, pages)
            return [page for sublist in data for page in sublist]
         
            
    def __getbulk_fom_disk__(self, file_subfix, subindices):
        data = []
        with open(f'wiki-pages/wiki-{file_subfix}.jsonl', 'r') as f:
            for i, line in enumerate(f):
                if i in subindices:
                    data.append(json.loads(line))
        return data
    
    def __get_all_from_disk__(self, index):
        file_subfix = str(index).zfill(3)
        data = []
        with open(f'wiki-pages/wiki-{file_subfix}.jsonl', 'r') as f:
            data = jsonlist.load(f)
        return data


    def __getitem_from_disk__(self, index):
        file_subfix, subindex = index//PAGES_PER_FILE, index%PAGES_PER_FILE
        # file_subfix has to be three digits
        file_subfix = str(file_subfix+1).zfill(3)

        if self.current_block['subfix'] != file_subfix:
            with open(f'wiki-pages/wiki-{file_subfix}.jsonl', 'r') as f:
                self.current_block['data'] = jsonlist.load(f)
            self.current_block['subfix'] = file_subfix

        # find the page in the file (avoid loading the whole file)
        try:
            return self.current_block['data'][subindex]
        except:
            print(f'Error: {index}, {file_subfix}, {subindex}')
            return None


    def __len__(self):            
        return len(self.dataset)

    def __getitem__(self, index):
        index = self.dataset[index]
        if self.in_mem:
            return index
        else: 
            return self.__getitem_from_disk__(index)
        

    def refresh(self):
        if self.reduced and self.seed is None:
            self.__initialization__()

    def get_random_ids(self, top=10):
        if self.in_mem:
            random.sample(self.dataset, top)
        else:
            # get random pages from a single random block
            indices = random.sample(range(PAGES_PER_FILE), top)
            random_block = random.randint(0, TOTAL_WIKI_PAGES//PAGES_PER_FILE - 1)
            indices = [i + PAGES_PER_FILE * random_block for i in indices]
            return [self.__getitem_from_disk__(index) for index in indices]

            