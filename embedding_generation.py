import pandas as pd
import numpy as np
from angle_emb import AnglE, Prompts
from sentence_transformers import SentenceTransformer
import os


class EmbeddingGenerator():

    def __init__(self, encoder='mpnet', version='v1', device='cuda'):
        self.name = f'{encoder}_{version}'
        os.makedirs(f'embeddings/{self.name}', exist_ok=True)
        if encoder == 'mpnet':
            self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').to(device)
        elif encoder == 'UAE-Large-V1':
            self.model = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').to(device)


    def generate(self, wiki_loader):
        # remove the existing embeddings
        for file in os.listdir(f'embeddings/{self.name}'):
            os.remove(f'embeddings/{self.name}/{file}')
        for i, pages in enumerate(wiki_loader):
            vec = self.model.encode(pages['text'])
            print(f'Block {i}/{len(wiki_loader)} done')
            np.save(f'embeddings/{self.name}/{i}.npy', vec)
        