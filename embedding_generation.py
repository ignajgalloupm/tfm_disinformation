import pandas as pd
import numpy as np
#from angle_emb import AnglE, Prompts
from sentence_transformers import SentenceTransformer

BLOCK_SIZE = 10

class EmbeddingGenerator():

    def __init__(self, encoder='mpnet', version='v1', device='cuda'):
        self.name = f'{encoder}_{version}'
        if encoder == 'mpnet':
            self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').to(device)
        elif encoder == 'UAE-Large-V1':
            self.model = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').to(device)


    def generate(self, wiki, batch_size=200):
        vecs = []
        for i in range(len(wiki)//batch_size):
            selected_rows = [wiki[i]['text'] for i in range(i*batch_size, (i+1)*batch_size)]
            vec = self.model.encode(selected_rows)
            vecs.append(vec)
            if i != 0 and i % BLOCK_SIZE == 0:
                print(f'Block {i}/{len(wiki)//BLOCK_SIZE} done')
                self.save_to_file(vecs, f'embeddings/{self.name}_{i//BLOCK_SIZE}.npy')
                vecs = []

        # the remaining ones
        selected_rows = [wiki[i]['text'] for i in range((i+1)*batch_size, len(wiki))]
        vec = self.model.encode(selected_rows)
        vecs.append(vec)
        self.save_to_file(vecs, f'embeddings/{self.name}_{(i+1)//BLOCK_SIZE}.npy')

    def save_to_file(self, vecs, path):
        vecs = [v for vec in vecs for v in vec]
        np.save(path, np.array(vecs))
        