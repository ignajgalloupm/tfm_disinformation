import pandas as pd
import numpy as np
from angle_emb import AnglE, Prompts
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator():

    def __init__(self, encoder='mpnet', version='v1', batch_size=200):
        self.name = f'{encoder}_{version}'
        if encoder == 'mpnet':
            self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').cuda()
        elif encoder == 'UAE-Large-V1':
            self.model = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()
        self.batch_size = batch_size

    def generate(self, wiki):
        vecs = []
        for i in range(len(wiki)//self.batch_size):
            selected_rows = [wiki[i]['text'] for i in range(i*self.batch_size, (i+1)*self.batch_size)]
            vec = self.model.encode(selected_rows)
            vecs.append(vec)

        # the remaining ones
        selected_rows = [wiki[i]['text'] for i in range((i+1)*self.batch_size, len(wiki))]
        vec = self.model.encode(selected_rows) 
        vecs.append(vec)

        # remove one dimension
        vecs = [v for vec in vecs for v in vec]
        np.save(f'embeddings/{self.name}.npy', np.array(vecs))