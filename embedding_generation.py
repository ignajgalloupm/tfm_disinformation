import numpy as np
from angle_emb import AnglE, Prompts
from sentence_transformers import SentenceTransformer
import os
import torch

# log and warning suppression
import logging
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"



class EmbeddingGenerator():

    def __init__(self, encoder='mpnet', version='v1', device='cuda'):
        self.name = f'{encoder}_{version}'
        os.makedirs(f'embeddings/{self.name}', exist_ok=True)
        if encoder == 'mpnet':
            model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').to(device)
        elif encoder == 'UAE-Large-V1':
            model = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').to(device)
        # set the model to half precision
        self.model = torch.compile(model.half())
        
    
    def __call__(self, texts):
        return self.model.encode(texts)