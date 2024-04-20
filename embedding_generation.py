import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import torch

# log and warning suppression
import logging
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"



class EmbeddingGenerator(torch.nn.Module):

    def __init__(self, encoder='mpnet', version='v1', device='cuda'):
        super(EmbeddingGenerator, self).__init__()
        self.name = f'{encoder}_{version}'
        os.makedirs(f'embeddings/{self.name}', exist_ok=True)
        if encoder == 'mpnet':
            model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        elif encoder == 'UAE-Large-V1':
            model = SentenceTransformer('WhereIsAI/UAE-Large-V1')
        else:
            raise ValueError('Invalid encoder')
        # set the model to half precision
        self.model = model.half().to(device)
        
    
    def forward(self, texts):
        return self.model.encode(texts, batch_size=128)
    

class NLI(torch.nn.Module):

    def __init__(self, encoder='mpnet', version='v1', device='cuda'):
        super(NLI, self).__init__()
        self.name = f'{encoder}_{version}'
        os.makedirs(f'embeddings/{self.name}', exist_ok=True)
        if encoder == 'mpnet':
            self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
            model = AutoModelForSequenceClassification.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        elif encoder == 'UAE-Large-V1':
            self.tokenizer = AutoTokenizer.from_pretrained('WhereIsAI/UAE-Large-V1')
            model = AutoModelForSequenceClassification.from_pretrained('WhereIsAI/UAE-Large-V1')
        # set the model to half precision
        self.model = model.half().to(device)
        
    
    def forward(self, embeddings):
        probs = torch.nn.functional.softmax(self.model(inputs_embeds=embeddings).logits, dim=-1)
        return probs