from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import torch.nn.functional as F
import os
import torch
from torch.cuda.amp import autocast

# log and warning suppression
import logging
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"



# class EmbeddingGenerator(torch.nn.Module):

#     def __init__(self, encoder='mpnet', version='v1', device='cuda'):
#         super(EmbeddingGenerator, self).__init__()
#         self.name = f'{encoder}_{version}'
#         os.makedirs(f'embeddings/{self.name}', exist_ok=True)
#         if encoder == 'mpnet':
#             model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
#         elif encoder == 'UAE-Large-V1':
#             model = SentenceTransformer('WhereIsAI/UAE-Large-V1')
#         else:
#             raise ValueError('Invalid encoder')
#         # set the model to half precision
#         self.model = model.half().to(device)
        
    
#     def forward(self, texts, grad=False):
#         if grad:
#             return self.model.encode(texts, batch_size=128, convert_to_tensor=True)
#         else:
#             return self.model.encode(texts, batch_size=128)
        


class EmbeddingGenerator(torch.nn.Module):

    def __init__(self, encoder='mpnet', version='v1', device='cuda'):
        super(EmbeddingGenerator, self).__init__()
        self.name = f'{encoder}_{version}'
        self.device = device
        os.makedirs(f'embeddings/{self.name}', exist_ok=True)
        if encoder == 'mpnet':
            model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
            self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
            self.embedding_size = 768
        elif encoder == 'UAE-Large-V1':
            model = AutoModel.from_pretrained('WhereIsAI/UAE-Large-V1')
            self.tokenizer = AutoTokenizer.from_pretrained('WhereIsAI/UAE-Large-V1')
            self.embedding_size = 1024
        else:
            raise ValueError('Invalid encoder')
        # set the model to half precision
        self.model = model.to(device)
        
    
    def forward(self, texts):
        #Mean Pooling - Take attention mask into account for correct averaging
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0] #First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        def sort_with_index(texts):
            indices = list(range(len(texts)))  # Create a list of indices corresponding to original order
            sorted_texts, indices = zip(*sorted(zip(texts, indices), key=lambda x: len(x[0])))
            return list(sorted_texts), list(indices)
        
        batch_size = 32
        if len(texts) > batch_size:
            # sort the texts by length but keep the original order
            texts, indices = sort_with_index(texts)
        # process the texts in batches of 128
        encodes = []
        for i in range(0, len(texts), batch_size):
            # Tokenize sentences
            encoded_input = self.tokenizer(texts[i:i+batch_size], padding=True, truncation=True, return_tensors='pt').to(self.device)
            # Compute token embeddings
            with autocast():
                model_output = self.model(**encoded_input)
            # Perform pooling
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            # Normalize embeddings
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            encodes.extend([e for e in sentence_embeddings])
        if len(texts) > batch_size:
            # reorder the embeddings to the original order
            encodes = [encodes[i] for i in indices]
        return torch.cat(encodes, dim=0).view([-1, self.embedding_size])

    

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
        self.model = model.to(device)
        
    
    def forward(self, embeddings):
        with autocast():
            probs = torch.nn.functional.softmax(self.model(inputs_embeds=embeddings).logits, dim=-1)
        return probs