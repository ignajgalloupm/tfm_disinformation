from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import torch.nn.functional as F
import os
import torch
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader

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

    def __init__(self, encoder='mpnet', version='v1', device='cuda', batch_size=16):
        super(EmbeddingGenerator, self).__init__()
        self.name = f'{encoder}_{version}'
        self.device = device
        self.batch_size = batch_size
        os.makedirs(f'embeddings/{self.name}', exist_ok=True)
        if encoder == 'mpnet':
            model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
            self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        elif encoder == 'UAE-Large-V1':
            model = AutoModel.from_pretrained('WhereIsAI/UAE-Large-V1')
            self.tokenizer = AutoTokenizer.from_pretrained('WhereIsAI/UAE-Large-V1')
        else:
            raise ValueError('Invalid encoder')
        # set the model to half precision
        self.model = model.to(device)

    # @autocast()
    # def forward(self, texts):        
    #     def sort_with_index(texts):
    #         indices = list(range(len(texts)))  # Create a list of indices corresponding to original order
    #         sorted_texts, indices = zip(*sorted(zip(texts, indices), key=lambda x: len(x[0])))
    #         return sorted_texts, indices
        
    #     if len(texts) > self.batch_size:
    #         # sort the texts by length but keep the original order
    #         texts, indices = sort_with_index(texts)
    #         # process the texts in batches of 128
    #         encodes = []
    #         for i in range(0, len(texts), self.batch_size):
    #             # Tokenize sentences
    #             encoded_input = self.tokenizer(texts[i:i+self.batch_size], padding=True, truncation=True, return_tensors='pt').to(self.device)
    #             # Compute token embeddings
    #             model_output = self.model(**encoded_input)
    #             # Perform pooling
    #             sentence_embeddings = self.__mean_pooling_normalization__(model_output, encoded_input['attention_mask'])
    #             encodes.extend([e for e in sentence_embeddings])
    #         encodes = [encodes[i] for i in indices]
    #         output = torch.stack(encodes)
    #     else:
    #         encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(self.device)
    #         # Compute token embeddings
    #         model_output = self.model(**encoded_input)
    #         # Perform pooling
    #         output = self.__mean_pooling_normalization__(model_output, encoded_input['attention_mask'])
    #     return output
 
    @autocast()
    def forward(self, texts):   
        encoded_input = self.tokenizer(texts, padding='max_length', truncation=True, return_tensors='pt')
        input_ids = encoded_input['input_ids']
        attention_mask = encoded_input['attention_mask']
        
        if len(texts) > self.batch_size:
            # sort the texts by length but keep the original order
            lengths = torch.sum(attention_mask, dim=1)
            #input_ids, attention_mask, indices = sort_with_index(input_ids, attention_mask, lengths)
            _, indices = torch.sort(lengths, descending=True)
            input_ids = input_ids[indices]
            attention_mask = attention_mask[indices]

            encodes = []
            for i in range(0, len(texts), self.batch_size):
                # get slice
                in_ids = input_ids[i:i+self.batch_size]
                at_mask = attention_mask[i:i+self.batch_size]
            
                # cut the unnecessary padding
                max_length = torch.max(torch.sum(at_mask, dim=1)).item()
                in_ids = in_ids[:, :max_length].to(self.device)
                at_mask = at_mask[:, :max_length].to(self.device)

            
                # Compute token embeddings
                model_output = self.model(input_ids=in_ids, attention_mask=at_mask)
                # Perform pooling and normalization
                sentence_embeddings = self.__mean_pooling_normalization__(model_output, at_mask)
                encodes.extend([e for e in sentence_embeddings])
            # reorder the embeddings to the original order
            encodes = [encodes[i] for i in indices]
            output = torch.stack(encodes)

        else:
            # Compute token embeddings
            in_ids = input_ids.to(self.device)
            at_mask = attention_mask.to(self.device)
            model_output = self.model(input_ids=in_ids, attention_mask=at_mask)
            # Perform pooling and normalization
            output = self.__mean_pooling_normalization__(model_output, at_mask)
        return output
    
    #Mean Pooling - Take attention mask into account for correct averaging
    def __mean_pooling_normalization__(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sentence_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return F.normalize(sentence_embeddings, p=2, dim=1)

    

class NLI(torch.nn.Module):

    def __init__(self, encoder='mpnet', version='v1', device='cuda'):
        super(NLI, self).__init__()
        self.name = f'{encoder}_{version}'
        os.makedirs(f'embeddings/{self.name}', exist_ok=True)
        if encoder == 'mpnet':
            self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
            model = AutoModelForSequenceClassification.from_pretrained('sentence-transformers/all-mpnet-base-v2', num_labels=1)
        elif encoder == 'UAE-Large-V1':
            self.tokenizer = AutoTokenizer.from_pretrained('WhereIsAI/UAE-Large-V1')
            model = AutoModelForSequenceClassification.from_pretrained('WhereIsAI/UAE-Large-V1', num_labels=1)
        # set the model to half precision
        self.model = model.to(device)
        
    def forward(self, embeddings):
        probs = self.model(inputs_embeds=embeddings).logits
        return probs