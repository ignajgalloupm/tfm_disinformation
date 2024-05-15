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
    #         # reorder the embeddings to the original order, indices indicate the original position of the embeddings
    #         _, inverse_indices = torch.sort(torch.tensor(indices))
    #         encodes = [encodes[i] for i in inverse_indices]
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
        if len(texts) > self.batch_size:
            encoded_input = self.tokenizer(texts, padding='max_length', truncation=True, return_tensors='pt')
            input_ids = encoded_input['input_ids']
            attention_mask = encoded_input['attention_mask']
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
                
                in_ids = in_ids[:, :max_length].detach().to(self.device)
                at_mask = at_mask[:, :max_length].detach().to(self.device)

                # Compute token embeddings
                model_output = self.model(input_ids=in_ids, attention_mask=at_mask)
                # Perform pooling and normalization
                sentence_embeddings = self.__mean_pooling_normalization__(model_output, at_mask)
                encodes.extend([e for e in sentence_embeddings])
            # reorder the embeddings to the original order, indices indicate the original position of the embeddings
            _, inverse_indices = torch.sort(indices)
            encodes = [encodes[i] for i in inverse_indices]
            output = torch.stack(encodes)

        else:
            # Compute token embeddings
            encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(self.device)
            # Compute token embeddings
            model_output = self.model(**encoded_input)
            # Perform pooling and normalization
            output = self.__mean_pooling_normalization__(model_output, encoded_input['attention_mask'])
        return output
    
    #Mean Pooling - Take attention mask into account for correct averaging
    def __mean_pooling_normalization__(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sentence_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return F.normalize(sentence_embeddings, p=2, dim=1)

    

# class NLI(torch.nn.Module):

#     def __init__(self, encoder='mpnet', version='v1', device='cuda'):
#         super(NLI, self).__init__()
#         self.name = f'{encoder}_{version}'
#         os.makedirs(f'embeddings/{self.name}', exist_ok=True)
#         if encoder == 'mpnet':
#             model = AutoModelForSequenceClassification.from_pretrained('sentence-transformers/all-mpnet-base-v2', num_labels=1)
#         elif encoder == 'UAE-Large-V1':
#             model = AutoModelForSequenceClassification.from_pretrained('WhereIsAI/UAE-Large-V1', num_labels=1)
#         # set the model to half precision
#         self.model = model.to(device)
        
#     def forward(self, embeddings):
#         probs = self.model(inputs_embeds=embeddings).logits
#         return probs



# 0.70
class NLI(torch.nn.Module):

    def __init__(self, encoder='mpnet', version='v1', device='cuda'):
        super(NLI, self).__init__()
        # input is batch_size x 11 x 768
        self.c1 = torch.nn.Linear(768*2, 600).to(device)
        self.c2 = torch.nn.Linear(600, 400).to(device)
        self.c3 = torch.nn.Linear(400, 300).to(device)

        self.l1 = torch.nn.Linear(300*10, 1500).to(device)
        self.l2 = torch.nn.Linear(1500, 1000).to(device)
        self.l3 = torch.nn.Linear(1000, 400).to(device)
        self.l4 = torch.nn.Linear(400, 100).to(device)
        self.l5 = torch.nn.Linear(100, 1).to(device)

        self.lnorm = torch.nn.LayerNorm(1500).to(device)
        self.dropout = torch.nn.Dropout(0.15).to(device)


    def forward(self, embeddings):

        # pair every embedding with the first one
        pairs = torch.cat([embeddings[:, 0].unsqueeze(1).repeat(1, 10, 1), embeddings[:, 1:]], dim=2)

        c1 = self.c1(pairs)
        c1 = F.leaky_relu(c1)
        c2 = self.c2(c1)
        c2 = F.leaky_relu(c2)
        c3 = self.c3(c2)
        c3 = F.leaky_relu(c3)

        l1 = self.l1(c3.view(c3.shape[0], -1))
        l1 = F.leaky_relu(self.lnorm(l1))
        l2 = self.l2(l1)
        l2 = F.leaky_relu(self.dropout(l2))
        l3 = self.l3(l2)
        l3 = F.leaky_relu(l3)
        l4 = self.l4(l3)
        l4 = F.leaky_relu(l4)
        l5 = self.l5(l4)
        return l5
    
