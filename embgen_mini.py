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
        


class EmbeddingGenerator(torch.nn.Module):

    def __init__(self, encoder='mpnet', version='v1', device='cuda', batch_size=16):
        super(EmbeddingGenerator, self).__init__()
        self.name = f'{encoder}_{version}'
        self.device = device
        self.batch_size = batch_size
        os.makedirs(f'embeddings/{self.name}', exist_ok=True)
        if encoder == 'mpnet':
            model = AutoModel.from_pretrained("avsolatorio/NoInstruct-small-Embedding-v0")
            self.tokenizer = AutoTokenizer.from_pretrained("avsolatorio/NoInstruct-small-Embedding-v0")
        else:
            raise ValueError('Invalid encoder')
        # set the model to half precision
        self.model = model.to(device)

        
 
    @autocast()
    def forward(self, texts, mode='sentence'):   

        assert mode in ("query", "sentence"), f"mode={mode} was passed but only `query` and `sentence` are the supported modes."

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
            

                if mode == "query":
                    vectors = model_output.last_hidden_state * at_mask.unsqueeze(2)
                    vectors = vectors.sum(dim=1) / at_mask.sum(dim=-1).view(-1, 1)
                else:
                    vectors = model_output.last_hidden_state[:, 0, :]
                encodes.extend([e for e in vectors])

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
#         if encoder == 'mpnet':
#             model = AutoModelForSequenceClassification.from_pretrained("avsolatorio/NoInstruct-small-Embedding-v0")
#         self.model = model.to(device)
        
#     def forward(self, embeddings):
#         print(embeddings.shape)
#         # no positional embeddings
#         probs = self.model(inputs_embeds=embeddings, position_ids=torch.zeros_like(embeddings).long())
#         print(probs.shape)
#         return probs


# 0.70
class NLI(torch.nn.Module):

    def __init__(self, encoder='mpnet', version='v1', device='cuda'):
        super(NLI, self).__init__()
        # input is batch_size x 11 x 768
        self.c1 = torch.nn.Linear(768, 600)
        self.c2 = torch.nn.Linear(600, 400)
        self.c3 = torch.nn.Linear(400, 300)

        self.l1 = torch.nn.Linear(300*10, 1500)
        self.l2 = torch.nn.Linear(1500, 1000)
        self.l3 = torch.nn.Linear(1000, 400)
        self.l4 = torch.nn.Linear(400, 100)
        self.l5 = torch.nn.Linear(100, 1)

        self.lnorm = torch.nn.LayerNorm(1500)
        self.dropout = torch.nn.Dropout(0.1)

        # move to device
        self.to(device)


    def forward(self, embeddings):

        # pair every embedding with the first one
        pairs = torch.cat([embeddings[:, 0].unsqueeze(1).repeat(1, 10, 1), embeddings[:, 1:]], dim=2)

        c1 = self.c1(pairs)
        c1 = F.leaky_relu(c1)
        c2 = self.c2(c1)
        c2 = F.leaky_relu(self.dropout(c2))
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


# class NLI(torch.nn.Module):

#     def __init__(self, encoder='mpnet', version='v1', device='cuda'):
#         super(NLI, self).__init__()
#         # input is batch_size x 11 x 768
#         self.l1 = torch.nn.Linear(384*11, 600)
#         self.l2 = torch.nn.Linear(600, 200)
#         self.l3 = torch.nn.Linear(200, 1)
#         self.dropout = torch.nn.Dropout(0.1)

#         # move to device
#         self.to(device)


#     def forward(self, embeddings):
            
#         x = self.l1(embeddings.view(embeddings.shape[0], -1))
#         x = F.leaky_relu(self.dropout(x))
#         x = self.l2(x)
#         x = F.leaky_relu(self.dropout(x))
#         x = self.l3(x)
#         return x
            

        
        

        
    
