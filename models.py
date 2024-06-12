from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import torch.nn.functional as F
import os
import torch
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader
from torch import nn

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
        # freeze the first half of the model
        for param in model.embeddings.parameters():
            param.requires_grad = False
        num_encoder_layers = len([p for p in model.encoder.parameters()])
        for i, param in enumerate(model.encoder.parameters()):
                if i < num_encoder_layers - 40:
                        param.requires_grad = False

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



class NLI_FullLinear(torch.nn.Module):

    def __init__(self, device='cuda'):
        super(NLI_FullLinear, self).__init__()
        # input is batch_size x 11 x 768
        self.l1 = torch.nn.Linear(768*11, 5000)
        self.l2 = torch.nn.Linear(5000, 2000)
        self.l3 = torch.nn.Linear(2768, 1000)
        self.l4 = torch.nn.Linear(1000, 500)
        self.l5 = torch.nn.Linear(500, 1)

        self.dropout = torch.nn.Dropout(0.1)
        self.lnorm = torch.nn.LayerNorm(2768)

        # move to device
        self.to(device)

    @torch.compile
    def forward(self, embeddings):
            
        x = self.l1(embeddings.view(embeddings.shape[0], -1))
        x = F.leaky_relu(self.dropout(x))
        x = F.leaky_relu(self.l2(x))
        x = torch.cat([x, embeddings[:, 10]], dim=1)
        x = self.lnorm(x)
        x = F.leaky_relu(self.l3(x))
        x = F.leaky_relu(self.l4(x))
        x = self.l5(x)
        return x



# 0.70
class NLI_PairsBasic(torch.nn.Module):

    def __init__(self, device='cuda'):
        super(NLI_PairsBasic, self).__init__()
        # input is batch_size x 11 x 768
        self.c1 = torch.nn.Linear(768*2, 1024)
        self.c2 = torch.nn.Linear(1024, 1024)
        self.c3 = torch.nn.Linear(1024, 768)

        self.l1 = torch.nn.Linear(768*10, 5096)
        self.l2 = torch.nn.Linear(5096, 2048)
        self.l3 = torch.nn.Linear(2048, 1024)
        self.l4 = torch.nn.Linear(1024, 512)
        self.l5 = torch.nn.Linear(512, 256)
        self.l6 = torch.nn.Linear(256, 64)
        self.l7 = torch.nn.Linear(64, 1)

        self.lnorm = torch.nn.LayerNorm(5096)
        self.dropout = torch.nn.Dropout(0.1)

        # move to device
        self.to(device)

    @torch.compile
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
        l5 = F.leaky_relu(l5)
        l6 = self.l6(l5)
        l6 = F.leaky_relu(l6)
        l7 = self.l7(l6)
        return l7
    


NUM_HEADS_1 = 16
DIMS_1 = 128
class NLI_Heads(torch.nn.Module):
    
    class head(torch.nn.Module):
        def __init__(self):
            super(NLI_Heads.head, self).__init__()
            # input is batch_size x 11 x 768
            self.c1 = torch.nn.Linear(768*2, 512)
            self.c2 = torch.nn.Linear(512, DIMS_1)
        
        def forward(self, pairs):
            c1 = self.c1(pairs)
            c1 = F.leaky_relu(c1)
            c2 = self.c2(c1)
            return c2


    def __init__(self, device='cuda'):
        super(NLI_Heads, self).__init__()
        
        self.heads = torch.nn.ModuleList([self.head() for _ in range(NUM_HEADS_1)])

        self.l1 = torch.nn.Linear(NUM_HEADS_1*10*DIMS_1, 2048)
        self.l2 = torch.nn.Linear(2048, 768)
        self.l3 = torch.nn.Linear(768, 256)
        self.l4 = torch.nn.Linear(256, 64)
        self.l5 = torch.nn.Linear(64, 1)

        self.lnorm = torch.nn.LayerNorm(NUM_HEADS_1*10*DIMS_1)
        self.dropout = torch.nn.Dropout(0.1)

        # move to device
        self.to(device)

    @torch.compile
    def forward(self, embeddings):

        pairs = torch.cat([embeddings[:, 0].unsqueeze(1).repeat(1, 10, 1), embeddings[:, 1:]], dim=2)
        # Create a list of inputs for each head
        inputs = [pairs] * len(self.heads)
        # Parallelize the computation of each head
        outputs = nn.parallel.parallel_apply(self.heads, inputs)
        # Concatenate the outputs of each head
        heads = torch.cat(outputs, dim=1)
    
        x = heads.view(heads.shape[0], -1)
        x = self.lnorm(x)
        x = self.l1(self.dropout(x))
        x = F.leaky_relu(x)
        x = self.l2(x)
        x = F.leaky_relu(self.dropout(x))
        x = self.l3(x)
        x = F.leaky_relu(x)
        x = self.l4(x)
        x = F.leaky_relu(x)
        x = self.l5(x)
        return x




NUM_HEADS = 8
NUM_MINI_HEADS = 4
DIMS = 512
MINI_DIMS = 64
class NLI_MiniHeads(torch.nn.Module):
    
    class head(torch.nn.Module):

        class mini_head(torch.nn.Module):
            def __init__(self):
                super(NLI_MiniHeads.head.mini_head, self).__init__()
                # input is batch_size x 11 x 768
                self.c1 = torch.nn.Linear(768*2, 512)
                self.c2 = torch.nn.Linear(512, MINI_DIMS)
            
            def forward(self, pairs):
                c1 = self.c1(pairs)
                c1 = F.leaky_relu(c1)
                c2 = self.c2(c1)
                return c2
            
        
        def __init__(self):
            super(NLI_MiniHeads.head, self).__init__()
            
            self.mini_heads = torch.nn.ModuleList([self.mini_head() for _ in range(NUM_MINI_HEADS)])

            self.l1 = torch.nn.Linear(NUM_MINI_HEADS*10*MINI_DIMS, 768)
            self.l2 = torch.nn.Linear(768, DIMS)


        def forward(self, pairs):
            # Create a list of inputs for each head
            inputs = [pairs] * len(self.mini_heads)

            # Parallelize the computation of each head
            outputs = nn.parallel.parallel_apply(self.mini_heads, inputs)

            # Concatenate the outputs of each head
            heads = torch.cat(outputs, dim=1)
        
            x = heads.view(heads.shape[0], -1)
            x = self.l1(x)
            x = F.leaky_relu(x)
            x = self.l2(x)
            return x



    def __init__(self, device='cuda'):
        super(NLI_MiniHeads, self).__init__()
        
        self.heads = torch.nn.ModuleList([self.head() for _ in range(NUM_HEADS)])

        self.l1 = torch.nn.Linear(NUM_HEADS*DIMS, 2048)
        self.l2 = torch.nn.Linear(2048, 768)
        self.l3 = torch.nn.Linear(768, 256)
        self.l4 = torch.nn.Linear(256, 64)
        self.l5 = torch.nn.Linear(64, 1)

        self.lnorm = torch.nn.LayerNorm(NUM_HEADS*DIMS)
        self.dropout = torch.nn.Dropout(0.1)

        # move to device
        self.to(device)

    @torch.compile
    def forward(self, embeddings):

        pairs = torch.cat([embeddings[:, 0].unsqueeze(1).repeat(1, 10, 1), embeddings[:, 1:]], dim=2)
        # Create a list of inputs for each head
        inputs = [pairs] * len(self.heads)
        # Parallelize the computation of each head
        outputs = nn.parallel.parallel_apply(self.heads, inputs)
        # Concatenate the outputs of each head
        heads = torch.cat(outputs, dim=1)
    
        x = heads.view(heads.shape[0], -1)
        x = self.lnorm(x)
        x = self.l1(self.dropout(x))
        x = F.leaky_relu(x)
        x = self.l2(x)
        x = F.leaky_relu(self.dropout(x))
        x = self.l3(x)
        x = F.leaky_relu(x)
        x = self.l4(x)
        x = F.leaky_relu(x)
        x = self.l5(x)
        return x