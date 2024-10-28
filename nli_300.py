import torch.nn.functional as F
import os
import torch
from torch import nn


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



class NLI_FullLinear_300k(torch.nn.Module):

    def __init__(self, device='cuda'):
        super(NLI_FullLinear_300k, self).__init__()
        # input is batch_size x 6 x 768
        self.l1 = torch.nn.Linear(768*6, 64)
        self.l2 = torch.nn.Linear(64, 32)
        self.l3 = torch.nn.Linear(32, 8)
        self.l4 = torch.nn.Linear(8, 1)

        self.lnorm = torch.nn.LayerNorm(64)
        self.dropout = torch.nn.Dropout(0.1)

        # move to device
        self.to(device)

   
    def forward(self, embeddings):
            
        x = self.l1(embeddings.view(embeddings.shape[0], -1))
        x = F.leaky_relu(x)
        x = self.lnorm(x)
        x = self.dropout(x)
        x = self.l2(x)
        x = F.leaky_relu(x)
        x = self.l3(x)
        x = F.leaky_relu(x)
        x = self.l4(x)
        return x
    


# 0.70
class NLI_PairsBasic_300k(torch.nn.Module):

    def __init__(self, device='cuda'):
        super(NLI_PairsBasic_300k, self).__init__()
        # input is batch_size x 11 x 768
        self.c1 = torch.nn.Linear(768*2, 128).to(device)
        self.l1 = torch.nn.Linear(128*5, 128).to(device)
        self.l2 = torch.nn.Linear(128, 32).to(device)
        self.l3 = torch.nn.Linear(32, 8).to(device)
        self.l4 = torch.nn.Linear(8, 1).to(device)

        self.lnorm = torch.nn.LayerNorm(128)
        self.dropout = torch.nn.Dropout(0.1).to(device)

        # move to device
        self.to(device)

  
    def forward(self, embeddings):

        # pair every embedding with the first one
        pairs = torch.cat([embeddings[:, 0].unsqueeze(1).repeat(1, 5, 1), embeddings[:, 1:]], dim=2)

        c1 = self.c1(pairs)
        c1 = F.leaky_relu(c1)
        c1 = self.dropout(c1)

        l1 = self.l1(c1.view(c1.shape[0], -1))
        l1 = F.leaky_relu(self.lnorm(l1))
        l2 = self.l2(l1)
        l2 = F.leaky_relu(l2)
        l3 = self.l3(l2)
        l3 = F.leaky_relu(l3)
        l4 = self.l4(l3)
        return l4
    


NUM_HEADS_1 = 4
DIMS_1 = 8
class NLI_Heads_300k(torch.nn.Module):
    
    class head(torch.nn.Module):
        def __init__(self):
            super(NLI_Heads_300k.head, self).__init__()
            # input is batch_size x 11 x 768
            self.c1 = torch.nn.Linear(768*2, 48)
            self.c2 = torch.nn.Linear(48, DIMS_1)
        
        def forward(self, pairs):
            c1 = self.c1(pairs)
            c1 = F.leaky_relu(c1)
            c2 = self.c2(c1)
            return c2


    def __init__(self, device='cuda'):
        super(NLI_Heads_300k, self).__init__()
        
        self.heads = torch.nn.ModuleList([self.head() for _ in range(NUM_HEADS_1)])

        self.l1 = torch.nn.Linear(NUM_HEADS_1*5*DIMS_1, 64)
        self.l2 = torch.nn.Linear(64, 8)
        self.l3 = torch.nn.Linear(8, 1)

        self.lnorm = torch.nn.LayerNorm(NUM_HEADS_1*5*DIMS_1)
        self.dropout = torch.nn.Dropout(0.1)

        # move to device
        self.to(device)


    def forward(self, embeddings):

        pairs = torch.cat([embeddings[:, 0].unsqueeze(1).repeat(1, 5, 1), embeddings[:, 1:]], dim=2)
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
        x = F.leaky_relu(x)
        x = self.l3(x)
        return x




NUM_HEADS = 2
NUM_MINI_HEADS = 4
DIMS = 8
MINI_DIMS = 4
class NLI_MiniHeads_300k(torch.nn.Module):
    
    class head(torch.nn.Module):

        class mini_head(torch.nn.Module):
            def __init__(self):
                super(NLI_MiniHeads_300k.head.mini_head, self).__init__()
                # input is batch_size x 11 x 768
                self.c1 = torch.nn.Linear(768*2, 24)
                self.c2 = torch.nn.Linear(24, MINI_DIMS)
            
            def forward(self, pairs):
                c1 = self.c1(pairs)
                c1 = F.leaky_relu(c1)
                c2 = self.c2(c1)
                return c2
            
        
        def __init__(self):
            super(NLI_MiniHeads_300k.head, self).__init__()
            
            self.mini_heads = torch.nn.ModuleList([self.mini_head() for _ in range(NUM_MINI_HEADS)])

            self.l1 = torch.nn.Linear(NUM_MINI_HEADS*5*MINI_DIMS, 16)
            self.l2 = torch.nn.Linear(16, DIMS)


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
        super(NLI_MiniHeads_300k, self).__init__()
        
        self.heads = torch.nn.ModuleList([self.head() for _ in range(NUM_HEADS)])

        self.l1 = torch.nn.Linear(NUM_HEADS*DIMS, 8)
        self.l2 = torch.nn.Linear(8, 1)

        self.lnorm = torch.nn.LayerNorm(NUM_HEADS*DIMS)
        self.dropout = torch.nn.Dropout(0.1)

        # move to device
        self.to(device)


    def forward(self, embeddings):

        pairs = torch.cat([embeddings[:, 0].unsqueeze(1).repeat(1, 5, 1), embeddings[:, 1:]], dim=2)
        # Create a list of inputs for each head
        inputs = [pairs] * len(self.heads)
        # Parallelize the computation of each head
        outputs = nn.parallel.parallel_apply(self.heads, inputs)
        # Concatenate the outputs of each head
        heads = torch.cat(outputs, dim=1)
    
        x = self.dropout(heads.view(heads.shape[0], -1))
        x = self.lnorm(x)
        x = self.l1(x)
        x = F.leaky_relu(x)
        x = self.l2(x)
        return x