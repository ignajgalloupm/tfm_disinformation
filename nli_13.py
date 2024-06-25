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



class NLI_FullLinear_13M(torch.nn.Module):

    def __init__(self, device='cuda'):
        super(NLI_FullLinear_13M, self).__init__()
        # input is batch_size x 6 x 768
        self.l1 = torch.nn.Linear(768*6, 2348)
        self.l2 = torch.nn.Linear(2348, 1024)
        self.l3 = torch.nn.Linear(1024, 512)
        self.l4 = torch.nn.Linear(512, 64)
        self.l5 = torch.nn.Linear(64, 1)

        self.dropout = torch.nn.Dropout(0.1)
        self.lnorm = torch.nn.LayerNorm(2348)

        # move to device
        self.to(device)

   
    def forward(self, embeddings):
            
        x = self.l1(embeddings.view(embeddings.shape[0], -1))
        x = self.lnorm(x)
        x = F.leaky_relu(x)
        x = self.l2(x)
        x = F.leaky_relu(self.dropout(x))
        x = F.leaky_relu(self.l3(x))
        x = F.leaky_relu(self.l4(x))
        x = self.l5(x)
        return x
    


# 0.70
class NLI_PairsBasic_13M(torch.nn.Module):

    def __init__(self, device='cuda'):
        super(NLI_PairsBasic_13M, self).__init__()
        # input is batch_size x 6 x 768
        self.c1 = torch.nn.Linear(768*2, 1024)
        self.c2 = torch.nn.Linear(1024, 768)

        self.l1 = torch.nn.Linear(768*5, 2048)
        self.l2 = torch.nn.Linear(2048, 1024)
        self.l3 = torch.nn.Linear(1024, 512)
        self.l4 = torch.nn.Linear(512, 256)
        self.l5 = torch.nn.Linear(256, 64)
        self.l6 = torch.nn.Linear(64, 1)

        self.lnorm = torch.nn.LayerNorm(2048)
        self.dropout = torch.nn.Dropout(0.1)

        # move to device
        self.to(device)

  
    def forward(self, embeddings):

        # pair every embedding with the first one
        pairs = torch.cat([embeddings[:, 0].unsqueeze(1).repeat(1, 5, 1), embeddings[:, 1:]], dim=2)

        c1 = self.c1(pairs)
        c1 = F.leaky_relu(c1)
        c2 = self.c2(c1)
        c2 = F.leaky_relu(self.dropout(c2))

        l1 = self.l1(c2.view(c2.shape[0], -1))
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
        return l6
    


NUM_HEADS_1 = 8
DIMS_1 = 64
class NLI_Heads_13M(torch.nn.Module):
    
    class head(torch.nn.Module):
        def __init__(self):
            super(NLI_Heads_13M.head, self).__init__()
            # input is batch_size x 6 x 768
            self.c1 = torch.nn.Linear(768*2, 512)
            self.c2 = torch.nn.Linear(512, DIMS_1)
        
        def forward(self, pairs):
            c1 = self.c1(pairs)
            c1 = F.leaky_relu(c1)
            c2 = self.c2(c1)
            return c2


    def __init__(self, device='cuda'):
        super(NLI_Heads_13M, self).__init__()
        
        self.heads = torch.nn.ModuleList([self.head() for _ in range(NUM_HEADS_1)])

        self.l1 = torch.nn.Linear(NUM_HEADS_1*5*DIMS_1, 2048)
        self.l2 = torch.nn.Linear(2048, 768)
        self.l3 = torch.nn.Linear(768, 256)
        self.l4 = torch.nn.Linear(256, 64)
        self.l5 = torch.nn.Linear(64, 1)

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
        x = F.leaky_relu(self.dropout(x))
        x = self.l3(x)
        x = F.leaky_relu(x)
        x = self.l4(x)
        x = F.leaky_relu(x)
        x = self.l5(x)
        return x




NUM_HEADS = 8
NUM_MINI_HEADS = 4
DIMS = 64
MINI_DIMS = 32
class NLI_MiniHeads_13M(torch.nn.Module):
    
    class head(torch.nn.Module):

        class mini_head(torch.nn.Module):
            def __init__(self):
                super(NLI_MiniHeads_13M.head.mini_head, self).__init__()
                # input is batch_size x 6 x 768
                self.c1 = torch.nn.Linear(768*2, 256)
                self.c2 = torch.nn.Linear(256, MINI_DIMS)
            
            def forward(self, pairs):
                c1 = self.c1(pairs)
                c1 = F.leaky_relu(c1)
                c2 = self.c2(c1)
                return c2
            
        
        def __init__(self):
            super(NLI_MiniHeads_13M.head, self).__init__()
            
            self.mini_heads = torch.nn.ModuleList([self.mini_head() for _ in range(NUM_MINI_HEADS)])

            self.l1 = torch.nn.Linear(NUM_MINI_HEADS*5*MINI_DIMS, 196)
            self.l2 = torch.nn.Linear(196, DIMS)


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
        super(NLI_MiniHeads_13M, self).__init__()
        
        self.heads = torch.nn.ModuleList([self.head() for _ in range(NUM_HEADS)])

        self.l1 = torch.nn.Linear(NUM_HEADS*DIMS, 196)
        self.l2 = torch.nn.Linear(196, 64)
        self.l3 = torch.nn.Linear(64, 1)

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
    
        x = heads.view(heads.shape[0], -1)
        x = self.lnorm(x)
        x = self.l1(self.dropout(x))
        x = F.leaky_relu(x)
        x = self.l2(x)
        x = F.leaky_relu(x)
        x = self.l3(x)

        return x