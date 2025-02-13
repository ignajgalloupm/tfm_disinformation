
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import os
import torch
from torch.amp import autocast
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class EmbeddingGenerator(torch.nn.Module):

    def __init__(self, encoder='gte', version='v1', device='cuda', batch_size=16):
        super(EmbeddingGenerator, self).__init__()
        self.name = f'{encoder}_{version}'
        self.device = device
        self.batch_size = batch_size
        os.makedirs(f'embeddings/{self.name}', exist_ok=True)
        if encoder == 'gte':
            model = AutoModel.from_pretrained('Alibaba-NLP/gte-large-en-v1.5',
                                              trust_remote_code=True,
                                              unpad_inputs=True,
                                              use_memory_efficient_attention=True)
            self.tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-large-en-v1.5')
        else:
            raise ValueError('Invalid encoder')
        # freeze the first half of the model
        for param in model.embeddings.parameters():
            param.requires_grad = False
        num_encoder_layers = len([p for p in model.encoder.parameters()])
        for i, param in enumerate(model.encoder.parameters()):
                if i < num_encoder_layers - 55:
                    param.requires_grad = False

        self.model = model.to(device)

 
    def forward(self, texts):   
        
        if len(texts) > self.batch_size:
            encoded_input = self.tokenizer(texts, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
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
                with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                    model_output = self.model(input_ids=in_ids, attention_mask=at_mask)
                # Perform pooling and normalization
                sentence_embeddings = model_output.last_hidden_state[:, 0]
                sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
                encodes.extend([e for e in sentence_embeddings])
            # reorder the embeddings to the original order, indices indicate the original position of the embeddings
            _, inverse_indices = torch.sort(indices)
            encodes = [encodes[i] for i in inverse_indices]
            output = torch.stack(encodes)

        else:
            # Compute token embeddings
            encoded_input = self.tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors='pt').to(self.device)
            # Compute token embeddings
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                model_output = self.model(**encoded_input)
            # Perform pooling and normalization
            output = model_output.last_hidden_state[:, 0]
            output = F.normalize(output, p=2, dim=1)
        return output
