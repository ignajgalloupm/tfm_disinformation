from other_utils import print_progress
from train_eval_utils2 import get_metrics, nli_step, emb_gen_step
from torch.cuda.amp import autocast
import torch
from torch.utils.data import DataLoader
import time
from sub_fever_dataset import Sub_Dataset, Sub_Collator


BATCH_SIZE = 8#3


class Train:

    def __init__(self, device, train_loader, loss_fn1, loss_fn2, optimizer, model_name):
        self.device = device
        self.train_loader = train_loader
        self.loss_fn1 = loss_fn1
        self.loss_fn2 = loss_fn2
        self.optimizer = optimizer
        self.model_name = model_name
        self.sub_dataloader = None


    # performs a single validation step
    @autocast()
    def train_step(self, input_batch, emb_gen, nli):
        #with torch.no_grad():
        outputs, loss1 = emb_gen_step(input_batch, emb_gen, self.loss_fn1, self.device)
        #cut the gradient flow
        # outputs = outputs.detach()

        # outputs = input_batch['claim_embs']
        # loss1 = torch.tensor(0.0).to(self.device)

        preds, loss2 = nli_step(input_batch, emb_gen, nli, outputs, self.loss_fn2, self.device)

        total_loss = loss1 + loss2

        total_loss.backward()

        result = {'preds': preds, 
                  'original_labels': input_batch['original_nli_targets'],
                  'dynamic_labels': input_batch['dynamic_nli_targets'], 
                  'percentage_retrieved': input_batch['percentage_retrieved'],
                  'loss1': loss1.item(),
                  'loss2': loss2.item(),
                  'total_loss': total_loss.item()}
        return result
    
    def train_super_step(self, sub_dataloader, emb_gen, nli, super_batch, epoch, super_i, tracking_train, tracking_eval=None, ma_ratio=None):
        emb_gen.train()
        nli.train()
        results = []
        # get the predictions and the targets for the validation set
        for i, data in enumerate(sub_dataloader, 0):
            result = self.train_step(data, emb_gen, nli)
            print(result)
            results.append(result)
            if i % super_batch == super_batch - 1:
                self.optimizer.step()
                self.optimizer.zero_grad()

                # save the results and show the progress
                metrics = get_metrics(results)
                results = []
                for k, v in metrics.items():
                    tracking_train[k].append(v)
                print_progress(epoch=epoch, batch=super_i * len(sub_dataloader) + i, num_total_batches=len(self.train_loader.dataset)//BATCH_SIZE, tracking_train=tracking_train, tracking_eval=tracking_eval, ma_ratio=ma_ratio)

        if len(results) > 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

            # save the results and show the progress
            metrics = get_metrics(results)
            for k, v in metrics.items():
                tracking_train[k].append(v)
            print_progress(epoch=epoch, batch=super_i * len(sub_dataloader) + i, num_total_batches=len(self.train_loader.dataset)//BATCH_SIZE, tracking_train=tracking_train, tracking_eval=tracking_eval, ma_ratio=ma_ratio)
        

    # evaluates the model on the validation set and saves the model if it is the best one yet
    def train_epoch(self, epoch, vdb, emb_gen, nli, tracking_train, tracking_eval=None, super_batch=10, ma_ratio=None):
        
        # get the predictions and the targets for the validation set
        for i, data in enumerate(self.train_loader, 0):
            if epoch % 2 == 0:
                emb_gen.eval()
                with torch.no_grad():
                    claim_embs = emb_gen(data['claims']).detach().half().cpu()
                data['claim_embs'] = claim_embs

                sub_dataset = Sub_Dataset(data, vdb, set_type='train')
                sub_collator = Sub_Collator()
                self.sub_dataloader = DataLoader(sub_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=sub_collator, num_workers=8)
            else:
                self.sub_dataloader.dataset.second = True
            self.train_super_step(self.sub_dataloader, emb_gen, nli, super_batch, epoch, i, tracking_train, tracking_eval, ma_ratio)

        if epoch % 2 == 1:
            vdb.refresh(emb_gen)
            if not vdb.wiki_loader.dataset.reduced:
                # sleep for 10 mins
                time.sleep(600)

        return True