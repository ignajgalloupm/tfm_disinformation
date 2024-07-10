from NLI_tests.other_utils import print_progress
from NLI_tests.train_eval_utils2 import get_metrics, nli_step
from torch.cuda.amp import autocast
import torch
from torch.utils.data import DataLoader
from NLI_tests.sub_fever_dataset import Sub_Dataset, Sub_Collator


BATCH_SIZE = 16#3


class Train:

    def __init__(self, device, train_loader, loss_fn2, optimizer, vdb, evdb, emb_gen):
        self.device = device
        self.loss_fn2 = loss_fn2
        self.optimizer = optimizer

        # get the predictions and the targets for the validation set
        for _, data in enumerate(train_loader, 0):
            emb_gen.eval()
            with torch.no_grad():
                claim_embs = emb_gen(data['claims']).detach().half().cpu()
            data['claim_embs'] = claim_embs

            sub_dataset = Sub_Dataset(data, vdb, set_type='train', evdb=evdb)
            sub_collator = Sub_Collator()
            self.sub_dataloader = DataLoader(sub_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=sub_collator, num_workers=8)


    # performs a single validation step
    @autocast()
    def train_step(self, input_batch, emb_gen, nli):
        outputs = input_batch['claim_embs']
        preds, loss2 = nli_step(input_batch, emb_gen, nli, outputs, self.loss_fn2, self.device)

        for l in loss2:
            l.backward()

        result = {'preds': preds, 
                  'original_labels': input_batch['original_nli_targets'],
                  'dynamic_labels': input_batch['dynamic_nli_targets'], 
                  'percentage_retrieved': input_batch['percentage_retrieved'],
                  'loss2': [l.item() for l in loss2]}
        return result
    

    # evaluates the model on the validation set and saves the model if it is the best one yet
    def train_epoch(self, epoch, emb_gen, nli, val, tracking_train, tracking_eval=None, super_batch=10, ma_ratio=None):
        results = []
        # get the predictions and the targets for the validation set
        for i, data in enumerate(self.sub_dataloader, 0):
            [n.train() for n in nli]
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
                print_progress(epoch=epoch, batch=i, num_total_batches=len(self.sub_dataloader), tracking_train=tracking_train, tracking_eval=tracking_eval, ma_ratio=ma_ratio)
            if i % (50000//BATCH_SIZE) == 0 and i > 0:
                val.valid_epoch(emb_gen=emb_gen, nli=nli, tracking_eval=tracking_eval)

        if len(results) > 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

            # save the results and show the progress
            metrics = get_metrics(results)
            for k, v in metrics.items():
                tracking_train[k].append(v)
            print_progress(epoch=epoch, batch=i, num_total_batches=len(self.sub_dataloader), tracking_train=tracking_train, tracking_eval=tracking_eval, ma_ratio=ma_ratio)
            
        return True