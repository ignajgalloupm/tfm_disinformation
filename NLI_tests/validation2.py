import torch
from NLI_tests.train_eval_utils2 import get_metrics, nli_step
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from NLI_tests.sub_fever_dataset import Sub_Dataset, Sub_Collator


class Validation:

    def __init__(self, device, eval_loader, loss_fn2, vdb, emb_gen):
        self.device = device
        self.eval_loader = eval_loader
        self.loss_fn2 = loss_fn2

        for _, data in enumerate(self.eval_loader, 0):
            with torch.no_grad():
                claim_embs = emb_gen(data['claims']).detach().half().cpu()
            data['claim_embs'] = claim_embs

            sub_dataset = Sub_Dataset(data, vdb, set_type='eval')
            sub_collator = Sub_Collator()
            self.sub_dataloader = DataLoader(sub_dataset, batch_size=64, shuffle=True, collate_fn=sub_collator, num_workers=8, pin_memory=True)


    # performs a single validation step
    @torch.no_grad()
    def valid_step(self, input_batch, emb_gen, nli):
        outputs = input_batch['claim_embs']
        preds, loss2 = nli_step(input_batch, emb_gen, nli, outputs, self.loss_fn2, self.device)

        result = {'preds': preds, 
                  'original_labels': input_batch['original_nli_targets'],
                  'dynamic_labels': input_batch['dynamic_nli_targets'], 
                  'percentage_retrieved': input_batch['percentage_retrieved'],
                  'loss2': [l.item() for l in loss2]}
        return result

        

    # evaluates the model on the validation set and saves the model if it is the best one yet
    def valid_epoch(self, emb_gen, nli, tracking_eval):
        emb_gen.eval()
        [n.eval() for n in nli]
        results = []
        # get the predictions and the targets for the validation set
        for i, data in enumerate(self.sub_dataloader, 0):
            result  = self.valid_step(data, emb_gen, nli)
            results.append(result)

        # save the results and show the progress
        metrics = get_metrics(results)
        for k, v in metrics.items():
            tracking_eval[k].append(v)

        return True