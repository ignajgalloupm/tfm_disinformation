import torch
from train_eval_utils import get_metrics, nli_step, emb_gen_step
from torch.cuda.amp import autocast



class Validation:

    def __init__(self, device, eval_loader, loss_fn1, loss_fn2, model_name):
        self.device = device
        self.eval_loader = eval_loader
        self.loss_fn1 = loss_fn1
        self.loss_fn2 = loss_fn2
        self.model_name = model_name


    # performs a single validation step
    @torch.no_grad()
    @autocast()
    def valid_step(self, input_batch, vdb, emb_gen, nli):
        outputs, dynamic_nli_targets, original_nli_targets, percentage_retrieved, loss1 = emb_gen_step(input_batch, vdb, emb_gen, self.loss_fn1, self.device)
        preds, loss2 = nli_step(vdb, nli, outputs, dynamic_nli_targets, self.loss_fn2, self.device)
        result = {'preds': preds, 
                'original_labels': original_nli_targets, 
                'dynamic_labels': dynamic_nli_targets, 
                'percentage_retrieved': percentage_retrieved,
                'loss1': loss1.item(),
                'loss2': loss2.item()}
        return result
        

    # evaluates the model on the validation set and saves the model if it is the best one yet
    def valid_epoch(self, vdb, emb_gen, nli, tracking_eval):
        emb_gen.eval()
        nli.eval()
        results = []
        # get the predictions and the targets for the validation set
        for _, data in enumerate(self.eval_loader, 0):
            result  = self.valid_step(data, vdb, emb_gen, nli)
            results.append(result)

        # save the results and show the progress
        metrics = get_metrics(results)
        for k, v in metrics.items():
            tracking_eval[k].append(v)


        # save the model if it is the best one, return True if it is not to stop the training
        # if loss_eval[-1] == min(loss_eval):
        #     if pre == False: # for the final model
        #         torch.save(model.state_dict(), '{0}_{1}.pt'.format(model_name, difficulty))
        #     elif pre == True: # for the fine-tuned model
        #         torch.save(model.bert.state_dict(), 'pre_{0}_{1}.pt'.format(model_name, difficulty))
        #     return False
        return True