import torch
from other_utils import print_progress
from train_eval_utils import get_metrics, nli_step, emb_gen_step




class Train:

    def __init__(self, device, train_loader, loss_fn1, loss_fn2, optimizer, model_name):
        self.device = device
        self.train_loader = train_loader
        self.loss_fn1 = loss_fn1
        self.loss_fn2 = loss_fn2
        self.optimizer = optimizer
        self.model_name = model_name


    # performs a single validation step
    def train_step(self, input_batch, vdb, emb_gen, nli):
        outputs, dynamic_nli_targets, original_nli_targets, percentage_retrieved, loss1 = emb_gen_step(input_batch, vdb, emb_gen, self.loss_fn1, self.device)
        
        # cut the gradient flow in outputs
        outputs = outputs.detach()

        preds, loss2 = nli_step(vdb, nli, outputs, dynamic_nli_targets, self.loss_fn2, self.device)
        total_loss = loss1 + loss2
        total_loss.backward()
        # clip the gradient
        # torch.nn.utils.clip_grad_norm_(nli.parameters(), 0.1)
        # torch.nn.utils.clip_grad_norm_(emb_gen.parameters(), 0.1)
        self.optimizer.step()
        self.optimizer.zero_grad()

        result = {'preds': preds, 
                'original_labels': original_nli_targets, 
                'dynamic_labels': dynamic_nli_targets, 
                'percentage_retrieved': percentage_retrieved,
                'loss1': loss1.item(),
                'loss2': loss2.item()}
        return result
        

    # evaluates the model on the validation set and saves the model if it is the best one yet
    def train_epoch(self, epoch, vdb, emb_gen, nli, tracking_train, tracking_eval=None, visualization_step=10):
        emb_gen.train()
        nli.train()
        results = []
        # get the predictions and the targets for the validation set
        for i, data in enumerate(self.train_loader, 0):
            result = self.train_step(data, vdb, emb_gen, nli)
            results.append(result)
            if i % visualization_step == visualization_step - 1:
                # save the results and show the progress
                metrics = get_metrics(results)
                results = []
                for k, v in metrics.items():
                    tracking_train[k].append(v)
                print_progress(epoch=epoch, batch=i, num_total_batches=len(self.train_loader), tracking_train=tracking_train, tracking_eval=tracking_eval, ma_ratio=0.1)

        # save the model if it is the best one, return True if it is not to stop the training
        # if loss_eval[-1] == min(loss_eval):
        #     if pre == False: # for the final model
        #         torch.save(model.state_dict(), '{0}_{1}.pt'.format(model_name, difficulty))
        #     elif pre == True: # for the fine-tuned model
        #         torch.save(model.bert.state_dict(), 'pre_{0}_{1}.pt'.format(model_name, difficulty))
        #     return False
        return True