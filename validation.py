import torch
from torch.cuda.amp import autocast
import numpy as np
from sklearn.metrics import f1_score
from other_utils import print_progress
from train_eval_utils import get_target_changes, get_negative_examples, get_metrics

PAGES_RETRIEVED = 50
PAGES_FOR_EVIDENCE = 10


class Validation:

    def __init__(self, device, eval_loader, vdb, emb_gen, nli, loss_fn1, loss_fn2, model_name):
        self.device = device
        self.eval_loader = eval_loader
        self.vdb = vdb
        self.emb_gen = emb_gen
        self.nli = nli
        self.loss_fn1 = loss_fn1
        self.loss_fn2 = loss_fn2
        self.model_name = model_name


    # performs a single validation step
    def valid_step(self, input_batch):
        batch_size = len(input_batch['claims'])

        # get embeddings of the claims
        with torch.no_grad():
            outputs = self.emb_gen(input_batch['claims'])

        # search for similar pages
        similar_pages = self.vdb.search_similar(outputs, PAGES_RETRIEVED, with_payload=True)

        similar_texts = []
        similar_ids = []
        for s in similar_pages:
            similar_texts.append([t.payload['text'] for t in s])
            similar_ids.append([t.payload['id'] for t in s])

        target_changes, percentage_retrieved = get_target_changes(input_batch, similar_ids, PAGES_FOR_EVIDENCE)
        origibal_nli_targets = [int(v == 'VERIFIABLE') for v in input_batch['verifiable']]

        # dinamically change the target
        #---------------------------------- only training
        dynamic_nli_targets = [int(t and tc) for t, tc in zip(origibal_nli_targets, target_changes)]
        #---------------------------------- only training

        all_evidence = [r['all_evidence'] if r['all_evidence'] != [None] else [] for r in input_batch['evidence']]
        evidence_pages = [self.vdb.search_ids(all_evidence[i]) for i in range(batch_size)]
        evidence_texts = [[t.payload['text'] for t in s] for s in evidence_pages]
        # pick as negative examples the texts of the last len(evidence_texts) of the 50 retrieved pages
        negative_examples = get_negative_examples(similar_texts, similar_ids, all_evidence)

        # combine all the batches
        unfolded_outputs = []
        unfolded_combined_texts = []
        unfolded_labels = []
        for i in range(batch_size):
            unfolded_outputs.extend([outputs[i]] * len(evidence_texts[i] + negative_examples[i]))
            unfolded_combined_texts.extend(evidence_texts[i] + negative_examples[i])
            unfolded_labels.extend([1] * len(evidence_texts[i]) + [0] * len(negative_examples[i]))
        
        # encode the combined texts in batches
        combined_embeddings = []
        for i in range(0, len(unfolded_combined_texts), batch_size):
            with torch.no_grad():
                combined_embeddings.extend(self.emb_gen(unfolded_combined_texts[i:i+batch_size]))

        combined_embeddings = torch.tensor(np.array(combined_embeddings)).to(self.device)
        unfolded_outputs = torch.tensor(np.array(unfolded_outputs)).to(self.device)
        unfolded_labels = torch.tensor(np.array(unfolded_labels)).to(self.device)

        loss1 = self.loss_fn1(combined_embeddings, unfolded_outputs, unfolded_labels)


        # input for the NLI model
        similar_embeddings = self.vdb.search_similar(outputs, PAGES_FOR_EVIDENCE, with_vector=True)
        similar_embeds = [[t.vector for t in s] for s in similar_embeddings]
        similar_embeds = torch.tensor(np.array(similar_embeds))
        outputs = torch.tensor(outputs).unsqueeze(1)

        # concat the output of the embedding generator
        nli_inputs = torch.cat([outputs, similar_embeds], dim=1).to(self.device)

        with torch.no_grad():
            nli_outputs = self.nli(nli_inputs.half())
        
        preds = torch.argmax(nli_outputs, dim=1).cpu().numpy()
        targets = torch.tensor(dynamic_nli_targets).half().to(self.device)

        # nli_outputs is 32,2 keep only the 1
        nli_outputs = nli_outputs[:, 1]

        # Convert lists of tensors to tensors
        loss2 = self.loss_fn2(nli_outputs, targets)
        
        result = {'preds': preds, 
                'original_labels': origibal_nli_targets, 
                'dynamic_labels': dynamic_nli_targets, 
                'percentage_retrieved': percentage_retrieved,
                'loss1': loss1.item(),
                'loss2': loss2.item()}
        
        return result
        

    # evaluates the model on the validation set and saves the model if it is the best one yet
    def valid_epoch(self, epoch, tracking_eval):
        self.emb_gen.eval()
        self.nli.eval()

        results = []
        # get the predictions and the targets for the validation set
        for _, data in enumerate(self.eval_loader, 0):
            result  = self.valid_step(data)
            results.append(result)

        # save the results and show the progress
        metrics = get_metrics(results)
        for k, v in metrics.items():
            tracking_eval[k].append(v)
        print_progress(epoch=epoch, batch=1, num_total_batches=1, tracking_eval=tracking_eval, ma_ratio=None)

        # save the model if it is the best one, return True if it is not to stop the training
        # if loss_eval[-1] == min(loss_eval):
        #     if pre == False: # for the final model
        #         torch.save(model.state_dict(), '{0}_{1}.pt'.format(model_name, difficulty))
        #     elif pre == True: # for the fine-tuned model
        #         torch.save(model.bert.state_dict(), 'pre_{0}_{1}.pt'.format(model_name, difficulty))
        #     return False
        return True