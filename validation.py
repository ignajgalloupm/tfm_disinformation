import torch
from torch.cuda.amp import autocast
import numpy as np
from sklearn.metrics import f1_score
from other_utils import print_progress
from train_eval_utils import get_target_changes, get_negative_examples

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
        # get embeddings of the claims
        with torch.no_grad():
            outputs = self.emb_gen(input_batch['claims'])

        # search for similar pages
        similar_pages = self.vdb.search_similar(outputs, PAGES_RETRIEVED)
        #print(similar_pages)
        similar_texts = [[t.payload['text'] for t in s] for s in similar_pages]
        similar_ids = [[t.payload['id'] for t in s[:PAGES_FOR_EVIDENCE]] for s in similar_pages]
        similar_embeds = [[t.vector for t in s] for s in similar_pages]

        target_changes, precentage_retrieved = get_target_changes(input_batch, similar_ids)
        targets = [v == 'VERIFIABLE' for v in input_batch['verifiable']]

        # dinamically change the target
        #---------------------------------- only training
        targets = [t and tc for t, tc in zip(targets, target_changes)]
        #---------------------------------- only training

        all_evidence = [r['all_evidence'] for r in input_batch['evidence']]
        evidence_pages = self.vdb.search_ids(all_evidence)
        evidence_texts = [[t.payload['text'] for t in s] for s in evidence_pages]
        # pick as negative examples the texts of the last len(evidence_texts) of the 50 retrieved pages
        negative_examples = get_negative_examples(similar_texts, similar_ids, all_evidence)

        # combine the positive and negative examples
        combined_texts = [s + n for s, n in zip(similar_texts, negative_examples)]

        # encode the combined texts in batches
        len_batch = len(input_batch['claims'])

        for i in range(0, len(combined_texts), len_batch):
            with torch.no_grad():
                combined_embeddings = self.emb_gen(combined_texts[i:i+len_batch])
            if i == 0:
                combined_embeds = combined_embeddings
            else:
                combined_embeds = torch.cat([combined_embeds, combined_embeddings], dim=0)


        # create the label tensor, half of the labels are 1 and half are 0



        loss1 = self.loss_fn1(outputs, similar_embeds)





        # input for the NLI model
        outputs = torch.tensor(outputs).unsqueeze(1).to(self.device)
        similar_embeds = torch.tensor(similar_embeds).to(self.device)

        # concat the output of the embedding generator
        nli_inputs = torch.cat([outputs, similar_embeds[::PAGES_FOR_EVIDENCE]], dim=1)

        with torch.no_grad():
            outputs = self.nli(nli_inputs.half())
        
        preds = torch.argmax(outputs, dim=1)
        targets = torch.tensor(targets).to(self.device)

        # Convert lists of tensors to tensors
        loss = self.loss_fn(outputs.squeeze(), targets.to(self.device))

        # calculate the f1 score, we apply a sigmoid to the output if the loss function is BCEWithLogitsLoss
        if self.loss_fn.__class__.__name__ == 'BCEWithLogitsLoss':
            outputs = torch.sigmoid(outputs)
        outputs = outputs.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        f1 = f1_score(targets, (outputs > 0.5).astype(int), average='macro')

        return {'preds': preds, 'targets': targets, 'target_changes': target_changes, 'precentage_retrieved': precentage_retrieved}

    # evaluates the model on the validation set and saves the model if it is the best one yet
    def valid_epoch(self, epoch, tracking):
        self.emb_gen.eval()
        self.nli.eval()

        # get the predictions and the targets for the validation set
        for _, data in enumerate(self.eval_loader, 0):
            result  = self.valid_step(data, self.emb_gen, self.nli)


        # save the results and show the progress
        loss_eval = tracking['loss_eval']
        f1_eval = tracking['f1_eval']
        loss_eval.append(loss.item()), f1_eval.append(f1)
        print_progress(epoch, 1, tracking, 1)

        # save the model if it is the best one, return True if it is not to stop the training
        if loss_eval[-1] == min(loss_eval):
            if pre == False: # for the final model
                torch.save(model.state_dict(), '{0}_{1}.pt'.format(model_name, difficulty))
            elif pre == True: # for the fine-tuned model
                torch.save(model.bert.state_dict(), 'pre_{0}_{1}.pt'.format(model_name, difficulty))
            return False
        return True