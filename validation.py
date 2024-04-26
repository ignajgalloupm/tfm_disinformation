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
        batch_size = len(input_batch['claims'])

        # get embeddings of the claims
        with torch.no_grad():
            outputs = emb_gen(input_batch['claims'])

        # search for similar pages
        similar_pages = vdb.search_similar(outputs, PAGES_RETRIEVED)
        #print(similar_pages)
        similar_texts = [[t.payload['text'] for t in s] for s in similar_pages]
        similar_ids = [[t.payload['id'] for t in s] for s in similar_pages]
        similar_embeds = [[t.vector for t in s[:PAGES_FOR_EVIDENCE]] for s in similar_pages]

        target_changes, precentage_retrieved = get_target_changes(input_batch, similar_ids, PAGES_FOR_EVIDENCE)
        nli_targets = [v == 'VERIFIABLE' for v in input_batch['verifiable']]

        # dinamically change the target
        #---------------------------------- only training
        nli_targets = [int(t and tc) for t, tc in zip(nli_targets, target_changes)]
        #---------------------------------- only training

        all_evidence = [r['all_evidence'] if r['all_evidence'] != [None] else [] for r in input_batch['evidence']]
        evidence_pages = [vdb.search_ids(all_evidence[i]) for i in range(batch_size)]
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
                combined_embeddings.extend(emb_gen(unfolded_combined_texts[i:i+batch_size]))

        combined_embeddings = torch.tensor(combined_embeddings).to(device)
        unfolded_outputs = torch.tensor(unfolded_outputs).to(device)
        unfolded_labels = torch.tensor(unfolded_labels).to(device)

        loss1 = loss_fn1(combined_embeddings, unfolded_outputs, unfolded_labels)
        print(loss1)


        # input for the NLI model
        outputs = torch.tensor(outputs).unsqueeze(1).to(device)
        similar_embeds = torch.tensor(similar_embeds).to(device)

        # concat the output of the embedding generator
        nli_inputs = torch.cat([outputs, similar_embeds], dim=1)

        with torch.no_grad():
            nli_outputs = nli(nli_inputs.half())
        
        preds = torch.argmax(nli_outputs, dim=1)
        targets = torch.tensor(nli_targets).half().to(device)

        # nli_outputs is 32,2 keep only the 1
        nli_outputs = nli_outputs[:, 1]

        # Convert lists of tensors to tensors
        loss2 = loss_fn2(nli_outputs, targets)
        print(loss2)

        # calculate the f1 score, we apply a sigmoid to the output if the loss function is BCEWithLogitsLoss
        if loss_fn2.__class__.__name__ == 'BCEWithLogitsLoss':
            nli_outputs = torch.sigmoid(nli_outputs)
        nli_outputs = nli_outputs.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        f1 = f1_score(targets, (nli_outputs > 0.5).astype(int), average='macro')

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