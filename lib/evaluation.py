import lib
import numpy as np
import torch
from tqdm import tqdm

class Evaluation(object):
    def __init__(self, model, loss_func, use_cuda, k=20, eval_hidden_reset=False, use_correct_mask_reset=False):
        self.model = model
        self.loss_func = loss_func
        self.topk = k
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.eval_hidden_reset = eval_hidden_reset
        self.use_correct_mask_reset = use_correct_mask_reset

    def eval(self, eval_data, batch_size):
        self.model.eval()
        losses = []
        recalls = []
        mrrs = []
        dataloader = lib.DataLoader(eval_data, batch_size, self.use_correct_mask_reset)
        with torch.no_grad():
            hidden = self.model.init_hidden()
            for ii, (input, target, mask) in tqdm(enumerate(dataloader), total=len(dataloader.dataset.df) // dataloader.batch_size, miniters = 1000):
                input = input.to(self.device)
                target = target.to(self.device)
                if self.eval_hidden_reset:
                    hidden[:,mask,:] = 0
                logit, hidden = self.model(input, hidden)
                logit_sampled = logit[:, target.view(-1)]
                loss = self.loss_func(logit_sampled)
                recall, mrr = lib.evaluate(logit, target, k=self.topk)

                losses.append(loss.item())
                recalls.append(recall)
                mrrs.append(mrr.item())
        mean_losses = np.mean(losses)
        mean_recall = np.mean(recalls)
        mean_mrr = np.mean(mrrs)

        return mean_losses, mean_recall, mean_mrr