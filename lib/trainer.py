import os
import lib
import time
import torch
import numpy as np
from tqdm import tqdm


class Trainer(object):
    def __init__(self, model, train_data, eval_data, optim, use_cuda, loss_func, batch_size, args, use_correct_mask_reset):
        self.model = model
        self.train_data = train_data
        self.optim = optim
        self.loss_func = loss_func
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.batch_size = batch_size
        self.args = args
        self.use_correct_mask_reset = use_correct_mask_reset

    def train(self, start_epoch, end_epoch, start_time=None):
        if start_time is None:
            self.start_time = time.time()
        else:
            self.start_time = start_time

        for epoch in range(start_epoch, end_epoch + 1):
            st = time.time()
            train_loss, events = self.train_epoch(epoch)
            train_time = time.time() - st
            print("epoch:{} loss: {:.6f} {:.2f} s {:.2f} e/s {:.2f} mb/s".format(epoch, train_loss, train_time, np.sum(events)/train_time, len(events)/train_time))
            checkpoint = {
                'model': self.model,
                'args': self.args,
                'epoch': epoch,
                'optim': self.optim,
            }
            model_name = os.path.join(self.args.checkpoint_dir, "model_{0:05d}.pt".format(epoch))
            torch.save(checkpoint, model_name)
            print("Save model as %s" % model_name)


    def train_epoch(self, epoch):
        self.model.train()
        losses = []
        events = []

        def reset_hidden(hidden, mask):
            """Helper function that resets hidden state when some sessions terminate"""
            if len(mask) != 0:
                hidden[:, mask, :] = 0
            return hidden

        hidden = self.model.init_hidden()
        dataloader = lib.DataLoader(self.train_data, self.batch_size, use_correct_mask_reset=self.use_correct_mask_reset)
        for ii, (input, target, mask) in tqdm(enumerate(dataloader), total=len(dataloader.dataset.df) // dataloader.batch_size, miniters = 1000):
            input = input.to(self.device)
            target = target.to(self.device)
            self.optim.zero_grad()
            hidden = reset_hidden(hidden, mask).detach()
            logit, hidden = self.model(input, hidden)
            # output sampling
            logit_sampled = logit[:, target.view(-1)]
            loss = self.loss_func(logit_sampled)
            losses.append(loss.item())
            loss.backward()
            self.optim.step()
            events.append(len(input))

        mean_losses = np.mean(losses)
        return mean_losses, events