import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging
import time
import os
from tqdm import tqdm

from drlearn.utils import AverageMeter

class NeuralNetModel(nn.Module):
    def __init__(self, game, args, input_shape):
        self.args = args
        self.game = game
        self.input_shape = input_shape
        super().__init__()
        if args.cuda:
            super().cuda()

    def fit(self, examples):
        """
        examples: list of examples, each example is of form (state, pi, v)
        """
        optimizer = optim.Adam(self.parameters())

        for epoch in range(self.args.epochs):
            logging.info('EPOCH ::: ' + str(epoch + 1))
            super().train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / self.args.batch_size)

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=self.args.batch_size)
                states, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                states = torch.FloatTensor(np.array(states).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if self.args.cuda:
                    states, target_pis, target_vs = states.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                # compute output
                out_pi, out_v = self(states)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), states.size(0))
                v_losses.update(l_v.item(), states.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, state):
        """
        state: np array with state
        """
        # timing
        start = time.time()

        # preparing input
        state = torch.FloatTensor(state.astype(np.float64))
        if self.args.cuda: state = state.contiguous().cuda()
        input_shape = (1, *self.input_shape)
        state = state.view(input_shape)
        super().eval()
        with torch.no_grad():
            pi, v = self(state)

        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]
    
    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_model(self, filename='cur.model'):
        folder = os.path.join(os.path.dirname(__file__), "../../saved_models")        
        filepath = os.path.join(folder, self.game.__class__.__name__+'.'+filename)
        if not os.path.exists(folder):
            logging.debug("saved_models directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            logging.debug("saved_models directory exists! ")
        torch.save({
            'state_dict': self.state_dict(),
        }, filepath)

    def load_model(self, filename='cur.model'):
        folder = os.path.join(os.path.dirname(__file__), "../../saved_models")        
        filepath = os.path.join(folder, self.game.__class__.__name__+'.'+filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if self.args.cuda else 'cpu'
        model = torch.load(filepath, map_location=map_location)
        self.load_state_dict(model['state_dict'])