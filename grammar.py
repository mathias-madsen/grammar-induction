import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import Sequence, Tuple, Optional, Dict, List


class Grammar(torch.nn.Module):

    def __init__(self, nrules: int, nchars: int) -> None:

        super().__init__()

        self.nrules = nrules
        self.nchars = nchars

        logtrans = torch.normal(torch.zeros([nrules, nrules, nrules]), 1)
        self.logtrans = torch.nn.Parameter(logtrans)

        logemits = torch.normal(torch.zeros([nrules, nchars]), 1)
        self.logemits = torch.nn.Parameter(logemits)

        logit_emitprobs = torch.normal(torch.zeros([nrules]), 1)
        self.logit_emitprobs = torch.nn.Parameter(logit_emitprobs)

    def get_logtrans_and_logemits(self) -> Tuple[torch.Tensor, torch.Tensor]:

        # binary probabilities of the binary choice:
        logemitprobs = -torch.nn.functional.softplus(-self.logit_emitprobs)
        logbranchprobs = -torch.nn.functional.softplus(-self.logit_emitprobs)

        # conditional probabilities, assuming the binary choice:
        logtnorms = torch.logsumexp(self.logtrans, axis=(1, 2), keepdims=True)
        logtrans = self.logtrans - logtnorms
        logenorms = torch.logsumexp(self.logemits, axis=1, keepdims=True)
        logemits = self.logemits - logenorms

        logtrans = logtrans + logbranchprobs[:, None, None]
        logemits = logemits + logemitprobs[:, None]

        return logtrans, logemits

    def get_trans_and_emits(self) -> Tuple[torch.Tensor, torch.Tensor]:

        # binary probabilities of the binary choice:
        emitprobs = torch.sigmoid(self.logit_emitprobs)

        # conditional probabilities, assuming the binary choice:
        logtransflat = self.logtrans.reshape([self.nrules, -1])
        transflat = torch.softmax(logtransflat, dim=1)
        cond_trans = transflat.reshape([self.nrules, self.nrules, self.nrules])
        cond_emits = torch.softmax(self.logemits, dim=1)

        # joint probabilities of the combined choice:
        trans = cond_trans * (1 - emitprobs[:, None, None])
        emits = cond_emits * emitprobs[:, None]

        transsums = trans.sum(axis=(1, 2))
        emitssums = emits.sum(axis=1)
        assert torch.all(transsums <= 1)
        assert torch.all(transsums >= 0)
        assert torch.all(emitssums <= 1)
        assert torch.all(emitssums >= 0)
        assert torch.allclose(transsums + emitssums, torch.ones([]))
    
        return trans, emits

    def sample(self, maxdepth: Optional[int] = None) -> List[int]:

        trans, emits = self.get_trans_and_emits()
        emitprobs = emits.sum(axis=1)

        trans = trans.detach().numpy()
        emits = emits.detach().numpy()
        emitprobs = emitprobs.detach().numpy()

        return self._subsample(trans, emits, emitprobs, 0, maxdepth)

    def _subsample(self, trans, emits, emitprobs, rootidx, maxdepth):

        depth_exceeded = maxdepth is not None and maxdepth < 1

        if np.random.uniform(0, 1) < emitprobs[rootidx] or depth_exceeded:
            # time to emit a terminal
            dist = emits[rootidx, :]
            dist = dist / dist.sum()
            termidx = np.random.choice(range(dist.size), p=dist)
            return [termidx]
        else:
            # do another splitting
            dist = trans[rootidx, :, :]
            dist = dist / dist.sum()
            pairidx = np.random.choice(range(dist.size), p=dist.flatten())
            pairidx1, pairidx2 = np.unravel_index(pairidx, shape=dist.shape)
            newmaxdepth = maxdepth - 1 if maxdepth is not None else None
            return (self._subsample(trans, emits, emitprobs, pairidx1, newmaxdepth) + 
                    self._subsample(trans, emits, emitprobs, pairidx2, newmaxdepth))
