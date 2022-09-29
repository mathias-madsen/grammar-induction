import numpy as np

from typing import Optional, List


class Grammar:

    def __init__(self, nrules: int, nchars: int) -> None:

        self.nrules = nrules
        self.nchars = nchars

        self.trans = np.random.gamma(1.0, size=(nrules, nrules, nrules))
        self.trans /= np.sum(self.trans, axis=(1, 2), keepdims=True)

        self.emits = np.random.gamma(1.0, size=(nrules, nchars))
        self.emits /= np.sum(self.emits, axis=1, keepdims=True)

        self.emitprobs = np.random.beta(1.0, 2.0, size=nrules)
        self.trans *= (1 - self.emitprobs[:, None, None])
        self.emits *= self.emitprobs[:, None]
    
    def sample(self, maxdepth: Optional[int] = None,
                     root: int = 0) -> List[int]:

        # check whether we have reached the maximum depth:
        bottomed = maxdepth is not None and maxdepth < 1

        if bottomed or np.random.uniform(0, 1) < self.emitprobs[root]:
            # emit a character:
            dist = self.emits[root, :] / np.sum(self.emits[root, :])
            return [np.random.choice(self.nchars, p=dist)]
        else:
            # recursively branch:
            dist = self.trans[root, :, :] / np.sum(self.trans[root, :, :])
            root12 = np.random.choice(dist.size, p=dist.flatten())
            root1, root2 = np.unravel_index(root12, shape=dist.shape)
            newdepth = None if maxdepth is None else maxdepth - 1
            return self.sample(newdepth, root1) + self.sample(newdepth, root2)
    
