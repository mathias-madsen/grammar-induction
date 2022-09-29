import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import Sequence, Tuple, Optional, Dict, List

from grammar import Grammar
from parsing import compile_likelihood_table


def sample_balanced_parens(maxdepth: Optional[int] = None) -> List[int]:
    """ Sample a list of 0s and 1s representing balanced left/right parens. """

    uniform = np.random.uniform(0, 1)
    newmaxdepth = maxdepth - 1 if maxdepth is not None else None

    if maxdepth is not None and maxdepth < 1:
        return [0, 1]
    elif 0 <= uniform < 1/3:
        return [0, 1]
    elif 1/3 <= uniform < 2/3:
        return (sample_balanced_parens(newmaxdepth) +
                sample_balanced_parens(newmaxdepth))
    elif 2/3 <= uniform < 1:
        return [0] + sample_balanced_parens(newmaxdepth) + [1]
    else:
        raise Exception("Unexpected toggle value: %r" % uniform)


def is_balanced_parens(sequence: Sequence[int]) -> bool:
    """ Check whether a list of 0/1s represents balanced L/R parens. """

    if any(x not in [0, 1] for x in sequence):
        raise ValueError("Expected binary list, got %r" % sequence)

    steps = [+1 if x == 0 else -1 for x in sequence]
    depths = np.cumsum(steps)

    return all(depths >= 0) and depths[-1] == 0


if __name__ == "__main__":

    model = Grammar(nchars=2, nrules=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    
    model.zero_grad()
    
    total_loss = 0.0
    total_sents = 0
    total_chars = 0
    
    for stepidx in range(1000):
    
        print("Step %s . . ." % stepidx)
        while total_sents < 20:
    
            indices = sample_balanced_parens(maxdepth=20)
            assert is_balanced_parens(indices)
            if len(indices) > 40:
                continue
            print(len(indices), end=", ", flush=True)
    
            trans, emits = model.get_trans_and_emits()
            likelihoods = compile_likelihood_table(trans, emits, indices)
            fullspanlikes = likelihoods[0, len(indices)]
            loss = -torch.log(fullspanlikes[0])
            loss.backward()
    
            total_loss += loss
            total_sents += 1
            total_chars += len(indices)
    
        optimizer.step()
        model.zero_grad()
    
        print("Loss per sentence: %.5f" % (total_loss / total_sents))
        print("Loss per character: %.5f" % (total_loss / total_chars))
        print("")
        total_loss = 0.0
        total_sents = 0
        total_chars = 0
    
        if (stepidx + 1) % 10 == 0:
    
            print("")
            print("Examples:")
            print("---------")
            judgments = []
            for _ in range(25):
                seq = model.sample(maxdepth=20)
                good = is_balanced_parens(seq)
                judgments.append(good)
                string = "".join("(" if t == 0 else ")" for t in seq)
                print(good, string)
            print("")
            k = sum(judgments)
            n = len(judgments)
            print("%s / %s (%.1f pct) were balanced" % (k, n, 100. * k / n))
            print("")