import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import Sequence, Tuple, Optional, Dict, List

from grammar import Grammar
from parsing import compile_log_likes


def sample_palindrome(alphabet_size: int,
                      maxdepth: Optional[int] = None) -> List[int]:
    """ Sample a list of ints which reads the same in both directions. """

    uniform = np.random.uniform(0, 1)
    newmaxdepth = maxdepth - 1 if maxdepth is not None else None
    letter = np.random.choice(range(alphabet_size))

    if uniform < 0.3 or (maxdepth is not None and maxdepth < 1):
        if np.random.uniform(0, 1) < 1/2:
            return [letter, letter]
        else:
            return [letter]
    else:
        middle = sample_palindrome(alphabet_size, newmaxdepth)
        return [letter] + middle + [letter]


def is_palindrome(sequence: Sequence[int]) -> bool:
    """ Whether a list of ints reads the same from left and right. """

    return sequence == sequence[::-1]


if __name__ == "__main__":

    alphabet_size = 2

    model = Grammar(nchars=alphabet_size, nrules=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    model.zero_grad()
    
    total_loss = 0.0
    total_sents = 0
    total_chars = 0
    
    for stepidx in range(1000):
    
        print("Step %s . . ." % stepidx)
        while total_sents < 128:
    
            indices = sample_palindrome(alphabet_size, maxdepth=20)
            assert is_palindrome(indices)
            if len(indices) > 200:
                continue
    
            logtrans, logemits = model.get_logtrans_and_logemits()
            loglikelihoods = compile_log_likes(logtrans, logemits, indices)
            fullspanloglikes = loglikelihoods[0, len(indices)]
            loss = -fullspanloglikes[0]  # neg log like under nonterm S
            loss.backward()
    
            total_loss += float(loss)
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
            k = 0
            n = 0
            for _ in range(100):
                seq = model.sample(maxdepth=20)
                halflen = len(seq) // 2
                matches = sum(x == y for x, y in zip(seq[:halflen], seq[::-1]))
                k += matches
                n += halflen
                string = "".join(chr(65 + i) for i in seq)
                print(string, end=", ", flush=True)
            print("")
            print("")
            print("%s / %s (%.1f pct) correctly contrained characters"
                  % (k, n, 100. * (k + 1e-3) / (n + 2e-3)))
            print("")