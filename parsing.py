import torch
import numpy as np
from collections import defaultdict
from typing import Sequence, Tuple, Dict, Union


ArrayLike = Union[np.ndarray, torch.Tensor]


def compile_likelihood_table(
            trans: ArrayLike,
            emits: ArrayLike,
            terminal_indices: Sequence[int],
            ) -> Dict[Tuple[int, int], ArrayLike]:
    """ Compile a table of sub-sentence likelihoods.
    
    The entry at index [s, t, k] is the likelihood that nonterminal
    `k` expands into the sequence of tokens `terminal_indices[s:t]`.
    """

    if type(trans) == torch.Tensor:
        pkg = torch
    elif type(trans) == np.ndarray:
        pkg = np
    else:
        raise ValueError("Unexpected type: %r" % type(trans))

    nrules, _ = emits.shape
    length = len(terminal_indices)
    # inside = np.zeros([length, length + 1, nrules])
    inside = defaultdict(float)

    for start, idx in enumerate(terminal_indices):
        inside[start, start + 1] = emits[:, idx]

    for width in range(2, len(terminal_indices) + 1):
        for start in range(0, len(terminal_indices) - width + 1):
            stop = start + width
            probterms = []
            for split in range(start + 1, stop):
                left = inside[start, split]
                right = inside[split, stop]
                forkprobs = trans * left[:, None] * right[None, :]
                probterms.append(forkprobs)
            inside[start, stop] = pkg.sum(forkprobs, axis=(1, 2))

    return inside


def compile_log_likes(
            logtrans: torch.Tensor,
            logemits: torch.Tensor,
            terminal_indices: Sequence[int],
            ) -> Dict[Tuple[int, int], torch.Tensor]:
    """ Compile a table of sub-sentence log-likelihoods.
    
    The entry at index [s, t, k] is the log-likelihood that nonterminal
    `k` expands into the sequence of tokens `terminal_indices[s:t]`.
    """

    inside = dict()

    for start, idx in enumerate(terminal_indices):
        inside[start, start + 1] = logemits[:, idx]

    for width in range(2, len(terminal_indices) + 1):
        for start in range(0, len(terminal_indices) - width + 1):
            stop = start + width
            probterms = []
            for split in range(start + 1, stop):
                left = inside[start, split]
                right = inside[split, stop]
                forkprobs = logtrans + left[:, None] + right[None, :]
                probterms.append(forkprobs)
            inside[start, stop] = torch.logsumexp(forkprobs, axis=(1, 2))

    return inside
