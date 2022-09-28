import torch
from typing import Sequence, Tuple, Dict


def compile_likelihood_table(
            trans: torch.Tensor,
            emits: torch.Tensor,
            terminal_indices: Sequence[int],
            ) -> Dict[Tuple[int, int], torch.Tensor]:

    nrules, _ = emits.shape
    inside = dict()

    for start, idx in enumerate(terminal_indices):
        inside[start, start + 1] = emits[:, idx]

    for width in range(2, len(terminal_indices) + 1):
        for start in range(0, len(terminal_indices) - width + 1):
            stop = start + width
            inside[start, stop] = torch.zeros([nrules])
            for split in range(start + 1, stop):
                left = inside[start, split]
                right = inside[split, stop]
                forkprobs = trans * left[:, None] * right[None, :]
                inside[start, stop] += torch.sum(forkprobs, axis=(1, 2))

    return inside

