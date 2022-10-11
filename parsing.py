import torch
import numpy as np
from collections import defaultdict
from typing import Sequence, Tuple, Dict, Union


def compile_log_likes(
            logtrans: Union[np.ndarray, torch.Tensor],
            logemits: Union[np.ndarray, torch.Tensor],
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


def compile_priors(trans, emits, indices, loglikes):

    nrules, nchars = emits.shape
    length = len(indices)
    outside = dict()
    outside[0, length] = torch.tensor([1.] + (nrules - 1)*[0.])

    for width in reversed(range(1, length)):
        for start in range(0, length - width + 1):
            stop = start + width
            joints = []
            marginals = []
            for left in range(0, start):
                sisters = torch.exp(loglikes[left, start])
                parents = outside[left, stop]
                joint = trans * sisters[None, :, None] * parents[:, None, None]
                joints.append(joint)
                marginals.append(torch.sum(joint, axis=(0, 1)))
            for right in range(stop + 1, length + 1):
                sisters = torch.exp(loglikes[stop, right])
                parents = outside[start, right]
                joint = trans * sisters[None, None, :] * parents[:, None, None]
                joints.append(joint)
                marginals.append(torch.sum(joint, axis=(0, 2)))
            summarginals = sum(marginals)  # List[Tensor] -> Tensor
            outside[start, stop] = summarginals / torch.sum(summarginals)
    
    return outside


def compile_log_priors(logtrans, _, indices, loglikes):

    length = len(indices)
    outside = dict()
    nrules, _, _ = logtrans.shape
    onehot = [1.] + (nrules - 1)*[0.]
    outside[0, length] = torch.log(torch.tensor(onehot))
    
    logjoints = []
    for width in reversed(range(1, length)):
        for start in range(0, length - width + 1):
            stop = start + width
            logmarginals = []
            for left in range(0, start):
                sisters = loglikes[left, start][None, :, None]
                parents = outside[left, stop][:, None, None]
                logjoint = logtrans + sisters + parents
                logjoints.append(logjoint)
                logmarginals.append(torch.logsumexp(logjoint, axis=(0, 1)))
            for right in range(stop + 1, length + 1):
                sisters = loglikes[stop, right][None, None, :]
                parents = outside[start, right][:, None, None]
                logjoint = logtrans + sisters + parents
                logjoints.append(logjoint)
                logmarginals.append(torch.logsumexp(logjoint, axis=(0, 2)))
            logmarginals = torch.stack(logmarginals, axis=0)
            outside[start, stop] = torch.logsumexp(logmarginals, axis=0)
    logjoints = torch.stack(logjoints, axis=0)
    logjoint = torch.logsumexp(logjoints, axis=0)
    assert logjoint.shape == (nrules, nrules, nrules)

    return outside, logjoint


def compile_log_joint(logtrans, logemits, indices, loglikes):
    """ Compute the probs of nonterminals over spans given context. """

    length = len(indices)
    outside = dict()
    nrules, _, _ = logtrans.shape
    onehot = [1.] + (nrules - 1)*[0.]
    outside[0, length] = torch.log(torch.tensor(onehot))

    for width in reversed(range(1, length)):
        for start in range(0, length - width + 1):
            stop = start + width
            logmarginals = []
            for left in range(0, start):
                logpriors = outside[left, stop][:, None, None]
                sisters = loglikes[left, start][None, :, None]
                logcond = sisters #- torch.logsumexp(sisters, dim=1, keepdim=True)
                logjoint = logtrans + logpriors + logcond
                logmarginals.append(torch.logsumexp(logjoint, axis=(0, 1)))
            for right in range(stop + 1, length + 1):
                logpriors = outside[start, right][:, None, None]
                sisters = loglikes[stop, right][None, None, :]
                logcond = sisters #- torch.logsumexp(sisters, dim=1, keepdim=True)
                logjoint = logtrans + logpriors + logcond
                logmarginals.append(torch.logsumexp(logjoint, axis=(0, 2)))
            logmarginals = torch.stack(logmarginals, axis=0)
            outside[start, stop] = torch.logsumexp(logmarginals, axis=0)

    return outside



if __name__ == "__main__":

    from pprint import pprint

    # R = 2
    # A = 3
    
    # logt = np.random.normal(size=(R, R, R))
    # logt -= np.max(logt, axis=(1, 2), keepdims=True)
    # logt -= np.log(np.sum(np.exp(logt), axis=(1, 2), keepdims=True))
    # logt = torch.tensor(logt)

    # loge = np.random.normal(size=(R, A))
    # loge -= np.max(loge, axis=1, keepdims=True)
    # loge -= np.log(np.sum(np.exp(loge), axis=1, keepdims=True))
    # loge = torch.tensor(loge)

    # idx = [np.random.choice(range(A)) for _ in range(7)]
    # inside = compile_log_likes(logt, loge, idx)

    # widening = sorted(inside.keys(), key=lambda st: st[1] - st[0])
    # for key in widening:
    #     print(key, inside[key])
    # print()

    # outside = compile_log_joint(logt, loge, idx, inside)

    # narrowing = sorted(inside.keys(), key=lambda st: st[0] - st[1])
    # for key in narrowing:
    #     expvalue = torch.exp(outside[key])
    #     print(key, expvalue, torch.sum(expvalue))
    # print()

    slength = 25

    nrules = 6
    nchars = 2

    trans = np.random.normal(size=(nrules, nrules, nrules)) ** 2
    trans /= np.sum(trans, axis=(1, 2), keepdims=True)

    for dist in trans:
        assert np.isclose(dist.sum(), 1.0)

    emits = np.random.normal(size=(nrules, nchars)) ** 2
    emits /= np.sum(emits, axis=1, keepdims=True)

    for dist in emits:
        assert np.isclose(dist.sum(), 1.0)

    eprobs = np.random.beta(1, 1, size=nrules)

    emits *= eprobs[:, None]
    trans *= eprobs[:, None, None]

    idx = [np.random.choice(range(nchars)) for _ in range(slength)]
    
    # inside

    inside = {(start, start + 1): emits[:, i]
              for start, i in enumerate(idx)}
    
    # NOTE that the likelihoods in inside can sum to more than 1
    # if several nonterminals would generate a certain sequence
    # with high probability; for instance, if both nonterminals
    # N0 and N1 produce the sequence ABCD deterministinally, then
    # the likelihood vector of ABCD under both is [1., 1., ...].
    #
    # However, they are always between 0 and 1 because we are
    # dealing with probability mass functions.

    for width in range(2, len(idx) + 1):
        for start in range(len(idx) - width + 1):
            stop = start + width
            joints = []
            for mid in range(start + 1, stop):
                assert 0 <= start < mid < stop <= len(idx)
                leftlike = inside[start, mid][:, None]
                rightlike = inside[mid, stop][None, :]
                joints.append(trans * leftlike * rightlike)
            joints = np.stack(joints)  # [split, parent, left, right]
            for k, jnt in enumerate(joints):
                assert jnt.shape == (nrules, nrules, nrules)
                assert jnt.sum() <= 1.0, jnt
            assert (start, stop) not in inside
            inside[start, stop] = np.sum(joints, axis=(0, 2, 3))
            assert np.all(inside[start, stop] >= 0.0)
            assert np.all(inside[start, stop] <= 1.0)
    
    for key, value in inside.items():
        print(key, value)
    print()

    # outside

    rootdist = np.array([1.] + (nrules - 1)*[0.])
    outside = {(0, len(idx)): rootdist}

    for width in reversed(range(1, len(idx))):  # [T - 1, ..., 1]
        for start in range(len(idx) - width + 1):
            stop = start + width
            assert (start, stop) not in outside
            outside[start, stop] = np.zeros(nrules)
            lefts = []
            for leftex in range(start):
                assert 0 <= leftex < start < stop <= len(idx)
                parent = outside[leftex, stop][:, None, None]
                sister = inside[leftex, start][None, :, None]
                assert sister.size == nrules
                assert np.all(sister <= 1.0)
                # cond = sister #/ np.sum(sister)
                lefts.append(trans * parent * sister)
            if len(lefts) > 0:
                lefts = np.stack(lefts, axis=0)
                outside[start, stop] += np.sum(lefts, axis=(0, 1, 2))
            rights = []
            for rightex in range(stop + 1, len(idx) + 1):
                assert 0 <= start < stop < rightex <= len(idx)
                parent = outside[start, rightex][:, None, None]
                sister = inside[stop, rightex][None, None, :]
                assert sister.size == nrules
                assert np.all(sister <= 1.0)
                # cond = sister #/ np.sum(sister)
                rights.append(trans * parent * sister)
            if len(rights) > 0:
                rights = np.stack(rights, axis=0)
                outside[start, stop] += np.sum(rights, axis=(0, 1, 3))
            # assert np.all(outside[start, stop] >= 0.0)
            # assert np.all(outside[start, stop] <= 1.0), outside[start, stop]

    for (start, stop), dist in outside.items():
        print((start, stop), dist)
    print()
