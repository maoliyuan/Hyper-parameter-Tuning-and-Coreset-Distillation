print(__doc__)
import matplotlib
#matplotlib.use('TkAgg')

import heapq
import numpy as np
import pandas as pd
import scipy as sp
from sklearn import metrics
import math
from scipy import spatial
import matplotlib.pyplot as plt
import time


class FacilityLocation:

    def __init__(self, D, V, alpha=1.):
        '''
        Args
        - D: np.array, shape [N, N], similarity matrix
        - V: list of int, indices of columns of D
        - alpha: float
        '''
        self.D = D
        self.curVal = 0
        self.curMax = np.zeros(len(D))
        self.gains = []
        self.alpha = alpha
        self.f_norm = self.alpha / self.f_norm(V)
        self.norm = 1. / self.inc(V, [])

    def f_norm(self, sset):
        return self.D[:, sset].max(axis=1).sum()

    def inc(self, sset, ndx):
        if len(sset + [ndx]) > 1:
            if not ndx:  # normalization
                return math.log(1 + self.alpha * 1)
            return self.norm * math.log(1 + self.f_norm * np.maximum(self.curMax, self.D[:, ndx]).sum()) - self.curVal
        else:
            return self.norm * math.log(1 + self.f_norm * self.D[:, ndx].sum()) - self.curVal

    def add(self, sset, ndx):
        cur_old = self.curVal
        if len(sset + [ndx]) > 1:
            self.curMax = np.maximum(self.curMax, self.D[:, ndx])
        else:
            self.curMax = self.D[:, ndx]
        self.curVal = self.norm * math.log(1 + self.f_norm * self.curMax.sum())
        self.gains.extend([self.curVal - cur_old])
        return self.curVal


def _heappush_max(heap, item):
    heap.append(item)
    heapq._siftdown_max(heap, 0, len(heap)-1)


def _heappop_max(heap):
    """Maxheap version of a heappop."""
    lastelt = heap.pop()  # raises appropriate IndexError if heap is empty
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        heapq._siftup_max(heap, 0)
        return returnitem
    return lastelt


def lazy_greedy_heap(F, V, B):
    curVal = 0
    sset = []
    vals = []

    order = []
    heapq._heapify_max(order)
    [_heappush_max(order, (F.inc(sset, index), index)) for index in V]

    while order and len(sset) < B:
        el = _heappop_max(order)
        improv = F.inc(sset, el[1])

        if improv >= 0:
            if not order:
                curVal = F.add(sset, el[1])
                sset.append(el[1])
                vals.append(curVal)
            else:
                top = _heappop_max(order)
                if improv >= top[0]:
                    curVal = F.add(sset, el[1])
                    sset.append(el[1])
                    vals.append(curVal)
                else:
                    _heappush_max(order, (improv, el[1]))
                _heappush_max(order, top)

    return sset, vals


def similarity(X, metric, rewards, reward_ratio=50):
    assert(X.shape[0] == rewards.shape[0])
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    rewards[rewards < 0] = sigmoid(rewards[rewards < 0])
    rewards[rewards >= 0] = rewards[rewards >= 0] + 0.5
    rewards = rewards.repeat(X.shape[0], axis=1)
    dists = metrics.pairwise_distances(X, metric=metric, n_jobs=1)
    if metric == 'cosine':
        S = 1 - dists
    elif metric == 'euclidean' or metric == 'l1':
        m = np.max(dists)
        S = m - dists
    else:
        raise ValueError(f'unknown metric: {metric}')

    S = S + rewards * reward_ratio
    S = S + rewards.T * reward_ratio
    
    return S


def get_facility_location_order(S, B, weights=None):
    '''
    Returns
    - order: np.array, shape [B], order of points selected by facility location
    - sz: np.array, shape [B], type int64, size of cluster associated with each selected point
    '''
    N = S.shape[0]
    V = list(range(N))
    F = FacilityLocation(S, V)
    order, _ = lazy_greedy_heap(F, V, B)
    order = np.asarray(order, dtype=np.int64)
    sz = np.zeros(B, dtype=np.float64)
    for i in range(N):  
        if weights is None:
            sz[np.argmax(S[i, order])] += 1
        else:
            sz[np.argmax(S[i, order])] += weights[i]
    return order, sz


def coreset_order(X, metric, B, rewards, weights=None):
    S = similarity(X, metric=metric, rewards=rewards)
    order, cluster_sz = get_facility_location_order(S, B, weights)
    return order, cluster_sz


def test_coreset_order():
    start = time.time()
    seed = 321
    np.random.seed(seed)
    n = 5000
    d = 20
    b = 10
    print(f"settings: n = {n}, d = {d}, b = {b}")
    X = np.random.rand(n, d)
    Y = np.random.rand(n, 3, 4)
    # print(f'first 5 Y: {Y[:5]}')
    order, cluster_sz = coreset_order(X, 'euclidean', b)
    t = time.time() - start
    print(f"time consumed: {t}")
    print(order)
    print(cluster_sz)
    select_Y = Y[order]
    Y[:b] = select_Y
    # print(f"after select, first 5 Y: {Y[:5]}")


if __name__ == '__main__':
    test_coreset_order()
