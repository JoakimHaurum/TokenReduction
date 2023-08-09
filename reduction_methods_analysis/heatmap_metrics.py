import numpy as np
from scipy.stats import spearmanr
from pyemd import emd_with_flow

def KL(map1, map2):
    assert map1.shape == map2.shape

    map1 = map1 / np.sum(map1)
    map2 = map2 / np.sum(map2)
    return np.sum(map2 * np.log2(map2/map1))

def JS(map1, map2):
    assert map1.shape == map2.shape

    map1 = map1 / np.sum(map1)
    map2 = map2 / np.sum(map2)

    map1 = map1.reshape(-1)
    map2 = map2.reshape(-1)

    map_avg = 0.5 * (map1+map2)
    return 0.5 * KL(map1, map_avg) + 0.5 * KL(map2, map_avg)

def PCC(map1, map2):
    assert map1.shape == map2.shape
    
    map1 = (map1 - np.mean(map1)) / np.std(map1, ddof=1)
    map2 = (map2 - np.mean(map2)) / np.std(map2, ddof=1)

    # MATLAB corr2 reimplementation
    map1 = map1-np.mean(map1)
    map2 = map2-np.mean(map2)
    score = np.sum(map1*map2) / np.sqrt(np.sum(map1*map1) * np.sum(map2*map2))

    return score

def SIM(map1, map2):
    assert map1.shape == map2.shape

    map1 = (map1-np.min(map1))/(np.max(map1)-np.min(map1))
    map1 = map1/np.sum(map1)

    map2 = (map2-np.min(map2))/(np.max(map2)-np.min(map2))
    map2 = map2/np.sum(map2)

    diff = np.minimum(map1, map2)

    return np.sum(diff)

def EMD(map1, map2, dist=None):
    assert map1.shape == map2.shape

    R, C = map1.shape

    if dist is None:
        dist = create_emd_dist(R,C)
    else:
        assert dist.shape[0] == R*C
        assert dist.shape[1] == R*C

    extra_mass = 0.
    map1 = map1.reshape(-1)
    map2 = map2.reshape(-1)

    map1 = map1/np.sum(map1)
    map2 = map2/np.sum(map2)

    score, flow = emd_with_flow(map1, map2, dist, extra_mass_penalty=extra_mass)

    return score

def SCC(map1, map2):
    assert map1.shape == map2.shape
    
    map1 = map1 / np.sum(map1)
    map2 = map2 / np.sum(map2)
    score = spearmanr(map1.reshape(-1), map2.reshape(-1)).correlation
    return score

def create_emd_dist(R,C):
    dist = np.zeros((R*C, R*C))
    j = 0
    for c1 in range(1,C+1):
        for r1 in range(1, R+1):
            i = 0
            for c2 in range(1, C+1):
                for r2 in range(1,R+1):
                    dist[i,j] = np.sqrt((r1-r2)**2+(c1-c2)**2)
                    i += 1
            j += 1
    return dist