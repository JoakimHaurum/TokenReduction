import numpy as np

def get_model_pair(model, loc, ratio):
    return model+"-"+"_".join(str(x) for x in loc)+"-"+"_" + str(ratio)

def map_cluster_centers(cluster1_stage, cluster2_stage, cluster1 = None, cluster2 = None, first_stage=True):
    if first_stage:
        cluster1 = np.asarray(cluster1_stage)
        cluster2 = np.asarray(cluster2_stage)
    elif len(cluster1_stage) == len(cluster2_stage):
        cluster1_tmp = cluster1.copy()
        cluster2_tmp = cluster2.copy()
        for cluster_idx in range(len(cluster1_stage)):
            cluster1_tmp[cluster1 == cluster_idx] = cluster1_stage[cluster_idx]
            cluster2_tmp[cluster2 == cluster_idx] = cluster2_stage[cluster_idx]
        cluster1 = cluster1_tmp
        cluster2 = cluster2_tmp
    else:
        cluster1_tmp = cluster1.copy()
        cluster2_tmp = cluster2.copy()
        for cluster_idx in range(len(cluster1_stage)):
            cluster1_tmp[cluster1 == cluster_idx] = cluster1_stage[cluster_idx]
        for cluster_idx in range(len(cluster2_stage)):
            cluster2_tmp[cluster2 == cluster_idx] = cluster2_stage[cluster_idx]
        cluster1 = cluster1_tmp
        cluster2 = cluster2_tmp

    return cluster1, cluster2