import gc
import multiprocessing
import time

import numpy as np
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist


def greedy_alignment(embed1, embed2, top_k, nums_threads, metric, normalize, csls_k, accurate):
    """
    Search align with greedy strategy.

    Parameters
    ----------
    embed1 : matrix_like
        An embedding matrix of size n1*d, where n1 is the number of embeddings and d is the dimension.
    embed2 : matrix_like
        An embedding matrix of size n2*d, where n2 is the number of embeddings and d is the dimension.
    top_k : list of integers
        Hits@k metrics for evaluating results.
    nums_threads : int
        The number of threads used to search align.用于搜索对齐的线程数
    metric : string
        The distance metric to use. It can be 'cosine', 'euclidean' or 'inner'.距离度量
    normalize : bool, true or false.
        Whether to normalize the input embeddings.表示是否对输入嵌入进行归一化
    csls_k : int
        K value for csls. If k > 0, enhance the similarity by csls.增强相似性的 K 值

    Returns
    -------
    alignment_rest :  list, pairs of aligned entities
    hits1 : float, hits@1 values for align results
    mr : float, MR values for align results
    mrr : float, MRR values for align results
    """
    t = time.time()
    # 计算相似性矩阵
    sim_mat = sim(embed1, embed2, metric=metric, normalize=normalize, csls_k=csls_k)
    num = sim_mat.shape[0]
    # 决定是否只用多线程
    if nums_threads > 1:
        # 如果启用多线程
        hits = [0] * len(top_k)
        mr, mrr = 0, 0
        alignment_rest = set()
        rests = list()
        # 将相似性矩阵划分为多个子任务
        search_tasks = task_divide(np.array(range(num)), nums_threads)
        # 创建线程池，以便并行计算
        pool = multiprocessing.Pool(processes=len(search_tasks))
        for task in search_tasks:
            mat = sim_mat[task, :]
            rests.append(pool.apply_async(calculate_rank, (task, mat, top_k, accurate, num)))
        pool.close()
        pool.join()
        for rest in rests:
            sub_mr, sub_mrr, sub_hits, sub_hits1_rest = rest.get()
            mr += sub_mr
            mrr += sub_mrr
            hits += np.array(sub_hits)
            alignment_rest |= sub_hits1_rest
    else:
        mr, mrr, hits, alignment_rest = calculate_rank(list(range(num)), sim_mat, top_k, accurate, num)
    assert len(alignment_rest) == num
    hits = np.array(hits) / num * 100
    for i in range(len(hits)):
        hits[i] = round(hits[i], 3)
    cost = time.time() - t
    if accurate:
        if csls_k > 0:
            print("accurate results with csls, hits@{} = {}%, mr = {:.3f}, mrr = {:.6f}, time = {:.3f} s ".
                  format(top_k, hits, mr, mrr, cost))
        else:
            print("accurate results: hits@{} = {}%, mr = {:.3f}, mrr = {:.6f}, time = {:.3f} s ".
                  format(top_k, hits, mr, mrr, cost))
    else:
        if csls_k > 0:
            print("quick results with csls, hits@{} = {}%, time = {:.3f} s ".format(top_k, hits, cost))
        else:
            print("quick results: hits@{} = {}%, time = {:.3f} s ".format(top_k, hits, cost))
    hits1 = hits[0]
    del sim_mat
    gc.collect()
    return alignment_rest, hits1, mr, mrr


def arg_sort(idx, sim_mat, prefix1, prefix2):
    candidates = dict()
    for i in range(len(idx)):
        x_i = prefix1 + str(idx[i])
        rank = (-sim_mat[i, :]).argsort()
        y_j = [prefix2 + str(r) for r in rank]
        candidates[x_i] = y_j
    return candidates


def calculate_rank(idx, sim_mat, top_k, accurate, total_num):
    # idx: 一个包含索引的列表，代表 gold 实体的索引。
    # sim_mat: 一个相似性矩阵，包含了两组嵌入之间的相似度。
    # top_k: 一个整数列表，表示计算 Hits@k 指标。
    # accurate: 一个布尔值，表示是否进行精确计算。
    # total_num: 整体实体数量。
    assert 1 in top_k
    mr = 0
    mrr = 0
    hits = [0] * len(top_k)
    hits1_rest = set()
    for i in range(len(idx)):
        gold = idx[i]
        if accurate:
            # sim_mat[i, :]：取相似度矩阵的第 i 行，表示实体 i 与其他所有实体的相似度
            # 得到降序排列后的索引
            rank = (-sim_mat[i, :]).argsort()
        else:
            # 获取前k个最大的元素的索引
            rank = np.argpartition(-sim_mat[i, :], np.array(top_k) - 1)
        # 存储命中第一个的实体对
        hits1_rest.add((gold, rank[0]))
        # 确保“gold”（即正确的实体索引）在“rank”中
        assert gold in rank
        print("gold+rank[0]:")
        print((gold, rank[0]))
        # 获取正确实体在排名中的位置索引
        rank_index = np.where(rank == gold)[0][0]
        print("rank_index:")
        print(rank_index)
        mr += (rank_index + 1)
        mrr += 1 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                hits[j] += 1
    mr /= total_num
    mrr /= total_num
    return mr, mrr, hits, hits1_rest

# 计算相似性矩阵
def sim(embed1, embed2, metric='inner', normalize=False, csls_k=0):
    """
    Compute pairwise similarity between the two collections of embeddings.

    Parameters
    ----------
    embed1 : matrix_like
        An embedding matrix of size n1*d, where n1 is the number of embeddings and d is the dimension.
    embed2 : matrix_like
        An embedding matrix of size n2*d, where n2 is the number of embeddings and d is the dimension.
    metric : str, optional, inner default.
        The distance metric to use. It can be 'cosine', 'euclidean', 'inner'.
    normalize : bool, optional, default false.
        Whether to normalize the input embeddings.
    csls_k : int, optional, 0 by default.
        K value for csls. If k > 0, enhance the similarity by csls.

    Returns
    -------
    sim_mat : An similarity matrix of size n1*n2.
    """
    if normalize:
        embed1 = preprocessing.normalize(embed1)
        embed2 = preprocessing.normalize(embed2)
    if metric == 'inner':
        # 矩阵乘法计算相似性
        sim_mat = np.matmul(embed1, embed2.T)  # numpy.ndarray, float32
    elif metric == 'cosine' and normalize:
        # 矩阵乘法计算相似性
        sim_mat = np.matmul(embed1, embed2.T)  # numpy.ndarray, float32
    elif metric == 'euclidean':
        # 欧氏距离计算相似性
        sim_mat = 1 - euclidean_distances(embed1, embed2)
        print(type(sim_mat), sim_mat.dtype)
        sim_mat = sim_mat.astype(np.float32)
    elif metric == 'cosine':
        # 余弦距离计算相似性
        sim_mat = 1 - cdist(embed1, embed2, metric='cosine')  # numpy.ndarray, float64
        sim_mat = sim_mat.astype(np.float32)
    elif metric == 'manhattan':
        # 曼哈顿距离计算相似性
        sim_mat = 1 - cdist(embed1, embed2, metric='cityblock')
        sim_mat = sim_mat.astype(np.float32)
    else:
        # 制定的相似性度量
        sim_mat = 1 - cdist(embed1, embed2, metric=metric)
        sim_mat = sim_mat.astype(np.float32)
    if csls_k > 0:
        #  CSLS 增强相似性
        sim_mat = csls_sim(sim_mat, csls_k)
    return sim_mat


def csls_sim(sim_mat, k):
    """
    Compute pairwise csls similarity based on the input similarity matrix.

    Parameters
    ----------
    sim_mat : matrix-like
        A pairwise similarity matrix.
    k : int
        The number of nearest neighbors.

    Returns
    -------
    csls_sim_mat : A csls similarity matrix of n1*n2.
    """

    nearest_values1 = calculate_nearest_k(sim_mat, k)
    nearest_values2 = calculate_nearest_k(sim_mat.T, k)
    csls_sim_mat = 2 * sim_mat - nearest_values1 - nearest_values2.T
    return csls_sim_mat


def calculate_nearest_k(sim_mat, k):
    sorted_mat = -np.partition(-sim_mat, k + 1, axis=1)  # -np.sort(-sim_mat1)
    nearest_k = sorted_mat[:, 0:k]
    return np.mean(nearest_k, axis=1, keepdims=True)


def csls_sim_multi_threads(sim_mat, k, nums_threads):
    tasks = task_divide(np.array(range(sim_mat.shape[0])), nums_threads)
    pool = multiprocessing.Pool(processes=len(tasks))
    rests = list()
    for task in tasks:
        rests.append(pool.apply_async(calculate_nearest_k, (sim_mat[task, :], k)))
    pool.close()
    pool.join()
    sim_values = None
    for res in rests:
        val = res.get()
        if sim_values is None:
            sim_values = val
        else:
            sim_values = np.append(sim_values, val)
    assert sim_values.shape[0] == sim_mat.shape[0]
    return sim_values


def task_divide(idx, n):
    total = len(idx)
    if n <= 0 or 0 == total:
        return [idx]
    if n > total:
        return [idx]
    elif n == total:
        return [[i] for i in idx]
    else:
        j = total // n
        tasks = []
        for i in range(0, (n - 1) * j, j):
            tasks.append(idx[i:i + j])
        tasks.append(idx[(n - 1) * j:])
        return tasks