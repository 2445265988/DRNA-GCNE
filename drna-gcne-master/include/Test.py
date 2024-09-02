import numpy as np
import scipy


def get_hits(vec, test_pair, top_k=(1, 5, 10, 50)):
    Lvec = np.array([vec[e1] for e1, e2 in test_pair])
    Rvec = np.array([vec[e2] for e1, e2 in test_pair])
    sim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='cityblock')
    top_lr = [0] * len(top_k)
    mr_lr = []
    mr_rl = []
    for i in range(Lvec.shape[0]):
        rank = sim[i, :].argsort()
        rank_index = np.where(rank == i)[0][0]
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
        mr_lr.append((rank_index + 1) if rank_index < Lvec.shape[0] else 0)
    top_rl = [0] * len(top_k)
    for i in range(Rvec.shape[0]):
        rank = sim[:, i].argsort()
        rank_index = np.where(rank == i)[0][0]
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_rl[j] += 1
        mr_rl.append((rank_index + 1) if rank_index < Rvec.shape[0] else 0)
    print('For each left:')
    for i in range(len(top_lr)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
    print('For each right:')
    for i in range(len(top_rl)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(test_pair) * 100))

    # 计算 MR
    mr_lr_avg = sum(mr_lr) / len(mr_lr)
    mr_rl_avg = sum(mr_rl) / len(mr_rl)

    # 计算 MMR
    mmr_lr = sum(1/mmr_l for mmr_l in mr_lr) / len(mr_lr)
    mmr_rl = sum(1/mmr_r for mmr_r in mr_rl) / len(mr_rl)
    print('Mean Reciprocal Rank (Left): %.4f' % mr_lr_avg)
    print('Mean Maximum Reciprocal Rank (Left): %.4f' % mmr_lr)
    print('Mean Reciprocal Rank (Right): %.4f' % mr_rl_avg)
    print('Mean Maximum Reciprocal Rank (Right): %.4f' % mmr_rl)

