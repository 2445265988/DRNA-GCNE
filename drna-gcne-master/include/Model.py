import math

import psutil
import wandb
import matplotlib.pyplot as plt
from .Init import *
from .Test import get_hits
import scipy
import json
from .util import no_weighted_adj
import scipy.spatial as spatial
import os
from .preprocess import generate_2steps_path
import tensorflow as tf
# import keras.backend as K
# import wandb



def rfunc(KG, e):
    head = {}
    tail = {}
    cnt = {}
    for tri in KG:
        if tri[1] not in cnt:
            cnt[tri[1]] = 1
            head[tri[1]] = set([tri[0]])
            tail[tri[1]] = set([tri[2]])
        else:
            cnt[tri[1]] += 1
            head[tri[1]].add(tri[0])
            tail[tri[1]].add(tri[2])
    r_num = len(head)
    head_r = np.zeros((e, r_num))
    tail_r = np.zeros((e, r_num))
    r_mat_ind = []
    r_mat_val = []
    for tri in KG:
        # print("this kg")
        # print(tri[0],tri[1],tri[2])
        head_r[tri[0]][tri[1]] = 1
        tail_r[tri[2]][tri[1]] = 1
        r_mat_ind.append([tri[0], tri[2]])
        r_mat_val.append(tri[1])
    r_mat = tf.SparseTensor(
        indices=r_mat_ind, values=r_mat_val, dense_shape=[e, e])

    return head, tail, head_r, tail_r, r_mat


def get_mat(e, KG):
    du = [{e_id} for e_id in range(e)]
    for tri in KG:
        if tri[0] != tri[2]:
            du[tri[0]].add(tri[2])
            du[tri[2]].add(tri[0])
    du = [len(d) for d in du]
    M = {}
    for tri in KG:
        if tri[0] == tri[2]:
            continue
        if (tri[0], tri[2]) not in M:
            M[(tri[0], tri[2])] = 1
        else:
            pass
        if (tri[2], tri[0]) not in M:
            M[(tri[2], tri[0])] = 1
        else:
            pass

    for i in range(e):
        M[(i, i)] = 1
    return M, du


# get a sparse tensor based on relational triples
def get_sparse_tensor(e, KG):
    print('getting a sparse tensor...')
    M, du = get_mat(e, KG)
    ind = []
    val = []
    M_arr = np.zeros((e, e))
    for fir, sec in M:
        ind.append((sec, fir))
        val.append(M[(fir, sec)] / math.sqrt(du[fir]) / math.sqrt(du[sec]))
        M_arr[fir][sec] = 1.0
    M = tf.SparseTensor(indices=ind, values=val, dense_shape=[e, e])

    return M, M_arr


# add a layer
# def add_diag_layer(inlayer, dimension, M, act_func, dropout=0.0, init=ones):
#     inlayer = tf.nn.dropout(inlayer, 1 - dropout)
#     dimension_in = inlayer.get_shape().as_list()[-1]
#     W = init([dimension_in, dimension])
#     print('adding a diag layer...')
#
#     node_features = tf.matmul(inlayer, W)
#     aggregated_features = tf.sparse_tensor_dense_matmul(M, node_features)
#     if act_func is None:
#         return aggregated_features
#     else:
#         return act_func(aggregated_features)

def add_diag_layer(inlayer, dimension, M, act_func, dropout=0.0, init=ones):
    inlayer = tf.nn.dropout(inlayer, 1 - dropout)
    print('adding a diag layer...')
    w0 = init([1, dimension])
    tosum = tf.sparse_tensor_dense_matmul(M, tf.multiply(inlayer, w0))
    if act_func is None:
        return tosum
    else:
        return act_func(tosum)


def add_full_layer(inlayer, dimension_in, dimension_out, M, act_func, dropout=0.0, init=glorot):
    inlayer = tf.nn.dropout(inlayer, 1 - dropout)
    print('adding a full layer...')
    w0 = init([dimension_in, dimension_out])
    tosum = tf.sparse_tensor_dense_matmul(M, tf.matmul(inlayer, w0))
    if act_func is None:
        return tosum
    else:
        return act_func(tosum)


def add_sparse_att_layer(inlayer, dual_layer, r_mat, act_func, e):
    dual_transform = tf.reshape(tf.layers.conv1d(
        tf.expand_dims(dual_layer, 0), 1, 1), (-1, 1))
    logits = tf.reshape(tf.nn.embedding_lookup(
        dual_transform, r_mat.values), [-1])
    print('adding sparse attention layer...')
    lrelu = tf.SparseTensor(indices=r_mat.indices,
                            values=tf.nn.leaky_relu(logits),
                            dense_shape=(r_mat.dense_shape))
    coefs = tf.sparse_softmax(lrelu)
    vals = tf.sparse_tensor_dense_matmul(coefs, inlayer)
    if act_func is None:
        return vals
    else:
        return act_func(vals)


def add_dual_att_layer(inlayer, inlayer2, adj_mat, act_func, hid_dim):
    in_fts = tf.layers.conv1d(tf.expand_dims(inlayer2, 0), hid_dim, 1)
    f_1 = tf.reshape(tf.layers.conv1d(in_fts, 1, 1), (-1, 1))
    f_2 = tf.reshape(tf.layers.conv1d(in_fts, 1, 1), (-1, 1))
    logits = f_1 + tf.transpose(f_2)
    print('adding dual attention layer...')
    adj_tensor = tf.constant(adj_mat, dtype=tf.float32)
    bias_mat = -1e9 * (1.0 - (adj_mat > 0))
    logits = tf.multiply(adj_tensor, logits)
    coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

    vals = tf.matmul(coefs, inlayer)
    if act_func is None:
        return vals
    else:
        return act_func(vals)


def add_self_att_layer(inlayer, adj_mat, act_func, hid_dim):
    in_fts = tf.layers.conv1d(tf.expand_dims(
        inlayer, 0), hid_dim, 1, use_bias=False)
    f_1 = tf.reshape(tf.layers.conv1d(in_fts, 1, 1), (-1, 1))
    f_2 = tf.reshape(tf.layers.conv1d(in_fts, 1, 1), (-1, 1))
    logits = f_1 + tf.transpose(f_2)
    print('adding self attention layer...')
    adj_tensor = tf.constant(adj_mat, dtype=tf.float32)
    logits = tf.multiply(adj_tensor, logits)
    bias_mat = -1e9 * (1.0 - (adj_mat > 0))
    coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

    vals = tf.matmul(coefs, inlayer)
    if act_func is None:
        return vals
    else:
        return act_func(vals)


def highway(layer1, layer2, dimension):
    kernel_gate = glorot([dimension,dimension])
    bias_gate = zeros([dimension])
    transform_gate = tf.matmul(layer1, kernel_gate) + bias_gate
    transform_gate = tf.nn.sigmoid(transform_gate)
    carry_gate = 1.0 - transform_gate
    return transform_gate*layer2 + carry_gate * layer1


def compute_r(inlayer, head_r, tail_r, dimension):
    head_l = tf.transpose(tf.constant(head_r, dtype=tf.float32))
    tail_l = tf.transpose(tf.constant(tail_r, dtype=tf.float32))
    L = tf.matmul(head_l, inlayer) / \
        tf.expand_dims(tf.reduce_sum(head_l, axis=-1), -1)
    R = tf.matmul(tail_l, inlayer) / \
        tf.expand_dims(tf.reduce_sum(tail_l, axis=-1), -1)
    r_embeddings = tf.concat([L, R], axis=-1)
    return r_embeddings


def get_dual_input(inlayer, head, tail, head_r, tail_r, dimension):
    dual_X = compute_r(inlayer, head_r, tail_r, dimension)
    print('computing the dual input...')
    count_r = len(head)
    dual_A = np.zeros((count_r, count_r))
    for i in range(count_r):
        for j in range(count_r):
            a_h = len(head[i] & head[j]) / len(head[i] | head[j])
            a_t = len(tail[i] & tail[j]) / len(tail[i] | tail[j])
            dual_A[i][j] = a_h + a_t
    return dual_X, dual_A


def get_input_layer(e, dimension, lang):
    print('adding the primal input layer...')
    with open(file='data/' + lang + '_en/' + lang + '_vectorList.json', mode='r', encoding='utf-8') as f:
        embedding_list = json.load(f)
        print(len(embedding_list), 'rows,', len(embedding_list[0]), 'columns.')
    input_embeddings = tf.convert_to_tensor(embedding_list)
    ent_embeddings = tf.Variable(input_embeddings)
    return tf.nn.l2_normalize(ent_embeddings, 1)

def contrastive_loss(y_true, y_pred, margin=1.0):
    square_pred = tf.reduce_sum(y_pred ** 2, axis=1, keepdims=True)
    dot_product = tf.matmul(y_true, y_pred, transpose_b=True)
    square_distance = tf.sqrt(square_pred + tf.transpose(square_pred) - 2 * dot_product)
    loss = y_true * square_distance + (1 - y_true) * tf.square(tf.maximum(margin - square_distance, 0))
    return tf.reduce_mean(loss)


"""对比损失"""
def get_loss_compare(outlayer, ILL, k):
    margin=1.0
    gamma=1.0
    print('getting loss...')
    # 提取实体ID
    left = ILL[:, 0]
    right = ILL[:, 1]
    t = len(ILL)
    left_x = tf.nn.embedding_lookup(outlayer, left)
    right_x = tf.nn.embedding_lookup(outlayer, right)
    # 对比损失
    positive_distance = tf.reduce_sum(tf.square(left_x - right_x), 1)

    # 正则化项，这里使用L2正则化
    positive_loss = tf.reduce_sum(tf.nn.relu(positive_distance))

    # 计算左负样本对之间的距离
    neg_left = tf.placeholder(tf.int32, [t * k], "neg_left")
    neg_right = tf.placeholder(tf.int32, [t * k], "neg_right")
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
    B = tf.reduce_sum(tf.square(neg_l_x - neg_r_x), 1)
    negative_loss_1 = tf.reduce_mean(tf.nn.relu(B - gamma))
    neg2_left = tf.placeholder(tf.int32, [t * k], "neg2_left")
    neg2_right = tf.placeholder(tf.int32, [t * k], "neg2_right")
    neg2_l_x = tf.nn.embedding_lookup(outlayer, neg2_left)
    neg2_r_x = tf.nn.embedding_lookup(outlayer, neg2_right)
    B2 = tf.reduce_sum(tf.square(neg2_l_x - neg2_r_x), 1)
    negative_loss_2 = tf.reduce_mean(tf.nn.relu(B2 - gamma))

    # 总损失是正样本损失和两组负样本损失的和
    loss = (positive_loss + negative_loss_1 + negative_loss_2) / (3 * k * t)
    return loss

def get_loss_dualaloss(outlayer, ILL, k, node_size):
    def squared_dist(x):
        A, B = x
        row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
        row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.
        row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
        row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.
        return row_norms_A + row_norms_B - 2 * tf.matmul(A, B, transpose_b=True)

    def reduce_std(tensor, axis=None, keepdims=False):
        """
        计算张量在指定维度上的元素的标准偏差。

        参数:
        - tensor: 要计算标准偏差的张量。
        - axis: 沿着哪个维度计算标准偏差，默认为None，表示计算所有元素的标准偏差。
        - keepdims: 是否保留维度。

        返回:
        - 每个指定维度的标准偏差张量。
        """
        # 计算均值
        mean = tf.reduce_mean(tensor, axis=axis, keepdims=keepdims)

        # 计算方差
        squared_diff = tf.square(tensor - mean)

        # 计算方差
        variance = tf.reduce_mean(squared_diff, axis=axis, keepdims=keepdims)

        # 计算标准偏差
        std_dev = tf.sqrt(variance)

        return std_dev

    gamma=1.0
    print('getting loss...')
    # 提取实体ID
    left = ILL[:, 0]
    right = ILL[:, 1]
    t = len(ILL)
    # 维度为【9000，300】
    left_x = tf.nn.embedding_lookup(outlayer, left)
    right_x = tf.nn.embedding_lookup(outlayer, right)

    # 计算绝对差之和，度量正样本对之间的距离
    # 度为【9000，1】
    positive_distance = tf.reduce_sum(tf.square(left_x - right_x), 1,keepdims=True)
    # with tf.Session() as sess:
    #     print("实际维度 of positive_distance:", sess.run(tf.shape(positive_distance)))
    # 计算左侧的负样本对之间的距离
    neg_left = tf.placeholder(tf.int32, [t * k], "neg_left")
    neg_right = tf.placeholder(tf.int32, [t * k], "neg_right")
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
    with tf.Session() as sess:
        print("实际维度 of neg_l_x:", sess.run(tf.shape(neg_l_x)))
    with tf.Session() as sess:
        print("实际维度 of neg_r_x:", sess.run(tf.shape(neg_r_x)))
    # negative_distance_1 = squared_dist([neg_l_x,neg_r_x])
    negative_distance_1 = tf.reduce_sum(tf.square(neg_l_x - neg_r_x), 1, keepdims=True)
    # 归一化副样本1损失
    neg1_dist_mean = tf.reduce_mean(negative_distance_1)
    neg1_dist_std = reduce_std(negative_distance_1)
    normalized_neg1_dist = (negative_distance_1-neg1_dist_mean)/(neg1_dist_std+1e-8)
    left_loss=positive_distance-normalized_neg1_dist+gamma
    # negative_distance_1 = tf.reduce_sum(negative_distance_1,1,keepdims=True)
    # with tf.Session() as sess:
    #     print("实际维度 of negative_distance_1:", sess.run(tf.shape(negative_distance_1)))
    # negative_distance_1 = tf.reshape(negative_distance_1, [t, k])
    neg2_left = tf.placeholder(tf.int32, [t * k], "neg2_left")
    neg2_right = tf.placeholder(tf.int32, [t * k], "neg2_right")
    neg2_l_x = tf.nn.embedding_lookup(outlayer, neg2_left)
    neg2_r_x = tf.nn.embedding_lookup(outlayer, neg2_right)
    negative_distance_2 = tf.reduce_sum(tf.square(neg2_l_x - neg2_r_x), 1, keepdims=True)

    # 归一化副样本1损失
    neg2_dist_mean = tf.reduce_mean(negative_distance_2)
    neg2_dist_std = reduce_std(negative_distance_2)
    normalized_neg2_dist = (negative_distance_2 - neg2_dist_mean) / (neg2_dist_std + 1e-8)
    right_loss = positive_distance - normalized_neg2_dist + gamma

    # negative_distance_2 = squared_dist([neg2_l_x,neg2_r_x])
    negative_distance_2 = tf.reduce_sum(tf.square(neg2_l_x - neg2_r_x), 1, keepdims=True)
    # negative_distance_2 = tf.reshape(negative_distance_2, [t, k])
    # 边界损失函数，使用relu激活函数和边界gamma
    # left_loss=positive_distance-negative_distance_1+gamma
    # right_loss=positive_distance-negative_distance_2+gamma
    # # 创建独热编码矩阵
    # l_one_hot = tf.one_hot(indices=left, depth=k)
    # r_one_hot = tf.one_hot(indices=right, depth=k)
    # left_loss = left_loss *(1 - l_one_hot- r_one_hot)
    # #
    # right_loss = right_loss * (1 - l_one_hot- r_one_hot)
    # right_loss = (right_loss - tf.stop_gradient(tf.reduce_mean(right_loss, axis=-1, keepdims=True))) / tf.stop_gradient(
    #     right_loss)
    # left_loss = (left_loss - tf.stop_gradient(tf.reduce_mean(left_loss, axis=-1, keepdims=True))) / tf.stop_gradient(
    #     left_loss)

    lamb, tau = 30, 10
    neg1_loss = tf.reduce_logsumexp(lamb * left_loss + tau, axis=-1)
    neg2_loss = tf.reduce_logsumexp(lamb * right_loss + tau, axis=-1)
    # pos_loss = tf.reduce_logsumexp(lamb * positive_distance + tau, axis=-1)
    # positive_loss=K.asd

    # 总损失是正样本损失和两组负样本损失的和
    # loss = (l_loss + r_loss) / (2 * k * t)
    return tf.reduce_mean(neg1_loss + neg2_loss )

"""温度缩放"""
def get_loss_wen(outlayer, ILL, gamma, k, temperature=1):
    print('getting loss...')
    # 提取实体ID
    left_ids = ILL[:, 0]
    right_ids = ILL[:, 1]
    t = len(ILL)

    # 获取正样本嵌入向量
    left_x = tf.nn.embedding_lookup(outlayer, left_ids)
    right_x = tf.nn.embedding_lookup(outlayer, right_ids)

    # 计算正样本对之间的平方距离
    pos_dist = tf.reduce_sum(tf.abs(left_x - right_x),1)

    # 计算损失的辅助函数
    def calc_loss_with_neg_samples(pos_dist, neg_left, neg_right, temperature):
        # 获取负样本嵌入向量
        neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
        neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)

        # 计算负样本对之间的平方距离
        neg_dist = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), axis=1)
        neg_dist = -tf.reshape(neg_dist, [t, k])

        # 温度缩放和平移，防止数值过小
        lamb, tau = 30, 10
        scaled_pos_dist = lamb*pos_dist+tau
        scaled_neg_dist = lamb*neg_dist+tau
        # 使用logsumexp技巧来稳定损失计算
        loss = tf.reduce_logsumexp(tf.nn.leaky_relu(tf.add(scaled_neg_dist, tf.reshape(scaled_pos_dist, [t, 1]))))
        return loss
    # 计算第一种负样本的损失
    neg_left1 = tf.placeholder(tf.int32, [t * k], "neg_left")
    neg_right1 = tf.placeholder(tf.int32, [t * k], "neg_right")
    loss1 = calc_loss_with_neg_samples(pos_dist, neg_left1, neg_right1, temperature)
    # 计算第二种负样本的损失
    neg_left2 = tf.placeholder(tf.int32, [t * k], "neg2_left")
    neg_right2 = tf.placeholder(tf.int32, [t * k], "neg2_right")
    loss2 = calc_loss_with_neg_samples(pos_dist, neg_left2, neg_right2, temperature)
    # 合并两种负样本的损失
    total_loss = (loss1 +loss2) / (2.0 * k * t)
    # # 平均损失
    # final_loss = total_loss / (k * t) + gamma
    return total_loss

"""平方损失"""
def get_loss_square(outlayer, ILL, gamma, k):
    print('getting loss...')
    # 提取实体ID
    left = ILL[:, 0]
    right = ILL[:, 1]
    t = len(ILL)
    # 对应的嵌入向量
    left_x = tf.nn.embedding_lookup(outlayer, left)
    right_x = tf.nn.embedding_lookup(outlayer, right)
    # 计算绝对差之和，度量正样本对之间的距离
    A = tf.reduce_sum(tf.square(left_x - right_x), 1)
    # A = tf.nn.softplus(-A))
    # 计算左负样本对之间的距离
    neg_left = tf.placeholder(tf.int32, [t * k], "neg_left")
    neg_right = tf.placeholder(tf.int32, [t * k], "neg_right")
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
    neg_distance = neg_l_x - neg_r_x
    B = tf.reduce_sum(tf.square(neg_l_x - neg_r_x), 1)
    # C 通过 B 计算负样本对的距离并重塑为 [t, k] 形状
    C = - tf.reshape(B, [t, k])
    # D 是一个常数项，等于 A 加上 gamma，重塑为 [t, 1] 形状。
    D = A + gamma
    # L1 计算正样本对的损失，L2 计算负样本对的损失
    L1 = tf.nn.leaky_relu(tf.add(C, tf.reshape(D, [t, 1])))
    neg_left = tf.placeholder(tf.int32, [t * k], "neg2_left")
    neg_right = tf.placeholder(tf.int32, [t * k], "neg2_right")
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
    B = tf.reduce_sum(tf.square(neg_l_x - neg_r_x), 1)
    C = - tf.reshape(B, [t, k])
    L2 = tf.nn.leaky_relu(tf.add(C, tf.reshape(D, [t, 1])))

    return (tf.reduce_sum(L1) + tf.reduce_sum(L2)) / (2.0 * k * t)


# def get_loss(outlayer, ILL, gamma, k):
#     print('getting loss...')
#     # 提取实体ID
#     left = ILL[:, 0]
#     right = ILL[:, 1]
#     t = len(ILL)
#     # 对应的嵌入向量
#     left_x = tf.nn.embedding_lookup(outlayer, left)
#     right_x = tf.nn.embedding_lookup(outlayer, right)
#     # 计算绝对差之和，度量正样本对之间的距离
#     A = tf.reduce_sum(tf.abs(left_x - right_x), 1)
#     # 计算左负样本对之间的距离
#     neg_left = tf.placeholder(tf.int32, [t * k], "neg_left")
#     neg_right = tf.placeholder(tf.int32, [t * k], "neg_right")
#     neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
#     neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
#     with tf.Session() as sess:
#         print("实际维度 of negative_distance_1:", sess.run(tf.shape(neg_r_x)))
#
#     shape_of_neg_r_x = tf.shape(neg_r_x)
#     with tf.Session() as sess:
#         print("实际维度 of neg_r_x:", sess.run(shape_of_neg_r_x))
#     B = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
#     # C 通过 B 计算负样本对的距离并重塑为 [t, k] 形状
#     C = - tf.reshape(B, [t, k])
#     # D 是一个常数项，等于 A 加上 gamma，重塑为 [t, 1] 形状。
#     D = A + gamma
#     # L1 计算正样本对的损失，L2 计算负样本对的损失
#     L1 = tf.reduce_logsumexp(tf.add(C, tf.reshape(D, [t, 1])), axis=-1)
#     neg_left = tf.placeholder(tf.int32, [t * k], "neg2_left")
#     neg_right = tf.placeholder(tf.int32, [t * k], "neg2_right")
#     neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
#     neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
#     B = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
#     C = - tf.reshape(B, [t, k])
#     L2 = tf.reduce_logsumexp(tf.add(C, tf.reshape(D, [t, 1])), axis=-1)
#     return (tf.reduce_mean(L1) + tf.reduce_mean(L2)) / (2.0 * t)
"""边际排序损失"""
def get_loss_rank(outlayer, ILL, gamma, k):
    print('getting loss...')
    # 提取实体ID
    left = ILL[:, 0]
    right = ILL[:, 1]
    t = len(ILL)
    # 对应的嵌入向量
    left_x = tf.nn.embedding_lookup(outlayer, left)
    right_x = tf.nn.embedding_lookup(outlayer, right)
    # 计算绝对差之和，度量正样本对之间的距离
    A = tf.reduce_sum(tf.multiply(left_x,right_x), 1)
    # 计算左负样本对之间的距离
    neg_left = tf.placeholder(tf.int32, [t * k], "neg_left")
    neg_right = tf.placeholder(tf.int32, [t * k], "neg_right")
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
    B = tf.reduce_sum(tf.multiply(neg_l_x,neg_r_x), 1)
    # C 通过 B 计算负样本对的距离并重塑为 [t, k] 形状
    C = tf.reshape(B, [t, k])
    # D 是一个常数项，等于 A 加上 gamma，重塑为 [t, 1] 形状。
    D = -A+gamma
    # L1 计算正样本对的损失，L2 计算负样本对的损失
    L1 = tf.nn.leaky_relu(tf.add(C, tf.reshape(D, [t, 1])))
    neg_left = tf.placeholder(tf.int32, [t * k], "neg2_left")
    neg_right = tf.placeholder(tf.int32, [t * k], "neg2_right")
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
    B = tf.reduce_sum(tf.multiply(neg_l_x,neg_r_x), 1)
    C = tf.reshape(B, [t, k])
    L2 = tf.nn.leaky_relu(tf.add(C, tf.reshape(D, [t, 1])))
    return (tf.reduce_sum(L1) + tf.reduce_sum(L2)) / (2.0 * k * t)

def get_loss(outlayer, ILL, gamma, k):
    print('getting loss...')
    # 提取实体ID
    left = ILL[:, 0]
    right = ILL[:, 1]
    t = len(ILL)
    # 对应的嵌入向量
    left_x = tf.nn.embedding_lookup(outlayer, left)
    right_x = tf.nn.embedding_lookup(outlayer, right)
    # 计算绝对差之和，度量正样本对之间的距离
    A = tf.reduce_sum(tf.abs(left_x - right_x), 1)
    # 计算左负样本对之间的距离
    neg_left = tf.placeholder(tf.int32, [t * k], "neg_left")
    neg_right = tf.placeholder(tf.int32, [t * k], "neg_right")
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
    with tf.Session() as sess:
        print("实际维度 of negative_distance_1:", sess.run(tf.shape(neg_r_x)))

    shape_of_neg_r_x = tf.shape(neg_r_x)
    with tf.Session() as sess:
        print("实际维度 of neg_r_x:", sess.run(shape_of_neg_r_x))
    B = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
    # C 通过 B 计算负样本对的距离并重塑为 [t, k] 形状
    C = - tf.reshape(B, [t, k])
    # D 是一个常数项，等于 A 加上 gamma，重塑为 [t, 1] 形状。
    D = A + gamma
    # L1 计算正样本对的损失，L2 计算负样本对的损失
    L1 = tf.nn.leaky_relu(tf.add(C, tf.reshape(D, [t, 1])))
    neg_left = tf.placeholder(tf.int32, [t * k], "neg2_left")
    neg_right = tf.placeholder(tf.int32, [t * k], "neg2_right")
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
    B = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
    C = - tf.reshape(B, [t, k])
    L2 = tf.nn.leaky_relu(tf.add(C, tf.reshape(D, [t, 1])))
    return (tf.reduce_sum(L1) + tf.reduce_sum(L2)) / (2.0 * k * t)



def merge_adj_matrices(adj1, adj2):
    # 确保两个邻接矩阵的形状相同


    # 将两个邻接矩阵相加
    merged_adj = adj1 + adj2

    # 将所有大于 1 的元素置为 1，确保最终的邻接矩阵中每个元素的值不会大于 1
    merged_adj.data = np.minimum(merged_adj.data, np.ones_like(merged_adj.data))

    return merged_adj



def build(dimension, act_func, alpha, beta, gamma, k, lang, e, ILL, KG):
    tf.reset_default_graph()
    # 从输入数据中获取原始节点的特征表示
    primal_X_0 = get_input_layer(e, dimension, lang)
    heads = set([triple[0] for triple in KG])
    tails = set([triple[2] for triple in KG])
    ents = heads | tails


    one_adj,_ = no_weighted_adj(len(ents), KG, is_two_adj= False)
    two_hop_triples = generate_2steps_path(KG)
    two_adj,_ = no_weighted_adj(len(ents), two_hop_triples, is_two_adj= False)

    # 创建 SparseTensor 对象
    sparse_tensor_one = tf.SparseTensor(indices=one_adj[0], values=one_adj[1], dense_shape=one_adj[2])
    # sparse_tensor_two = tf.SparseTensor(indices=two_adj[0], values=two_adj[1], dense_shape=two_adj[2])
    # M_sparse = tf.sparse_add(sparse_tensor_one, sparse_tensor_two)
    M_sparse = sparse_tensor_one
    M = tf.cast(M_sparse, tf.float32)
    node_size = len(ents)


    # M, M_arr = get_sparse_tensor(e, KG)
    # 生成头实体、尾实体、头关系、尾关系以及关系矩阵。
    head, tail, head_r, tail_r, r_mat = rfunc(KG, e)
    print('first interaction...')
    # 获取双图输入层和双图邻接矩阵。
    dual_X_1, dual_A_1 = get_dual_input(
        primal_X_0, head, tail, head_r, tail_r, dimension)
    # 对双图进行自注意力操作，得到双图的隐藏表示。
    dual_H_1 = add_self_att_layer(dual_X_1, dual_A_1, tf.nn.relu, 600)
    # 对原始图进行稀疏注意力操作，得到原始图的隐藏表示。
    primal_H_1 = add_sparse_att_layer(
        primal_X_0, dual_H_1, r_mat, tf.nn.relu, e)
    # 使用残差连接更新原始节点特征表示。
    primal_X_1 = primal_X_0 + alpha * primal_H_1

    print('second interaction...')
    # 与第一个交互阶段类似，对更新后的原始节点特征再次进行交互操作。
    dual_X_2, dual_A_2 = get_dual_input(
        primal_X_1, head, tail, head_r, tail_r, dimension)
    dual_H_2 = add_dual_att_layer(
        dual_H_1, dual_X_2, dual_A_2, tf.nn.relu, 600)
    primal_H_2 = add_sparse_att_layer(
        primal_X_1, dual_H_2, r_mat, tf.nn.relu, e)
    primal_X_2 = primal_X_0 + beta * primal_H_2

    print('gcn layers...')
    shape = tf.shape(primal_X_2)
    with tf.Session() as sess:
        dimension1 = sess.run(shape)
    print("dimension维度")
    print(dimension1)
    gcn_layer_1 = add_diag_layer(
        primal_X_2, dimension, M, act_func, dropout=0.1)
    gcn_layer_1 = highway(primal_X_2, gcn_layer_1, dimension1[1])
    gcn_layer_2 = add_diag_layer(
        gcn_layer_1, dimension, M, act_func, dropout=0.1)
    shape = tf.shape(gcn_layer_1)
    with tf.Session() as sess:
        dimension2 = sess.run(shape)
    output_layer = highway(gcn_layer_1, gcn_layer_2, dimension2[1])
    # loss = get_loss_rank(output_layer, ILL, gamma, k)
    loss = get_loss(output_layer, ILL, gamma, k)
    # loss = get_loss_compare(output_layer, ILL, k)
    # loss = get_loss_wen(output_layer, ILL, gamma, k)
    # loss = get_loss_dualaloss(output_layer, ILL, k, node_size)
    print(loss)
    return output_layer, loss


# get negative samples
# 负样本生成-可改进
def get_neg(ILL, output_layer, k):
    neg = []
    t = len(ILL)
    ILL_vec = np.array([output_layer[e1] for e1 in ILL])
    KG_vec = np.array(output_layer)
    # 得到每个实体与字典中所有实体的曼哈顿距离
    sim = spatial.distance.cdist(ILL_vec, KG_vec, metric='cityblock')
    for i in range(t):
        # 得到每个实体与字典中所有实体的前k个相似实体
        rank = sim[i, :].argsort()
        # rank = sim[i, :].argsort()[:: , -1]
        neg.append(rank[0:k])

    neg = np.array(neg)
    neg = neg.reshape((t * k,))
    return neg

def compute_validation_loss_old(session, output_layer, ILL_valid, gamma, k):
    # 假设 ILL_valid 是类似 ILL 的结构，其中包含需要验证的实体对
    t_valid = len(ILL_valid)
    ILL_valid = np.array(ILL_valid)

    left_ids = ILL_valid[:, 0]
    right_ids = ILL_valid[:, 1]

    left_embeddings = tf.nn.embedding_lookup(output_layer, left_ids)
    right_embeddings = tf.nn.embedding_lookup(output_layer, right_ids)

    # 计算基本损失，这里简单使用 L1 距离
    basic_loss = tf.reduce_mean(tf.abs(left_embeddings - right_embeddings))

    # 计算负采样损失，这部分可能需要根据你的具体情况调整
    neg_left_ids = np.random.randint(0, output_layer.shape[0], size=(t_valid * k,))
    neg_right_ids = np.random.randint(0, output_layer.shape[0], size=(t_valid * k,))
    neg_left_embeddings = tf.nn.embedding_lookup(output_layer, neg_left_ids)
    neg_right_embeddings = tf.nn.embedding_lookup(output_layer, neg_right_ids)

    neg_loss = tf.reduce_mean(tf.abs(neg_left_embeddings - neg_right_embeddings))

    # 组合损失并计算总损失
    total_loss = basic_loss + gamma * neg_loss
    total_loss_val = session.run(total_loss)  # 计算具体的损失值

    return total_loss_val


def compute_validation_loss(sess, output_layer, ILL_valid, gamma, k):
    # 假设 ILL_valid 是类似 ILL 的结构，其中包含需要验证的实体对
    t_valid = len(ILL_valid)
    ILL_valid = np.array(ILL_valid)
    left_ids = ILL_valid[:, 0]
    right_ids = ILL_valid[:, 1]
    # 计算负采样损失
    t_valid = len(ILL_valid)
    ILL_valid = np.array(ILL_valid)
    v_L = np.ones((t_valid, k)) * (ILL_valid[:, 0].reshape((t_valid, 1)))
    v_neg_left = v_L.reshape((t_valid * k,))
    v_neg_left = tf.convert_to_tensor(v_neg_left, dtype=tf.int32)
    v_L = np.ones((t_valid, k)) * (ILL_valid[:, 1].reshape((t_valid, 1)))
    v_neg2_right = v_L.reshape((t_valid * k,))
    v_neg2_right = tf.convert_to_tensor(v_neg2_right, dtype=tf.int32)
    out = sess.run(output_layer)
    v_neg2_left = get_neg(ILL_valid[:, 1], out, k)
    v_neg_right = get_neg(ILL_valid[:, 0], out, k)


    # 计算基本损失，这里简单使用 L1 距离
    left_embeddings = tf.nn.embedding_lookup(output_layer, left_ids)
    right_embeddings = tf.nn.embedding_lookup(output_layer, right_ids)
    A = tf.reduce_mean(tf.square(left_embeddings - right_embeddings),1)


    # 使用占位符和 embedding_lookup 构建计算图
    neg_left_embeddings = tf.nn.embedding_lookup(output_layer,v_neg_left)
    neg_right_embeddings = tf.nn.embedding_lookup(output_layer, v_neg_right)
    neg2_left_embeddings = tf.nn.embedding_lookup(output_layer, v_neg2_left)
    neg2_right_embeddings = tf.nn.embedding_lookup(output_layer, v_neg2_right)

    B = tf.reduce_sum(tf.square(neg_left_embeddings - neg_right_embeddings),1)
    C = -tf.reshape(B, [t_valid, k])
    D = A + gamma
    L1 = tf.nn.relu(tf.add(C, tf.reshape(D, [t_valid, 1])))
    B = tf.reduce_sum(tf.square(neg2_left_embeddings - neg2_right_embeddings),1)
    C = -tf.reshape(B, [t_valid, k])
    L2 = tf.nn.relu(tf.add(C, tf.reshape(D, [t_valid, 1])))
    # 计算总损失
    total_loss = (tf.reduce_sum(L1) + tf.reduce_sum(L2)) / (2.0 * k * t_valid)
    total_loss_val = sess.run(total_loss)
    return total_loss_val


class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait = 0
        self.stopped = False

    def check(self, val_loss):
        if val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped = True

def training(output_layer, loss, learning_rate, epochs, ILL, e, k, test,validation, gamma):
    # 初始化 wandb
    wandb.init(project="drna-gcne")
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    early_stopping = EarlyStopping(patience=5, min_delta=0.01)
    print('initializing...')
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    print('running...')
    J = []
    t = len(ILL)
    ILL = np.array(ILL)
    L = np.ones((t, k)) * (ILL[:, 0].reshape((t, 1)))
    neg_left = L.reshape((t * k,))
    L = np.ones((t, k)) * (ILL[:, 1].reshape((t, 1)))
    neg2_right = L.reshape((t * k,))
    for i in range(epochs):
        if i % 10 == 0:
            out = sess.run(output_layer)
            neg2_left = get_neg(ILL[:, 1], out, k)
            neg_right = get_neg(ILL[:, 0], out, k)
            feeddict = {"neg_left:0": neg_left,
                        "neg_right:0": neg_right,
                        "neg2_left:0": neg2_left,
                        "neg2_right:0": neg2_right}

        _, th = sess.run([train_step, loss], feed_dict=feeddict)
        if i % 10 == 0:
            th, outvec = sess.run([loss, output_layer], feed_dict=feeddict)
            J.append(th)
            get_hits(outvec, test)
            val_loss = compute_validation_loss_old(sess, output_layer, validation, gamma, k)
            # 记录损失到 wandb
            wandb.log({"Training Loss": th})
            # Simulate validation loss calculation (implement this function based on your validation data)

            early_stopping.check(val_loss)
            if early_stopping.stopped:
                print("Early stopping triggered at epoch:", i + 1)
                break
        print('%d/%d' % (i + 1, epochs), 'epochs...', th)



    outvec = sess.run(output_layer)
    # saver = tf.train.Saver()
    # saver.save(sess, 'model/checkpoints/model.ckpt')
    sess.close()
    # 绘制损失函数
    plt.figure(figsize=(10, 5))  # 设置图表大小
    plt.plot(J, label='Training Loss')  # 绘制损失值
    plt.title('Training Loss over Epochs')  # 设置图表标题
    plt.xlabel('Epochs')  # 设置x轴标签
    plt.ylabel('Loss')  # 设置y轴标签
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格
    plt.show()  # 显示图表a
    return outvec, J


def continue_training(output_layer, loss, learning_rate, new_epochs, new_ILL, e, k, new_test, checkpoint_dir):
    # 创建Saver对象
    saver = tf.train.Saver()
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    # 初始化模型和变量
    print('initializing...')
    init = tf.global_variables_initializer()

    # 创建一个Session对象
    sess = tf.Session()
    sess.run(init)
    print('running...')

    # 检查checkpoint目录是否存在，并加载模型
    if os.path.isdir(checkpoint_dir):
        last_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if last_checkpoint:
            saver.restore(sess, last_checkpoint)
            print("Model loaded successfully from {}".format(last_checkpoint))
        else:
            print("No model found in the directory. Please train the model first.")
            return None, []
    else:
        print("Checkpoint directory does not exist. Please train the model first.")
        return None, []

    # 准备新的训练数据
    ILL = np.array(new_ILL)
    t = len(ILL)
    L = np.ones((t, k)) * (ILL[:, 0].reshape((t, 1)))
    neg_left = L.reshape((t * k,))
    L = np.ones((t, k)) * (ILL[:, 1].reshape((t, 1)))
    neg2_right = L.reshape((t * k,))

    # 继续训练
    J = []
    for i in range(new_epochs):
        if i % 10 == 0:
            out = sess.run(output_layer)
            neg2_left = get_neg(ILL[:, 1], out, k)
            neg_right = get_neg(ILL[:, 0], out, k)
            feeddict = {"neg_left:0": neg_left,
                        "neg_right:0": neg_right,
                        "neg2_left:0": neg2_left,
                        "neg2_right:0": neg2_right}

        _, th = sess.run([train_step, loss], feed_dict=feeddict)

        if i % 10 == 0:
            th, outvec = sess.run([loss, output_layer], feed_dict=feeddict)
            J.append(th)
            get_hits(outvec, new_test)

        print('%d/%d' % (i + 1, new_epochs), 'epochs...', th)

    # 保存最终模型
    saver.save(sess, os.path.join(checkpoint_dir, 'model.ckpt'))

    # 关闭会话
    sess.close()

    # 返回模型输出和损失列表
    return outvec, J

# 使用示例
# 假设你已经有了新的数据new_ILL和new_test，以及训练的参数
# new_epochs 表示你想要继续训练的额外epoch数量
# checkpoint_dir 是模型保存的目录
# output_layer, loss, learning_rate, e, k 等参数需要与之前的训练保持一致

