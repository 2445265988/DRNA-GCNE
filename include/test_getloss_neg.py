import math

import scipy
import json

import scipy.spatial as spatial
import os
import tensorflow as tf


def get_loss_maxneg(outlayer, ILL, gamma, k):

    print('getting loss...')
    # 提取实体ID
    left = ILL[:, 0]
    right = ILL[:, 1]
    t = len(ILL)
    # 对应的嵌入向量
    left_x = tf.nn.embedding_lookup(outlayer, left)
    right_x = tf.nn.embedding_lookup(outlayer, right)
    # 计算绝对差之和，度量正样本对之间的距离
    positive_distance = tf.reduce_sum(tf.square(left_x - right_x), 1)
    # 正样本损失：最小化正样本距离
    # 我们希望正样本距离小于 gamma，所以使用最大函数确保距离不会超过 gamma
    positive_loss = tf.reduce_mean(tf.maximum(gamma - positive_distance, 0))
    # 计算左负样本对之间的距离
    neg_left = tf.placeholder(tf.int32, [t * k], "neg_left")
    neg_right = tf.placeholder(tf.int32, [t * k], "neg_right")
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
    negative_distance_1 = tf.reduce_sum(tf.square(neg_l_x - neg_r_x), 1)
    # 负样本损失：最大化负样本距离
    # 我们希望负样本距离大于 gamma，所以使用最大函数确保距离不会小于 gamma
    negative_loss_1 = tf.reduce_mean(tf.maximum(negative_distance_1 - gamma, 0))

    neg2_left = tf.placeholder(tf.int32, [t * k], "neg2_left")
    neg2_right = tf.placeholder(tf.int32, [t * k], "neg2_right")
    neg2_l_x = tf.nn.embedding_lookup(outlayer, neg2_left)
    neg2_r_x = tf.nn.embedding_lookup(outlayer, neg2_right)
    negative_distance_2 = tf.reduce_sum(tf.square(neg2_l_x - neg2_r_x), 1)
    negative_loss_2 = tf.reduce_mean(tf.nn.relu(negative_distance_2 - gamma))
    # 总损失是正样本损失和两组负样本损失的和
    loss = positive_loss + negative_loss_1 + negative_loss_2
    return loss