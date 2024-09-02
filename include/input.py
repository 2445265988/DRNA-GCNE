import os

import numpy as np
import time

from align.kg import KG


def read_input(folder):
    triples_set1 = read_triples(folder + 'triples_1')
    triples_set2 = read_triples(folder + 'triples_2')
    triples1 = KG(triples_set1)
    triples2 = KG(triples_set2)
    # 计算两个知识图中所有实体的总数
    total_ent_num = len(triples1.ents | triples2.ents)
    # 计算两个知识图中所有关系的总数
    total_rel_num = len(triples1.props | triples2.props)
    # 计算两个知识图中所有三元组的总数
    total_triples_num = len(triples1.triple_list) + len(triples2.triple_list)
    print('total ents:', total_ent_num)
    print('total rels:', len(triples1.props), len(triples2.props), total_rel_num)
    print('total triples: %d + %d = %d' % (len(triples1.triples), len(triples2.triples), total_triples_num))
    # 断言读取关系文件
    ref_ent1, ref_ent2 = read_references(folder + 'ref_ent_ids')
    # 关系实体的列表长度
    # 如果 assert 后面的条件为假，Python 解释器会引发 AssertionError 异常
    assert len(ref_ent1) == len(ref_ent2)
    print("To aligned entities:", len(ref_ent1))
    sup_ent1, sup_ent2 = read_references(folder + 'sup_ent_ids')
    return triples1, triples2, sup_ent1, sup_ent2, ref_ent1, ref_ent2, total_triples_num, total_ent_num, total_rel_num


def read_dbp15k_input(folder):
    triples_set1 = read_triples(folder + 'triples_1')
    triples_set2 = read_triples(folder + 'triples_2')
    kg1 = KG(triples_set1)
    kg2 = KG(triples_set2)
    total_ent_num = len(kg1.ents | kg2.ents)
    total_rel_num = len(kg1.props | kg2.props)
    total_triples_num = len(kg1.triple_list) + len(kg2.triple_list)
    print('total ents:', total_ent_num)
    print('total rels:', len(kg1.props), len(kg2.props), total_rel_num)
    print('total triples: %d + %d = %d' % (len(kg1.triples), len(kg2.triples), total_triples_num))
    if os.path.exists(folder + 'ref_pairs'):
        # 参考实体对
        ref_ent1, ref_ent2 = read_references(folder + 'ref_pairs')
    else:
        ref_ent1, ref_ent2 = read_references(folder + 'ref_ent_ids')
    assert len(ref_ent1) == len(ref_ent2)
    print("To aligned entities:", len(ref_ent1))
    if os.path.exists(folder + 'sup_pairs'):
        # 已知的对齐实体对
        sup_ent1, sup_ent2 = read_references(folder + 'sup_pairs')
    else:
        sup_ent1, sup_ent2 = read_references(folder + 'sup_ent_ids')
    #     ****************************id mapping*************************
    rel_id_mapping = get_id_mapping(folder)
    return kg1, kg2, sup_ent1, sup_ent2, ref_ent1, ref_ent2, total_triples_num, total_ent_num, total_rel_num, rel_id_mapping


def read_triples(file):
    triples = set()
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            params = line.strip('\n').split('\t')
            assert len(params) == 3
            h = int(params[0])
            r = int(params[1])
            t = int(params[2])
            triples.add((h, r, t))
        f.close()
    return triples


def read_references(file):
    ref1, ref2 = list(), list()
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            params = line.strip('\n').split('\t')
            assert len(params) == 2
            e1 = int(params[0])
            e2 = int(params[1])
            ref1.append(e1)
            ref2.append(e2)
        f.close()
        assert len(ref1) == len(ref2)
    return ref1, ref2


def div_list(ls, n):
    ls_len = len(ls)
    if n <= 0 or 0 == ls_len:
        return [ls]
    if n > ls_len:
        return [ls]
    elif n == ls_len:
        return [[i] for i in ls]
    else:
        j = ls_len // n
        k = ls_len % n
        ls_return = []
        for i in range(0, (n - 1) * j, j):
            ls_return.append(ls[i:i + j])
        ls_return.append(ls[(n - 1) * j:])
        return ls_return

# 将三元组转换为头尾实体集合
def triples2ht_set(triples):
    ht_set = set()
    for h, r, t in triples:
        ht_set.add((h, t))
    print("the number of ht: {}".format(len(ht_set)))
    return ht_set

# 合并两个字典
def merge_dic(dic1, dic2):
    return {**dic1, **dic2}

# 生成邻接矩阵的函数
def generate_adjacency_mat(triples1, triples2, ent_num, sup_ents):
    adj_mat = np.mat(np.zeros((ent_num, len(sup_ents)), dtype=np.int32))
    #  将三元组转换为头尾实体集合
    ht_set = triples2ht_set(triples1) | triples2ht_set(triples2)
    # # 遍历实体和支持实体的组合
    for i in range(ent_num):
        for j in sup_ents:
            if (i, j) in ht_set:
                #  如果 (i, j) 在头尾实体集合中，说明存在关系，将对应位置设为 1
                adj_mat[i, sup_ents.index(j)] = 1
    print("shape of adj_mat: {}".format(adj_mat.shape))
    print("the number of 1 in adjacency matrix: {}".format(np.count_nonzero(adj_mat)))
    return adj_mat

# 生成一个输入矩阵
def generate_adj_input_mat(adj_mat, d):
    # 生成一个大小为 (adj_mat.shape[1], d) 的随机权重矩阵 W
    W = np.random.randn(adj_mat.shape[1], d)
    # 计算输入矩阵 M，即邻接矩阵 adj_mat 与权重矩阵 W 的乘积
    M = np.matmul(adj_mat, W)
    # 打印输入矩阵的形状
    print("shape of input adj_mat: {}".format(M.shape))
    # 返回生成的输入矩阵 M
    return M

# 生成实体属性嵌入的函数
def generate_ent_attrs_sum(ent_num, ent_attrs1, ent_attrs2, attr_embeddings):
    # 记录函数开始时间
    t1 = time.time()
    # 初始化实体属性嵌入矩阵
    ent_attrs_embeddings = None
    for i in range(ent_num):
        # 获取当前实体 i 的属性索引，合并 ent_attrs1 和 ent_attrs2 中的属性
        attrs_index = list(ent_attrs1.get(i, set()) | ent_attrs2.get(i, set()))
        # 断言：确保属性索引集合非空
        assert len(attrs_index) > 0
        # 根据属性索引获取对应的属性嵌入，并对它们进行求和
        attrs_embeds = np.sum(attr_embeddings[attrs_index,], axis=0)
        # 将当前实体的属性嵌入追加到 ent_attrs_embeddings 中
        if ent_attrs_embeddings is None:
            ent_attrs_embeddings = attrs_embeds
        else:
            ent_attrs_embeddings = np.row_stack((ent_attrs_embeddings, attrs_embeds))
    print("shape of ent_attr_embeds: {}".format(ent_attrs_embeddings.shape))
    print("generating ent features costs: {:.3f} s".format(time.time() - t1))
    return ent_attrs_embeddings


def get_id_mapping(folder):

    rel_ids_1 = folder + "rel_ids_1"
    rel_ids_2 = folder + "rel_ids_2"
    kg1_id_dict = dict()
    kg2_id_dict = dict()
    # 读取第一个知识图谱的关系ID映射文件
    # kg1_id_dict[http://dbpedia.org/resource/Gela]=4.0
    with open(rel_ids_1, 'r', encoding='utf8') as f:
        for line in f:
            params = line.strip('\n').split('\t')
            kg1_id_dict[params[1]] = int()
            kg1_id_dict[params[1]] = int(params[0])
        f.close()
    # 读取第一个知识图谱的关系ID映射文件
    with open(rel_ids_2, 'r', encoding='utf8') as f:
        for line in f:
            params = line.strip('\n').split('\t')
            kg2_id_dict[params[1]] = int()
            kg2_id_dict[params[1]] = int(params[0])
        f.close()

    rt_dict = dict()
    #  从文件夹路径中提取新的文件夹目录名
    # 更新最新的关系ID映射
    fold = folder.split("/")[-2]
    new_dir = folder.split("mtranse")[0] + fold + "/"
    if os.path.exists(new_dir):
        new_ids_1 = new_dir + "rel_ids_1"
        new_ids_2 = new_dir + "rel_ids_2"
        with open(new_ids_1, "r", encoding="utf8") as f:
            for line in f:
                params = line.strip("\n").split("\t")
                if kg1_id_dict[params[1]] not in rt_dict.keys():
                    rt_dict[kg1_id_dict[params[1]]] = int()
                rt_dict[kg1_id_dict[params[1]]] = int(params[0])
            f.close()
        with open(new_ids_2, "r", encoding="utf8") as f:
            for line in f:
                params = line.strip("\n").split("\t")
                if kg2_id_dict[params[1]] not in rt_dict.keys():
                    rt_dict[kg2_id_dict[params[1]]] = int()
                rt_dict[kg2_id_dict[params[1]]] = int(params[0])
            f.close()
    # 如果新的文件夹不存在，直接将关系ID映射关系设为对应的ID
    else:
        for value in kg1_id_dict.values():
            if value not in rt_dict.keys():
                rt_dict[value] = int()
                rt_dict[value] = value
        for value in kg2_id_dict.values():
            if value not in rt_dict.keys():
                rt_dict[value] = int()
                rt_dict[value] = value

    return rt_dict
