import math


class KG:
    def __init__(self, triples, ori_triples=None):
        self.triples = set(triples)
        self.triple_list = list(self.triples)
        self.triples_num = len(self.triples)

        self.heads = set([triple[0] for triple in self.triple_list])
        self.props = set([triple[1] for triple in self.triple_list])
        self.tails = set([triple[2] for triple in self.triple_list])
        self.ents = self.heads | self.tails

        print("triples num", self.triples_num)

        print("head ent num", len(self.heads))
        print("total ent num", len(self.ents))

        self.prop_list = list(self.props)
        self.ent_list = list(self.ents)
        self.prop_list.sort()
        self.ent_list.sort()

        if ori_triples is None:
            self.ori_triples = None
        else:
            self.ori_triples = set(ori_triples)

        self._generate_related_ents()
        self._generate_triple_dict()
        self._generate_ht()
        self.__generate_weight()

    # 生成头实体与尾实体的相关实体字典
    def _generate_related_ents(self):
        # 存储每个头实体对应的相关尾实体集合
        self.out_related_ents_dict = dict()
        # 存储每个尾实体对应的相关头实体集合。
        self.in_related_ents_dict = dict()
        for h, r, t in self.triple_list:
            out_related_ents = self.out_related_ents_dict.get(h, set())
            out_related_ents.add(t)
            self.out_related_ents_dict[h] = out_related_ents

            in_related_ents = self.in_related_ents_dict.get(t, set())
            in_related_ents.add(h)
            self.in_related_ents_dict[t] = in_related_ents

    # 头实体与关系、尾实体与关系的字典
    def _generate_triple_dict(self):
        self.rt_dict, self.hr_dict = dict(), dict()
        for h, r, t in self.triple_list:
            rt_set = self.rt_dict.get(h, set())
            rt_set.add((r, t))
            self.rt_dict[h] = rt_set
            hr_set = self.hr_dict.get(t, set())
            hr_set.add((h, r))
            self.hr_dict[t] = hr_set

    # 头尾实体对集合
    def _generate_ht(self):
        self.ht = set()
        for h, r, t in self.triples:
            self.ht.add((h, t))

    #生成加权的三元组列表
    def __generate_weight(self):
        triple_num = dict()
        n = 0
        for h, r, t in self.triples:
            # 如果尾实体在头实体集合中，则证明该三元组是一个输出实体
            # 在这种情况下，关系的起始点（头实体）与终点（尾实体）之间存在某种关联，通常可以理解为头实体经过关系的作用，导致了尾实体的出现。检查尾实体是否在头实体集合中有助于识别这种关联性
            if t in self.heads:
                n = n + 1
                triple_num[h] = triple_num.get(h, 0) + 1
                triple_num[t] = triple_num.get(t, 0) + 1
        self.weighted_triples = list()
        self.additional_triples = list()
        # 计算平均每个头实体的输出次数，采用向上取整
        ave = math.ceil(n / len(self.heads))
        print("ave outs:", ave)
        # 生成了一个新的加权三元组列表
        # weighted_triples，其中每个三元组的最后一个元素是该三元组的权重。
        # 对于那些在头实体集合中的尾实体，并且头实体输出次数不超过平均值的情况下，赋予其更高的权重（2.0），其余三元组的权重为默认值（1）
        for h, r, t in self.triples:
            w = 1
            if t in self.heads and triple_num[h] <= ave:
                w = 2.0
                self.additional_triples.append((h, r, t))
            self.weighted_triples.append((h, r, t, w))
        print("additional triples:", len(self.additional_triples))
        # self.train_triples
