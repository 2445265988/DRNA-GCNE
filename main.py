import psutil
import tensorflow as tf
from include.Config import Config
# 改进的模型
from include.Model import training,build
# 基线模型
# from include.Model import build
from include.Test import get_hits
from include.Load import *

import warnings
warnings.filterwarnings("ignore")



seed = 12306
np.random.seed(seed)
tf.set_random_seed(seed)

if __name__ == '__main__':
    # 获取内存使用情况
    # memory = psutil.virtual_memory()


    # print("总内存: %s, 可用内存: %s, 已使用内存: %s, 使用百分比: %d%%" % (
    #     memory.total,
    #     memory.available,
    #     memory.used,
    #     memory.percent
    # ))
    e = len(set(loadfile(Config.e1, 1)) | set(loadfile(Config.e2, 1)))
    ILL = loadfile(Config.ill, 2)
    illL = len(ILL)
    np.random.shuffle(ILL)
    # train = np.array(ILL[:illL*0.6])
    # 计算划分比例
    train_size = int(illL * 0.6)
    val_size = int(illL * 0.2)
    test_size = illL - train_size - val_size
    # 从打乱后的数据中提取训练及验证集测试集
    train = np.array(ILL[:train_size])
    val = np.array(ILL[train_size:train_size + val_size])

    test = np.array(ILL[train_size + val_size:])
    KG1 = loadfile(Config.kg1, 3)
    KG2 = loadfile(Config.kg2, 3)
    # 获取内存使用情况
    # memory = psutil.virtual_memory()
    # print("总内存: %s, 可用内存: %s, 已使用内存: %s, 使用百分比: %d%%" % (
    #     memory.total,
    #     memory.available,
    #     memory.used,
    #     memory.percent
    # ))
    output_layer, loss = build(
        Config.dim, Config.act_func, Config.alpha, Config.beta, Config.gamma, Config.k, Config.language[0:2], e,
        train, KG1 + KG2)
    #
    # 获取内存使用情况
    # memory = psutil.virtual_memory()
    #
    # # 打印中文结果
    # print("总内存: %s, 可用内存: %s, 已使用内存: %s, 使用百分比: %d%%" % (
    #     memory.total,
    #     memory.available,
    #     memory.used,
    #     memory.percent
    # ))
    vec, J = training(output_layer, loss, 0.001,
                      Config.epochs, train, e, Config.k, train, val, Config.gamma)
    # vec, J = training(output_layer, loss, 0.001,
    #                   Config.epochs, train, e, Config.k, train)
    print('loss:', J)
    print('Result:')
    get_hits(vec, test)
