# coding=utf-8
import torch

# 数据路径
base_dir = 'resource/data/'
# 模型保存路径
save_dir = 'resource/models/'

categories = ['car_related',
              'appearance',
              'interior',
              'config',
              'space',
              'control',
              'comfort',
              'power',
              'energy_consumption',
              'car_use',
              'budget',
              'offer',
              'discount',
              'loan',
              'insurance',
              'final_price',
              'car_price_other',
              'other']

gpu_NO = 1
cuda_NO = 'cuda:' + str(gpu_NO)


class TextCNNConfig(object):
    """TextCNN模型配置参数"""

    def __init__(self):
        self.n_vocab = 0  # 词表大小，在运行时赋值
        self.embed = 200  # 字向量维度
        self.pad_size = 128  # 每句话处理成的长度(短填长切)
        self.num_classes = len(categories)  # 类别数

        self.dropout = 0.5  # 随机失活
        self.batch_size = 32  # mini-batch大小:128
        self.learning_rate = 1e-5  # 学习率

        self.filter_sizes = (5, 3)  # 卷积核尺寸
        self.num_filters = 1024  # 卷积核数量(channels数)

        self.num_epochs = 100  # epoch数:100
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练

    model_name = 'TextCNN-dim18'
    class_list = categories  # 类别名单
    embedding_pretrained = None  # 预训练词向量

    # 数据集路径
    vocab_path = base_dir + 'vocab.txt'  # 词表
    train_path = base_dir + 'train.txt'  # 训练集
    dev_path = base_dir + 'dev.txt'  # 验证集
    test_path = base_dir + 'test.txt'  # 测试集

    # 最佳验证结果保存路径
    save_path = save_dir + model_name + '.ckpt'  # 模型训练结果

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
