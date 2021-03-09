import torch.nn as nn
import torch


class VGG(nn.Module):
    # features是提取的特征，num_classes分1000类(imagenet)，初始化权重
    def __init__(self, features, num_classes=1000, init_weights=False):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            # 随机失活50%神经元
            nn.Dropout(p=0.5),
            # 标准的vgg最后都会生成512*7*7的特征矩阵，最终应该展平成4096，但是这里为了减少训练参数
            nn.Linear(512 * 7 * 7, 2048),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Linear(2048, num_classes)
        )
        # 如果init_weights=true时，则会初始化权重
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)
        # N x 512*7*7
        x = self.classifier(x)
        return x

    # 不清楚为什么要这么初始化
    # https://blog.csdn.net/dss_dssssd/article/details/83959474
    # 定义的初始化权重函数
    def _initialize_weights(self):
        # 遍历网络当中每一个子模块，也就是遍历每一层
        for m in self.modules():
            # isinstance(a,b)：a是否是b类型
            # 如果当前层是卷积层
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # 使用这个函数来初始化卷积核的权重
                nn.init.xavier_uniform_(m.weight)
                # 如果卷积核的的偏置不为0
                if m.bias is not None:
                    # 置为0
                    nn.init.constant_(m.bias, 0)
            # 如果是全连接层
            elif isinstance(m, nn.Linear):
                # 初始化全连接层的权重
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                # 偏置置为0
                nn.init.constant_(m.bias, 0)

# 生成 提取特征 网络
# 传入的参数是List类型，就是把cfgs那个字典的其中一条传过来
def make_features(cfg: list):
    # 一个空的List
    layers = []
    in_channels = 3
    for v in cfg:
        # 如果传入的是最大池化下采样
        if v == "M":
            # 就把最大池化下采样加到layers这个空的List中
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        # 如果不是，就按照cfg列表的结构，加到layers这个空列表中
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    # 最后把整个vgg卷积的过程的列表layers，通过非关键字参数的形式传入
    # 这个*代表非关键字参数
    return nn.Sequential(*layers)


# 标准的vgg11 13 16 19，数字代表特征图个数，字母代表MaxPooling
cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

# 实例化vgg网络
# args,*args,**kargs用法：https://www.cnblogs.com/kaishirenshi/p/8610662.html
# **kwargs是可变长度的字典对象
def vgg(model_name="vgg16", **kwargs):
    try:
        # key=vgg16传入字典中
        cfg = cfgs[model_name]
    except:
        print("Warning: model number {} not in cfgs dict!".format(model_name))
        exit(-1)
    model = VGG(make_features(cfg), **kwargs)
    return model
