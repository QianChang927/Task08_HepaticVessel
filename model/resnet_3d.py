import torch
from torch import nn

# 类型注解所用库
from torch import Tensor
from torch.nn import Module, Sequential
from typing import Sequence, Literal, Any

class ResNet3D(nn.Module):
    """
    ResNet的3D版本
    """
    def __init__(
        self,
        block: Module,
        layers: Sequence[int],
        in_channels: int,
        out_channels: int,
        planes: Sequence[int]=None,
        skip_type: Literal['A', 'B']= 'B',
        norm_type: Module=None,
        norm_args: dict=None
    ) -> None:
        """
        ResNet3D构造函数
        :param block: 选用的基本块类
        :param layers: 4个卷积块中基本块的层数
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param planes: 每层输出通道数
        :param skip_type: 短路连接类型，'A' -填充方法增加通道数，'B' -卷积方法增加通道数
        :param norm_type: 归一化层类
        :param norm_args: 归一化层需要特殊处理的参数
        :return:
        """
        super(ResNet3D, self).__init__()
        if planes is None: planes = [64, 128, 256, 512]
        if norm_type is None: norm_type = nn.Identity
        if norm_args is None: norm_args = {}
        bias = False if norm_type != nn.Identity else True

        assert len(layers) == len(planes)
        assert len(layers) >= 4
        assert skip_type in ['A', 'B']

        self.in_planes = 64
        self.bias = bias

        self.block = block
        self.layers = layers[:4]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.planes = planes[:4]
        self.skip_type = skip_type
        self.norm_type = norm_type
        self.norm_args = norm_args

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, self.in_planes, kernel_size=7, stride=2, padding=3, bias=bias),
            norm_type(self.in_planes, **norm_args),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.__build_layers(planes[0], layers[0], 1)
        self.layer2 = self.__build_layers(planes[1], layers[1], 2)
        self.layer3 = self.__build_layers(planes[2], layers[2], 2)
        self.layer4 = self.__build_layers(planes[3], layers[3], 2)
        self.segment = self.__build_segment()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        :param x: 传入张量
        :return: 传出张量
        """
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.segment(x)
        return x

    def __build_layers(self, planes: int, block_num: int, stride: int=1) -> Sequential:
        """
        构建中间层
        :param planes: 中间层输出通道
        :param block_num: 中间层数量
        :param stride: 中间层步长
        :return: 中间层
        """
        block, bias = self.block, self.bias
        norm_type, norm_args = self.norm_type, self.norm_args

        if stride != 1 or self.in_planes != planes * block.expansion:
            if self.skip_type == 'A':
                from functools import partial
                shortcut = partial(
                    ResNet3D.__downsample_zero_pads,
                    planes=planes * block.expansion,
                    stride=stride
                )
            else:
                shortcut = nn.Sequential(
                    nn.Conv3d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, padding=0, bias=bias),
                    norm_type(planes * block.expansion, **norm_args)
                )
        else: shortcut = None

        layers = [block(self.in_planes, planes, stride, shortcut, norm_type, norm_args)]
        self.in_planes = planes * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_planes, planes, norm_type=norm_type, norm_args=norm_args))

        return nn.Sequential(*layers)

    def __build_segment(self) -> Sequential:
        """
        构建分割层
        """
        bias, block, planes = self.bias, self.block, self.planes
        norm_type, norm_args = self.norm_type, self.norm_args
        in_planes, out_planes = planes[0], planes[-1]

        layers = [nn.Sequential(
            nn.ConvTranspose3d(out_planes * block.expansion, in_planes, kernel_size=2, stride=2, bias=bias),
            norm_type(in_planes, **norm_args),
            nn.ReLU(inplace=True)
        )]

        for _ in range(len(planes)):
            layers.append(nn.Sequential(
                nn.ConvTranspose3d(in_planes, in_planes, kernel_size=2, stride=2, bias=bias),
                norm_type(in_planes, **norm_args),
                nn.ReLU(inplace=True)
            ))

        layers.append(nn.Conv3d(in_planes, self.out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        return nn.Sequential(*layers)

    @staticmethod
    def __downsample_zero_pads(x: Tensor, planes: int, stride: int) -> Tensor:
        """
        A方法实现下采样，利用零填充
        :param x: 输入张量
        :param planes: 输出通道数
        :param stride: 步长
        :return: 输出张量
        """
        import torch.nn.functional as F
        spatial_out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        out_shape = spatial_out.shape
        if len(out_shape) > 4:
            zero_pads = torch.zeros(out_shape[0], planes - out_shape[1], *out_shape[2:], dtype=x.dtype, device=x.device)
        else:
            zero_pads = torch.zeros(planes - out_shape[0], *out_shape[1:], dtype=x.dtype, device=x.device)
        out = torch.cat([spatial_out.data, zero_pads], dim=1)
        out.requires_grad_()
        return out


class BasicBlock(nn.Module):
    """
    用于构建ResNet-18/34网络（小型网络）
    """
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int=1,
        shortcut: Module=None,
        norm_type: Module=None,
        norm_args: dict=None
    ) -> None:
        """
        构造函数
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param stride: 卷积层步长，用于实现降采样，不进行降采样时为1
        :param shortcut: 短路连接层，用于与卷积层的降采样步骤保持一致
        :param norm_type: 归一化层类
        :param norm_args: 归一化层需要特殊处理的参数
        :return:
        """
        super(BasicBlock, self).__init__()
        if shortcut is None: shortcut = nn.Identity(in_channels, out_channels)
        if norm_type is None: norm_type = nn.Identity
        if norm_args is None: norm_args = {}
        bias = False if norm_type != nn.Identity else True

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias),
            norm_type(out_channels, **norm_args),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels * self.expansion, kernel_size=3, stride=1, padding=1, bias=bias),
            norm_type(out_channels * self.expansion, **norm_args)
        )
        self.act = nn.ReLU(inplace=True)
        self.shortcut = shortcut

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        :param x: 传入张量
        :return: 传出张量
        """
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.add(x, residual)
        x = self.act(x)
        return x


class BottleNeck(nn.Module):
    """
    用于构建ResNet-50/101/152网络（大型网络）
    """
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        shortcut: Module = None,
        norm_type: Module=None,
        norm_args: dict=None
    ) -> None:
        """
        构造函数
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param stride: 卷积层步长，用于实现降采样，不进行降采样时为1
        :param shortcut: 短路连接层，用于与卷积层的降采样步骤保持一致
        :param norm_type: 归一化层类
        :param norm_args: 归一化层需要特殊处理的参数
        :return:
        """
        super(BottleNeck, self).__init__()
        if shortcut is None: shortcut = nn.Identity(in_channels, out_channels)
        if norm_type is None: norm_type = nn.Identity
        if norm_args is None: norm_args = {}
        bias = False if norm_type != nn.Identity else True

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
            norm_type(out_channels, **norm_args),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias),
            norm_type(out_channels, **norm_args),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0, bias=bias),
            norm_type(out_channels * self.expansion, **norm_args)
        )
        self.act = nn.ReLU(inplace=True)
        self.shortcut = shortcut

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        :param x: 传入张量
        :return: 传出张量
        """
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.add(x, residual)
        x = self.act(x)
        return x


class ResNet:
    """
    ResNet构造类
    """
    @staticmethod
    def resnet18(**kwargs: Any) -> ResNet3D:
        return ResNet3D(BasicBlock, [2, 2, 2, 2], **kwargs)

    @staticmethod
    def resnet34(**kwargs: Any) -> ResNet3D:
        return ResNet3D(BasicBlock, [3, 4, 6, 3], **kwargs)

    @staticmethod
    def resnet50(**kwargs: Any) -> ResNet3D:
        return ResNet3D(BottleNeck, [3, 4, 6, 3], **kwargs)

    @staticmethod
    def resnet101(**kwargs: Any) -> ResNet3D:
        return ResNet3D(BottleNeck, [3, 4, 23, 3], **kwargs)

    @staticmethod
    def resnet152(**kwargs: Any) -> ResNet3D:
        return ResNet3D(BottleNeck, [3, 8, 36, 3], **kwargs)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    resnet_3d = ResNet3D(
        block=BottleNeck,
        layers=[3, 4, 6, 3],
        in_channels=1,
        out_channels=2,
        skip_type='B',
        norm_type=nn.BatchNorm3d,
        norm_args={'affine': True}
    ).to(device)
    print(resnet_3d)

    tensor: Tensor = torch.randn([8, 1, 128, 128, 64]).to(device)
    output: Tensor = resnet_3d(tensor)
    print(output.shape)