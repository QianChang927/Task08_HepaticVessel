import torch
from torch import nn

# 类型注解所用库
from torch import Tensor
from torch.nn import Module, ModuleList
from typing import Literal

class UNet3D(nn.Module):
    """
    3D U-Net类，总体框架分为编码部分和解码部分
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_channels: list=None,
        norm_type: Module=None,
        norm_args: dict=None
    ) -> None:
        """
        3D U-Net类构造函数
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param n_channels: 中间层通道数
        :param norm_type: 归一化层类
        :param norm_args: 归一化层需要特殊处理的参数
        :return:
        """
        super(UNet3D, self).__init__()
        if norm_type is None: norm_type = nn.Identity
        if norm_args is None: norm_args = {}
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_type = norm_type
        self.norm_args = norm_args

        if n_channels is None: n_channels = [64, 128, 256, 512]
        if len(n_channels) <= 1: raise ValueError('n_channels should have at least 2 elements')
        self.n_channels = n_channels

        self.in_block = DoubleConv(in_channels, n_channels[0], norm_type=norm_type, norm_args=norm_args)
        self.encoder = self.__build_layers(mode='encoder')
        self.decoder = self.__build_layers(mode='decoder')
        self.out_block = OutConv(n_channels[0], out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        :param x: 传入张量
        :return: 计算结果
        """
        # 特征缓存
        features = []

        # 输入层
        x = self.in_block(x)
        features.append(x)

        # 编码过程
        for layer in self.encoder:
            x = layer(x)
            features.append(x)

        # 瓶颈层处理，丢弃此层的特征避免错误残差计算
        features.pop()

        # 解码过程
        for layer in self.decoder:
            x = layer(x, features.pop())

        # 输出层
        x = self.out_block(x)

        return x

    def __build_layers(self, mode: Literal['encoder', 'decoder']) -> ModuleList:
        """
        构造中间层
        :param mode: 构造模式，选择构造编码器或解码器
        :return: 构造出的中间网络层
        """
        layers = nn.ModuleList()
        n_channels = self.n_channels if mode == 'encoder' else tuple(reversed(self.n_channels))
        for in_channels, out_channels in zip(n_channels[:-1], n_channels[1:]):
            if mode == 'encoder':
                layer = DownSample(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    norm_type=self.norm_type,
                    norm_args=self.norm_args
                )
            elif mode == 'decoder':
                layer = UpSample(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    skip_channels=out_channels,
                    norm_type=self.norm_type,
                    norm_args=self.norm_args
                )
            else:
                raise ValueError('mode should be either `encoder` or `decoder`')
            layers.append(layer)
        return layers


class DoubleConv(nn.Module):
    """
    双层卷积，包含两个卷积层
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: Module=None,
        norm_args: dict=None
    ) -> None:
        """
        双层卷积类构造函数
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param norm_type: 归一化层类
        :param norm_args: 归一化层需要特殊处理的参数
        :return:
        """
        super(DoubleConv, self).__init__()
        if norm_type is None: norm_type = nn.Identity
        if norm_args is None: norm_args = {}
        mid_channels = out_channels // 2
        bias = False if norm_type != nn.Identity else True

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=bias),
            norm_type(num_features=mid_channels, **norm_args),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=bias),
            norm_type(num_features=out_channels, **norm_args),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        :param x: 传入张量
        :return: 计算结果
        """
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DownSample(nn.Module):
    """
    编码器，执行下采样操作
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: Module=None,
        norm_args: dict=None
    ) -> None:
        """
        编码器类构造函数
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param norm_type: 归一化层类
        :param norm_args: 归一化层需要特殊处理的参数
        :return:
        """
        super(DownSample, self).__init__()
        if norm_type is None: norm_type = nn.Identity
        if norm_args is None: norm_args = {}

        self.down = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels, norm_type=norm_type, norm_args=norm_args)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        :param x: 传入张量
        :return: 计算结果
        """
        x = self.down(x)
        return x


class UpSample(nn.Module):
    """
    解码器，执行上采样操作
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        norm_type: Module=None,
        norm_args: dict=None
    ) -> None:
        """
        编码器类构造函数
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param skip_channels: 传递张量编码器的通道数
        :param norm_type: 归一化层类
        :param norm_args: 归一化层需要特殊处理的参数
        :return:
        """
        super(UpSample, self).__init__()
        if norm_type is None: norm_type = nn.Identity
        if norm_args is None: norm_args = {}

        self.up = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels, norm_type=norm_type, norm_args=norm_args)

    def forward(self, x: Tensor, encoder: Tensor) -> Tensor:
        """
        前向传播
        :param x: 传入张量
        :param encoder: 从编码器传来用于连接的张量
        :return: 计算结果
        """
        x = self.up(x)
        x = torch.cat([encoder, x], dim=1 if len(x.shape) == 5 else 0)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    """
    输出层，包含一个卷积层
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ) -> None:
        """
        输出层类构造函数
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :return:
        """
        super(OutConv, self).__init__()
        self.out = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        :param x: 传入张量
        :return: 计算结果
        """
        x = self.out(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = UNet3D(
        in_channels=1,
        out_channels=2,
        n_channels=[64, 128, 256, 512, 1024],
        norm_type=nn.InstanceNorm3d,
        norm_args={ 'affine': True }
    ).to(device)
    print(model)

    tensor = torch.randn([8, 1, 128, 128, 64]).to(device)
    output = model(tensor)
    print(output.shape)