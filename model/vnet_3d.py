import torch
from torch import nn

# 类型注解所用库
from torch import Tensor
from torch.nn import Module, ModuleList
from typing import Sequence, Literal

class VNet3D(nn.Module):
    """
    3D V-Net类，总体框架分为编码部分和解码部分
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_channels: Sequence[int]=None,
        layer_nums: Sequence[int]=None,
        norm_type: Module=None,
        norm_args: dict=None
    ) -> None:
        """
        3D V-Net类构造函数
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param n_channels: 中间层通道数
        :param layer_nums: 中间卷积层数
        :param norm_type: 归一化层类
        :param norm_args: 归一化层需要特殊处理的参数
        :return:
        """
        super(VNet3D, self).__init__()
        if norm_type is None: norm_type = nn.Identity
        if norm_args is None: norm_args = {}
        if n_channels is None: n_channels = [16, 32, 64, 128, 256]
        if layer_nums is None: layer_nums = [1, 2, 3, 3, 3]
        assert len(n_channels) == len(layer_nums)
        assert len(n_channels) >= 2

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_channels = list(n_channels)
        self.layer_nums = list(layer_nums)
        self.norm_type = norm_type
        self.norm_args = norm_args

        self.in_block = InputBlock(in_channels, n_channels[0], norm_type=norm_type, norm_args=norm_args)
        self.encoder = self.__build_layers('encoder')
        self.decoder = self.__build_layers('decoder')
        self.out_block = OutputBlock(n_channels[0] * 2, out_channels, norm_type=norm_type, norm_args=norm_args)

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        :param x: 传入张量
        :return: 传出张量
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

    def __build_layers(self, build_mode: Literal['encoder', 'decoder']) -> ModuleList:
        """
        构建编码器/解码器
        :param build_mode: 构建模式
        :return: 构建的编码器/解码器
        """
        assert build_mode in ['encoder', 'decoder']
        if build_mode == 'encoder':
            layer_nums = self.layer_nums[1:]
            in_channels = self.n_channels[:len(layer_nums)]
            out_channels = self.n_channels[-len(layer_nums):]
        else:
            layer_nums = list(reversed(self.layer_nums[:-1]))
            # n_channels:   [256,   256,    128,    64,     32,     16]
            # in_channels:  [256,   256,    128,    64]
            # out_channels: [128,   64,     32,     16]
            n_channels = list(reversed(self.n_channels + [self.n_channels[-1]]))
            in_channels = n_channels[:len(layer_nums)]
            out_channels = n_channels[-len(layer_nums):]

        layers = nn.ModuleList()
        for in_ch, out_ch, layer_num in zip(in_channels, out_channels, layer_nums):
            if build_mode == 'encoder':
                layers.append(DownSample(in_ch, out_ch, layer_num, norm_type=self.norm_type, norm_args=self.norm_args))
            else:
                layers.append(UpSample(in_ch, out_ch, out_ch, layer_num, norm_type=self.norm_type, norm_args=self.norm_args))
        return layers


class ConvBlock(nn.Module):
    """
    卷积块
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        layer_num: int,
        norm_type: Module=None,
        norm_args: dict=None
    ) -> None:
        """
        卷积块构造函数
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param layer_num: 卷积层数
        :param norm_type: 归一化层类
        :param norm_args: 归一化层需要特殊处理的参数
        :return:
        """
        super(ConvBlock, self).__init__()
        if norm_type is None: norm_type = nn.Identity
        if norm_args is None: norm_args = {}
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layer_num = layer_num
        self.norm_type = norm_type
        self.norm_args = norm_args
        self.__build_layers()

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        :param x: 输入张量
        :return: 输出张量
        """
        for conv in self.conv:
            x = conv(x)
        return x

    def __build_layers(self) -> None:
        """
        构造中间卷积层
        :return:
        """
        self.conv = nn.ModuleList()
        in_channels = self.in_channels
        out_channels = self.out_channels
        bias = False if self.norm_type != nn.Identity else True
        for i in range(self.layer_num):
            conv_block = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 5, 1, 2, bias=bias),
                self.norm_type(out_channels, **self.norm_args)
            )
            if i + 1 < self.layer_num:
                conv_block.append(nn.PReLU())
            self.conv.append(conv_block)
            in_channels = out_channels


class InputBlock(nn.Module):
    """
    传入块
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: Module=None,
        norm_args: dict=None
    ) -> None:
        """
        传入块构造函数
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param norm_type: 归一化层类
        :param norm_args: 归一化层需要特殊处理的参数
        :return:
        """
        super(InputBlock, self).__init__()
        if norm_type is None: norm_type = nn.Identity
        if norm_args is None: norm_args = {}
        self.conv = ConvBlock(in_channels, out_channels, 1, norm_type=norm_type, norm_args=norm_args)
        self.act = nn.PReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        :param x: 传入张量
        :return: 传出张量
        """
        x_out = self.conv(x)
        x = torch.add(x, x_out)
        x = self.act(x)
        return x


class DownSample(nn.Module):
    """
    编码器
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        layer_num: int,
        norm_type: Module=None,
        norm_args: dict=None
    ) -> None:
        """
        编码器构造函数
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param layer_num: 卷积层数
        :param norm_type: 归一化层类
        :param norm_args: 归一化层需要特殊处理的参数
        :return:
        """
        super(DownSample, self).__init__()
        if norm_type is None: norm_type = nn.Identity
        if norm_args is None: norm_args = {}
        bias = False if norm_type != nn.Identity else True

        self.down = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 2, 2, 0, bias=bias),
            norm_type(out_channels, **norm_args),
            nn.PReLU()
        )
        self.conv = ConvBlock(out_channels, out_channels, layer_num, norm_type=norm_type, norm_args=norm_args)
        self.act = nn.PReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        :param x: 输入张量
        :return: 输出张量
        """
        x = self.down(x)
        x = torch.add(x, self.conv(x))
        x = self.act(x)
        return x


class UpSample(nn.Module):
    """
    解码器，注意输出张量通道数为out_channels + skip_channels
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        layer_num: int,
        norm_type: Module=None,
        norm_args: dict=None
    ) -> None:
        """
        解码器构造函数
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param skip_channels: 跳跃连接通道数
        :param layer_num: 卷积层数
        :param norm_type: 归一化层类
        :param norm_args: 归一化层需要特殊处理的参数
        :return:
        """
        super(UpSample, self).__init__()
        if norm_type is None: norm_type = nn.Identity
        if norm_args is None: norm_args = {}
        bias = False if norm_type != nn.Identity else True

        self.up = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, 2, 2, 0, bias=bias),
            norm_type(out_channels, **norm_args),
            nn.PReLU()
        )
        true_out_channels = out_channels + skip_channels
        self.conv = ConvBlock(true_out_channels, true_out_channels, layer_num, norm_type=norm_type, norm_args=norm_args)
        self.act = nn.PReLU()

    def forward(self, x: Tensor, x_skip: Tensor) -> Tensor:
        """
        前向传播
        :param x: 传入张量
        :param x_skip: 跳跃连接传入张量
        :return: 传出张量
        """
        x = self.up(x)
        x_cat = torch.cat((x, x_skip), dim=1)
        x = self.conv(x_cat)
        x = torch.add(x, x_cat)
        x = self.act(x)
        return x


class OutputBlock(nn.Module):
    """
    输出块
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: Module=None,
        norm_args: dict=None
    ) -> None:
        """
        输出块构造函数
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param norm_type: 归一化层类
        :param norm_args: 归一化层需要特殊处理的参数
        :return:
        """
        super(OutputBlock, self).__init__()
        if norm_type is None: norm_type = nn.Identity
        if norm_args is None: norm_args = {}

        self.conv1 = ConvBlock(in_channels, out_channels, 1, norm_type=norm_type, norm_args=norm_args)
        self.act = nn.PReLU()
        self.conv2 = ConvBlock(out_channels, out_channels, 1, norm_type=norm_type, norm_args=norm_args)

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        :param x: 传入张量
        :return: 传出张量
        """
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    vnet_3d = VNet3D(
        in_channels=1,
        out_channels=2,
        n_channels=None,
        layer_nums=None,
        norm_type=nn.InstanceNorm3d,
        norm_args={'affine': True}
    ).to(device)
    print(vnet_3d)

    tensor: Tensor = torch.randn([8, 1, 128, 128, 64]).to(device)
    output: Tensor = vnet_3d(tensor)
    print(output.shape)