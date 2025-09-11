import argparse
from argparse import ArgumentParser, Namespace

class ArgParser:
    """
    参数解析类
    """
    def __init__(self) -> None:
        """
        参数解析类构造函数，用于生成ArgumentParser便于后续处理
        :return:
        """
        def __parse_tuple(value: str) -> tuple:
            """
            私有函数，为ArgumentParser解析元组
            :param value: 传入解析的字符串，形式要求: "(number1, number2, ...)" 或 "number1, number2, ..."
            :return: 解析后的元组
            """
            try:
                value = value.strip()
                if value.startswith('(') and value.endswith(')'):
                    value = value[1:-1]
                value = value.split(',')
                value = [s.strip() for s in value]
                return tuple(map(int, value))
            except:
                raise argparse.ArgumentTypeError(f'Invalid tuple forms: {value}. Use "(1, 2, 3)" or "1, 2, 3"')

        parser = ArgumentParser()

        # 文件相关设置
        parser.add_argument('--root_dir', type=str, help='数据集所在位置')
        parser.add_argument('--save_dir', type=str, default='./checkpoint', help='模型、配置及进度保存位置(默认为`./checkpoint`)')

        # DataReader相关设置
        parser.add_argument('--remains', type=int, default=None, help='训练-验证集的数据个数，该参数为None时选取所有数据作为训练-验证集')
        parser.add_argument('--val_scale', type=float, default=0.25, help='训练-验证集中验证集所占的比例(默认为0.25)，训练/验证集中均至少含有一个数据')
        parser.add_argument('--num_workers_cache', type=int, default=4, help='CacheDataset加载数据的进程数')
        parser.add_argument('--num_workers_loader', type=int, default=4, help='DataLoader加载数据的进程数')
        parser.add_argument('--crop_size', type=__parse_tuple, default="(128, 128, 64)", help='数据预处理中填充/Resize后的张量大小')
        parser.add_argument('--samp_size', type=__parse_tuple, default="(128, 128, 64)", help='数据预处理中随机采样后patch的张量大小')

        # Model相关设置
        parser.add_argument('--model', type=str, choices=['UNet3D', 'VNet3D', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152', 'UNetMONAI', 'VNetMONAI'], help='需要调用的模型名')
        parser.add_argument('--judge_channel', type=int, default=-1, help='损失函数训练通道选择')
        parser.add_argument('--in_channels', type=int, default=1, help='模型输入通道')
        parser.add_argument('--out_channels', type=int, default=2, help='模型输出通道')
        parser.add_argument('--n_channels', type=__parse_tuple, default="(16, 32, 64, 128, 256)", help='模型中间层的通道数设置(默认为"(16, 32, 64, 128, 256)")')
        parser.add_argument('--norm_layer', type=str, choices=['BatchNorm', 'InstanceNorm', 'None'], default='BatchNorm', help='归一化层的类型(默认为`BatchNorm`)')
        parser.add_argument('--resnet_type', type=str, choices=['A', 'B'], default='B', help='ResNet网络跳跃连接类型')

        # Optimizer相关设置
        parser.add_argument('--optimizer', type=str, choices=['Adam', 'SGD'], default='Adam', help='优化器的类型(默认为`Adam`)')
        parser.add_argument('--lr', type=float, default=1e-03, help='优化器的初始学习率(默认为1e-03)')

        # Trainer相关设置
        parser.add_argument('--epochs', type=int, default=1000, help='训练轮次(默认为1000)')
        parser.add_argument('--batch', type=int, default=2, help='训练过程中DataLoader的batch_size')
        parser.add_argument('--batch_other', type=int, default=1, help='验证/测试过程中DataLoader的batch_size')
        parser.add_argument('--shuffle', action='store_true', help='是否启用随机化')
        parser.add_argument('--seed', type=int, default=0, help='shuffle为False时生效，用于固定随机数种子(默认为0)')
        parser.add_argument('--roi_size', type=__parse_tuple, default="(128, 128, 64)", help='验证过程中滑动窗口的大小')
        parser.add_argument('--sw_batch', type=int, default=2, help='验证过程中滑动窗口的batch_size')
        parser.add_argument('--overlap', type=float, default=0.25, help='验证过程中滑动窗口的重叠率')

        self.parser = parser

    def parse_args(self) -> Namespace:
        """
        解析参数
        :return: 解析出的可读取对象
        """
        return self.parser.parse_args()

if __name__ == '__main__':
    parser = ArgParser()
    args = parser.parse_args()