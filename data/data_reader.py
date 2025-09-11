import os
import json
import glob
import random
import inspect

from monai import transforms
from monai.data import CacheDataset, DataLoader

# 类型注解所用库
from typing import Literal
from argparse import Namespace
from monai.transforms import Compose

class DataReaderMSD:
    """
    用于读取MSD相关数据集的DataReader类
    """
    def __init__(
        self,
        root_dir: str,
        args: Namespace=None,
        remains: int=None,
        val_scale: float=0.1,
        shuffle: bool=False,
        num_workers_cache: int=4,
        num_workers_loader: int=4,
        batch: int=2,
        batch_other: int=1,
        crop_size: tuple = (64, 64, 64),
        samp_size: tuple = (64, 64, 64),
        train_transforms: Compose = None,
        valid_transforms: Compose = None,
        test_transforms: Compose = None
    ) -> None:
        """
        读取MSD相关数据集的类构造函数
        :param root_dir: 数据集所在位置
        :param args: 命令行传入参数
        :param remains: 训练-验证集的数据个数，该参数为None时选取所有数据作为训练-验证集
        :param val_scale: 训练-验证集中验证集所占的比例，训练/验证集中均至少含有一个数据
        :param shuffle: 是否启用随机化
        :param num_workers_cache: CacheDataset加载数据的进程数
        :param num_workers_loader: DataLoader加载数据的进程数
        :param batch: 训练过程中DataLoader的batch_size
        :param batch_other: 验证/测试过程中DataLoader的batch_size
        :param crop_size: 数据预处理中填充/Resize后的张量大小
        :param samp_size: 数据预处理中随机采样后patch的张量大小
        :param train_transforms: 训练预处理
        :param valid_transforms: 验证预处理
        :param test_transforms: 测试预处理
        :return:
        """
        # 初始化参数
        self.root_dir = root_dir
        self.args = args
        self.remains = remains
        self.val_scale = val_scale
        self.shuffle = shuffle
        self.num_workers_cache = num_workers_cache
        self.num_workers_loader = num_workers_loader
        self.batch = batch
        self.batch_other = batch_other
        self.crop_size = crop_size
        self.samp_size = samp_size
        # ArgumentParser有效时更新参数
        if isinstance(self.args, Namespace):
            self.__update_params()
        # 检查参数合法性
        self.__check_available()
        # 生成文件路径字典列表
        self.config = self.__get_config()
        self.train_valid_files = self.__get_files('train')
        _keys = list(self.train_valid_files[0].keys())
        # 参数功能实装
        self.train_files = []
        self.valid_files = []
        self.test_files = self.__get_files('test', _key=_keys[0])
        self.__enable_params()
        # 生成数据预处理Transforms
        _transforms = TransformsMSD(keys=_keys, keys_test=_keys[0], crop_size=self.crop_size, samp_size=self.samp_size)
        self.train_transforms = train_transforms if train_transforms is not None else _transforms.train_transforms
        self.valid_transforms = valid_transforms if valid_transforms is not None else _transforms.valid_transforms
        self.test_transforms = test_transforms if test_transforms is not None else _transforms.test_transforms

    def get_dataloader(self, loader_type: Literal['train', 'valid', 'test']) -> DataLoader:
        """
        生成DataLoader
        :param loader_type: 需要生成的DataLoader类型
        :return: 生成的DataLoader
        """
        assert loader_type in ['train', 'valid', 'test']
        if not hasattr(self, f'{loader_type}_cache'):
            self.__get_cache(loader_type)
        _dataset = getattr(self, f'{loader_type}_cache')
        _batch_size = self.batch if loader_type == 'train' else self.batch_other
        return DataLoader(dataset=_dataset, num_workers=self.num_workers_loader, batch_size=_batch_size, shuffle=self.shuffle)

    def __update_params(self) -> None:
        """
        通过self.args更新已有的类属性成员
        :return:
        """
        params = inspect.signature(self.__init__).parameters
        params = list(params.keys())
        for param in params:
            if not hasattr(self.args, param): continue
            setattr(self, param, getattr(self.args, param))

    def __check_available(self) -> None:
        """
        检查属性合法性，属性非法则抛出异常
        :return:
        """
        assert (isinstance(self.remains, int) and self.remains >= 1) or self.remains is None
        assert isinstance(self.val_scale, float) and 0 <= self.val_scale <= 1
        assert isinstance(self.num_workers_cache, int) and self.num_workers_cache >= 1
        assert isinstance(self.num_workers_loader, int) and self.num_workers_loader >= 1
        assert isinstance(self.batch, int) and self.batch >= 1
        assert isinstance(self.batch_other, int) and self.batch_other >= 1

    def __enable_params(self) -> None:
        """
        使参数生效
        :return:
        """
        if self.shuffle:
            random.shuffle(self.train_valid_files)
            random.shuffle(self.test_files)

        if self.remains:
            self.train_valid_files = self.train_valid_files[:self.remains]

        if len(self.train_valid_files) <= 1:
            raise RuntimeError('train and valid files should be at least 2 file')

        split_nums = int(len(self.train_valid_files) * self.val_scale)
        if split_nums < 1: split_nums = 1

        self.train_files = self.train_valid_files[:-split_nums]
        self.valid_files = self.train_valid_files[-split_nums:]

    def __get_config(self) -> dict:
        """
        通过读取MSD数据集中*.json文件获取数据集相关信息
        :return: 以字典形式保存的*.json文件
        """
        json_pattern = os.path.join(self.root_dir, '*.json')
        json_path = next(glob.iglob(json_pattern))
        with open(json_path) as f:
            config = json.load(f)
        return config

    def __get_files(self, file_type: Literal['train', 'test'], _key: str='image') -> list:
        """
        获取*.json中数据路径信息列表
        :param file_type: 需要获取的数据类型
        :param _key: 原始路径信息列表不为字典列表时的键
        :return: 解析后的路径信息列表
        """
        assert file_type in ['train', 'test']
        _key_map = {'train': 'training', 'test': 'test'}
        config_list: list = self.config[_key_map[file_type]]
        for i, config in enumerate(config_list):
            if not isinstance(config, dict):
                config_list[i] = {_key: os.path.join(self.root_dir, config)}
                continue
            for key, value in config.items():
                config_list[i][key] = os.path.join(self.root_dir, value)
        return config_list

    def __get_cache(self, cache_type: Literal['train', 'valid', 'test']) -> None:
        """
        生成CacheDataset以节省数据加载时间
        :param cache_type: 需要生成的缓存数据集类型
        :return:
        """
        assert cache_type in ['train', 'valid', 'test']
        _transform = getattr(self, f'{cache_type}_transforms')
        assert _transform is not None
        _data = getattr(self, f'{cache_type}_files')
        assert _data is not None and len(_data) > 0
        setattr(self, f'{cache_type}_cache', CacheDataset(data=_data, transform=_transform, num_workers=self.num_workers_cache))


class TransformsMSD:
    """
    适用于DataReaderMSD的数据预处理transforms
    """
    def __init__(
        self,
        keys: list,
        keys_test: str,
        crop_size: tuple,
        samp_size: tuple
    ) -> None:
        """
        生成transforms的类构造函数
        :param keys: 训练/验证集的键
        :param keys_test: 测试集的键
        :param crop_size: 数据预处理中填充/Resize后的张量大小
        :param samp_size: 数据预处理中随机采样后patch的张量大小
        """
        self.keys = keys
        self.keys_test = keys_test
        self.crop_size = crop_size
        self.samp_size = samp_size
        self.train_transforms = None
        self.valid_transforms = None
        self.test_transforms = None
        self._min = -100
        self._max = 200
        self._pixdim = (0.8, 0.8, 1.5)
        self.__create_transforms()

    def __create_transforms(self) -> None:
        """
        生成transforms
        :return:
        """
        self.train_transforms = Compose([
            transforms.LoadImaged(keys=self.keys),
            transforms.EnsureChannelFirstd(keys=self.keys),
            transforms.Orientationd(keys=self.keys, axcodes='RAS'),
            transforms.Spacingd(
                keys=self.keys,
                pixdim=self._pixdim,
                mode=('bilinear', 'nearest')
            ),
            transforms.ScaleIntensityRanged(
                keys=self.keys[0],
                a_min=self._min,
                a_max=self._max,
                b_min=0.0,
                b_max=1.0,
                clip=True
            ),
            transforms.Resized(
                keys=self.keys,
                spatial_size=self.crop_size
            ),

            transforms.RandCropByPosNegLabeld(
                keys=self.keys,
                image_key=self.keys[0],
                label_key=self.keys[-1],
                spatial_size=self.samp_size,
                pos=1,
                neg=1,
                num_samples=4,
                image_threshold=0.0
            ),
            transforms.RandFlipd(
                keys=self.keys,
                spatial_axis=[0, 1, 2],
                prob=0.5
            ),
            transforms.RandRotate90d(
                keys=self.keys,
                prob=0.5
            ),
            transforms.RandGaussianNoised(
                keys=self.keys,
                prob=0.2
            ),
            transforms.ToTensord(
                keys=self.keys
            ),
            transforms.EnsureTyped(
                keys=self.keys,
                data_type='tensor'
            )
        ])

        self.valid_transforms = Compose([
            transforms.LoadImaged(keys=self.keys),
            transforms.EnsureChannelFirstd(keys=self.keys),
            transforms.Orientationd(keys=self.keys, axcodes='RAS'),
            transforms.Spacingd(
                keys=self.keys,
                pixdim=self._pixdim,
                mode=('bilinear', 'nearest')
            ),
            transforms.ScaleIntensityRanged(
                keys=self.keys[0],
                a_min=self._min,
                a_max=self._max,
                b_min=0.0,
                b_max=1.0,
                clip=True
            ),
            transforms.Resized(
                keys=self.keys,
                spatial_size=self.crop_size
            )
        ])

        self.test_transforms = Compose([
            transforms.LoadImaged(keys=self.keys),
            transforms.EnsureChannelFirstd(keys=self.keys),
            transforms.Orientationd(keys=self.keys, axcodes='RAS'),
            transforms.Spacingd(
                keys=self.keys,
                pixdim=self._pixdim,
                mode=('bilinear', 'nearest')
            ),
            transforms.ScaleIntensityRanged(
                keys=self.keys[0],
                a_min=self._min,
                a_max=self._max,
                b_min=0.0,
                b_max=1.0,
                clip=True
            ),
            transforms.Resized(
                keys=self.keys,
                spatial_size=self.crop_size
            )
        ])


if __name__ == '__main__':
    from parser import ArgParser
    parser = ArgParser()
    args = parser.parse_args()
    data_reader = DataReaderMSD(root_dir='../../Task_Dataset/Task03_Liver', args=args)
    train_loader = data_reader.get_dataloader('train')
    valid_loader = data_reader.get_dataloader('valid')
