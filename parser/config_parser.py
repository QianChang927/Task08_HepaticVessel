import os
import re
import sys
import json
import zlib
import inspect

# 类型注解所用库
from typing import Protocol, Literal, Sequence, Mapping
from functools import partial
from torch import device
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from argparse import Namespace
from data import DataReaderMSD
from train import Trainer, EarlyStopping

# 类型注解相关协议
class Stringfiable(Protocol):
    def __str__(self) -> str: ...

class Listifiable(Protocol):
    def tolist(self) -> list: ...

class InnerClass(Protocol):
    def __class__(self): ...

class ConfigParser:
    """
    配置文件解析类，用于解析各种运行配置，并保存为JSON文件
    """
    def __init__(
        self,
        save_dir: str,
        device: device,
        args: Namespace=None,
        data_reader: DataReaderMSD=None,
        model: Module=None,
        loss_fn: Module=None,
        optimizer: Optimizer=None,
        scheduler: LRScheduler=None,
        trainer: Trainer=None,
        early_stopping: EarlyStopping=None
    ) -> None:
        # 初始化参数
        self.save_dir = save_dir
        self.device = device
        self.args = args
        self.data_reader = data_reader
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.trainer = trainer
        self.early_stopping = early_stopping
        self.config_path = os.path.join(self.save_dir, 'config.json')
        self.save_path = os.path.join(self.save_dir, 'config')
        os.makedirs(self.save_path, exist_ok=True)
        # 解析config
        self.__parse_args()
        self.__parse_transforms()
        self.__parse_model()
        for __object in ['loss_fn', 'optimizer', 'scheduler', 'trainer', 'early_stopping']:
            self.__parse_others(__object)
        # 写入json文件
        for __object in ['transforms', 'model', 'loss_fn', 'optimizer', 'scheduler']:
            self.__save_config(__object, True)
        for __object in ['trainer', 'early_stopping', 'args']:
            self.__save_config(__object, False)

    def __parse_args(self) -> None:
        """
        解析Namespace内的属性并保存为config
        :return:
        """
        self.config_args = {'system': sys.platform, 'device': self.device.type}
        args_config = vars(self.args)
        # 特殊模型处理
        if args_config.get('model') in ['VNetMONAI']:
            args_config['norm_layer'] = 'BatchNorm'
        self.config_args.update(args_config)

    def __parse_transforms(self) -> None:
        """
        解析data_reader内的transforms结构
        :return:
        """
        if self.data_reader is None: return
        self.config_transforms = {'train': [], 'valid': [], 'test': []}
        for key in self.config_transforms.keys():
            if not hasattr(self.data_reader, f'{key}_transforms'): continue
            transforms = getattr(self.data_reader, f'{key}_transforms').transforms
            for transform in transforms:
                transform_keys = list(inspect.signature(type(transform)).parameters)
                transform_dict = ConfigParser.__get_total_dict(transform.__dict__)
                transform_name = ConfigParser.get_obj_name(transform)
                transform_info = {}
                for k in transform_keys:
                    dict_key = ConfigParser.__search_key(transform_dict, k)
                    if dict_key is None: continue
                    transform_info[dict_key] = transform_dict[dict_key]
                self.config_transforms[key].append({transform_name: transform_info})

    def __parse_model(self):
        """
        解析模型配置信息
        :return:
        """
        if self.model is None: return
        model_name = ConfigParser.get_obj_name(self.model)
        self.config_model = {model_name: str(self.model)}

    def __parse_others(self, parse_mode: Literal['loss_fn', 'optimizer', 'scheduler', 'trainer', 'early_stopping']) -> None:
        """
        解析其余的配置信息
        :param parse_mode: 解析的对象
        :return:
        """
        assert parse_mode in ['loss_fn', 'optimizer', 'scheduler', 'trainer', 'early_stopping']
        config_obj = getattr(self, f'{parse_mode}')
        if config_obj is None: return
        config_keys = list(inspect.signature(type(config_obj)).parameters)
        config_dict = ConfigParser.__get_total_dict(config_obj.__dict__)
        config_name = ConfigParser.get_obj_name(config_obj)
        config_info = {}
        for key in config_keys:
            if hasattr(self, key): continue
            dict_key = ConfigParser.__search_key(config_dict, key)
            if dict_key is None: continue
            config_info[dict_key] = config_dict[dict_key]
        setattr(self, f'config_{parse_mode}', {config_name: config_info})

    def __save_config(self, key_name: str, add_to_config_json: bool=True) -> None:
        """
        将配置字典写入json文件中
        :param key_name: 需要写入的文件，同时作为保存到config.json的键名
        :param add_to_config_json: 是否添加到config.json中
        :return:
        """
        def __serialize(obj: type | Listifiable | Stringfiable) -> list | str:
            """
            处理JSON不可序列化的非基本数据类型
            :param obj: 需要序列化的对象
            :return: 转化后可序列化的对象
            """
            if type(obj).__name__ == 'type':
                return obj.__module__ + '.' + obj.__name__
            elif hasattr(obj, 'tolist') and callable(obj.tolist):
                obj_list = obj.tolist()
                return ['NaN' if isinstance(item, float) and item != item else item for item in obj_list]
            elif hasattr(obj, '__str__') and callable(obj.__str__):
                return str(obj)
            raise TypeError(f"Type {type(obj)} is not JSON serializable")

        def __dict_to_hash(obj: dict) -> str:
            """
            字典转哈希值
            :param obj: 需要转化成哈希值的字典
            :return: 转化后的哈希值
            """
            json_str = json.dumps(obj, sort_keys=True, allow_nan=False, default=__serialize)
            crc32_hash = zlib.crc32(json_str.encode('utf-8'))
            return format(crc32_hash & 0xFFFFFFFF, '08X')

        if not hasattr(self, f'config_{key_name}'): return
        config_dict = getattr(self, f'config_{key_name}')
        config_path = os.path.join(self.save_path, f'{key_name}.json')

        key_is_args = key_name == 'args'
        if key_is_args: config_path = self.config_path
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=4, allow_nan=False, default=__serialize)

        if not key_is_args and add_to_config_json:
            hash_hex_str = __dict_to_hash(config_dict)
            self.config_args.update({f'{key_name}-obj': hash_hex_str})

    @staticmethod
    def get_obj_name(obj: partial | InnerClass) -> str:
        """
        获取类名
        :param obj: 需要获取名字的类
        :return: 类名
        """
        if isinstance(obj, partial):
            ori_class = obj.func
            return ori_class.__name__
        else:
            return obj.__class__.__name__

    @staticmethod
    def __get_total_dict(obj: dict) -> dict:
        """
        获取嵌套对象的所有字典信息，若嵌套对象含有相同属性，只保留更高层对象的属性
        :param obj: 需要获取的顶层字典
        :return: 展开后的所有字典
        """
        def __update(origin_dict: dict, nested_dict: dict) -> None:
            """
            更新字典
            :param origin_dict: 需要更新的字典
            :param nested_dict: 新字典
            :return:
            """
            for key, value in nested_dict.items():
                if key in origin_dict: continue
                origin_dict.update({key: value})

        def __flatten(origin_dict: dict) -> dict:
            """
            扁平化字典
            :param origin_dict: 需要扁平化的字典
            :return: 扁平化后的字典
            """
            for key, value in origin_dict.copy().items():
                if isinstance(value, Mapping):
                    origin_dict.pop(key)
                    if len(value) > 0: origin_dict.update(__flatten(value))
            return origin_dict

        nested = re.compile(r'.*0[xX][0-9a-fA-F]+', flags=re.DOTALL)
        obj_flattened = __flatten(obj.copy())
        result = obj_flattened.copy()

        for key, value in obj_flattened.items():
            value_str = str(value)
            value_seq = [value] if not isinstance(value, Sequence) else value

            if not nested.match(value_str): continue
            for v in value_seq:
                nest_dict = ConfigParser.__get_total_dict(v.__dict__)
                __update(result, nest_dict)
            result.pop(key)
        return result

    @staticmethod
    def __search_key(origin_dict, key) -> str:
        """
        在字典中搜索key可能的变化值
        :param origin_dict: 需要搜索的字典
        :param key: 需要搜索的键
        :return: 键的可能新值
        """
        try:
            _ = origin_dict[key]
            return key
        except KeyError:
            key_similar = None
            for dict_key in origin_dict.keys():
                if key not in dict_key: continue
                key_similar = dict_key
                break
            return key_similar


if __name__ == '__main__':
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config_parser = ConfigParser(save_dir='../checkpoint', device=device)