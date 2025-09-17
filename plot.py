import os
import math
import json
import glob
import torch
import numpy as np

from collections.abc import Iterable
from matplotlib import pyplot as plt
from datetime import datetime

# 类型注解所用库
from typing import Callable, Sequence, Mapping, Any

OMIT_DICT = {
    'BatchNorm': 'BN',
    'InstanceNorm': 'IN'
}

class Plot:
    """
    曲线绘制类
    """
    def __init__(
        self,
        root_dir: str,
        process_seq: Sequence[dict] | dict=None,
        sign_filter: Mapping[str, Sequence[str]] =None,
        channel: int | Sequence[int]=1,
        plot_col: int=3,
        file_ext: str='.pt',
        file_seg: str='_',
        fig_size: tuple[int, int]=(20, 6)
    ) -> None:
        """
        绘制类构造函数
        :param root_dir: log根目录位置
        :param process_seq: log根目录预处理序列，内部元素形如{'func': Callable, 'args': Sequence}
        :param sign_filter: log图例标注过滤序列，{'mode': 'omit/select', 'args': Sequence'}
        :param channel: 评判标准含多维数据时取第几个维度
        :param plot_col: 子图每列张数
        :param file_ext: 曲线文件后缀
        :param file_seg: 曲线文件前缀
        :param fig_size: 绘制图像大小
        """
        # 参数初始化
        self.root_dir = root_dir
        self.process_seq = process_seq
        self.sign_filter = sign_filter
        self.channel = channel
        self.plot_col = plot_col
        self.file_ext = file_ext if '*' in file_ext else f"*{file_ext}"
        self.file_seg = file_seg
        self.fig_size = fig_size
        # 获取日志文件夹
        self.__get_log_dirs()
        # 获取标签字典
        self.__get_log_signs()
        # 生成子图布局
        self.__load_log_info()

    def plot(self):
        assert len(self.log_dirs)
        for log_dir in self.log_dirs:
            plt.figure('Criteria', self.fig_size)
            row, col = self.log_subplot[log_dir]
            plot_dict = self.log_content[log_dir]

            subplot_index = 1
            subplot_dict = {}

            for key, value in plot_dict.items():
                if key not in subplot_dict:
                    subplot_dict[key] = subplot_index
                    subplot_index += 1

                index = subplot_dict[key]
                plt.subplot(row, col, index)

                plt.title(key)
                plt.xlabel('epoch')

                np_value = np.array(value)
                np_shape = np_value.shape

                need_plot = True
                if len(np_shape) > 1:
                    if isinstance(self.channel, int):
                        self.channel = min(self.channel, np_shape[-1])
                        value = np_value[:, self.channel].tolist()
                    else:
                        need_plot = False
                        for channel in self.channel:
                            channel = min(channel, np_shape[-1])
                            value = np_value[:, channel].tolist()
                            plt.plot([x + 1 for x in range(len(value))], value, label=self.log_signs[log_dir] + f'_CHANNEL_{channel + 1}')
                            plt.grid(True)
                            plt.legend()

                if need_plot:
                    plt.plot([x + 1 for x in range(len(value))], value, label=self.log_signs[log_dir])
                    plt.grid(True)
                    plt.legend()

                if 'dice' in key: print(f'{self.log_signs[log_dir]}-best {key}: {max(value)}')
                elif 'loss' in key: print(f'{self.log_signs[log_dir]}-best {key}: {min(value)}')
        plt.show()

    def __get_log_dirs(self) -> None:
        """
        生成需要处理的日志文件夹位置
        :return:
        """
        self.log_dirs = os.listdir(self.root_dir)
        if self.process_seq is None: return
        if not isinstance(self.process_seq, Sequence): self.process_seq = [self.process_seq]
        for preprocess in self.process_seq:
            process_func: Callable = preprocess.get('func', None)
            process_args: Sequence = preprocess.get('args', None)
            if process_func is None: continue
            self.log_dirs = process_func(self.root_dir, self.log_dirs, process_args)

    def __get_log_signs(self) -> None:
        """
        生成每个日志文件对应的标签图例
        :return:
        """
        '''
        思路：
        1.  log_dir1 <=> { 'attr1': 'value1', 'attr2': 'value2' } -> { 'attr1': { 'value1': ['log_dir1', ...] }, 'attr2': { 'value2': ['log_dir1', ...] } }
        2.  统计attr*中键的个数
            1)  若不唯一，则differ_dict[log_dir*]['attr*'] = 'value*'
            2)  若唯一，则differ_dict忽略attr*
        3.  统计attr*中['log_dir1', ...]的总长度，比较其与len(log_dirs)的大小
            1)  若不一致，则differ_dict[log_dir*]['attr*'] = 'value*'
            2)  若一致，则differ_dict忽略attr*
        4.  differ_dict[log_dir*] = dict_to_str(differ_dict[log_dir*])
        '''
        def __dict_to_str(origin_dict: dict) -> str:
            """
            字典转字符串
            :param origin_dict: 需要转换的字典
            :return: 字符串
            """
            if not isinstance(origin_dict, dict):
                return str(origin_dict)
            target_str = ''
            for key, value in origin_dict.items():
                if not isinstance(value, str) and isinstance(value, Iterable):
                    value = '-'.join(map(str, value))
                target_str += f'{OMIT_DICT.get(key, key.upper())}_{OMIT_DICT.get(value, value)}_'
            return target_str[:-1]

        def __skip(key: str) -> bool:
            """
            判断是否略过该键
            :param key: 需要判断的键
            :return: 是否略过
            """
            if not self.sign_filter: return False
            skip_list = self.sign_filter.get('args', [])
            skip_type = self.sign_filter.get('mode', None)
            if not skip_type: return False
            import fnmatch
            match_flag = False
            for skip_key in skip_list:
                if fnmatch.fnmatch(key, skip_key):
                    match_flag = True
                    break
            if skip_type == 'omit': return match_flag
            elif skip_type == 'select': return not match_flag
            else: raise ValueError(f'skip_type: {skip_type} must be `omit` or `select`')

        # 无日志情况排除
        if not len(self.log_dirs): return

        # 初始化图例
        self.log_signs = {}

        # 生成反置键值对
        revert_dict = {}
        for log_dir in self.log_dirs:
            config = Plot.get_config_json(os.path.join(self.root_dir, log_dir))
            key_omit = ['root_dir', 'save_dir']
            for key, value in config.items():
                if key in key_omit: continue
                if __skip(key): continue
                if not isinstance(value, str) and isinstance(value, Sequence):
                    value = '-'.join(map(str, value))
                if key not in revert_dict: revert_dict[key] = {}
                if value not in revert_dict[key]: revert_dict[key][value] = []
                revert_dict[key][value].append(log_dir)

        # 生成图例字典
        for key, value in revert_dict.items():
            if len(value.keys()) == 1:
                log_dir_length = 0
                for log_dirs in value.values():
                    log_dir_length += len(log_dirs)
                if log_dir_length == len(self.log_dirs):
                    continue
            for v, log_dirs in value.items():
                for log_dir in log_dirs:
                    if log_dir not in self.log_signs:
                        self.log_signs[log_dir] = {}
                    self.log_signs[log_dir][key] = v

        # 图例字典值转换
        for key, value in self.log_signs.items():
            self.log_signs[key] = __dict_to_str(value)

        # 防止不同log_dir的__log_signs完全一致
        revert_dict = {}
        for key, value in self.log_signs.items():
            if value not in revert_dict:
                revert_dict[value] = []
            revert_dict[value].append(key)
            if len(revert_dict[value]) > 1:
                self.log_signs[key] += f'_FILE_{key}'

        # 防止为空
        for log_dir in self.log_dirs:
            signs = self.log_signs.get(log_dir, {})
            if not signs: self.log_signs[log_dir] = f'FILE_{log_dir}'

    def __load_log_info(self):
        """
        生成子图布局及日志文件信息
        :return:
        """
        self.log_content = {}
        self.log_subplot = {}

        for log_dir in self.log_dirs:
            plot_files = glob.glob(os.path.join(self.root_dir, log_dir, self.file_ext))
            if not len(plot_files): continue

            self.log_content[log_dir] = {}
            for plot_file in plot_files:
                plot_dict = torch.load(plot_file)
                file_prefix = os.path.basename(plot_file).split(self.file_seg)[0]
                for key, value in plot_dict.items():
                    self.log_content[log_dir][f"{file_prefix} {key}"] = value

            key_nums = len(self.log_content[log_dir])
            self.log_subplot[log_dir] = (math.ceil(key_nums / self.plot_col), min(key_nums, self.plot_col))

    @staticmethod
    def get_config_json(dir_path: str) -> dict:
        """
        获取配置文件config.json内容
        :param dir_path: 文件夹名
        :return: config.json
        """
        config_path = os.path.join(dir_path, 'config.json')
        if not os.path.exists(config_path):
            raise FileExistsError('config.json not exists!')

        config = {}
        with open(config_path, 'r', encoding='UTF-8') as f:
            config = json.load(f)
        return config


class ModifyMethods:
    @staticmethod
    def filter_kwargs(root_dir: str, log_dirs: list[str], args: list[Any]) -> list[str]:
        """
        筛选config中的kwargs，保留/丢弃符合条件的log_dir
        :param root_dir:
        :param log_dirs:
        :param args: 此参数为空时直接返回log_dirs，[(Optional)mode['omit', 'select'], {key1: value1, key2: value2, ...}, {key3: value3, ...}, ...]
        :return: new_log_dirs
        """

        if not args or not isinstance(args, list):
            return log_dirs

        if isinstance(args[0], str):
            mode = args.pop(0).strip().lower()
        else:
            mode = 'select'  # omit: 省略, select: 选取

        if len(args) < 1:
            return log_dirs

        if mode == 'omit':
            filter_init = False
            filter_lambda = lambda x, y: x or y
            new_log_dirs = log_dirs.copy()
        elif mode == 'select':
            filter_init = True
            filter_lambda = lambda x, y: x and y
            new_log_dirs = []
        else:
            raise ValueError('`mode` should be `omit` or `select`')

        def _modify(target_list, element):
            if mode == 'omit':
                target_list.remove(element)
            else:
                target_list.append(element)

        def _judge(_flag) -> bool:
            if mode == 'omit':
                return _flag
            else:
                return not _flag

        for log_dir in log_dirs:
            config = Plot.get_config_json(os.path.join(root_dir, log_dir))
            flag = filter_init

            for kwargs in args:
                if _judge(flag): break
                for key, value in kwargs.items():
                    if _judge(flag): break
                    if isinstance(value, str) or not isinstance(value, Iterable):
                        value = [value]
                    flag = filter_lambda(flag, config.get(key, None) in value)

            if flag:
                _modify(new_log_dirs, log_dir)

        return new_log_dirs

    @staticmethod
    def filter_ctime(root_dir: str, log_dirs: list[str], args: list[Any]) -> list[str]:
        """
        筛选文件创建时间，保留/丢弃符合条件的log_dir
        :param root_dir:
        :param log_dirs:
        :param args: 此参数为空时直接返回log_dirs，[(Optional)mode['omit', 'select'], time_start, time_end]
        :return: new_log_dirs
        """

        if not args or not isinstance(args, list):
            return log_dirs

        if isinstance(args[0], str):
            mode = args.pop(0).strip().lower()
        else:
            mode = 'select'  # omit: 省略, select: 选取

        if len(args) < 2:
            return log_dirs

        if mode == 'omit':
            new_log_dirs = log_dirs.copy()
        elif mode == 'select':
            new_log_dirs = []
        else:
            raise ValueError('`mode` should be `omit` or `select`')

        time_start = args.pop(0)
        time_end = args.pop(0)

        def _check(file_ctime):
            return time_start <= file_ctime < time_end

        def _modify(target_list, element):
            if mode == 'omit':
                target_list.remove(element)
            else:
                target_list.append(element)

        def _get_ctime(file_path):
            create_time = os.path.getctime(file_path)
            return datetime.fromtimestamp(create_time)

        for log_dir in log_dirs:
            file_path = os.path.join(root_dir, log_dir)
            file_ctime = _get_ctime(file_path)
            if _check(file_ctime):
                _modify(new_log_dirs, log_dir)

        return new_log_dirs


if __name__ == '__main__':
    plot = Plot(
        root_dir='./checkpoint',
        process_seq=[
            {'func': ModifyMethods.filter_ctime, 'args': [datetime(2025, 9, 12), datetime(2025, 9, 30)]},
            # {'func': ModifyMethods.filter_kwargs, 'args': ['omit', {'resnet_type': 'A'}]},
            {'func': ModifyMethods.filter_kwargs, 'args': ['select', {'lr': 1e-04}]}
        ],
        sign_filter={'mode': 'omit', 'args': ['*obj']},
        channel=[0, 1],
        plot_col=3,
        file_ext='.pt',
        file_seg='_',
        fig_size=(20, 6)
    )
    plot.plot()