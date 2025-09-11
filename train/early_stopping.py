import os
import torch

# 类型注解所用库
from argparse import Namespace
from torch.nn import Module
from typing import Literal, Callable, Sequence

class CONFIG:
    """
    用于保存常量的配置类
    """
    JUDGE_CHANNEL: int = -1


class EarlyStopping:
    """
    早停机制实现类，包含早停机制、进度输出及进度保存
    """
    def __init__(
        self,
        model: Module,
        save_dir: str,
        patience: int=None,
        min_delta: float=0.,
        stop_criterion: Literal['train', 'valid']='train',
        save_criterion: Literal['train', 'valid']='valid',
        save_interval: int=10,
        verbose: bool=False,
        train_compare: Callable[[dict | float, dict | float], float]=None,
        valid_compare: Callable[[dict | float, dict | float], float]=None,
        judge_channel: int=-1,
        save_log: bool=True,
        args: Namespace=None
    ) -> None:
        """
        早停机制类构造函数
        :param model: 实例化后的模型类
        :param save_dir: 文件保存位置
        :param patience: 连续patience个epoch满足条件后停止，为None时表示不需要早停
        :param min_delta: 模型优化的最小阈值
        :param stop_criterion: 早停评估标准，以训练还是验证的数据为评判标准
        :param save_criterion: 保存评估标准，用于保存最佳模型数据
        :param save_interval: 进度保存间隔
        :param verbose: 是否输出额外信息
        :param train_compare: 训练过程比较函数，输出优越差
        :param valid_compare: 验证过程比较函数，输出优越差
        :param judge_channel: 训练/验证过程损失/dice通道选择
        :param save_log: 是否保存日志
        :param args: 命令行参数解析器
        :return:
        """
        if args is not None:
            CONFIG.JUDGE_CHANNEL = args.judge_channel
        else:
            CONFIG.JUDGE_CHANNEL = judge_channel

        self.model = model
        self.save_dir = save_dir
        self.patience = patience
        self.min_delta = min_delta
        self.stop_criterion = stop_criterion
        self.save_criterion = save_criterion
        self.save_interval = save_interval
        self.verbose = verbose
        self.save_log = save_log

        self.train_compare = train_compare if train_compare else EarlyStoppingMethods.train_compare
        self.valid_compare = valid_compare if valid_compare else EarlyStoppingMethods.valid_compare

        self.counter = 0
        self.early_stop = False

        self.train_criteria = {}
        self.valid_criteria = {}

        self.save_criteria = None
        self.stop_criteria = None
        self.best_epoch = -1

    def __call__(self, epoch: int, new_criteria: dict, judge_mode: Literal['train', 'valid']) -> None:
        """
        类调用方法
        :param epoch: 当前轮次
        :param new_criteria: 当前评判指标
        :param judge_mode: 评判模式
        :return:
        """
        def __display(criteria: dict, concat: int = 2) -> None:
            """
            输出字典内容
            :param criteria: 需要输出的内容
            :param concat: 一行输出concat个键值
            :return:
            """
            cache = ''
            i = 0
            for key, value in criteria.items():
                i += 1
                if isinstance(value, Sequence): value = [f'{v:.5f}' for v in value]
                else: value = f'{value:.5f}'
                cache += f"{f'{judge_mode} {key}: {value}': <30}"
                if i % concat == 0:
                    self.smart_print(cache)
            if i % concat:
                self.smart_print(cache)

        def __save():
            """
            更新最佳评判标准并保存模型及评判标准变化曲线
            :return:
            """
            self.save_criteria = new_criteria
            self.best_epoch = epoch
            self.__save_model()
            self.__save_criteria(judge_mode)

        if judge_mode == 'train':
            compare = self.train_compare
        elif judge_mode == 'valid':
            compare = self.valid_compare
        else:
            raise ValueError('judge_mode must be "train" or "valid"')

        self.__update(new_criteria, judge_mode)
        __display(new_criteria)

        if (epoch + 1) % self.save_interval == 0:
            self.__save_criteria(judge_mode)

        if judge_mode == self.stop_criterion:
            if self.stop_criteria is None or compare(self.stop_criteria, new_criteria) > self.min_delta:
                self.counter = 0
                self.stop_criteria = new_criteria
            else:
                self.counter += 1
                if self.verbose:
                    self.smart_print(f"Early stopping: {self.counter}/{self.patience}")
                if self.counter >= self.patience:
                    self.early_stop = True

        elif judge_mode == self.save_criterion:
            if self.save_criteria is None or compare(self.save_criteria, new_criteria) > 0:
                __save()

    def end_display(self) -> None:
        """
        结束信息输出
        :return:
        """
        self.__save_model('last')
        self.smart_print(f"best criteria: {self.save_criteria} at epoch {self.best_epoch + 1}")

    def smart_print(self, message: str) -> None:
        """
        用于输出信息+保存日志
        :param message: 需要输出的信息
        :return:
        """
        print(message)
        if not self.save_log or self.save_dir is None:
            return
        with open(os.path.join(self.save_dir, 'log.txt'), 'a') as f:
            f.write(message + '\n')

    def __update(self, new_criteria: dict, update_mode: Literal['train', 'valid']) -> None:
        """
        用于更新字典内容
        :param new_criteria: 新字典
        :param update_mode: 更新模式
        :return:
        """
        if update_mode == 'train':
            past_criteria = self.train_criteria
        elif update_mode == 'valid':
            past_criteria = self.valid_criteria
        else:
            raise ValueError('update_mode must be "train" or "valid"')

        for key, value in new_criteria.items():
            if key not in past_criteria:
                past_criteria[key] = []
            past_criteria[key].append(value)

    def __save_model(self, save_mode: Literal['best', 'last']='best') -> None:
        """
        保存模型
        :return:
        """
        if self.verbose: self.smart_print(f"Saving model to {self.save_dir}...")
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, f'model_{save_mode}.pth'))

    def __save_criteria(self, save_mode: Literal['train', 'valid']) -> None:
        """
        保存评判标准变化曲线
        :param save_mode: 保存模式
        :return:
        """
        if save_mode == 'train':
            criteria = self.train_criteria
        elif save_mode == 'valid':
            criteria = self.valid_criteria
        else:
            raise ValueError('save_mode must be "train" or "valid"')
        torch.save(criteria, os.path.join(self.save_dir, f'{save_mode}_criteria.pt'))


class EarlyStoppingMethods:
    """
    早停机制默认静态方法类
    """
    @staticmethod
    def train_compare(criteria_1: dict | float, criteria_2: dict | float) -> float:
        """
        训练标准优越度计算：返回criteria_1与criteria_2的优越差值
        :param criteria_1: 源字典|浮点数
        :param criteria_2: 新字典|浮点数
        :return: >0 criteria_2更优越; <0 criteria_1更优越; =0 二者无区别
        """
        cri_1 = criteria_1.get('loss', 0.) if isinstance(criteria_1, dict) else criteria_1
        cri_2 = criteria_2.get('loss', 0.) if isinstance(criteria_2, dict) else criteria_2
        return float(cri_1 - cri_2)

    @staticmethod
    def valid_compare(criteria_1: dict | Sequence | float, criteria_2: dict | Sequence | float) -> float:
        """
        验证标准优越度计算：返回criteria_1与criteria_2的优越差值
        :param criteria_1: 源字典|浮点数
        :param criteria_2: 新字典|浮点数
        :return: >0 criteria_2更优越; <0 criteria_1更优越; =0 二者无区别
        """
        cri_1 = criteria_1.get('dice', 0.) if isinstance(criteria_1, dict) else criteria_1
        if isinstance(cri_1, Sequence): cri_1 = cri_1[CONFIG.JUDGE_CHANNEL]
        cri_2 = criteria_2.get('dice', 0.) if isinstance(criteria_2, dict) else criteria_2
        if isinstance(cri_2, Sequence): cri_2 = cri_2[CONFIG.JUDGE_CHANNEL]
        return float(cri_2 - cri_1)