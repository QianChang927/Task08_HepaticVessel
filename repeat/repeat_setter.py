import os
import random
import numpy as np
import torch
from monai.utils import set_determinism

class RepeatSetter:
    """
    可重复性设置器
    """
    def __init__(self, seed: int=0) -> None:
        """
        设置器构造函数
        :param seed: 固定的随机数种子
        :return:
        """
        self.seed = seed

    def __call__(self) -> None:
        """
        类调用方法
        :return:
        """
        self.__set_python()
        self.__set_numpy()
        self.__set_torch()
        self.__set_monai()
        print(f"Reproducibility set, using seed: {self.seed}")

    def __set_python(self):
        """
        类私有函数，设置python内置库及系统环境的可重复性
        :return:
        """
        # random库
        random.seed(self.seed)
        # 系统环境
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

    def __set_numpy(self) -> None:
        """
        类私有函数，设置numpy的可重复性
        :return:
        """
        np.random.seed(self.seed)

    def __set_torch(self) -> None:
        """
        类私有函数，设置torch的可重复性，包括cuda及cudnn
        :return:
        """
        # torch
        torch.manual_seed(self.seed)
        # cuda
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        # cudnn
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def __set_monai(self) -> None:
        """
        类私有函数，设置monai库的可重复性
        :return:
        """
        set_determinism(seed=self.seed)

if __name__ == '__main__':
    repeat_setter = RepeatSetter(seed=0)
    repeat_setter()