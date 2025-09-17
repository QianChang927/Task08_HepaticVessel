import os
import torch
import warnings

warnings.filterwarnings(action="ignore", category=UserWarning)

# 类型注解所用库
from typing import Callable
from torch import device, Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from argparse import Namespace
from monai.data import DataLoader
from .early_stopping import EarlyStopping

def smart_print(message: str) -> None:
    """
    用于输出信息+保存日志
    :param message: 需要输出的信息
    :return:
    """
    print(message)
    if not CONFIG.SAVE_LOG or CONFIG.SAVE_DIR is None:
        return
    with open(os.path.join(CONFIG.SAVE_DIR, 'log.txt'), 'a') as f:
        f.write(message + '\n')


class CONFIG:
    """
    用于保存常量的配置类
    """
    ROI_SIZE: tuple[int, int, int] = (64, 64, 64)
    SW_BATCH_SIZE: int = 2
    OUT_CHANNELS: int = 2
    JUDGE_CHANNEL: int = -1
    SAVE_LOG: bool = True
    SAVE_DIR: str = None


class Trainer:
    def __init__(
        self,
        model: Module,
        loss_fn: Module,
        optimizer: Optimizer,
        early_stopping: EarlyStopping,
        train_loader: DataLoader,
        valid_loader: DataLoader=None,
        save_dir: str=None,
        scheduler: LRScheduler=None,
        device: device=None,
        train_process: Callable[[Module, DataLoader, Callable[[dict, device], tuple[Tensor, Tensor]], Module, Optimizer, LRScheduler, device], dict|None]=None,
        valid_process: Callable[[Module, DataLoader, Callable[[dict, device], tuple[Tensor, Tensor]], Module, Optimizer, LRScheduler, device], dict|None]=None,
        batch_process: Callable[[dict, device], tuple[Tensor, Tensor]]=None,
        valid_interval: int=5,
        judge_channel: int=-1,
        save_log: bool=True,
        args: Namespace=None
    ) -> None:
        """
        训练器构造函数
        :param model: 所使用的神经网络
        :param loss_fn: 损失函数
        :param optimizer: 优化器
        :param early_stopping: 早停机制，包括进度输出功能
        :param train_loader: 所使用的训练数据集
        :param valid_loader: 所使用的验证数据集
        :param save_dir: 模型保存文件夹
        :param scheduler: 控制学习率变化的调度器
        :param device: 训练所用设备
        :param train_process: 训练过程函数：train_process(model, data_loader, batch_process, loss_fn, optimizer, scheduler, device) -> dict|None
        :param valid_process: 验证过程函数：valid_process(model, data_loader, batch_process, loss_fn, optimizer, scheduler, device) -> dict|None
        :param batch_process: batch处理函数：batch_process(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]
        :param valid_interval: 验证间隔
        :param judge_channel: 训练/验证过程损失/dice通道选择
        :param save_log: 是否保存日志
        :param args: 命令行参数解析器
        :return:
        """
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        CONFIG.SAVE_LOG = save_log
        CONFIG.SAVE_DIR = save_dir

        if args is not None:
            CONFIG.ROI_SIZE = args.roi_size
            CONFIG.SW_BATCH_SIZE = args.sw_batch
            CONFIG.OUT_CHANNELS = args.out_channels
            CONFIG.JUDGE_CHANNEL = args.judge_channel
        else:
            CONFIG.JUDGE_CHANNEL = judge_channel

        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping = early_stopping

        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.save_dir = save_dir
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

        self.train_process = train_process if train_process else TrainerMethods.train
        self.valid_process = valid_process if valid_process else TrainerMethods.valid
        self.batch_process = batch_process if batch_process else TrainerMethods.parse_batch
        self.valid_interval = valid_interval

    def run(self, epochs: int=100) -> None:
        """
        运行函数
        :param epochs: 训练轮次
        :return:
        """

        for epoch in range(epochs):
            smart_print(f"{f'Epoch {epoch + 1}/{epochs}':-^60}")
            smart_print(f"lr: {self.optimizer.state_dict()['param_groups'][0]['lr']:.8f}")

            self.model.train()
            train_criteria = self.train_process(self.model, self.train_loader, self.batch_process,
                                                self.loss_fn, self.optimizer, self.scheduler, self.device)
            if train_criteria is not None:
                self.early_stopping(epoch, train_criteria, 'train')

            if epoch % self.valid_interval != 0:
                continue

            self.model.eval()
            with torch.no_grad():
                valid_criteria = self.valid_process(self.model, self.valid_loader, self.batch_process,
                                                    self.loss_fn, self.optimizer, self.scheduler, self.device)
                if valid_criteria is not None:
                    self.early_stopping(epoch, valid_criteria, 'valid')

            if self.early_stopping.early_stop:
                smart_print(f"{'':-^60}")
                smart_print(f"Early stop: {epoch + 1}/{epochs}")
                self.early_stopping.end_display()
                break


class TrainerMethods:
    """
    训练器的默认静态方法类
    """
    @staticmethod
    def convert_to_onehot(tensor: Tensor, num_classes: int) -> Tensor:
        """
        将原有张量转为指定num_classes的one-hot编码张量
        :param tensor 需要转化的张量[B, C, H, W, D]
        :param num_classes one-hot编码张量的类别数
        :return: 转化后的one-hot编码张量
        """
        one_hot = torch.zeros(tensor.shape[0], num_classes, *tensor.shape[2:], dtype=torch.float32, device=tensor.device)
        for ch in range(num_classes):
            one_hot[:, ch:] = (tensor == ch).float()
        return one_hot

    @staticmethod
    def dice_coef(pred: Tensor, label: Tensor, epsilon: float = 1e-6, include_background: bool=True) -> list[float]:
        """
        计算Dice系数
        :param pred 预测张量，若要启用hard-dice模式须传入argmax+one-hot编码后的结果
        :param label 真实标签，通道与pred不匹配时将自动进行one-hot编码
        :param epsilon 防止分母为0的阈值
        :param include_background 是否包含背景（one-hot编码中0为背景）
        :return: 每个通道的Dice系数
        """
        # 不存在batch_size信息时增加一维，保持维度一致
        if pred.dim() == 4: pred = pred.unsqueeze(0)
        if label.dim() == 4: label = label.unsqueeze(0)

        # 处理label为one-hot标签
        if pred.shape != label.shape:
            label_onehot = TrainerMethods.convert_to_onehot(label, num_classes=pred.shape[1])
        else: label_onehot = label

        # 形状check
        assert pred.shape == label_onehot.shape, f"Shape mismatch: pred {pred.shape} label {label_onehot.shape}"

        # 忽略背景，请注意之后若不重新分配内存，请勿直接修改pred和label的数值
        start_channel = ~int(include_background) & 1
        pred = pred[:, start_channel:]
        label_onehot = label_onehot[:, start_channel:]

        # 展平为[batch_size, channels, N]方便计算
        pred_flat = pred.contiguous().view(pred.shape[0], pred.shape[1], -1)
        label_flat = label_onehot.contiguous().view(label_onehot.shape[0], label_onehot.shape[1], -1)

        # 根据公式计算dice系数，注意epsilon的作用
        part_inter = (pred_flat * label_flat).sum(-1)
        part_union = pred_flat.sum(-1) + label_flat.sum(-1)
        dice_batch = (2. * part_inter + epsilon) / (part_union + epsilon)

        # 对batch取平均
        dice_mean = dice_batch.mean(dim=0)
        return dice_mean

    @staticmethod
    def train(
        model: Module,
        data_loader: DataLoader,
        batch_process: Callable[[dict, device], tuple[Tensor, Tensor]],
        loss_fn: Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        device: device
    ) -> dict:
        """
        训练函数的默认实现
        :param model: 所使用的神经网络，若要在GPU上训练，应在调用此函数前转移
        :param data_loader: 所使用的训练数据集
        :param batch_process: batch解析函数，batch_process(batch, device)
        :param loss_fn: 损失函数，若要在GPU上训练，应在调用此函数前转移
        :param optimizer: 优化器
        :param scheduler: 控制动态学习率的调度器
        :param device: 训练所用设备
        :return: {'loss': epoch_loss}
        """
        train_step = 0
        epoch_loss = 0
        epoch_dice = 0

        for batch in data_loader:
            images, labels = batch_process(batch, device)
            train_step += 1

            _loss, _outputs = torch.Tensor(), torch.Tensor()
            def closure():
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                if loss.shape[0] > 1:
                    loss = loss[CONFIG.JUDGE_CHANNEL]
                nonlocal _loss, _outputs
                _loss, _outputs = loss.item(), outputs
                loss.backward()
                return loss

            optimizer.zero_grad()
            optimizer.step(closure)

            epoch_dice += TrainerMethods.dice_coef(torch.softmax(_outputs, dim=1), labels)
            epoch_loss += _loss
            smart_print(f"{train_step}/{len(data_loader)}, train loss: {_loss:.4f}")

        epoch_dice /= train_step
        epoch_loss /= train_step
        if hasattr(epoch_dice, 'tolist'): epoch_dice = epoch_dice.tolist()
        if hasattr(epoch_loss, 'tolist'): epoch_loss = epoch_loss.tolist()
        return {'loss': epoch_loss, 'dice': epoch_dice}

    @staticmethod
    def valid(
        model: Module,
        data_loader: DataLoader,
        batch_process: Callable[[dict, device], tuple[Tensor, Tensor]],
        loss_fn: Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        device: device
    ) -> dict:
        """
        验证函数的默认实现
        :param model: 所使用的神经网络，若要在GPU上训练，应在调用此函数前转移
        :param data_loader: 所使用的训练数据集
        :param batch_process: batch解析函数，batch_process(batch, device)
        :param loss_fn: 损失函数，若要在GPU上训练，应在调用此函数前转移
        :param optimizer: 优化器
        :param scheduler: 控制动态学习率的调度器
        :param device: 训练所用设备
        :return: {'dice': dice}
        """
        from monai.inferers import sliding_window_inference
        from monai.metrics import DiceMetric
        from monai.data import decollate_batch
        from monai import transforms

        post_pred = transforms.Compose([
            transforms.Activations(softmax=True),
            transforms.AsDiscrete(argmax=True, to_onehot=CONFIG.OUT_CHANNELS)
        ])

        post_label = transforms.Compose([
            transforms.AsDiscrete(to_onehot=CONFIG.OUT_CHANNELS)
        ])

        import inspect
        init_params = inspect.signature(DiceMetric).parameters
        valid_keys = set(init_params.keys())
        loss_fn_dict = loss_fn.__dict__
        filtered_dict = {key: value for key, value in loss_fn_dict.items() if key in valid_keys}
        filtered_dict.update({'reduction': 'none'})
        dice_metric = DiceMetric(**filtered_dict)

        for batch in data_loader:
            images, labels = batch_process(batch, device)
            valid_outputs = sliding_window_inference(images, CONFIG.ROI_SIZE, CONFIG.SW_BATCH_SIZE, model)
            valid_outputs = [post_pred(i) for i in decollate_batch(valid_outputs)]
            valid_labels = [post_label(i) for i in decollate_batch(labels)]
            dice_metric(y_pred=valid_outputs, y=valid_labels)

        dice = dice_metric.aggregate().mean(dim=0)
        dice_metric.reset()

        if scheduler is not None:
            from typing import Sequence
            dice_shape = dice.shape
            if not isinstance(dice_shape, Sequence):
                dice_shape = [dice_shape]

            if len(dice_shape) > 1 or dice_shape[0] > 1:
                dice_judge = dice[CONFIG.JUDGE_CHANNEL].item()
                dice = dice.tolist()
            else:
                dice_judge = dice.item()
                dice = dice.item()
            scheduler.step(dice_judge)

        return {'dice': dice}

    @staticmethod
    def parse_batch(batch: dict, device: device) -> tuple[Tensor, Tensor]:
        """
        解析batch的默认函数
        :param batch: 需要解析的batch
        :param device: 解析后的tensor数据存放在device上
        :return: 返回解析后的batch，该默认函数的返回类型为(Tensor, Tensor)
        """
        image, label = batch['image'], batch['label']
        label[label > 1] = 1
        return image.to(device), label.to(device)
