import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import setSeed
from utils.hyperparameter import HyperParameter
from utils.logger import get_logger
from utils.dataset import myDataset
from utils.model import myModel
from eval_train import train, eval

# ====================== step 1/5 前期准备 ====================== #
param = HyperParameter()
log_path = os.path.join(param.log_dir, param.train_name + '_log.txt')
logger = get_logger(log_path)
setSeed(param.seed)

use_cuda = param.cuda_available and torch.cuda.is_available()
device = torch.device('cpu')
if use_cuda:
    logger.info("cuda is available")
    device = torch.device('cuda')

param.parameter_log(logger)

# ====================== step 2/5 准备数据 ====================== #
myDataset = myDataset(data_dir=None)
train_size = int(len(myDataset) * param.trainset_ratio)
test_size = len(myDataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(myDataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=param.batch_size)
test_loader = DataLoader(test_dataset, batch_size=param.batch_size)

# ====================== step 3/5 准备模型 ====================== #
model = myModel().to(device)
optimizer = optim.SGD(model.parameters(), lr=param.lr, momentum=param.momentum)

# ====================== step 4/5 开始训练 ====================== #
for epoch_idx in range(1, param.epochs + 1):
    train(train_loader, model, optimizer, device, param, logger)
    eval(test_loader, model, device, param, logger)


# ====================== step 5/5 保存模型 ====================== #
if param.save_model:
    save_path = os.path.join(param.checkpoint_dir, param.train_name + "_model.pt")
    torch.save(model.state_dict(), save_path)
