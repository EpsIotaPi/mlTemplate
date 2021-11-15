import torch
import torch.nn as nn
import torch.nn.functional as F


def train(train_loader, model, optimizer, device, param, logger, **kwargs):
    model.train()
    for batch_idx, (batch, target) in enumerate(train_loader):
        if batch_idx % param.log_interval == 0:
           logger.info("")


def eval(test_loader, model, device, param, logger, **kwargs):
    model.eval()
    eval_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            pass
    eval_loss /= len(test_loader.dataset)
    logger.info("")


