## loss = 真实数据分类损失 + 虚拟数据分类损失 + 特征对齐损失

from torch.utils.data import DataLoader
from federatedscope.register import register_trainer
from federatedscope.core.trainers.torch_trainer import GeneralTorchTrainer
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.scheduler_builder import get_scheduler
import torch
import torch.nn as nn
import copy
import numpy as np
import torch.nn.functional as F
from itertools import cycle
import math


class FedPGTrainer(GeneralTorchTrainer):

    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):

        super(FedPGTrainer, self).__init__(model,
                                           data,
                                           device,
                                           config,
                                           only_for_eval,
                                           monitor)

        # 创建supcon_loss对象
        self.supcon_loss = SupConLoss(contrast_mode='all', base_temperature=0.07, device=device)
        self.vir_dataloader = None

        # 使权重系数随轮次减小
        self.weight_vdata = self.cfg.fedpg.weight_vdata
        self.weight_align = self.cfg.fedpg.weight_align
        self.decay = self.cfg.fedpg.weight_decay

        # 轮次
        self.state = 0

    def set_state(self, state):
        self.state = state

    def reset_vir_dataloader(self, vir_dataloader):

        self.vir_dataloader = iter(cycle(vir_dataloader))
        # print("==== Client# get the v_dataset!! ====")

    def get_feature_dim(self):
        # 生成数据
        dataloader = self.ctx.data['test']
        x_size = dataloader.dataset[0][0].size()
        x = torch.full(x_size, 1.0)
        x = x.unsqueeze(0)  # batch_size = 1

        self.ctx.model.eval()
        feature_dim = self.ctx.model.get_z(x)
        self.ctx.model.train()

        return feature_dim.size()[1]

    def _hook_on_fit_start_init(self, ctx):
        super()._hook_on_fit_start_init(ctx)
        
        # self.weight_vdata = self.cfg.fedpg.weight_vdata * (self.decay ** (self.state))
        # self.weight_align = self.cfg.fedpg.weight_align * (self.decay ** (self.state))

        # self.weight_vdata = self.cfg.fedpg.weight_vdata * math.cos(self.state *2)
        # self.weight_align = self.cfg.fedpg.weight_align * math.cos(self.state *2)

        # self.weight_vdata = 0 if self.state // 5 != 0 else self.cfg.fedpg.weight_vdata
        # self.weight_align = 0 if self.state // 5 != 0 else self.cfg.fedpg.weight_align
        # print("weight_vdata, align = ", self.weight_vdata, ", ", self.weight_align)

        if self.state > self.cfg.fedpg.upper_bound:
            self.weight_vdata = 0
            self.weight_align = 0 
            
        # print("state: ", self.state)

    # get a batch data from v_dataloader
    def _hook_on_batch_start_init(self, ctx):
        super()._hook_on_batch_start_init(ctx)

        if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE] and self.vir_dataloader is not None:
            ctx.v_data_batch = CtxVar(next(self.vir_dataloader), LIFECYCLE.BATCH)

    def proxy_align_loss(self, ctx, x, label, v_x, v_label):
        x_feature = ctx.model.get_z(x)
        v_x_feature = ctx.model.get_z(v_x)

        features = torch.cat((x_feature, v_x_feature.clone().detach()), dim=0)
        new_features = F.normalize(features, dim=1).unsqueeze(1)
        nosie_features = F.normalize(v_x_feature, dim=1).unsqueeze(1)

        labels = torch.cat((label, v_label), dim=0)

        loss_align = self.supcon_loss(new_features, labels, temperature=0.07, mask=None)
        loss_noise = self.supcon_loss(nosie_features, v_label, temperature=0.07, mask=None)
        return loss_align + 0.1* loss_noise

    # 前向传播函数
    def _hook_on_batch_forward(self, ctx):

        x, label = [_.to(ctx.device) for _ in ctx.data_batch]

        # 单通道数据变三通道，使得resnet等模型可用
        # if x.shape[1] == 1:
        #     assert self.cfg.data.type in ["mnist", "femnist", "fmnist", "femnist-digit"]
        #     x = x.repeat(1, 3, 1, 1)

        # print("x[0] : ", x[0], x[0].size())

        pred = ctx.model(x)
        if len(label.size()) == 0:
            label = label.unsqueeze(0)

        ctx.y_true = CtxVar(label, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(pred, LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(ctx.criterion(pred, label), LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(label), LIFECYCLE.BATCH)

        if hasattr(ctx, 'v_data_batch') and (self.weight_vdata > 0 or self.weight_align > 0):

            # 虚拟数据集上的训练loss
            v_x, v_label = [_.to(ctx.device) for _ in ctx.v_data_batch]

            # if v_x.shape[1] == 1:
            #     v_x = v_x.repeat(1, 3, 1, 1)

            # print("v_x[0] : ", v_x[0], v_x[0].size())
            # print(v_label)
            v_pred = ctx.model(v_x)
            loss_v = ctx.criterion(v_pred, v_label)
            ctx.loss_vdata = CtxVar(self.weight_vdata * loss_v, LIFECYCLE.BATCH)

            # 真实数据集和虚拟数据集的特征latent之间的距离损失
            loss_align = self.proxy_align_loss(ctx, x, label, v_x, v_label)
            ctx.loss_align = CtxVar(self.weight_align * loss_align , LIFECYCLE.BATCH)

            # print("ctx.loss_batch = {}, ctx.loss_v = {}, ctx.loss_align = {}".format(ctx.loss_batch, ctx.loss_v, ctx.loss_align))
            ctx.loss_batch = CtxVar(ctx.loss_batch + ctx.loss_vdata + ctx.loss_align, LIFECYCLE.BATCH)


def call_fedpg_trainer(trainer_type):
    if trainer_type == 'fedpg_trainer':
        trainer_builder = FedPGTrainer
        return trainer_builder


register_trainer('fedpg_trainer', call_fedpg_trainer)


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    # 损失与同类标签的特征向量间距离正相关，与异类标签的特征向量间距离负相关

    def __init__(self, contrast_mode='all',
                 base_temperature=0.07, device=None):
        super(SupConLoss, self).__init__()
        # self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.device = device

    def forward(self, features, labels=None, temperature=0.07, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        # device = (torch.device('cuda')
        #           if features.is_cuda
        #           else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), temperature)
        # logging.info(f"In SupCon, anchor_dot_contrast.shape: {anchor_dot_contrast.shape}, anchor_dot_contrast: {anchor_dot_contrast}")
        # logging.info(f"In SupCon, anchor_dot_contrast.shape: {anchor_dot_contrast.shape}, anchor_dot_contrast: {anchor_dot_contrast.mean()}")
        # logging.info(f"In SupCon, anchor_dot_contrast.device: {anchor_dot_contrast.device}, self.device: {self.device}")

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        # logging.info(f"In SupCon, exp_logits.shape: {exp_logits.shape}, exp_logits: {exp_logits.mean()}")
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        if torch.any(torch.isnan(log_prob)):
            log_prob[torch.isnan(log_prob)] = 0.0
        # logging.info(f"In SupCon, log_prob.shape: {log_prob.shape}, log_prob: {log_prob.mean()}")

        mask_sum = mask.sum(1)
        mask_sum[mask_sum == 0] += 1

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        # loss
        loss = - (temperature / self.base_temperature) * mean_log_prob_pos
        # loss[torch.isnan(loss)] = 0.0
        if torch.any(torch.isnan(loss)):
            loss[torch.isnan(loss)] = 0.0
        #     # logging.info(f"In SupCon, features.shape: {features.shape}, loss: {loss}")
        #     raise RuntimeError
        loss = loss.view(anchor_count, batch_size).mean()

        return loss