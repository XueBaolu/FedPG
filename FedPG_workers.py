from federatedscope.core.auxiliaries.model_builder import get_shape_from_data
from federatedscope.core.workers.base_worker import Worker
from federatedscope.core.workers.client import Client
from federatedscope.register import register_worker
from federatedscope.core.workers.server import Server
from federatedscope.core.message import Message
from torch.utils.data import TensorDataset, DataLoader
from federatedscope.core.workers.base_server import BaseServer
from federatedscope.core.auxiliaries.utils import merge_dict_of_results, \
    Timeout, merge_param_dict
import copy
from collections import defaultdict
import numpy as np
import random
import torch
import torch.optim as optim
from gan import Generative, get_virtual_dataloader, regularization
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ==============================================================================#
#################################### server ####################################
# ==============================================================================#
## sever：1. 聚合局部原型；2. 训练GAN；3. 生成并发放虚拟数据
## 定义传输本地和全局原型的message类
class FedPGServr(Server):

    def __init__(self,
                 ID=-1,
                 state=0,
                 config=None,
                 data=None,
                 model=None,
                 client_num=5,
                 total_round_num=10,
                 device='cpu',
                 strategy=None,
                 unseen_clients_id=None,
                 **kwargs):

        super(FedPGServr, self).__init__(ID,
                                         state,
                                         config,
                                         data,
                                         model,
                                         client_num,
                                         total_round_num,
                                         device,
                                         strategy,
                                         unseen_clients_id,
                                         **kwargs)

        # 接收客户端的本地原型
        self.register_handlers(
            msg_type='local_proto_msg',
            callback_func=self.callback_for_local_proto_msg)

        # 列表，元素为本轮客户端的本地原型字典
        self.local_proto_msgs = []

        # 定义GAN
        self.num_classes = self._cfg.model.out_channels
        self.feature_dim = self.trainer.get_feature_dim()
        self.input_shape = get_shape_from_data(self.data, self._cfg.model)
        c, h, w = self.input_shape[-3], self.input_shape[-2], self.input_shape[-1]

        self.gan_model = Generative(noise_dim=512,
                                    num_classes=self.num_classes,
                                    hidden_dim=[128, 256, 512, 1024],
                                    output_dim=c * h * w,
                                    image_size=[c, h, w],
                                    device=self.device).to(self.device)

        self.old_global_proto = None
        self.global_proto = None

    # server聚合本地原型
    def set_global_proto(self, label_proto_dict_list):
        all_proto = defaultdict(list)
        global_proto = {}

        for label_proto_dict in label_proto_dict_list:
            for label, proto in label_proto_dict.content.items():
                all_proto[label].append(proto)

        for label, protos in all_proto.items():
            protos_tensor = torch.stack(protos)
            global_proto[label] = torch.mean(protos_tensor, dim=0)

        ## 可能需要添加生成虚拟原型的代码

        self.global_proto = global_proto
        # global_proto = {'label': tensor(global_proto)}

    # 生成并显示图片
    def show_generated_images(self, num_per_class=5):
        # 调用训练好的generator，生成虚拟数据
        self.gan_model.eval()
        labels = torch.tensor(list(range(self.num_classes)) * num_per_class)
        generated_images = self.gan_model.get_output(labels)
        self.gan_model.train()
    
        # 显示虚拟图片
        generated_images = generated_images.to('cpu').detach().numpy()
        fig, axes = plt.subplots(self.num_classes, num_per_class, figsize=(8, 8))
        for i, ax in enumerate(axes.flatten()):
            img = generated_images[i].transpose(1, 2, 0)  # 通道顺序调整为 H x W x C
            ax.imshow((img + 1) / 2)  # 将数据从 [-1, 1] 转换到 [0, 1]
            ax.axis('off')
        plt.show()


    def callback_funcs_model_para(self, message: Message):
        """
        The handling function for receiving model parameters, which triggers \
        ``check_and_move_on`` (perform aggregation when enough feedback has \
        been received). This handling function is widely used in various FL \
        courses.

        Arguments:
            message: The received message.
        """
        if self.is_finish:
            return 'finish'

        round = message.state
        sender = message.sender
        timestamp = message.timestamp
        content = message.content
        self.sampler.change_state(sender, 'idle')

        # dequantization
        if self._cfg.quantization.method == 'uniform':
            from federatedscope.core.compression import \
                symmetric_uniform_dequantization
            if isinstance(content[1], list):  # multiple model
                sample_size = content[0]
                quant_model = [
                    symmetric_uniform_dequantization(x) for x in content[1]
                ]
            else:
                sample_size = content[0]
                quant_model = symmetric_uniform_dequantization(content[1])
            content = (sample_size, quant_model)

        # update the currency timestamp according to the received message
        assert timestamp >= self.cur_timestamp  # for test
        self.cur_timestamp = timestamp

        if round == self.state:
            if round not in self.msg_buffer['train']:
                self.msg_buffer['train'][round] = dict()
            # Save the messages in this round
            self.msg_buffer['train'][round][sender] = content
        elif round >= self.state - self.staleness_toleration:
            # Save the staled messages
            self.staled_msg_buffer.append((round, sender, content))
        else:
            # Drop the out-of-date messages
            logger.info(f'Drop a out-of-date message from round #{round}')
            self.dropout_num += 1

        if self._cfg.federate.online_aggr:
            self.aggregator.inc(content)

        if self.state < self._cfg.fedpg.upper_bound:
            return False

        else:
            move_on_flag = self.check_and_move_on()
            if self._cfg.asyn.use and self._cfg.asyn.broadcast_manner == \
                    'after_receiving':
                self.broadcast_model_para(msg_type='model_para',
                                          sample_client_num=1)
    
            return move_on_flag
        
    
    def check_and_move_on(self,
                          check_eval_result=False,
                          min_received_num=None):
        """
        To check the message_buffer. When enough messages are receiving, \
        some events (such as perform aggregation, evaluation, and move to \
        the next training round) would be triggered.

        Arguments:
            check_eval_result (bool): If True, check the message buffer for \
                evaluation; and check the message buffer for training \
                otherwise.
            min_received_num: number of minimal received message, used for \
                async mode
        """
        if min_received_num is None:
            if self._cfg.asyn.use:
                min_received_num = self._cfg.asyn.min_received_num
            else:
                min_received_num = self._cfg.federate.sample_client_num
        assert min_received_num <= self.sample_client_num

        if check_eval_result and self._cfg.federate.mode.lower(
        ) == "standalone":
            # in evaluation stage and standalone simulation mode, we assume
            # strong synchronization that receives responses from all clients
            min_received_num = len(self.comm_manager.get_neighbors().keys())

        move_on_flag = True  # To record whether moving to a new training
        # round or finishing the evaluation
        if self.check_buffer(self.state, min_received_num, check_eval_result):
            if not check_eval_result:
                # Receiving enough feedback in the training process
                aggregated_num = self._perform_federated_aggregation()
                self.state += 1
                if self.state % self._cfg.eval.freq == 0 and self.state != \
                        self.total_round_num:
                    #  Evaluate
                    logger.info(f'Server: Starting evaluation at the end '
                                f'of round {self.state - 1}.')
                    self.eval()

                # 保存服务器模型
                if self.best_model is not None and self.state in self._cfg.federate.server_save_round:
                    ckpt = {'cur_round': self.state, 'model': self.model.state_dict()}
                    path = self._cfg.federate.save_to + 'mid_global_{}.pt'.format(self.state)
                    torch.save(ckpt, path)

                # if self.best_model is not None:
                #     best_model_path = self._cfg.federate.save_to + 'best_global.pt'
                #     torch.save(self.best_model, best_model_path)
                #     print("Server save the {}th global model".format(self.state))
                
                if self.state < self.total_round_num:
                    # Move to next round of training
                    logger.info(
                        f'----------- Starting a new training round (Round '
                        f'#{self.state}) -------------')
                    

                    # 设置trainer.state
                    self.trainer.set_state(self.state)
                    # 获取虚拟数据
                    # print("server: len(local_proto_msgs) = ", len(self.local_proto_msgs))
                    if self.state <= self._cfg.fedpg.upper_bound:
                        self.gen_vir_data()
                        self.local_proto_msgs.clear()
                        
                    # Clean the msg_buffer
                    self.msg_buffer['train'][self.state - 1].clear()
                    self.msg_buffer['train'][self.state] = dict()
                    self.staled_msg_buffer.clear()
                    # Start a new training round
                    self._start_new_training_round(aggregated_num)
                else:
                    # Final Evaluate
                    logger.info('Server: Training is finished! Starting '
                                'evaluation.')

                    # 显示虚拟数据
                    self.show_generated_images()
                    self.eval()

            else:
                # Receiving enough feedback in the evaluation process
                self._merge_and_format_eval_results()
                if self.state >= self.total_round_num:
                    self.is_finish = True

        else:
            move_on_flag = False

        return move_on_flag

    def callback_for_local_proto_msg(self, message: Message):

        # Message(msg_type='local_proto_msg',
        #         sender=self.ID,
        #         receiver=[self.server_id],
        #         timestamp=0,
        #         content=local_proto)
        # local_proto = {label: proto}
        # print("server receives local prototypes of client#", message.sender)
        self.local_proto_msgs.append(message)
        
        if len(self.local_proto_msgs) == self.sample_client_num:
            move_on_flag = self.check_and_move_on()
            if self._cfg.asyn.use and self._cfg.asyn.broadcast_manner == \
                    'after_receiving':
                self.broadcast_model_para(msg_type='model_para',
                                          sample_client_num=1)
        
        # print("local_distribution_msgs ++")

    def gen_vir_data(self):
        # 聚合局部原型，得到全局原型
        # print("server: len(local_proto_msgs) = ", len(self.local_proto_msgs))
        self.set_global_proto(self.local_proto_msgs)

        # 训练GAN
        self.global_proto = self.train_gan(self.gan_model, self.models[0], self.global_proto, self.old_global_proto)
        # 生成虚拟数据dataloader
        self.vir_dataloader = get_virtual_dataloader(self.gan_model, self.num_classes,
                                                     self._cfg.fedpg.vdata_per_class,
                                                     self._cfg.fedpg.v_batch_size)
        # 向客户端广播虚拟数据
        receiver = list(self.comm_manager.neighbors.keys())
        self.comm_manager.send(
            Message(msg_type='vir_dataloader_msg',
                    sender=self.ID,
                    receiver=receiver,
                    state=None,
                    timestamp=self.cur_timestamp,
                    content=self.vir_dataloader))

        self.old_global_proto = self.global_proto.to('cpu')
        # print("old_global_proto.size() = ", self.old_global_proto.size())
        # self.global_proto = None

    # 定义GAN的训练函数
    # 服务器训练GAN
    def train_gan(self, generator, global_model, global_proto, old_global_proto):
        # print("Start to train GAN")
        n_round = 300# default=300
        batch_size = 128 # default=64
        
        scale = 2# 蒸馏温度default=2
        reg = 0.001# 距离权重default=0.001

        generator.reset_global_model(copy.deepcopy(global_model))
        generator.global_model.to(self.device)
        generator.reset_proto_dist(self.num_classes, self.feature_dim, global_proto, old_global_proto)
        gan_optimizer = optim.Adam(generator.parameters(), lr=0.00002, betas=(0.5, 0.999))# lr=0.00002
        gan_scheduler = optim.lr_scheduler.StepLR(gan_optimizer, step_size=20, gamma=0.95)# step=20,gamma=0.95
        
        generator.train()
        generator.global_model.eval()

        # ori_proto = generator.proto_dist.proto.clone()
        # print("ori_proto: ", ori_proto)
        for _ in range(n_round):
            gan_optimizer.zero_grad()
            labels = [random.randint(0, self.num_classes - 1) for _ in range(batch_size)]
            labels = torch.LongTensor(labels).to(self.device)

            z, proto, dist, proto_loss = generator(labels)
            log_p_y = F.log_softmax(scale * dist, dim=1)
            
            loss1 = F.nll_loss(log_p_y, labels)
            loss2 = reg * regularization(z, proto, labels)
            # loss2=0
            loss = proto_loss + loss1 + loss2

            loss.backward()
            gan_optimizer.step()
            gan_scheduler.step()

        # save gan model
        path = '../models/gan/' +'for_{}.pt'.format(self._cfg.data.type)
        torch.save(generator, path)

        tra_proto = generator.proto_dist.proto.clone()
        # print("tra_proto: ", tra_proto)
        # print("change: ", tra_proto - ori_proto)
        return tra_proto


#################################### client ####################################
# ==============================================================================#
## client：1）收到新一轮全局模型时，上传局部原型；2）利用虚拟数据辅助本地训练
import logging
logger = logging.getLogger(__name__)


def local_train_dataloader2label_tensor_dict(data):
    label_tensor_dict = {}

    dataloader = data['train']
    for batch in dataloader:
        images, labels = batch

        for label, image in zip(labels, images):
            label = int(label)  # 将标签转换为整数，如果标签是字符串或其他类型
            if label not in label_tensor_dict:
                label_tensor_dict[label] = []

            label_tensor_dict[label].append(image)

    return label_tensor_dict
    # {'label':[tensor, ..., tensor]}


class FedPGClient(Client):
    def __init__(self,
                 ID=-1,
                 server_id=None,
                 state=-1,
                 config=None,
                 data=None,
                 model=None,
                 device='cpu',
                 strategy=None,
                 is_unseen_client=False,
                 *args,
                 **kwargs):

        super(FedPGClient, self).__init__(ID,
                                          server_id,
                                          state,
                                          config,
                                          data,
                                          model,
                                          device,
                                          strategy,
                                          is_unseen_client,
                                          *args,
                                          **kwargs)

        self.register_handlers(
            msg_type='vir_dataloader_msg',
            callback_func=self.callback_for_vir_dataloader_msg)

        self.label_tensor_dict = local_train_dataloader2label_tensor_dict(self.data)
        self.n_class = self._cfg.model.out_channels
        self.n_support = 20

    # 更新本地训练时使用的虚拟数据集
    def callback_for_vir_dataloader_msg(self, message: Message):
        vir_dataloader = message.content
        self.trainer.reset_vir_dataloader(vir_dataloader)

    # 修改 “处理全局模型消息” 函数
    # 收到全局模型时，计算并上传本地原型
    def callback_funcs_for_model_para(self, message: Message):
        # 设置trainer.state
        self.trainer.set_state(message.state)
        
        # 更新本地模型，并上传
        super().callback_funcs_for_model_para(message)   
        
        # 计算本地原型
        if self.state <= self._cfg.fedpg.upper_bound:
            local_proto = self.get_local_proto(self.model)
            # 上传本地原型
            self.comm_manager.send(
                Message(msg_type='local_proto_msg',
                        sender=self.ID,
                        receiver=[message.sender],
                        state=self.state,
                        timestamp=message.timestamp,
                        content=local_proto))
            # print("client#", self.ID, " upload local proto")

    # 为在接受到新一轮全局模型时，随机选择‘支持集’来计算本地原型
    # 根据客户端本地数据的dataloader，创建以label为key，以对应tensor集合为value的字典

    # client抽取支持集并计算原型
    def get_local_proto(self, global_model):
        label_proto_dict = {}
        global_model.eval()

        for label in range(self.n_class):
            if label in self.label_tensor_dict:
                class_tensor = self.label_tensor_dict[label]
                index = [random.randint(0, len(class_tensor)-1) for _ in range(self.n_support)] if len(class_tensor) > self.n_support else range(len(class_tensor))
                support_tensor = torch.stack([class_tensor[i] for i in index])
                proto = torch.mean(global_model.get_z(support_tensor), dim=0)
                label_proto_dict[label] = proto

        global_model.train()
        return label_proto_dict

def call_fedpg_worker(method):
    if method == 'fedpg':
        worker_builder = {'client': FedPGClient, 'server': FedPGServr}
        return worker_builder


register_worker('fedpg', call_fedpg_worker)