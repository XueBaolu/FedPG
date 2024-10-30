# 导入环境
import sys
sys.path.append(r'./FederatedScope-master')
# 配置参数的定义
from federatedscope.core.configs.config import global_cfg
cfg = global_cfg.clone()

## fedpg算法参数
from xbl_codes.fed_proto_gan import trainer
from xbl_codes.fed_proto_gan.server import FedPGServr, FedPGClient
cfg.expname = 'mode1_test'
cfg.federate.method = 'fedpg'
cfg.trainer.type = 'fedpg_trainer'
cfg.fedpg.vdata_per_class = 50 
cfg.fedpg.weight_vdata = 0.5
cfg.fedpg.weight_align = 0.1

# 模型参数
cfg.model.type = 'resnet18'
cfg.model.in_channels = 3
cfg.model.out_channels = 10

# 数据参数
from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.cv.dataset  import cifar

cfg.data.type = 'cifar10'
cfg.data.root = './data'

cfg.data.splits = [0.85, 0.05, 0.1]

# 必须要加的超参数，否则acc=0.0(但是对于femnist来说不需要lda划分
cfg.data.splitter = 'lda'
cfg.data.splitter_args = [{'alpha':0.05}]

cfg.data.shuffle = True
cfg.data.batch_size = 128

# 训练参数
cfg.train.optimizer.lr = 0.01

cfg.train.optimizer.weight_decay = 0.0
cfg.grad.grad_clip = 5.0
cfg.early_stop.patience = 10

cfg.criterion.type = 'CrossEntropyLoss'

# 系统参数
# cfg.wandb.use = True
cfg.wandb.client_train_info = False

cfg.use_gpu = True
cfg.seed = 908

cfg.eval.freq = 1
cfg.eval.metrics = ['acc', 'f1']
cfg.eval.best_res_update_round_wise_key = 'test_acc'


cfg.federate.mode = 'standalone'
# cfg.federate.method = 'fedavg'
# cfg.federate.method = 'global'

cfg.federate.make_global_eval = True
cfg.federate.merge_test_data = True
cfg.federate.merge_val_data = True

cfg.federate.total_round_num = 200

cfg.train.batch_or_epoch = 'epoch'
cfg.train.local_update_steps = 1


cfg.federate.client_num = 10
cfg.federate.sample_client_num = 1

# cfg.federate.save_to = 'models/femnist_models/' # 模型保存位置

# 传参，记录
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
setup_seed(cfg.seed)
update_logger(cfg)

## 加载数据
data, modified_cfg = get_data(cfg.clone())
cfg.merge_from_other_cfg(modified_cfg)

# # FL开工！
from federatedscope.core.fed_runner import FedRunner
from federatedscope.core.auxiliaries.worker_builder import get_server_cls, get_client_cls

Fed_runner = FedRunner(data=data,
                       server_class=get_server_cls(cfg),
                       client_class=get_client_cls(cfg),
                       config=cfg.clone())
Fed_runner.run()