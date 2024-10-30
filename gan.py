import random
import numpy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F
from utils import euclidean_dist, proto_loss
from torch.utils.data import TensorDataset, DataLoader


class Generative(nn.Module):
    def __init__(self, noise_dim, num_classes, hidden_dim, output_dim, image_size, device) -> None:
        # 输入noise ++ label，输出output_dim = channel * h * w的虚拟图片的张量
        super().__init__()

        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.device = device
        self.channel, self.height, self.width = image_size[0], image_size[1], image_size[2]
        self.proto_dist = None
        self.global_model = None

        self.layer1 = nn.Sequential(
            nn.Linear(noise_dim + num_classes, hidden_dim[0]),
            nn.BatchNorm1d(hidden_dim[0]),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.BatchNorm1d(hidden_dim[1]),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim[1], hidden_dim[2]),
            nn.BatchNorm1d(hidden_dim[2]),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer4 = nn.Sequential(
            nn.Linear(hidden_dim[2], hidden_dim[3]),
            nn.BatchNorm1d(hidden_dim[3]),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer5 = nn.Sequential(
            nn.Linear(hidden_dim[3], output_dim),
            nn.Tanh()
        )

    def forward(self, labels):
        images = self.get_output(labels)
        z = self.global_model.get_z(images)
        proto, dist, proto_loss = self.proto_dist(z)
        
        return z, proto, dist, proto_loss
        
    def get_output(self, labels):
        batch_size = labels.shape[0]
        eps = torch.rand((batch_size, self.noise_dim), device=self.device)
        y_input = F.one_hot(labels, self.num_classes).to(self.device)
        x = torch.cat((eps, y_input), dim=1)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        out = self.layer5(x4)
        out = out.view(batch_size, self.channel, self.height, self.width)

        return out

    def reset_proto_dist(self, n_classes, feat_dim, global_proto, old_global_proto=None):
        self.proto_dist = ProtoDist(n_classes, feat_dim, global_proto, old_global_proto).to(self.device)

    def reset_global_model(self, global_model):
        self.global_model = global_model



class ProtoDist(nn.Module):
    def __init__(self, n_classes, feat_dim, global_proto, old_global_proto=None):
        super(ProtoDist, self).__init__()
        self.n_classes = n_classes
        self.feat_dim = feat_dim

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # if old_global_proto is None:
        #     proto = torch.stack([global_proto[i].clone().detach() if i in global_proto
        #                               else torch.randn(feat_dim) for i in range(n_classes)]).cuda()
        #
        # else:
        #     proto = torch.stack([global_proto[i].clone().detach() if i in global_proto
        #                               else old_global_proto[i, :] for i in range(n_classes)]).cuda()

        ## all trainable
        # self.proto = nn.Parameter(torch.randn(self.n_classes, feat_dim).cuda(), requires_grad=True)

    
        ## trainable and approx to prototype
        self.index = []
        global_proto_list = []
        for i in range(n_classes):
            if i in global_proto:
                global_proto_list.append(global_proto[i].clone())
                self.index.append(i)
            else:
                global_proto_list.append(torch.zeros(feat_dim))

        self.global_proto = nn.Parameter(torch.stack(global_proto_list).cuda(), requires_grad=False)
        self.proto = nn.Parameter(torch.randn(self.n_classes, feat_dim).cuda(), requires_grad=True)

        ## partitial trainable
        
        # self.proto =  nn.Parameter(proto, requires_grad=True)
        # for i in range(n_classes):
        #     if i in global_proto:
        #         self.register_buffer(f'freezed_proto_{i}', proto[i, :].detach())
        #         self.proto.data[i, :] = self.proto.data[i, :].detach()
        #         self.proto.data[i, :].requires_grad_(False)
        
        # nnpara = nn.ParameterList()
        # for i in range(n_classes):
        #     if i in global_proto:
        #         nnpara.append(nn.Parameter(global_proto[i], requires_grad=False))
        #     else:
        #         nnpara.append(nn.Parameter(torch.randn(feat_dim).cuda(), requires_grad=True))
        # self.proto = torch.tensor(nnpara).cuda()

        ## detach
        # self.proto = proto
                

    def forward(self, z):
        # dists = euclidean_dist(z, self.proto)

        features_square=torch.sum(torch.pow(z,2),1, keepdim=True)
        centers_square=torch.sum(torch.pow(torch.t(self.proto),2),0, keepdim=True)
        features_into_centers=2*torch.matmul(z, torch.t(self.proto))
        dist=features_square+centers_square-features_into_centers

        proto_dist = euclidean_dist(self.proto, self.global_proto)
        loss0 = proto_loss(self.index, proto_dist)
        # loss0 = 0

        return self.proto, -dist, loss0

def regularization(features, centers, labels):
        distance=(features-centers[labels])

        distance=torch.sum(torch.pow(distance,2),1, keepdim=True)

        distance=(torch.sum(distance, 0, keepdim=True))/features.shape[0]

        return distance


# 显示output图片
def show_output(output):
    # 显示虚拟图片
    output = output.detach().numpy()
    fig, axes = plt.subplots(5, math.ceil(output.shape[0] / 5), figsize=(8, 8))
    for i, ax in enumerate(axes.flatten()):
        img = output[i].transpose(1, 2, 0)  # 通道顺序调整为 H x W x C
        ax.imshow((img + 1) / 2)  # 将数据从 [-1, 1] 转换到 [0, 1]
        ax.axis('off')
    plt.show()


# 生成虚拟数据并加载为dataloader
def get_virtual_dataloader(generator, num_classes=10, num_per_class=50, v_batch_size=16):
    # 调用训练好的generator，生成虚拟数据
    generator.eval()
    labels = torch.tensor(list(range(num_classes)) * num_per_class)
    generated_images = generator.get_output(labels)
    generated_images = generated_images.detach()
    generator.train()

    # 把虚拟数据读取为dataloader
    dataset = TensorDataset(generated_images, labels)
    vir_dataloader = DataLoader(dataset, batch_size=v_batch_size, shuffle=True)
    return vir_dataloader


# if __name__ == '__main__':
#     noise_dim = 128
#     num_classes = 16
#     hidden_dim = [128, 256, 512, 1024]
#     output_dim = 3 * 28 * 28
#     image_size = [3, 28, 28]  # C * H * W
#     model = Generative(noise_dim=noise_dim, num_classes=num_classes, hidden_dim=hidden_dim,
#                        output_dim=output_dim, image_size=image_size, device="cpu")
#
#     # input = torch.randint(0, num_classes-1, (5,))
#     # print(input.shape)
#
#     # out = model(input)
#     # print(out.shape)
#
#     # show_output(out)
#     show_generated_images(model)