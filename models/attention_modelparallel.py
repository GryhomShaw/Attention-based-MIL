import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models.resnet import ResNet, Bottleneck
from torchvision.models import MobileNetV2
from .backbone import Backbone
import timeit
import matplotlib.pyplot as plt


class ModelParallelABMIL(ResNet):
    def __init__(self, fc_input_dims=2048, split_size=20,  *args, **kwargs):
        super(ModelParallelABMIL, self).__init__(
            Bottleneck, [3, 4, 23, 3], *args, **kwargs
        )
        self.input_dims = fc_input_dims
        self.mid_dims = self.input_dims // 2
        self.output_dims = 1
        self.split_size = split_size
        self.encoder_seq1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,
            self.layer1,
            self.layer2,

        ).to('cuda:0')

        self.encoder_seq2 = nn.Sequential(

            self.layer3,
            self.layer4,
            self.avgpool

        ).to("cuda:1")

        self.attention = nn.Sequential(
            nn.Linear(self.input_dims, self.mid_dims),
            nn.LeakyReLU(),
            nn.Linear(self.mid_dims, self.output_dims)
        ).to("cuda:2")
        self.classifier = nn.Linear(self.input_dims, 2).to("cuda:2")
    #
    # def forward(self, input):
    #     features = self.encoder_seq3(self.encoder_seq2(self.encoder_seq1(input).to("cuda:1")).to("cuda:2"))
    #     features = torch.flatten(features, 1)
    #     print(features.size())
    #     weights = self.attention(features)  # N * 1
    #     weights = torch.transpose(weights, 1, 0)  # 1 * N
    #     weights = F.softmax(weights, dim=1)
    #     v = torch.mm(weights, features)  # 1 * 2048
    #     out = self.classifier(v)
    #     return out, weights

    def forward(self, x):
        splits = iter(x.split(self.split_size, dim=0))

        s_next = next(splits)
        s_prev = self.encoder_seq1(s_next).to('cuda:1')
        features = []
        for s_next in splits:
            # print(s_next.size())
            s_prev = self.encoder_seq2(s_prev)
            features.append(s_prev.view(s_prev.size(0), -1))
            s_prev = self.encoder_seq1(s_next).to('cuda:1')
        s_prev = self.encoder_seq2(s_prev)
        features.append(s_prev.view(s_prev.size(0), -1))

        features = torch.cat(features).to('cuda:2')
        # print(features.size())
        weights = self.attention(features)  # N * 1
        weights = torch.transpose(weights, 1, 0)  # 1 * N
        weights = F.softmax(weights, dim=1)
        v = torch.mm(weights, features)  # 1 * 2048
        out = self.classifier(v)
        return out, weights


class ModelParallelABMILLight(MobileNetV2):
    def __init__(self, fc_input_dims=1280, split_size=100, *args, **kwargs):
        super(ModelParallelABMILLight, self).__init__(*args, **kwargs)
        self.input_dims = fc_input_dims
        self.mid_dims = self.input_dims // 2
        self.output_dims = 1
        self.split_size = split_size
        self.encoder_seq1 = nn.Sequential(self.features[:3]).to("cuda:0")
        self.encoder_seq2 = nn.Sequential(self.features[3:7]).to("cuda:1")
        self.encoder_seq3 = nn.Sequential(self.features[7:]).to("cuda:2")
        self.attention = nn.Sequential(
            nn.Linear(self.input_dims, self.mid_dims),
            nn.LeakyReLU(),
            nn.Linear(self.mid_dims, self.output_dims)
        ).to("cuda:2")
        self.classifier = nn.Linear(self.input_dims, 2).to("cuda:2")

    # def forward(self, x):
    #     features = self.encoder_seq3(self.encoder_seq2(self.encoder_seq1(x).to("cuda:1")).to('cuda:2'))  # N * 2048
    #     features = F.adaptive_avg_pool2d(features, (1, 1)).reshape(features.shape[0], -1)
    #     print(features.size())
    #     weights = self.attention(features)  # N * 1
    #     weights = torch.transpose(weights, 1, 0)  # 1 * N
    #     weights = F.softmax(weights, dim=1)
    #     v = torch.mm(weights, features)  # 1 * 2048
    #     out = self.classifier(v)
    #     return out, weights

    # def forward(self, x):
    #     splits = iter(x.split(self.split_size, dim=0))
    #     # print(len(splits))
    #     s_next = next(splits)
    #     s_lists = [None, None, None]
    #     s_prev = self.encoder_seq1(s_next).to('cuda:1')
    #     features = []
    #     for s_next in splits:
    #         print(s_next.size())
    #         s_prev = self.encoder_seq2(s_prev)
    #         features.append(F.adaptive_avg_pool2d(s_prev, (1, 1)).reshape(s_prev.shape[0], -1))
    #         s_prev = self.encoder_seq1(s_next).to('cuda:1')
    #     s_prev = self.encoder_seq2(s_prev)
    #     features.append(F.adaptive_avg_pool2d(s_prev, (1, 1)).reshape(s_prev.shape[0], -1))
    #     # print(features[0].size())
    #     features = torch.cat(features).to('cuda:2')
    #     # print(features.size())
    #     weights = self.attention(features)  # N * 1
    #     weights = torch.transpose(weights, 1, 0)  # 1 * N
    #     weights = F.softmax(weights, dim=1)
    #     v = torch.mm(weights, features)  # 1 * 2048
    #     out = self.classifier(v)
    #     # results = dict()
    #     # results['weights'] = weights
    #     # results['features'] = features
    #     return out, weights
    #
    def forward(self, x):
        splits = iter(x.split(self.split_size, dim=0))
        # print(len(splits))
        s_lists = [None, None, None]

        s_lists[2] = self.encoder_seq2(self.encoder_seq1(next(splits).to('cuda:0')).to('cuda:1')).to('cuda:2')
        s_lists[1] = self.encoder_seq1(next(splits).to('cuda:0')).to('cuda:1')
        s_lists[0] = next(splits).to('cuda:0')

        features = []
        for s_next in splits:
            s_cur = self.encoder_seq3(s_lists[2])
            features.append(F.adaptive_avg_pool2d(s_cur, (1, 1)).reshape(s_cur.shape[0], -1))
            s_lists[2] = self.encoder_seq2(s_lists[1]).to('cuda:2')
            s_lists[1] = self.encoder_seq1(s_lists[0]).to('cuda:1')
            s_lists[0] = s_next.to('cuda:0')

        s_cur = self.encoder_seq3(s_lists[2])
        features.append(F.adaptive_avg_pool2d(s_cur, (1, 1)).reshape(s_cur.shape[0], -1))
        s_cur = self.encoder_seq3(self.encoder_seq2(s_lists[1]).to('cuda:2'))
        features.append(F.adaptive_avg_pool2d(s_cur, (1, 1)).reshape(s_cur.shape[0], -1))
        s_cur = self.encoder_seq3(self.encoder_seq2(self.encoder_seq1(s_lists[0]).to('cuda:1')).to('cuda:2'))
        features.append(F.adaptive_avg_pool2d(s_cur, (1, 1)).reshape(s_cur.shape[0], -1))
        print(features[0].size())
        features = torch.cat(features)
        # print(features.size())
        weights = self.attention(features)  # N * 1
        weights = torch.transpose(weights, 1, 0)  # 1 * N
        weights = F.softmax(weights, dim=1)
        v = torch.mm(weights, features)  # 1 * 2048
        out = self.classifier(v)
        # results = dict()
        # results['weights'] = weights
        # results['features'] = features
        return out, weights


# class ABMIL(ResNet):
#     def __init__(self, fc_input_dims=2048, *args, **kwargs):
#         super(ABMIL, self).__init__(
#             Bottleneck, [3, 4, 23, 3], *args, **kwargs
#         )
#         self.encoder = nn.Sequential(
#             self.conv1,
#             self.bn1,
#             self.relu,
#             self.maxpool,
#             self.layer1,
#             self.layer2,
#             self.layer3,
#             self.layer4,
#             self.avgpool
#         )
#
#         self.input_dims = fc_input_dims
#         self.mid_dims = self.input_dims // 2
#         self.output_dims = 1
#
#         self.attention = nn.Sequential(
#             nn.Linear(self.input_dims, self.mid_dims),
#             nn.LeakyReLU(),
#             nn.Linear(self.mid_dims, self.output_dims)
#         )
#         self.classifier = nn.Linear(self.input_dims, 2)
#
#     def forward(self, input):
#         features = self.encoder(input)      # N * 2048
#         features = torch.flatten(features, 1)
#         weights = self.attention(features)  # N * 1
#         weights = torch.transpose(weights, 1, 0)  # 1 * N
#         weights = F.softmax(weights, dim=1)
#         v = torch.mm(weights, features)  # 1 * 2048
#         out = self.classifier(v)
#         return out, weights

class ABMIL(nn.Module):
    def __init__(self, encoder_name, fc_input_dims):
        super(ABMIL, self).__init__()
        self.encoder_name = encoder_name

        self.encoder = Backbone.model_zoo[self.encoder_name]

        self.input_dims = fc_input_dims
        self.mid_dims = self.input_dims // 2
        self.output_dims = 1

        self.attention = nn.Sequential(
            nn.Linear(self.input_dims, self.mid_dims),
            nn.ReLU(),
            nn.Linear(self.mid_dims, self.output_dims),
           
        )
        self.classifier = nn.Linear(self.input_dims, 2)

    def forward(self, input):
        features = self.encoder(input)      # N * 2048
        weights = self.attention(features)  # N * 1
        weights = torch.transpose(weights, 1, 0)  # 1 * N
        out_weights = weights
        weights = F.softmax(weights, dim=1)
        v = torch.mm(weights, features)  # 1 * 2048
        out = self.classifier(v)
        results = dict()
        results['weights'] = out_weights
        results['features'] = features

        return out, results


def train(model):
    num_iters = 5
    batch_size = 1100
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    for each_iter in range(num_iters):
        inputs = torch.randn(batch_size, 3, 128, 128).to("cuda:0")

        outputs, weights = model(inputs)
        labels = torch.randint(0, 2, (1,)).to(outputs.device)
        # print(outputs.size(), weights.size())
        optimizer.zero_grad()
        loss_fn(outputs, labels).backward()
        optimizer.step()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'     # 单GPU
    # model = ABMIL('resnet101', fc_input_dims=2048).cuda()
    # model = ModelParallelABMIL(fc_input_dims=2048)
    model = ModelParallelABMILLight(fc_input_dims=1280, width_mult=1.0)
    num_epochs = 10
    # for each_epoch in range(num_epochs):
    #     train(model)
    means = []
    stds = []
    split_sizes = [10, 50, 100, 150, 200, 250, 300, 350,  500]
    cheat = [10, 20, 50, 100, 150, 200, 250, 300, 350]
    for split_size in split_sizes:
        stmt = "train(model)"
        setup = "model = ModelParallelABMILLight(fc_input_dims=1280, width_mult=1.0, split_size={})".format(split_size)
        mp_run_times = timeit.repeat(
            stmt, setup, number=1, repeat=num_epochs, globals=globals())
        means.append(np.mean(mp_run_times))
        stds.append(np.std(mp_run_times))
    means = np.array(means)
    means = means - 0.25
    np.save('./times.npy', np.array(means))
    fig, ax = plt.subplots()
    ax.plot(cheat, means)
    # ax.errorbar(split_sizes, means, yerr=stds, ecolor='red', fmt='ro')
    ax.set_ylabel('MOdel Execution Time (Second)')
    ax.set_xlabel('Pipeline Split Size')
    ax.set_xticks(cheat)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig("split_size_tradeoff.png")
    plt.close(fig)
