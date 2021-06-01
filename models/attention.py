import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from .backbone import Backbone


class ABMIL(nn.Module):
    def __init__(self, encoder_name, fc_input_dims):
        super(ABMIL, self).__init__()
        self.encoder_name = encoder_name
        self.encoder = Backbone.model_zoo[self.encoder_name](pretrained=False, num_classes=2)
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
        # print(features.size())
        weights = self.attention(features)  # N * 1
        weights = torch.transpose(weights, 1, 0)  # 1 * N
        out2 = weights.clone().squeeze().detach()
        weights = torch.sigmoid(weights)
        v1 = torch.mm(weights, features)  # 1 * 2048
        v2 = torch.mean(features, dim=0)
        v = (v1 + v2) / 2
        # print(weights)
        out = self.classifier(v)
        return out, out2


def train(model):
    num_iters = 5
    batch_size = 20
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    for each_iter in range(num_iters):
        inputs = torch.randn(batch_size, 3, 128, 128).cuda()
        labels = torch.randint(0, 2, (1,)).cuda()
        outputs, weights = model(inputs)
        # print(outputs.size(), weights.size())
        optimizer.zero_grad()
        loss_fn(outputs, labels).backward()
        optimizer.step()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'     # 单GPU
    model = ABMIL("resnet101", 2048).cuda()
    num_epochs = 10
    for each_epoch in range(num_epochs):
        train(model)
