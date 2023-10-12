import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterLoss(nn.Module):
    def __init__(self, num_classes, dim):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.centers = nn.Parameter(torch.rand(self.num_classes, self.dim))

    def forward(self, x, y):
        batch_size = x.size(0)  # torch size (128,2)
        d1 = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes)
        d2 = torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        d = d1 + d2
        d.addmm(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long().cuda()
        y = y.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = y.eq(classes.expand(batch_size, self.num_classes))

        dist = d * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        

        return loss