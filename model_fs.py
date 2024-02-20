from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import kmeans_lloyd


class ResNet18(nn.Module):
    def __init__(self, pretrained=False, num_classes=7, drop_rate=0):
        super(ResNet18, self).__init__()
        self.drop_rate = drop_rate
        resnet = models.resnet18(pretrained=True)
        self.layer0 = nn.Sequential(*list(resnet.children())[0:4])
        self.layer1 = nn.Sequential(*list(resnet.children())[4:5])
        self.layer2 = nn.Sequential(*list(resnet.children())[5:6])
        self.layer3 = nn.Sequential(*list(resnet.children())[6:7])
        self.layer4 = nn.Sequential(*list(resnet.children())[7:9])
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # x = self.features(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.drop_rate > 0:
            x = nn.Dropout(self.drop_rate)(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)

        return out, x


class ResNet18_Scale(nn.Module):
    def __init__(self, pretrained=True, num_classes=7, drop_rate=0):
        super(ResNet18_Scale, self).__init__()
        self.drop_rate = drop_rate
        resnet = models.resnet18(pretrained)
        self.layer0 = nn.Sequential(*list(resnet.children())[0:4])
        self.layer1 = nn.Sequential(*list(resnet.children())[4:5])
        self.layer2 = nn.Sequential(*list(resnet.children())[5:6])
        self.layer3 = nn.Sequential(*list(resnet.children())[6:7])
        self.layer4 = nn.Sequential(*list(resnet.children())[7:8])
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # before avgpool 512x1
        self.arrangement4 = nn.PixelShuffle(16)
        self.arrangement3 = nn.PixelShuffle(8)
        self.arrangement2 = nn.PixelShuffle(4)
        self.arrangement1 = nn.PixelShuffle(2)
        # self.arm = Amend_raf()
        self.fc = nn.Linear(121, num_classes)


        self.conv4_1 = nn.Conv2d(2, 128, kernel_size=32, stride=8, padding=0, bias=False)
        self.conv3_1 = nn.Conv2d(4, 128, kernel_size=16, stride=10, padding=2, bias=False)
        self.conv2_1 = nn.Conv2d(8, 128, kernel_size=8, stride=11, padding=3, bias=False)
        self.conv1_1 = nn.Conv2d(16, 128, kernel_size=4, stride=11, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(128)
        self.conv = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv_w = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.ln = nn.Linear(128, 1)
        self.sm = nn.Sigmoid()

        self.conv2_2 = nn.Sequential(nn.Sigmoid(), nn.Conv2d(128, 4, kernel_size=3, stride=1, padding=1, bias=True))
        self.conv4_2 = nn.Sequential(nn.Sigmoid(), nn.Conv2d(128, 4, kernel_size=3, stride=1, padding=1, bias=True))
        self.conv1_2 = nn.Sequential(nn.Sigmoid(), nn.Conv2d(128, 4, kernel_size=3, stride=1, padding=1, bias=True))
        self.conv3_2 = nn.Sequential(nn.Sigmoid(), nn.Conv2d(128, 4, kernel_size=3, stride=1, padding=1, bias=True))
        self.img_p = nn.AdaptiveAvgPool2d(56)
        self.img_l = nn.Linear(56 * 56, 484)  # output layer
        self.predictor = nn.Sequential(nn.Linear(484, 1024, bias=False),
                                       nn.BatchNorm1d(1024),
                                       nn.ReLU(inplace=True),  # hidden layer
                                       nn.Linear(1024, 484))  # output layer


    def forward(self, x):

        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x12 = x1
        x2 = self.layer2(x1)
        x22 = x2
        x3 = self.layer3(x2)
        x32 = x3
        x4 = self.layer4(x3)
        x42 = x4

        x0 = torch.mean(x, 1, keepdim=True)
        x0 = self.img_p(x0)
        x0 = x0.view(x0.size(0), -1)
        x0 = self.img_l(x0)

        x1 = self.arrangement1(x1)
        x1 = self.conv1_1(x1)
        x2 = self.arrangement2(x2)
        x2 = self.conv2_1(x2)
        x3 = self.arrangement3(x3)
        x3 = self.conv3_1(x3)
        x4 = self.arrangement4(x4)  # BX32X28X28
        x4 = self.conv4_1(x4)
        x_cat = x4 + x3 + x2 + x1

        x4 = self.conv4_2(x4)
        x3 = self.conv3_2(x3)
        x2 = self.conv2_2(x2)
        x1 = self.conv1_2(x1)
        x4 = x4.view(x4.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x3 = x3.view(x3.size(0), -1)
        x1 = x1.view(x1.size(0), -1)
        x_out = [x1, x2, x3, x4]

        x42 = hash_coding(x42)
        x32 = hash_coding(x32)
        x22 = hash_coding(x22)
        x12 = hash_coding(x12)


        batch_size = x.size(0)
        cen_all = []
        m2_all = []
        m1_all = []
        m3_all = []
        x22 = x22.cuda()
        x32 = x32.cuda()
        x12 = x12.cuda()
        for dim in range(batch_size):
            label_cs, centroid = kmeans_lloyd.lloyd(x42[dim], 4, device=5, tol=1e-4)
            label_cs = label_cs.cuda()
            centroid = centroid.cuda()

            d20 = F.cosine_similarity(x22[dim], centroid[0].unsqueeze(0), dim=1, eps=1e-8).unsqueeze(0)
            d21 = F.cosine_similarity(x22[dim], centroid[1].unsqueeze(0), dim=1, eps=1e-8).unsqueeze(0)
            d22 = F.cosine_similarity(x22[dim], centroid[2].unsqueeze(0), dim=1, eps=1e-8).unsqueeze(0)
            d23 = F.cosine_similarity(x22[dim], centroid[3].unsqueeze(0), dim=1, eps=1e-8).unsqueeze(0)

            d30 = F.cosine_similarity(x32[dim], centroid[0].unsqueeze(0), dim=1, eps=1e-8).unsqueeze(0)
            d31 = F.cosine_similarity(x32[dim], centroid[1].unsqueeze(0), dim=1, eps=1e-8).unsqueeze(0)
            d32 = F.cosine_similarity(x32[dim], centroid[2].unsqueeze(0), dim=1, eps=1e-8).unsqueeze(0)
            d33 = F.cosine_similarity(x32[dim], centroid[3].unsqueeze(0), dim=1, eps=1e-8).unsqueeze(0)

            d10 = F.cosine_similarity(x12[dim], centroid[0].unsqueeze(0), dim=1, eps=1e-8).unsqueeze(0)
            d11 = F.cosine_similarity(x12[dim], centroid[1].unsqueeze(0), dim=1, eps=1e-8).unsqueeze(0)
            d12 = F.cosine_similarity(x12[dim], centroid[2].unsqueeze(0), dim=1, eps=1e-8).unsqueeze(0)
            d13 = F.cosine_similarity(x12[dim], centroid[3].unsqueeze(0), dim=1, eps=1e-8).unsqueeze(0)

            data2_cs = torch.cat([d20, d21, d22, d23], 0).t()
            _, label2_cs = torch.max(data2_cs, 1)

            data3_cs = torch.cat([d30, d31, d32, d33], 0).t()
            _, label3_cs = torch.max(data3_cs, 1)

            data1_cs = torch.cat([d10, d11, d12, d13], 0).t()
            _, label1_cs = torch.max(data1_cs, 1)
            # print(label1_cs)

            m2 = []
            m3 = []
            m1 = []
            for j in range(4):
                if min(x22[dim][(label2_cs == j)].shape) == 0:
                    m2.append(centroid[j])
                else:
                    m2.append(x22[dim][(label2_cs == j)].mean(dim=0))
                if min(x32[dim][(label3_cs == j)].shape) == 0:
                    m3.append(centroid[j])
                else:
                    m3.append(x32[dim][(label3_cs == j)].mean(dim=0))
                if min(x12[dim][(label1_cs == j)].shape) == 0:
                    m1.append(centroid[j])
                else:
                    m1.append(x12[dim][(label1_cs == j)].mean(dim=0))
            m2 = torch.stack(m2)
            m3 = torch.stack(m3)
            m1 = torch.stack(m1)
            cen_all.append(centroid)
            m2_all.append(m2)
            m3_all.append(m3)
            m1_all.append(m1)

        cen_all = torch.stack(cen_all)

        m2_all = torch.stack(m2_all)
        m3_all = torch.stack(m3_all)
        m1_all = torch.stack(m1_all)


        x_cat = self.conv(x_cat)
        x_cat = self.bn(x_cat)
        global_mean = x_cat.mean(dim=[0, 1])  # 11X11
        xmean = torch.mean(x_cat, 1, keepdim=True)  # 128X1X11X11
        x = xmean + global_mean

        if self.drop_rate > 0:
            x = nn.Dropout(self.drop_rate)(x)

        x = x.view(x.size(0), -1)
        out = self.fc(x)

        return out, cen_all, m2_all, m3_all, m1_all, x_out, x

        # return out, cen_all, m2_all, m3_all, m1_all, x_out, x, label1_cs, label2_cs, label3_cs, label_cs



def hash_coding(output):
    # s_index = s_index.float()
    # print type(s_index)
    feat_avg_shape = (8, 8)  # after downsampling featmap size
    feat_size = feat_avg_shape[0]
    output_size0 = output.size(0)
    output_size1 = output.size(1)
    feat_avg_4_4_pool_torch = F.avg_pool2d(output, kernel_size=3).view(output_size0, output_size1,
                                                                       -1)
    top2_ind = torch.topk(feat_avg_4_4_pool_torch, 2)  # get top 2 index
    top2_ind_row = (top2_ind[1] // feat_size).float()
    top2_ind_col = (top2_ind[1] % feat_size).float()

    top2_pos = torch.cat((top2_ind_row[:, :, 0].unsqueeze(dim=-1), top2_ind_col[:, :, 0].unsqueeze(dim=-1),
                          # get first and second position (x1,y1,x2,y2)
                          top2_ind_row[:, :, 1].unsqueeze(dim=-1), top2_ind_col[:, :, 1].unsqueeze(dim=-1)), dim=-1)
    return top2_pos

