# -*- coding: utf-8 -*-
"""
@file      :  FpnAttentionEncoder.py
@Time      :  2022/11/5 18:04
@Software  :  PyCharm
@summary   :
@Author    :  Bajian Xiang
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import sqrt
from torchvision.models.resnet import ResNet
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.autograd import Variable
from model.backbone import build_backbone


def cal_count_len(valid_len):
    count_len = [0]
    for i in valid_len:
        count_len.append(count_len[-1] + i)
    return count_len


def diag_below_false(x):
    for i in range(len(x)):
        for j in range(i):
            x[i][j] = 0
    return x


# x = create_attention_mask([4, 5], 5, 4)
def create_attention_mask(valid_len, max_len, num_head):
    """
    :param valid_len: example: [4, 5]
    :param max_len:   max(valid_len) -> example: 5
    :param num_head:  nn.MultiHeadAttention的超参 -> example:4
    :param batch:     attention的input的batch == len(valid_len) * 7(倍频程频率数量) -> example: 2 * 7 == 14
    :return:          mask:Tensor([batch * num_head, max_len, max_len)] -> [2 * 4, 5, 5]
    """
    mask_list = []
    for sample_len in valid_len:

        # if sample_len < max_len:
        #     # 当前sample的slice数量小于最大slice数量时
        #     temp_mask = torch.ones(max_len, max_len)
        #     temp_mask[:sample_len, :sample_len] = 0
        #     # sample_len=4, max_len=5时, temp_mask为:
        #     # tensor([[0., 0., 0., 0., 1.],
        #     #         [0., 0., 0., 0., 1.],
        #     #         [0., 0., 0., 0., 1.],
        #     #         [0., 0., 0., 0., 1.],
        #     #         [1., 1., 1., 1., 1.]])
        # else:
        # 当前sample的slice数为最大slice数时，生成全是0的tensor，shape = [
        temp_mask = torch.zeros(max_len, max_len)
        # for i in range(len(temp_mask)):
        #     temp_mask[i][i] = 0

        for _ in range(num_head):
            mask_list.append(temp_mask.unsqueeze(0))  # 为每个注意力头分配一个mask, shape = [num_head, 5, 5]

    mask = torch.cat(mask_list, dim=0)  # shape = [sample_num * num_head, 5, 5]
    mask = mask.bool()
    return mask


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# Feed-Forward
class PositionalWiseFeedForward(nn.Module):

    def __init__(self, ln_out, resnet_out):
        super(PositionalWiseFeedForward, self).__init__()
        self.fc1_out = resnet_out  # 2048
        self.fc2_out = 512
        self.fc3_out = 256
        self.fc4_out = 64
        self.ln_out = ln_out
        self.resnet_out = resnet_out

        self.layer_norm = nn.LayerNorm(normalized_shape=resnet_out, elementwise_affine=False, eps=0)
        self.fc = nn.Sequential(
            nn.Linear(in_features=self.resnet_out, out_features=self.fc1_out),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=self.fc1_out),
            nn.Dropout(0.3),

            nn.Linear(in_features=self.fc1_out, out_features=self.fc2_out),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=self.fc2_out),
            nn.Dropout(0.2),

            nn.Linear(in_features=self.fc2_out, out_features=self.fc3_out),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=self.fc3_out),
            nn.Dropout(0.2),

            nn.Linear(in_features=self.fc3_out, out_features=self.fc4_out),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=self.fc4_out),
            nn.Dropout(0.1),

            nn.Linear(in_features=self.fc4_out, out_features=self.ln_out),
        )

    def forward(self, x, valid_len, max_len):
        """
        :param x:  [batch * frequency_num, max_len, embed_dim] example:[5, 2, 2048]
        :param valid_len: example:[4, 5]
        :param max_len:   example:5
        :return: tensor(original_batch, frequency_num)  example:[9, 7]
        """
        # 把 [2, 5, 2048] = [5, 2048] + [5, 2048] -> [5, 2048] + [4, 2048] -> [9, 2048] 还原数据本身的格式
        attention_feature = []
        for i, sample_len in enumerate(valid_len):
            temp_sample = x[i]  # [5, 2048]

            # if sample_len < max_len:
            #     temp_sample = temp_sample[:sample_len]  # [4, 2048]

            attention_feature.append(temp_sample)
        x = torch.cat(attention_feature, dim=0)  # [9, 2048]
        x = self.fc(self.layer_norm(x))  # [9, 7]
        return x


class AttentionFPN(nn.Module):

    def __init__(self, num_blocks, num_classes, back_bone, pretrained=False, ln_out=1):
        super(AttentionFPN, self).__init__()
        self.in_planes = 64
        self.num_classes = num_classes

        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)

        self.back_bone = build_backbone(back_bone)
        #
        # # Semantic branch
        # self.semantic_branch = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(128, self.num_classes, kernel_size=1, stride=1, padding=0)

        # Multihead Attention After Resnet50
        self.resnet_out = 2048
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.t60_attn = nn.MultiheadAttention(embed_dim=self.resnet_out, num_heads=4,
                                              batch_first=True)  # heads必须要能被embed_dim整除
        # self.t60_layernorm = nn.LayerNorm(normalized_shape=self.resnet_out, eps=0,
        #                                  elementwise_affine=False)  # LayerNorm最后一个维度，也就是embed_dim
        self.t60_fc = PositionalWiseFeedForward(ln_out, self.resnet_out)

    def _upsample(self, x, h, w):
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

    def _make_layer(self, Bottleneck, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(Bottleneck(self.in_planes, planes, stride))
            self.in_planes = planes * Bottleneck.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y


    def forward(self, x):
        #valid_len = [1]
        valid_len = [1 for i  in range(4)]

        low_level_features = self.back_bone(x)

        c5 = low_level_features[4]
        a1 = torch.flatten(self.avgpool(c5), 1)
        count_len = cal_count_len(valid_len)
        a1 = [a1[count_len[index]: count_len[index + 1]] for index in range(len(count_len) - 1)]

        a1 = torch.concat(a1).unsqueeze(1)
        #a1 = pad_sequence(a1, batch_first=True)

        max_len = a1.size(1)

        attention_mask = create_attention_mask(valid_len=valid_len, max_len=max_len, num_head=self.t60_attn.num_heads)
        attention_mask = attention_mask.to(
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        attn_output, attn_output_weights = self.t60_attn(query=a1, key=a1, value=a1,
                                                         attn_mask=attention_mask)
        t60 = self.t60_fc(attn_output + a1, valid_len, max_len)

        return t60

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


if __name__ == "__main__":
    # 这个num_classes实际上是out_channel
    # model = AttentionFPN(num_blocks=[2, 4, 23, 3], num_classes=3, back_bone="resnet50", pretrained=False)
    # input = torch.rand(9, 3, 224, 224)
    # valid_len = [5, 4]
    # t60_out, dereverb_out = model(input, valid_len)  # [1, 3, 224, 224]
    # print(t60_out.shape)
    mask = create_attention_mask([1, 1, 1, 1], 1, 1)
    print(mask)