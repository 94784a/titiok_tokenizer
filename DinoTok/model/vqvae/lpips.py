"""Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models"""

import os, hashlib
import requests
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import models
from collections import namedtuple

URL_MAP = {"vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"}

CKPT_MAP = {"vgg_lpips": "vgg.pth"}

MD5_MAP = {"vgg_lpips": "d507d7349b931f0638a25a48a722f98a"}


def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def md5_hash(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def get_ckpt_path(name, root, check=False):
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        print("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path


def get_ckpt_path_local(name, root, check=False):
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        print("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path


class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(
        self,
        use_dropout=True,
        vgg_ckpt_path=None,
        checkpoint="/high_perf_store2/users/xiexuezhen/paper_code_base/projects/model/reconstruction_heads/losses/cache/vgg.pth",
    ):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vg16 features

        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained(checkpoint=checkpoint)
        self.net = vgg16(
            pretrained=True, requires_grad=False, ckpt_path=vgg_ckpt_path
        )  #
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips", checkpoint=None):
        if checkpoint is None:
            ckpt = get_ckpt_path(
                name, os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
            )
        else:
            ckpt = checkpoint
        self.load_state_dict(
            torch.load(ckpt, map_location=torch.device("cpu")), strict=False
        )
        print("loaded pretrained LPIPS loss from {}".format(ckpt))

    @classmethod
    def from_pretrained(cls, name="vgg_lpips"):
        if name != "vgg_lpips":
            raise NotImplementedError
        model = cls()
        ckpt = get_ckpt_path(
            name, os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
        )
        model.load_state_dict(
            torch.load(ckpt, map_location=torch.device("cpu")), strict=False
        )
        return model

    def forward(self, input, target):
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(
                outs1[kk]
            )
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [
            spatial_average(lins[kk].model(diffs[kk]), keepdim=True)
            for kk in range(len(self.chns))
        ]
        val = torch.stack(res, dim=0).sum(dim=0)
        # val = res[0]
        # for l in range(1, len(self.chns)):
        #     val += res[l]
        return val


# class ScalingLayer(nn.Module):
#     def __init__(self):
#         super(ScalingLayer, self).__init__()
#         self.register_buffer(
#             "shift", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None]
#         )
#         self.register_buffer(
#             "scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None]
#         )

#     def forward(self, inp):
#         return (inp - self.shift) / self.scale


class ScalingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # 不再 register_buffer
        self._shift = torch.tensor([-0.030, -0.088, -0.188]).view(1, 3, 1, 1)
        self._scale = torch.tensor([0.458, 0.448, 0.450]).view(1, 3, 1, 1)

    def forward(self, inp):
        shift = self._shift.to(dtype=inp.dtype, device=inp.device)
        scale = self._scale.to(dtype=inp.dtype, device=inp.device)
        return (inp - shift) / (scale + 1e-12)


class NetLinLayer(nn.Module):
    """A single linear layer which does a 1x1 conv"""

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = (
            [
                nn.Dropout(),
            ]
            if (use_dropout)
            else []
        )
        layers += [
            nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),
        ]
        self.model = nn.Sequential(*layers)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True, ckpt_path=None):
        """
        Args:
            requires_grad (bool): 是否需要训练 VGG 参数
            pretrained (bool): 是否加载预训练权重
            ckpt_path (str): 如果提供，强制从本地 checkpoint 加载，不联网
        """
        super(vgg16, self).__init__()

        # 1. 创建基本模型
        vgg = models.vgg16(weights=None)  # 不从网上下载
        if pretrained and ckpt_path is not None:
            state_dict = torch.load(ckpt_path, map_location="cpu")
            vgg.load_state_dict(state_dict)
        # 关闭所有 ReLU 的 inplace
        for m in vgg.features.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False

        vgg_pretrained_features = vgg.features

        # 2. 按 slice 切分
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        self.N_slices = 5

        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        # 3. 是否需要梯度
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple(
            "VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"]
        )
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True) + eps)
    return x / (norm_factor + eps)


def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)
