# https://github.com/naoto0804/pytorch-AdaIN/blob/master/function.py

  
import torch
import torch.nn.functional as F

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_std = feat.view(N, C, -1).std(dim=2) + eps
    feat_mean = feat.view(N, C, -1).mean(dim=2)
    feat_std, feat_mean = feat_std.view(N,C,1,1), feat_mean.view(N,C,1,1)
    return feat_mean, feat_std

def AdaIN(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean) / content_std
    return normalized_feat * style_std + style_mean
