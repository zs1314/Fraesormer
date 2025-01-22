import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.vision_transformer import VisionTransformer
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import time
from typing import Tuple, Union
from functools import partial


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class DWConv2d(nn.Module):

    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        x = x.permute(0, 3, 1, 2)  # (b c h w)
        x = self.conv(x)  # (b c h w)
        x = x.permute(0, 2, 3, 1)  # (b h w c)
        return x


class ATK_SPA(nn.Module):
    def __init__(self, dim, pdim, num_heads=8, bias=False):
        super(ATK_SPA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(pdim, pdim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(pdim * 3, pdim * 3, kernel_size=3, stride=1, padding=1, groups=pdim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # 门控网络，用于动态调整 K 值
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(dim // 2, 1, kernel_size=1),  # 输出动态 K
            nn.Sigmoid()
        )

        self.attn1 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)

        self.dim = dim
        self.pdim = pdim

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # BCHW

        b, c, h, w = x.shape
        x1, x2 = torch.split(x, [self.pdim, self.dim - self.pdim], dim=1)

        qkv = self.qkv_dwconv(self.qkv(x1))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, C, _ = q.shape

        # 动态调整 K 的大小，基于输入特征
        dynamic_k = int(C * self.gate(x).view(b, -1).mean())  # 对输出进行全局平均池化

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        # 创建掩码
        mask = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)

        # 使用动态 K 值
        index = torch.topk(attn, k=dynamic_k, dim=-1, largest=True)[1]
        mask.scatter_(-1, index, 1.)
        attn = torch.where(mask > 0, attn, torch.full_like(attn, float('-inf')))

        attn = attn.softmax(dim=-1)

        out1 = (attn @ v)
        out2 = (attn @ v)
        out3 = (attn @ v)
        out4 = (attn @ v)

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(torch.cat([out, x2], dim=1))  # B C H W
        out = out.permute(0, 2, 3, 1)
        return out


class DWConv3x3(nn.Module):
    def __init__(self, dim=768):
        super(DWConv3x3, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x):
        B, H, W, C = x.shape  # 修改为B H W C格式
        x = x.permute(0, 3, 1, 2).contiguous()  # 将B H W C转换为B C H W
        x = self.dwconv(x)  # 卷积操作
        x = x.permute(0, 2, 3, 1).contiguous()  # 将B C H W转换回B H W C
        return x


class DWConv5x5(nn.Module):
    def __init__(self, dim=768):
        super(DWConv5x5, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, bias=True, groups=dim)

    def forward(self, x):
        B, H, W, C = x.shape  # 修改为B H W C格式
        x = x.permute(0, 3, 1, 2).contiguous()  # 将B H W C转换为B C H W
        x = self.dwconv(x)  # 卷积操作
        x = x.permute(0, 2, 3, 1).contiguous()  # 将B C H W转换回B H W C
        return x


class MultiScaleDWConv(nn.Module):
    def __init__(self, dim, scale=(1, 3, 5, 7)):
        super().__init__()
        self.scale = scale
        self.channels = []
        self.proj = nn.ModuleList()
        for i in range(len(scale)):
            if i == 0:
                channels = dim - dim // len(scale) * (len(scale) - 1)
            else:
                channels = dim // len(scale)
            conv = nn.Conv2d(channels, channels,
                             kernel_size=scale[i],
                             padding=scale[i] // 2,
                             groups=channels)
            self.channels.append(channels)
            self.proj.append(conv)

    def forward(self, x):
        x = torch.split(x, split_size_or_sections=self.channels, dim=1)
        out = []
        for i, feat in enumerate(x):
            out.append(self.proj[i](feat))
        x = torch.cat(out, dim=1)
        return x


class HSSFGN(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(hidden_features),
        )
        self.dw = MultiScaleDWConv(dim=hidden_features)
        self.act = act_layer()
        self.norm = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, in_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_features),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B, H, W, C = x.shape  # 修改为B H W C格式
        x = x.permute(0, 3, 1, 2).contiguous()
        x = v = self.fc1(x)  # 线性变换并拆分
        x = self.dw(x) + x
        x = self.norm(self.act(x))

        x = x * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x


class Block(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, partial_dim: int, drop_path=0., layerscale=False,
                 layer_init_values=1e-5):
        super().__init__()
        self.layerscale = layerscale
        self.embed_dim = embed_dim
        self.retention_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)

        self.retention = ATK_SPA(dim=embed_dim, pdim=partial_dim, num_heads=num_heads)
        self.drop_path = DropPath(drop_path)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.ffn = HSSFGN(in_features=embed_dim, hidden_features=ffn_dim)
        self.pos = DWConv2d(embed_dim, 3, 1, 1)

        if layerscale:
            self.gamma_1 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim), requires_grad=True)

    def forward(
            self,
            x: torch.Tensor,
    ):
        x = x + self.pos(x)
        if self.layerscale:
            x = x + self.drop_path(
                self.gamma_1 * self.retention(self.retention_layer_norm(x)))
            x = x + self.drop_path(self.gamma_2 * self.ffn(self.final_layer_norm(x)))
        else:
            x = x + self.drop_path(
                self.retention(self.retention_layer_norm(x)))
            x = x + self.drop_path(self.ffn(self.final_layer_norm(x)))
        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(dim, out_dim, 3, 2, 1)
        self.norm = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        '''
        x: B H W C
        '''
        x = x.permute(0, 3, 1, 2).contiguous()  # (b c h w)
        x = self.reduction(x)  # (b oc oh ow)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (b oh ow oc)
        return x


class BasicLayer(nn.Module):

    def __init__(self, embed_dim, out_dim, depth, num_heads, partial_dim,
                 ffn_dim=96., drop_path=0., norm_layer=nn.LayerNorm,
                 downsample: PatchMerging = None, use_checkpoint=False,
                 layerscale=False, layer_init_values=1e-5, ):

        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, ffn_dim, partial_dim,
                  drop_path[i] if isinstance(drop_path, list) else drop_path, layerscale, layer_init_values)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=embed_dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        b, h, w, d = x.size()
        for blk in self.blocks:
            if self.use_checkpoint:
                tmp_blk = partial(blk)
                x = checkpoint.checkpoint(tmp_blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: torch.Tensor):
        '''
        x: (b c h w)
        '''
        x = x.permute(0, 2, 3, 1).contiguous()  # (b h w c)
        x = self.norm(x)  # (b h w c)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, 3, 2, 1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, 3, 1, 1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, 3, 2, 1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.BatchNorm2d(embed_dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).permute(0, 2, 3, 1)  # (b h w c)
        return x


class Fraesormer(nn.Module):

    def __init__(self, in_chans=3, num_classes=172,
                 embed_dims=None, depths=None, num_heads=None, mlp_ratios=None, drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, partial_dims=None,
                 patch_norm=True, use_checkpoints=None, projection=1024,
                 layerscales=None, layer_init_values=1e-6):
        super().__init__()

        if layerscales is None:
            layerscales = [False, False, False, False]
        if use_checkpoints is None:
            use_checkpoints = [False, False, False, False]
        if partial_dims is None:
            partial_dims = [16, 48, 96, 192]
        if mlp_ratios is None:
            mlp_ratios = [3, 3, 3, 3]
        if num_heads is None:
            num_heads = [3, 6, 12, 24]
        if depths is None:
            depths = [2, 2, 6, 2]
        if embed_dims is None:
            embed_dims = [96, 192, 384, 768]
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dims[0]
        self.patch_norm = patch_norm
        self.num_features = embed_dims[-1]
        self.mlp_ratios = mlp_ratios

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dims[0],
                                      norm_layer=norm_layer if self.patch_norm else None)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                embed_dim=embed_dims[i_layer],
                out_dim=embed_dims[i_layer + 1] if (i_layer < self.num_layers - 1) else None,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                ffn_dim=int(mlp_ratios[i_layer] * embed_dims[i_layer]),
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoints[i_layer],
                layerscale=layerscales[i_layer],
                layer_init_values=layer_init_values,
                partial_dim=partial_dims[i_layer]
            )
            self.layers.append(layer)

        self.proj = nn.Linear(self.num_features, projection)
        self.norm = nn.BatchNorm2d(projection)
        self.swish = MemoryEfficientSwish()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(projection, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            try:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            except:
                pass

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)

        for layer in self.layers:
            x = layer(x)

        x = self.proj(x)  # (b h w c)
        x = self.norm(x.permute(0, 3, 1, 2)).flatten(2, 3)  # (b c h*w)
        x = self.swish(x)

        x = self.avgpool(x)  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


@register_model
# tiny
def fraesormer_t(num_classes):
    model = Fraesormer(
        num_classes=num_classes,
        embed_dims=[32, 64, 128, 256],
        depths=[2, 2, 6, 2],
        num_heads=[4, 4, 8, 16],
        mlp_ratios=[3, 3, 3, 3],
        drop_path_rate=0.1,
        partial_dims=[16, 48, 64, 128],
        layerscales=[False, False, False, False]
    )
    model.default_cfg = _cfg()
    return model


@register_model
# base
def fraesormer_b(num_classes):
    model = Fraesormer(
        num_classes=num_classes,
        embed_dims=[64, 128, 256, 512],
        depths=[1, 1, 5, 1],
        num_heads=[4, 4, 8, 16],
        mlp_ratios=[3, 3, 3, 3],
        partial_dims=[48, 64, 128, 256],
        drop_path_rate=0.1,
        layerscales=[False, False, False, False]
    )
    model.default_cfg = _cfg()
    return model


@register_model
# large
def fraesormer_l(num_classes):
    model = Fraesormer(
        num_classes=num_classes,
        embed_dims=[64, 128, 256, 512],
        depths=[2, 2, 6, 3],
        num_heads=[4, 4, 8, 16],
        mlp_ratios=[4, 4, 3, 3],
        partial_dims=[48, 64, 128, 256],
        drop_path_rate=0.15,
        layerscales=[False, False, False, False]
    )
    model.default_cfg = _cfg()
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    from ptflops import get_model_complexity_info

    model = fraesormer_b(num_classes=256).to('cuda')
    macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print(f"Total FLOPs: {macs}")
    print(f"Total params: {params}")

    print("Params:", count_parameters(model) / 1e6)

    input = torch.randn(1, 3, 224, 224).to('cuda')
    out = model(input)
    print(out.shape)

    from thop import profile
    from thop import clever_format

    input = torch.randn(1, 3, 224, 224).to('cuda')
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)
