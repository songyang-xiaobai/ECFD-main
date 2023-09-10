import torch.nn.functional as F
from torch import nn
import torch
from functools import partial
import torch.utils.checkpoint as checkpoint
from ..builder import HEADS
import numpy as np
from mmcv.runner import auto_fp16, force_fp32
from ..builder import build_loss
import torch.utils.checkpoint as cp
from ops.modules import MSDeformAttn
from .decode_head import BaseDecodeHead
from timm.models.layers import to_2tuple, DropPath
from ..losses import accuracy
from mmseg.ops import resize


@HEADS.register_module()
class ECFD(BaseDecodeHead):
    def __init__(self,
                 pretrain_img_size=(224, 224),
                 patch_size=4,
                 patch_ratio=4,
                 embed_dim=768,
                 pos_drop=0.,
                 n_points=4,
                 with_cp=False,
                 L=3,
                 # CICM
                 CICM_num_heads=6,
                 CICM_n_level=3,
                 # SICM
                 SICM_num_heads=6,
                 SICM_n_levels=1,
                 SICM_mlp_ratio=0.25,
                 SICM_drop=0.,
                 SICM_drop_path=0.,
                 deform_ratio=1.,
                 with_mlp=True,
                 init_values=0.,
                 add_context_information=True,
                 loss_dice=None, **kwargs):
        super(ECFD, self).__init__(**kwargs)
        self.pretrain_img_size = pretrain_img_size
        self.patch_size = patch_size
        self.add_context_information = add_context_information
        self.embed_dim = embed_dim

        self.L = L
        self.c1_conv1x1 = nn.Conv2d(self.in_channels[0], self.channels, kernel_size=1, bias=True)
        self.c2_conv1x1 = nn.Conv2d(self.in_channels[1], embed_dim, kernel_size=1, bias=True)
        self.c3_conv1x1 = nn.Conv2d(self.in_channels[2], embed_dim, kernel_size=1, bias=True)
        self.c4_conv1x1 = nn.Conv2d(self.in_channels[3], embed_dim, kernel_size=1, bias=True)
        self.RCE = ReinforceContextEmbedding(
            img_size=pretrain_img_size, patch_size=patch_size, embed_dim=embed_dim, patch_ratio=patch_ratio,
            in_chans=self.in_channels, is_fuse=True, align_corners=self.align_corners)

        num_patches = self.RCE.num_patches
        self.num_tokens = 1
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=pos_drop)
        self.blocks = nn.ModuleList(
            [CrossLayerFusionBlock(
                # atten
                # vit_correct
                CICM_num_heads=CICM_num_heads,
                CICM_n_level=CICM_n_level,
                # conv_correct
                SICM_num_heads=SICM_num_heads,
                SICM_n_levels=SICM_n_levels,
                SICM_mlp_ratio=SICM_mlp_ratio,
                SICM_drop=SICM_drop,
                SICM_drop_path=SICM_drop_path,
                dim=embed_dim,
                with_mlp=with_mlp, n_points=n_points, deform_ratio=deform_ratio,
                act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                init_values=init_values, with_cp=with_cp) for i in range(self.L)])
        self.norm2 = nn.SyncBatchNorm(self.channels)
        self.norm3 = nn.SyncBatchNorm(self.channels)
        self.norm4 = nn.SyncBatchNorm(self.channels)

        self.c2_deconv1x1 = nn.Conv2d(embed_dim, self.channels, kernel_size=1, bias=True)
        self.c3_deconv1x1 = nn.Conv2d(embed_dim, self.channels, kernel_size=1, bias=True)
        self.c4_deconv1x1 = nn.Conv2d(embed_dim, self.channels, kernel_size=1, bias=True)
        self.fpn = FCFPNHead(self.channels,
                             fpn_inchannels=[self.channels, self.channels, self.channels, self.channels],
                             fpn_dim=self.channels, norm_layer=nn.SyncBatchNorm)
        self.clgd = CLGD(self.channels, self.channels, norm_layer=nn.SyncBatchNorm, align_corners=self.align_corners)
        self.up = nn.ConvTranspose2d(embed_dim, self.channels, 2, 2)

        if loss_dice is not None:
            self.loss_dice = build_loss(loss_dice)
        else:
            self.loss_dice = None

    def cls_seg(self, feat):
        """final feature classification."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def forward(self, inputs):

        # spatial_feature =
        c2 = self.c2_conv1x1(inputs[1])
        c3 = self.c3_conv1x1(inputs[2])
        c4 = self.c4_conv1x1(inputs[3])

        bs, dim, _, _ = c2.shape

        c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
        c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
        c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s
        c = torch.cat([c2, c3, c4], dim=1)
        # spatial size
        space_shape = [inputs[1].shape[2:], inputs[2].shape[2:], inputs[3].shape[2:]]
        deform_inputs1, deform_inputs2 = self.deform_inputs(inputs)
        # Reinforced Context Embedding
        x, H, W = self.RCE(inputs)

        bs, n, dim = x.shape
        pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H, W)
        x = self.pos_drop(x + pos_embed)
        outs = list()
        for i in range(self.L):
            c, x = self.blocks[i](x, c, deform_inputs1, deform_inputs2, space_shape)
            outs.append(x.transpose(1, 2).view(bs, dim, H, W).contiguous())

        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, inputs[1].shape[2], inputs[1].shape[3]).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, inputs[2].shape[2], inputs[2].shape[3]).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, inputs[3].shape[2], inputs[3].shape[3]).contiguous()

        x = x.permute(0, 2, 1).view(bs, dim, H, W).contiguous()
        r_c = self.up(x)
        # change channel number
        c1 = self.c1_conv1x1(inputs[0])
        c1 = self.clgd(c1, r_c)
        # c1 += multix[0]

        if self.add_context_information:
            x2, x3, x4 = outs
            x2 = resize(input=x2, size=space_shape[0], mode='bilinear', align_corners=self.align_corners)
            x4 = resize(input=x4, size=space_shape[2], mode='bilinear', align_corners=self.align_corners)
            c2, c3, c4 = c2 + x2, c3 + x3, c4 + x4

        # change channel number
        c2 = self.norm2(self.c2_deconv1x1(c2))
        c3 = self.norm3(self.c3_deconv1x1(c3))
        c4 = self.norm4(self.c4_deconv1x1(c4))
        out = self.fpn([c1, c2, c3, c4])
        out = self.cls_seg(out)

        return out

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing, only ``pam_cam`` is used."""
        return self.forward(inputs)

    def deform_inputs(self, inputs):
        bs, c, h1, w1 = inputs[1].shape
        _, _, h2, w2 = inputs[2].shape
        _, _, h3, w3 = inputs[3].shape
        spatial_shapes = torch.as_tensor([(h2, w2)], dtype=torch.long, device=inputs[0].device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        reference_points = get_reference_points([(h1, w1),
                                                 (h2, w2),
                                                 (h3, w3)], inputs[0].device)
        deform_inputs1 = [reference_points, spatial_shapes, level_start_index]

        spatial_shapes = torch.as_tensor([(h1, w1),
                                          (h2, w2),
                                          (h3, w3)], dtype=torch.long, device=inputs[0].device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        reference_points = get_reference_points([(h2, w2)], inputs[0].device)
        deform_inputs2 = [reference_points, spatial_shapes, level_start_index]

        return deform_inputs1, deform_inputs2

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_img_size[0] // self.patch_size, self.pretrain_img_size[1] // self.patch_size, -1).permute(
            0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=self.align_corners). \
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    @force_fp32(apply_to=('seg_logit',))
    def losses(self, seg_logit, seg_label):

        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)
        loss['loss_seg'] = self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index)
        if self.loss_dice is not None:
            loss['loss_dice'] = self.loss_dice(seg_logit, seg_label)
        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        return loss

    @staticmethod
    def binary_mask_from_label(seg_label, num_classes):
        """Convert segmentation label to onehot.

        Args:
            seg_label (Tensor): Segmentation label of shape (N, H, W).
            num_classes (int): Number of classes.

        Returns:
            Tensor: Onehot labels of shape (N, num_classes).
        """

        batch_size = seg_label.size(0)
        onehot_labels = seg_label.new_zeros((batch_size, num_classes))
        for i in range(batch_size):
            hist = seg_label[i].float().histc(
                bins=num_classes, min=0, max=num_classes - 1)
            onehot_labels[i] = hist > 0
        return onehot_labels


def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points


class ReinforceContextEmbedding(nn.Module):
    """2D Image to Patch Embedding."""

    def __init__(self, img_size=224, patch_size=16, in_chans=[256, 512, 1024, 2048], patch_ratio=2,
                 embed_dim=768, norm_layer=None, flatten=True, is_fuse=False, align_corners=False):
        super().__init__()
        if not isinstance(img_size, tuple):
            img_size = to_2tuple(img_size)
        if not isinstance(patch_size, tuple):
            patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.is_fuse = is_fuse
        self.align_corners = align_corners

        if self.is_fuse:
            # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_ratio, stride=patch_ratio)
            self.layer2_embedding = nn.Conv2d(in_chans[1], in_chans[0], kernel_size=1)
            self.layer3_embedding = nn.Conv2d(in_chans[2], in_chans[0], kernel_size=1)
            self.layer4_embedding = nn.ConvTranspose2d(in_chans[3], in_chans[0], kernel_size=1)
            self.fuse_layer = nn.Conv2d(in_chans[0] * 3, embed_dim, kernel_size=1)
        else:
            self.proj = nn.Conv2d(2048, embed_dim, kernel_size=1)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, inputs):
        if self.is_fuse:
            # x = self.proj(x)
            layer2_embedding = self.layer2_embedding(inputs[1])
            layer3_embedding = self.layer3_embedding(inputs[2])
            layer4_embedding = self.layer4_embedding(inputs[3])
            upsample_embedding = self.upsample((layer2_embedding, layer3_embedding, layer4_embedding))
            x = self.fuse_layer(upsample_embedding)
        else:
            x = self.proj(inputs[-1])
        _, _, H, W = x.shape
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x, H, W

    def upsample(self, inputs):
        out_shape = inputs[1].shape
        outputs = []
        for i in range(len(inputs)):
            temp = resize(inputs[i],
                          size=out_shape[2:],
                          mode='bilinear',
                          align_corners=self.align_corners)
            outputs.append(temp)
        upsample_embedding = torch.cat(outputs, dim=1)
        return upsample_embedding


class CLGD(nn.Module):
    """
    Cross-level Gating Decoder
    """

    def __init__(self, in_channels, out_channels, norm_layer, align_corners, inter_channels=32):
        super(CLGD, self).__init__()

        # inter_channels = 32
        self.conv_low = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                      norm_layer(inter_channels),
                                      nn.ReLU())  # skipconv

        self.conv_cat = nn.Sequential(nn.Conv2d(in_channels + inter_channels, in_channels, 3, padding=1, bias=False),
                                      norm_layer(in_channels),
                                      nn.ReLU())  # fusion1

        self.conv_att = nn.Sequential(nn.Conv2d(in_channels + inter_channels, 1, 1),
                                      nn.Sigmoid())  # att

        self.conv_out = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                                      norm_layer(out_channels),
                                      nn.ReLU())  # fusion2
        self.align_corners = align_corners

        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x, y):
        """
            inputs :
                x : low level feature(N,C,H,W)  y:high level feature(N,C,H,W)
            returns :
                out :  cross-level gating decoder feature
        """
        low_lvl_feat = self.conv_low(x)
        high_lvl_feat = F.interpolate(y, low_lvl_feat.size()[2:], mode='bilinear', align_corners=self.align_corners)
        feat_cat = torch.cat([low_lvl_feat, high_lvl_feat], 1)

        low_lvl_feat_refine = self.gamma * self.conv_att(feat_cat) * low_lvl_feat
        low_high_feat = torch.cat([low_lvl_feat_refine, high_lvl_feat], 1)
        low_high_feat = self.conv_cat(low_high_feat)

        low_high_feat = self.conv_out(low_high_feat)

        return low_high_feat


class FCFPNHead(nn.Module):
    def __init__(self, out_channels, norm_layer=nn.BatchNorm2d, fpn_inchannels=[256, 512, 1024, 2048],
                 fpn_dim=256, up_kwargs={'mode': 'bilinear', 'align_corners': True}, drop=0.1):
        super(FCFPNHead, self).__init__()
        # bilinear interpolate options
        assert up_kwargs is not None

        self._up_kwargs = up_kwargs
        fpn_lateral = []
        for fpn_inchannel in fpn_inchannels[:-1]:
            fpn_lateral.append(nn.Sequential(
                nn.Conv2d(fpn_inchannel, fpn_dim, kernel_size=1, bias=False),
                norm_layer(fpn_dim),
                nn.ReLU(inplace=True),
            ))
        self.fpn_lateral = nn.ModuleList(fpn_lateral)
        fpn_out = []
        for _ in range(len(fpn_inchannels) - 1):
            fpn_out.append(nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
                norm_layer(fpn_dim),
                nn.ReLU(inplace=True),
            ))
        self.fpn_out = nn.ModuleList(fpn_out)
        self.c4conv = nn.Sequential(nn.Conv2d(fpn_inchannels[-1], fpn_dim, 3, padding=1, bias=False),
                                    norm_layer(fpn_dim),
                                    nn.ReLU())
        inter_channels = len(fpn_inchannels) * fpn_dim
        self.conv5 = nn.Sequential(nn.Conv2d(inter_channels, out_channels, 3, padding=1, bias=False),
                                   norm_layer(out_channels),
                                   nn.ReLU())

    def forward(self, multix):
        c4 = multix[-1]
        # se_pred = False
        if hasattr(self, 'extramodule'):
            # if self.extramodule.se_loss:
            #    se_pred = True
            #    feat, se_out = self.extramodule(feat)
            # else:
            c4 = self.extramodule(c4)
        feat = self.c4conv(c4)
        c1_size = multix[0].size()[2:]
        feat_up = F.interpolate(feat, c1_size, **self._up_kwargs)
        fpn_features = [feat_up]
        # c4, c3, c2, c1
        for i in reversed(range(len(multix) - 1)):
            feat_i = self.fpn_lateral[i](multix[i])
            feat = F.interpolate(feat, feat_i.size()[2:], **self._up_kwargs)
            feat = feat + feat_i
            # interpolate to the same size with c1
            feat_up = F.interpolate(self.fpn_out[i](feat), c1_size, **self._up_kwargs)
            fpn_features.append(feat_up)
        fpn_features = torch.cat(fpn_features, 1)
        # if se_pred:
        #    return (self.conv5(fpn_features), se_out)
        return self.conv5(fpn_features)


class CrossLayerFusionBlock(nn.Module):
    def __init__(self,
                 # CICM
                 CICM_num_heads=6,
                 CICM_n_level=3,
                 # SICM
                 SICM_num_heads=6,
                 SICM_n_levels=1,
                 SICM_mlp_ratio=0.25,
                 SICM_drop=0.,
                 SICM_drop_path=0.,
                 dim=768,
                 with_mlp=True, n_points=4, deform_ratio=1.,
                 act_layer=nn.GELU, norm_cfg=None, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 init_values=0., with_cp=False):
        super(CrossLayerFusionBlock, self).__init__()
        self.with_cp = with_cp
        self.norm_cfg = norm_cfg

        self.CICM = ContextInformationCorrectionModule(dim=dim, num_heads=CICM_num_heads,
                                                       n_points=n_points,
                                                       n_levels=CICM_n_level,
                                                       deform_ratio=deform_ratio,
                                                       norm_layer=norm_layer, init_values=init_values,
                                                       with_cp=with_cp)
        self.SICM = SpatialInformationCorrectionModule(dim=dim, num_heads=SICM_num_heads,
                                                       n_levels=SICM_n_levels,
                                                       mlp_ratio=SICM_mlp_ratio,
                                                       drop=SICM_drop,
                                                       drop_path=SICM_drop_path,
                                                       deform_ratio=deform_ratio,
                                                       n_points=n_points,
                                                       with_mlp=with_mlp,
                                                       norm_layer=norm_layer, with_cp=with_cp)

    # space_shape = [inputs[1].shape[2:], inputs[2].shape[2:], inputs[3].shape[2:]]
    def forward(self, F_c, F_sp, deform_input1, deform_input2, space_shape):
        F_c = self.CICM(F_c, deform_input2[0], F_sp, deform_input2[1], deform_input2[2])
        F_sp = self.SICM(F_sp, deform_input1[0], F_c, deform_input1[1], deform_input1[2], space_shape)
        return F_sp, F_c




class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, space_shape):
        x = self.fc1(x)
        x = self.dwconv(x, space_shape)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, space_shape):
        B, N, C = x.shape
        H1, W1 = space_shape[0]
        H2, W2 = space_shape[1]
        H3, W3 = space_shape[2]

        x1 = x[:, 0:H1 * W1, :].transpose(1, 2).view(B, C, H1, W1).contiguous()
        x2 = x[:, H1 * W1:(H1 * W1 + H2 * W2), :].transpose(1, 2).view(B, C, H2, W2).contiguous()
        x3 = x[:, (H1 * W1 + H2 * W2):, :].transpose(1, 2).view(B, C, H3, W3).contiguous()
        x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
        x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
        x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


class ContextInformationCorrectionModule(nn.Module):
    def __init__(self, dim=768, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0.,
                 with_cp=False):
        super(ContextInformationCorrectionModule, self).__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x, reference_points, feat, spatial_shapes, level_start_index):
        def _inner_forward(x, feat):

            attn = self.attn(self.query_norm(x), reference_points,
                             self.feat_norm(feat), spatial_shapes,
                             level_start_index, None)
            return x + self.gamma * attn

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x, feat)
        else:
            x = _inner_forward(x, feat)

        return x


class SpatialInformationCorrectionModule(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_mlp=True, mlp_ratio=0.25, drop=0., drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False):
        super(SpatialInformationCorrectionModule, self).__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)  # , ratio=deform_ratio
        self.with_mlp = with_mlp
        self.with_cp = with_cp
        if with_mlp:
            self.Mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
            self.mlp_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, space_shape):
        def _inner_forward(query, feat):

            attn = self.attn(self.query_norm(query), reference_points,
                             self.feat_norm(feat), spatial_shapes,
                             level_start_index, None)
            query = query + attn

            x = self.Mlp(self.mlp_norm(query), space_shape)
            query = query + self.drop_path(x)

            return query

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query
