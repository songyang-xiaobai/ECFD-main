# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='ConvNeXt',
        in_chans=3,
        depths=[3, 3, 9, 3], 
        dims=[96, 192, 384, 768], 
        drop_path_rate=0.2,
        layer_scale_init_value=1.0,
        out_indices=[0, 1, 2, 3],
    ),
    decode_head=dict(
        type='ECFD',
        pretrain_img_size=(224, 224),
        in_channels=[256, 512, 1024, 2048],
        in_index=[0, 1, 2, 3],
        patch_size=4,
        patch_ratio=4,
        channels=256,
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
        with_mlp=True,
        add_context_information=True,
        init_values=0.,
        num_classes=19,
        norm_cfg=norm_cfg,
        input_transform='multiple_select',
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
