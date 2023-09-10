# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


_base_ = [
    '../_base_/models/ecfd_convnext.py', '../_base_/datasets/pascal_context_59.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
crop_size = (480, 480)

model = dict(
    backbone=dict(
        type='ConvNeXt',
        in_chans=3,
        depths=[3, 3, 27, 3],
        dims=[192, 384, 768, 1536],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        out_indices=[0, 1, 2, 3],
    ),
    decode_head=dict(
        type='ECFD',
        pretrain_img_size=crop_size,
        in_channels=[192, 384, 768, 1536],
        channels=192,
        embed_dim=384,
        L=3,
        CICM_num_heads=6,
        SICM_num_heads=6,
        with_cp=False,
        num_classes=59,
        add_context_information=False,
        SICM_mlp_ratio=0.5
    ),
    auxiliary_head=dict(
        in_channels=768,
        num_classes=59
    ),
    test_cfg=dict(mode='slide', crop_size=(480, 480), stride=(320, 320)),
)

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}))
# optimizer = dict(
#     _delete_=True,
#     type='AdamW',
#     lr=0.00006,
#     betas=(0.9, 0.999),
#     weight_decay=0.01,
#     paramwise_cfg=dict(
#         custom_keys={
#             'pos_block': dict(decay_mult=0.),
#             'norm': dict(decay_mult=0.),
#             'head': dict(lr_mult=1.)
#         }))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=2,val=dict(split='ImageSets/SegmentationContext/5/val_0.txt'))
