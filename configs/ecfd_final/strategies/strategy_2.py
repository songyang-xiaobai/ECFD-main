_base_ = [
    '../_base_/models/ecfd_r50.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
samples_per_gpu = 2
crop_size = (512, 1024)
n_min = samples_per_gpu * crop_size[0] * crop_size[1] // 16
model = dict(
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='ECFDGeneral',
        pretrain_img_size=crop_size,
        in_channels=[256, 512, 1024, 2048],
        embed_dim=192,
        L=3,
        CICM_num_heads=6,
        SICM_num_heads=6,
        with_cp=False,
        SICM_mlp_ratio=0.5,
        num_classes=19,
    ),
    auxiliary_head=dict(num_classes=19),
    train_cfg=dict())
# data = dict(test=dict(split='ImageSets/SegmentationContext/final/val_0.txt'))
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0001,
                 paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
data=dict(samples_per_gpu=2)
evaluation = dict(interval=8000, metric='mIoU')