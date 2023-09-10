_base_ = [
    '../_base_/models/ecfd_swin.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
samples_per_gpu = 2
crop_size = (512, 1024)
n_min = samples_per_gpu * crop_size[0] * crop_size[1] // 16
model = dict(
    backbone=dict(
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=False
    ),
    decode_head=dict(
        type='ECFD',
        pretrain_img_size=crop_size,
        in_channels=[192, 384, 768, 1536],
        channels=192,
        embed_dim=192,
        L=3,
        CICM_num_heads=6,
        SICM_num_heads=6,
        with_cp=False,
        num_classes=19,
        SICM_mlp_ratio=0.5
    ),
    auxiliary_head=dict(in_channels=768, num_classes=19),
    train_cfg=dict(),
    )
# test_cfg=dict(mode='slide', crop_size=crop_size, stride=(512, 512))
# data = dict(test=dict(split='ImageSets/SegmentationContext/final/val_0.txt'))
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
data=dict(samples_per_gpu=2)

