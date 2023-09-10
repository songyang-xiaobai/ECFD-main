_base_ = [
    '../_base_/models/ecfd_swin.py', '../_base_/datasets/pascal_context_59.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
crop_size = (480, 480)

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
        in_channels=[192, 384, 768, 1536],
        pretrain_img_size=crop_size,
        channels=192,
        embed_dim=192,
        L=3,
        add_context_information=False,
        CICM_num_heads=6,
        SICM_num_heads=6,
        SICM_mlp_ratio=0.5,
        with_cp=False,
        num_classes=59
    ),
    auxiliary_head=dict(in_channels=768, num_classes=59),
    test_cfg=dict(mode='slide', crop_size=(480, 480), stride=(320, 320))
)

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
data=dict(samples_per_gpu=2,val=dict(split='ImageSets/SegmentationContext/5/val_0.txt'))
#  test=dict(split='ImageSets/SegmentationContext/3/val_2.txt')