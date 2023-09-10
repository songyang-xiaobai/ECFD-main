_base_ = [
    '../_base_/models/ecfd_r50.py', '../_base_/datasets/pascal_context_59.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
crop_size = (480, 480)
model = dict(
             decode_head=dict(
                 type='ECFD',
                 pretrain_img_size=crop_size,
                 in_channels=[256, 512, 1024, 2048],
                 embed_dim=384,
                 L=3,
                 add_context_information=False,
                 CICM_num_heads=6,
                 SICM_num_heads=6,
                 SICM_mlp_ratio=0.5,
                 with_cp=False,
                 num_classes=59
             ),
             auxiliary_head=dict(num_classes=59),
             # model training and testing settings
             train_cfg=dict(),
             test_cfg=dict(mode='slide', crop_size=crop_size, stride=(320, 320)))
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0001,
                 paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
data=dict(samples_per_gpu=2, val=dict(split='ImageSets/SegmentationContext/5/val_0.txt'),)
# data = dict(test=dict(split='ImageSets/SegmentationContext/final/val_0.txt'))  test=dict(split='ImageSets/SegmentationContext/4/val_3.txt')