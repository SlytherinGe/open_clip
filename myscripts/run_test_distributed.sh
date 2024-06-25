export ANNOTATION_DB=/mnt/FastDisk/GeJunYao/VLP/databases/backups/2024-5-23/annotation.db


torchrun --nproc_per_node=4 train_val_test.py \
 --train-num-samples=2048000 \
 --test-dataset-root=/mnt/FastDisk/Datasets/RSBenchmarks \
 --model=ViT-L-14 \
 --dataset-type=webdataset \
 --annotation-db=$ANNOTATION_DB \
 --report-to=wandb \
 --wandb-project-name=openclip-RSVLD \
 --save-frequency=1 \
 --batch-size=2048 \
 --use-bn-sync \
 --epochs=20 \
 --workers=8 \
 --gather-with-grad \
 --log-every-n-steps=10 \
 --dataset-resampled \
 --force-image-size=224 \
 --zero-shot-at-start=True \
 --torchcompile \
 --test-batch-size=256   \
 --zeroshot-frequency=5 \
 --pretrained=laion2b_s32b_b82k \
 --train-data="/mnt/FastDisk/Datasets/RSVLD1M/Patches{{0000..{max_rsteller_tar_id:04d}}}.tar::/mnt/FastDisk/Datasets/SkyScript_wds/{{0000..{max_skyscript_tar_id:04d}}}.tar::/mnt/SrvDataDisk/DatasetMultiModal/LAION400M/laion400m-data/{{00000..01200}}.tar" \
 --warmup=500 \
 --wd=0.1 \
 --lr=1e-5 \
 --data-regulazation-portion=1 \
 --domain-data-portion=1 