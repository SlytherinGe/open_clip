export TEST_DATASET_ROOT=/mnt/FastDisk/Datasets/RSBenchmarks
# export MODEL=ViT-B-32
export PRETRAINED_CHECKPOINT=laion2b_s34b_b88k
export NAME=test_wandb_func
export MODEL=ViT-B-16
# export PRETRAINED_CHECKPOINT='/mnt/SrvUserDisk/Gejunyao/VLP/open_clip/ckpts/SkyCLIP_ViT_L14_top50pct/epoch_20.pt'
export NUM_GPUS=4
export TRAIN_DATASET=webdataset
export ANNOTATION_DB="/mnt/FastDisk/GeJunYao/VLP/databases/annotation.db"
export NUM_WORKERS=8
export BATCH_SIZE=2048


# '/mnt/SrvDataDisk/DatasetRemoteSensing/SkyScript/SkyCLIP_ViT_B32_top50pct/epoch_20.pt'
python train_val_test.py \
    --train-num-samples=2048000 \
    --test-dataset-root=/mnt/FastDisk/Datasets/RSBenchmarks \
    --datasets-for-testing 'rsicb'\
    --model=ViT-B-16 \
    --dataset-type=webdataset \
    --annotation-db=/mnt/FastDisk/GeJunYao/VLP/databases/backups/2024-5-23/annotation.db \
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
    --test-batch-size=256 \
    --zeroshot-frequency=5 \
    --pretrained=laion2b_s34b_b88k \
    --train-data "/mnt/FastDisk/Datasets/RSVLD1M/Patches{0000..0238}.tar::/mnt/FastDisk/Datasets/SkyScript_wds/{0000..0510}.tar::/mnt/SrvDataDisk/DatasetMultiModal/LAION400M/laion400m-data/{00000..04800}.tar" \
    --warmup=500 \
    --wd=0.1