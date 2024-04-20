export TEST_DATASET_ROOT=/mnt/SrvDataDisk/DatasetRemoteSensing
export MODEL=ViT-L-14
export DATASETS_TESTING='aid' 'eurosat'
export NAME=test-train-ViT-L-14
# export PRETRAINED_CHECKPOINT=laion2b_s34b_b79k
export PRETRAINED_CHECKPOINT=laion2B-s32B-b82K
export NUM_GPUS=4
export TRAIN_DATASET=sqlite
export ANNOTATION_DB="/mnt/FastDisk/GeJunYao/VLP/databases/backups/2024-4-15(washed)/annotation.db"
export METADATA_DB="/mnt/FastDisk/GeJunYao/VLP/databases/backups/2024-4-15(washed)/metadata.db"
# export ANNOTATION_DB=/mnt/SrvUserDisk/Gejunyao/VLP/test_downloader/annotation.db
# export METADATA_DB=/mnt/SrvUserDisk/Gejunyao/VLP/test_downloader/metadata.db
# export IMG_DATA_BACKEND="dict(type='disk',root='/mnt/SrvDataDisk/RSVLD')"
export NUM_WORKERS=16
export IMG_DATA_BACKEND="dict(type='mongo',mongo_uri='mongodb://localhost:27017')"
export BATCH_SIZE=256

# '/mnt/SrvDataDisk/DatasetRemoteSensing/SkyScript/SkyCLIP_ViT_B32_top50pct/epoch_20.pt'
python train_val_test.py \
    --test-dataset-root $TEST_DATASET_ROOT \
    --model $MODEL \
    --datasets-for-testing $DATASETS_TESTING \
    --pretrained $PRETRAINED_CHECKPOINT \
    --train-dataset $TRAIN_DATASET \
    --annotation-db $ANNOTATION_DB \
    --metadata-db $METADATA_DB \
    --img-data-backend $IMG_DATA_BACKEND \
    --report-to "wandb,tensorboard" \
    --wandb-project-name "openclip-test" \
    --save-frequency 1 \
    --val-frequency 1 \
    --warmup 1000 \
    --batch-size $BATCH_SIZE \
    --lr "1e-9" \
    --wd 0.1 \
    --epochs=20 \
    --workers=$NUM_WORKERS \
    --aug-cfg use_timm=True color_jitter=0.4 scale="(0.67, 1.0)" ratio="(0.5, 2.0)" \
    --gather-with-grad \
    --accum-freq 1 \
    --seed 42 

        # --grad-checkpointing \