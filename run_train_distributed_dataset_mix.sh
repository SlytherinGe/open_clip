export TEST_DATASET_ROOT=/mnt/FastDisk/Datasets/RSBenchmarks
# export MODEL=ViT-B-32
export PRETRAINED_CHECKPOINT=laion2B-s32B-b82K
export NAME=train-ViT-L-14-dataset-mix-1-025-0025
export MODEL=ViT-L-14
# export PRETRAINED_CHECKPOINT='/mnt/SrvUserDisk/Gejunyao/VLP/open_clip/ckpts/SkyCLIP_ViT_L14_top50pct/epoch_20.pt'
export NUM_GPUS=4
export TRAIN_DATASET=webdataset
export ANNOTATION_DB="/mnt/FastDisk/GeJunYao/VLP/databases/annotation.db"
export NUM_WORKERS=8
export BATCH_SIZE=2048


# '/mnt/SrvDataDisk/DatasetRemoteSensing/SkyScript/SkyCLIP_ViT_B32_top50pct/epoch_20.pt'
torchrun --nproc_per_node=$NUM_GPUS train_val_test.py \
    --test-dataset-root $TEST_DATASET_ROOT \
    --model $MODEL \
    --name $NAME \
    --datasets-for-testing 'rsicb'\
    --pretrained $PRETRAINED_CHECKPOINT \
    --dataset-type $TRAIN_DATASET \
    --annotation-db $ANNOTATION_DB \
    --report-to "wandb,tensorboard" \
    --wandb-project-name "openclip-RSVLD" \
    --save-frequency 1 \
    --zeroshot-frequency 5 \
    --warmup 1000 \
    --batch-size $BATCH_SIZE \
    --use-bn-sync \
    --lr "1e-5" \
    --wd 0.1 \
    --epochs=20 \
    --workers=$NUM_WORKERS \
    --aug-cfg use_timm=True color_jitter=0.4 scale="(0.67, 1.0)" ratio="(0.5, 2.0)" \
    --gather-with-grad \
    --accum-freq 1 \
    --log-every-n-steps 10 \
    --torchcompile \
    --train-num-samples 2000000 \
    --force-image-size=224 \
    --zero-shot-at-start True \
    --test-batch-size 256 \
    --skyscript-portion=0.25 \
    --laion-portion=0.025 \
    --dataset-resampled \
    --train-data "/mnt/FastDisk/Datasets/RSVLD1M/Patches{0000..0238}.tar::/mnt/FastDisk/Datasets/SkyScript_wds/{0000..0510}.tar::/mnt/SrvDataDisk/DatasetMultiModal/LAION400M/laion400m-data/{00000..04800}.tar" \
    # --grad-checkpointing \
    # --train-data "/mnt/FastDisk/Datasets/RSVLD1M/Patches{0000..0238}.tar" \
    # num of RSVLD 1190742
    # --datasets-for-testing 'aid' 'eurosat' 'nwpu' 'millionaid' 'rsicb' 'fmow' 'patternnet' 'SkyScript_cls'\