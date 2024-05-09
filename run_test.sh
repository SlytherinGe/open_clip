export TEST_DATASET_ROOT=/mnt/FastDisk/Datasets/RSBenchmarks
export MODEL=ViT-L-14
export DATASETS_TESTING='aid' 'eurosat' 'nwpu' 'millionaid'
export NAME=test-ViT-B-32
# export PRETRAINED_CHECKPOINT=laion2b_s34b_b79k
export PRETRAINED_CHECKPOINT=/mnt/SrvUserDisk/Gejunyao/VLP/open_clip/logs/train-ViT-L-14-dataset-mix-1-025-005/checkpoints/epoch_9.pt

# '/mnt/SrvDataDisk/DatasetRemoteSensing/SkyScript/SkyCLIP_ViT_B32_top50pct/epoch_20.pt'
python train_val_test.py \
    --test-dataset-root $TEST_DATASET_ROOT \
    --model $MODEL \
    --datasets-for-testing 'aid' 'eurosat' 'nwpu' 'millionaid' 'rsicb' 'fmow' 'patternnet' 'SkyScript_cls'\
    --pretrained $PRETRAINED_CHECKPOINT \
    --test-batch-size 2048 \
    # --report-to "wandb,tensorboard" \
    # --wandb-project-name "openclip-test" \

        # --datasets-for-testing 'aid' 'eurosat' 'nwpu' 'millionaid' 'fmow' 'patternnet' 'SkyScript_cls'\