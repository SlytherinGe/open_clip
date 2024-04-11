export TEST_DATASET_ROOT=/mnt/SrvDataDisk/DatasetRemoteSensing
export MODEL=ViT-B-32
export DATASETS_TESTING='aid'
export NAME=test-aid-ViT-B-32
export PRETRAINED_CHECKPOINT=laion2b_s34b_b79k

# '/mnt/SrvDataDisk/DatasetRemoteSensing/SkyScript/SkyCLIP_ViT_B32_top50pct/epoch_20.pt'
python train_val_test.py \
    --test-dataset-root $TEST_DATASET_ROOT \
    --model $MODEL \
    --datasets-for-testing $DATASETS_TESTING \
    --pretrained $PRETRAINED_CHECKPOINT \
    --report-to "wandb,tensorboard" \
    --wandb-project-name "openclip-test" \