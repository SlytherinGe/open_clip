export TEST_DATASET_ROOT=/mnt/SrvDataDisk/DatasetRemoteSensing
export MODEL=ViT-L-14
export DATASETS_TESTING='aid' 'eurosat' 'nwpu' 'millionaid'
export NAME=test-ViT-B-32
# export PRETRAINED_CHECKPOINT=laion2b_s34b_b79k
export PRETRAINED_CHECKPOINT=/mnt/SrvUserDisk/Gejunyao/VLP/open_clip/logs/2024_04_15-16_16_35-model_ViT-L-14-lr_1e-09-b_256-j_16-p_amp/checkpoints/epoch_20.pt

# '/mnt/SrvDataDisk/DatasetRemoteSensing/SkyScript/SkyCLIP_ViT_B32_top50pct/epoch_20.pt'
python train_val_test.py \
    --test-dataset-root $TEST_DATASET_ROOT \
    --model $MODEL \
    --datasets-for-testing 'aid' 'eurosat' 'nwpu' 'millionaid'\
    --pretrained $PRETRAINED_CHECKPOINT \
    --report-to "wandb,tensorboard" \
    --wandb-project-name "openclip-test" \