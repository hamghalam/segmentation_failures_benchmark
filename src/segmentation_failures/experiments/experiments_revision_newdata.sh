#!/bin/bash

maybe_cluster="--cluster"
# maybe_cluster=""

# =================================================================================================
# INVESTIGATING new dataset
group=revision_newdataset_2408

# =================================================================================================
# SEGMENTATION
for fold in 0 1 2 3 4; do
    for seed in 0 1 2 3 4; do
        # Retina 2D
        python launcher.py --task train_seg --dataset retina --fold $fold --seed $seed --backbone dynamic_unet_dropout --group $group $maybe_cluster

        # RETOUCH
        python launcher.py --task train_seg --dataset retouch_cirrus --fold $fold --seed $seed --backbone dynamic_unet_dropout --group $group $maybe_cluster

        # OCTA-500
        python launcher.py --task train_seg --dataset octa500 --fold $fold --seed $seed --backbone dynamic_unet_dropout --group $group $maybe_cluster

        # MVSeg
        python launcher.py --task train_seg --dataset mvseg23 --fold $fold --seed $seed --backbone dynamic_unet_dropout --group $group $maybe_cluster
    done
done

# =================================================================================================
# CROSS-VALIDATION PIXEL CSF (prepare regression methods)

seed=0
for dataset in octa500; do
    for fold in 0 1 2 3 4;
    do
        python launcher.py --task  validate_pixel_csf --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel baseline \
            --group $group $maybe_cluster
        python launcher.py --task  validate_pixel_csf --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel deep_ensemble \
            --group $group $maybe_cluster
    done
done

# THEN
for dataset in octa500; do
    python prepare_auxdata.py --expt_group $group --dataset $dataset $maybe_cluster
done

# =================================================================================================
# STAGE 2 TRAINING

seed=0
for dataset in octa500; do
    for fold in 0 1 2 3 4; do
        python launcher.py --task train_image_csf --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_image mahalanobis_gonzalez \
            --group $group $maybe_cluster
        # python launcher.py --task train_image_csf --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel baseline --csf_image quality_regression \
        #     --group $group $maybe_cluster
        # python launcher.py --task train_image_csf --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel baseline --csf_image vae_mask_only \
        #     --group $group $maybe_cluster
        # aggregation
        python launcher.py --task train_image_csf --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel baseline --csf_aggregation predictive_entropy+radiomics \
            --group $group $maybe_cluster --cpu
        python launcher.py --task train_image_csf --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel baseline --csf_aggregation predictive_entropy+heuristic \
            --group $group $maybe_cluster --cpu
        python launcher.py --task train_image_csf --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel deep_ensemble --csf_aggregation predictive_entropy+radiomics \
            --group $group $maybe_cluster --cpu
        python launcher.py --task train_image_csf --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel deep_ensemble --csf_aggregation predictive_entropy+heuristic \
            --group $group $maybe_cluster --cpu
    done
done


# =================================================================================================
# INFERENCE PIXEL CSF
seed=0
for dataset in octa500; do
    for fold in 0 1 2 3 4; do
        python launcher.py --task test_pixel_csf --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel baseline \
            --group $group $maybe_cluster
        python launcher.py --task test_pixel_csf --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel mcdropout \
            --group $group $maybe_cluster
        python launcher.py --task test_pixel_csf --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel deep_ensemble \
            --group $group $maybe_cluster
    done
done


# =================================================================================================
# FAILURE DETECTION TESTING
seed=0
for dataset in octa500; do
    for fold in 0 1 2 3 4; do
        python launcher.py --task test_fd --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel baseline --csf_aggregation all_simple \
            --group $group $maybe_cluster --cpu
        python launcher.py --task test_fd --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel mcdropout --csf_aggregation all_simple \
            --group $group $maybe_cluster --cpu
        python launcher.py --task test_fd --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel deep_ensemble --csf_aggregation all_simple \
            --group $group $maybe_cluster --cpu
        python launcher.py --task test_fd --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_image mahalanobis_gonzalez \
            --group $group $maybe_cluster
        python launcher.py --task test_fd --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel deep_ensemble --csf_image quality_regression \
            --group $group $maybe_cluster
        python launcher.py --task test_fd --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel deep_ensemble --csf_image vae_mask_only \
            --group $group $maybe_cluster
        python launcher.py --task test_fd --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel baseline --csf_aggregation predictive_entropy+radiomics \
            --group $group $maybe_cluster --cpu
        python launcher.py --task test_fd --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel baseline --csf_aggregation predictive_entropy+heuristic \
            --group $group $maybe_cluster --cpu
        python launcher.py --task test_fd --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel deep_ensemble --csf_aggregation predictive_entropy+radiomics \
            --group $group $maybe_cluster --cpu
        python launcher.py --task test_fd --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel deep_ensemble --csf_aggregation predictive_entropy+heuristic \
            --group $group $maybe_cluster --cpu
    done
done
