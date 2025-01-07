#!/bin/bash

group=segpaper
# maybe_cluster="--cluster"

# =================================================================================================
# SEGMENTATION
seed=0
for fold in {0..4}; do
    for seed in {0..4}; do
        # liver
        python launcher.py --task train_seg --dataset liver --fold $fold --seed $seed --backbone dynamic_unet_dropout --group $group 
    done
done

# # =================================================================================================
# # CROSS-VALIDATION PIXEL CSF (prepare regression methods)
# # simple brats
# seed=0
# for fold in {0..4};
# do
#     python launcher.py --task  validate_pixel_csf --dataset simple_fets22_corrupted --fold $fold --seed $seed --backbone monai_unet_dropout --csf_pixel baseline \
#         --group $group $maybe_cluster
#     python launcher.py --task  validate_pixel_csf --dataset simple_fets22_corrupted --fold $fold --seed $seed --backbone monai_unet_dropout --csf_pixel deep_ensemble \
#         --group $group $maybe_cluster
# done

# # 3d datasets: mnms prostate_gonzalez kits23 covid_gonzalez brats19_lhgg
# seed=0
# for dataset in mnms prostate_gonzalez kits23 covid_gonzalez brats19_lhgg; do
#     for fold in {0..4};
#     do
#         python launcher.py --task  validate_pixel_csf --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel baseline \
#             --group $group $maybe_cluster
#         python launcher.py --task  validate_pixel_csf --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel deep_ensemble \
#             --group $group $maybe_cluster
#     done
# done

# # =================================================================================================
# # PREPARE AUXDATA
# for dataset in simple_fets22_corrupted mnms prostate_gonzalez kits23 covid_gonzalez brats19_lhgg; do
#     python prepare_auxdata.py --expt_group $group --dataset $dataset $maybe_cluster
# done

# # =================================================================================================
# # STAGE 2 TRAINING
# # simple brats
# dataset=simple_fets22_corrupted
# seed=0
# for fold in {0..4}
# do
#     python launcher.py --task train_image_csf --dataset $dataset --fold $fold --seed $seed --backbone monai_unet_dropout --csf_image mahalanobis_gonzalez \
#         --group $group $maybe_cluster
#     python launcher.py --task train_image_csf --dataset $dataset --fold $fold --seed $seed --backbone monai_unet_dropout --csf_image quality_regression --csf_pixel baseline \
#         --group $group $maybe_cluster
#     python launcher.py --task train_image_csf --dataset $dataset --fold $fold --seed $seed --backbone monai_unet_dropout --csf_image vae_mask_only --csf_pixel baseline \
#         --group $group $maybe_cluster
#     # aggregation
#     python launcher.py --task train_image_csf --dataset $dataset --fold $fold --seed $seed --backbone monai_unet_dropout --csf_pixel baseline --csf_aggregation predictive_entropy+radiomics \
#         --group $group $maybe_cluster --cpu
#     python launcher.py --task train_image_csf --dataset $dataset --fold $fold --seed $seed --backbone monai_unet_dropout --csf_pixel baseline --csf_aggregation predictive_entropy+heuristic \
#         --group $group $maybe_cluster --cpu
#     python launcher.py --task train_image_csf --dataset $dataset --fold $fold --seed $seed --backbone monai_unet_dropout --csf_pixel deep_ensemble --csf_aggregation predictive_entropy+radiomics \
#         --group $group $maybe_cluster --cpu
#     python launcher.py --task train_image_csf --dataset $dataset --fold $fold --seed $seed --backbone monai_unet_dropout --csf_pixel deep_ensemble --csf_aggregation predictive_entropy+heuristic \
#         --group $group $maybe_cluster --cpu
# done

# # 3d data: mnms prostate_gonzalez kits23 covid_gonzalez brats19_lhgg
# seed=0
# for dataset in mnms prostate_gonzalez kits23 covid_gonzalez brats19_lhgg; do
#     for fold in {0..4}; do
#         python launcher.py --task train_image_csf --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_image mahalanobis_gonzalez \
#             --group $group $maybe_cluster
#         python launcher.py --task train_image_csf --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel baseline --csf_image quality_regression \
#             --group $group $maybe_cluster
#         python launcher.py --task train_image_csf --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel baseline --csf_image vae_mask_only \
#             --group $group $maybe_cluster
#         # aggregation
#         python launcher.py --task train_image_csf --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel baseline --csf_aggregation predictive_entropy+radiomics \
#             --group $group $maybe_cluster --cpu
#         python launcher.py --task train_image_csf --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel baseline --csf_aggregation predictive_entropy+heuristic \
#             --group $group $maybe_cluster --cpu
#         python launcher.py --task train_image_csf --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel deep_ensemble --csf_aggregation predictive_entropy+radiomics \
#             --group $group $maybe_cluster --cpu
#         python launcher.py --task train_image_csf --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel deep_ensemble --csf_aggregation predictive_entropy+heuristic \
#             --group $group $maybe_cluster --cpu
#     done
# done


# # =================================================================================================
# # INFERENCE PIXEL CSF
# dataset=simple_fets22_corrupted
# seed=0
# for fold in {0..4}
# do
#     python launcher.py --task test_pixel_csf --dataset $dataset --fold $fold --seed $seed --backbone monai_unet_dropout --csf_pixel baseline \
#         --group $group $maybe_cluster
#     python launcher.py --task test_pixel_csf --dataset $dataset --fold $fold --seed $seed --backbone monai_unet_dropout --csf_pixel mcdropout \
#         --group $group $maybe_cluster
#     python launcher.py --task test_pixel_csf --dataset $dataset --fold $fold --seed $seed --backbone monai_unet_dropout --csf_pixel deep_ensemble \
#         --group $group $maybe_cluster
# done

# seed=0
# for dataset in mnms prostate_gonzalez kits23 covid_gonzalez brats19_lhgg; do
#     for fold in {0..4}; do
#         python launcher.py --task test_pixel_csf --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel baseline \
#             --group $group $maybe_cluster
#         python launcher.py --task test_pixel_csf --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel mcdropout \
#             --group $group $maybe_cluster
#         python launcher.py --task test_pixel_csf --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel deep_ensemble \
#             --group $group $maybe_cluster
#     done
# done


# # =================================================================================================
# # FAILURE DETECTION TESTING
# # Simple FeTS
# seed=0
# dataset=simple_fets22_corrupted
# for fold in {0..4}
# do
#     python launcher.py --task test_fd --dataset $dataset --fold $fold --seed $seed --backbone monai_unet_dropout --csf_pixel baseline --csf_aggregation all_simple \
#         --group $group --cluster --cpu
#     python launcher.py --task test_fd --dataset $dataset --fold $fold --seed $seed --backbone monai_unet_dropout --csf_pixel mcdropout --csf_aggregation all_simple \
#         --group $group --cluster --cpu
#     python launcher.py --task test_fd --dataset $dataset --fold $fold --seed $seed --backbone monai_unet_dropout --csf_pixel deep_ensemble --csf_aggregation all_simple \
#         --group $group --cluster --cpu
#     python launcher.py --task test_fd --dataset $dataset --fold $fold --seed $seed --backbone monai_unet_dropout --csf_image mahalanobis_gonzalez \
#         --group $group $maybe_cluster
#     python launcher.py --task test_fd --dataset $dataset --fold $fold --seed $seed --backbone monai_unet_dropout --csf_pixel baseline --csf_image quality_regression \
#         --group $group $maybe_cluster
#     python launcher.py --task test_fd --dataset $dataset --fold $fold --seed $seed --backbone monai_unet_dropout --csf_pixel deep_ensemble --csf_image quality_regression \
#         --group $group $maybe_cluster
#     python launcher.py --task test_fd --dataset $dataset --fold $fold --seed $seed --backbone monai_unet_dropout --csf_pixel deep_ensemble --csf_image vae_mask_only \
#         --group $group $maybe_cluster
#     python launcher.py --task test_fd --dataset $dataset --fold $fold --seed $seed --backbone monai_unet_dropout --csf_pixel baseline --csf_aggregation predictive_entropy+radiomics \
#         --group $group $maybe_cluster --cpu
#     python launcher.py --task test_fd --dataset $dataset --fold $fold --seed $seed --backbone monai_unet_dropout --csf_pixel deep_ensemble --csf_aggregation predictive_entropy+radiomics \
#         --group $group $maybe_cluster --cpu
#     python launcher.py --task test_fd --dataset $dataset --fold $fold --seed $seed --backbone monai_unet_dropout --csf_pixel baseline --csf_aggregation predictive_entropy+heuristic \
#         --group $group $maybe_cluster --cpu
#     python launcher.py --task test_fd --dataset $dataset --fold $fold --seed $seed --backbone monai_unet_dropout --csf_pixel deep_ensemble --csf_aggregation predictive_entropy+heuristic \
#         --group $group $maybe_cluster --cpu
# done

# # 3d datasets: mnms prostate_gonzalez kits23 covid_gonzalez brats19_lhgg
# seed=0
# for dataset in mnms prostate_gonzalez kits23 covid_gonzalez brats19_lhgg; do
#     for fold in {0..4}; do
#         python launcher.py --task test_fd --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel baseline --csf_aggregation all_simple \
#             --group $group --cluster --cpu
#         python launcher.py --task test_fd --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel mcdropout --csf_aggregation all_simple \
#             --group $group --cluster --cpu
#         python launcher.py --task test_fd --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel deep_ensemble --csf_aggregation all_simple \
#             --group $group --cluster --cpu
#         python launcher.py --task test_fd --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_image mahalanobis_gonzalez \
#             --group $group $maybe_cluster
#         python launcher.py --task test_fd --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel baseline --csf_image quality_regression \
#             --group $group $maybe_cluster
#         python launcher.py --task test_fd --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel deep_ensemble --csf_image quality_regression \
#             --group $group $maybe_cluster
#         python launcher.py --task test_fd --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel deep_ensemble --csf_image vae_mask_only \
#             --group $group $maybe_cluster
#         python launcher.py --task test_fd --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel baseline --csf_aggregation predictive_entropy+radiomics \
#             --group $group $maybe_cluster --cpu
#         python launcher.py --task test_fd --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel baseline --csf_aggregation predictive_entropy+heuristic \
#             --group $group $maybe_cluster --cpu
#         python launcher.py --task test_fd --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel deep_ensemble --csf_aggregation predictive_entropy+radiomics \
#             --group $group $maybe_cluster --cpu
#         python launcher.py --task test_fd --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel deep_ensemble --csf_aggregation predictive_entropy+heuristic \
#             --group $group $maybe_cluster --cpu
#     done
# done
