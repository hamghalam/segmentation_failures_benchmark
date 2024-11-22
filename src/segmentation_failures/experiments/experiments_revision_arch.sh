#!/bin/bash

maybe_cluster="--cluster"

# =================================================================================================
# INVESTIGATING architecture improvements
group=revision_architecture_2408

# # SEGMENTATION
# for fold in 0 1 2; do
#     for seed in 0 1 2 3 4; do
#         # # MnMs
#         # python launcher.py --task train_seg --dataset mnms --fold $fold --seed $seed --backbone dynamic_unet_dropout --group $group $maybe_cluster
#         # python launcher.py --task train_seg --dataset mnms --fold $fold --seed $seed --backbone dynamic_resencunet_dropout --group $group $maybe_cluster
#         python launcher.py --task train_seg --dataset mnms --fold $fold --seed $seed --backbone dynamic_wideunet_dropout --group $group $maybe_cluster \
#             --gmem 24G
#         # # KiTS
#         # python launcher.py --task train_seg --dataset kits23 --fold $fold --seed $seed --backbone dynamic_unet_dropout --group $group $maybe_cluster
#         # python launcher.py --task train_seg --dataset kits23 --fold $fold --seed $seed --backbone dynamic_resencunet_dropout --group $group $maybe_cluster \
#         #     --overwrites datamodule.nnunet_plans_id=nnUNetResEncUNetLPlans datamodule.patch_size="[160,224,192]" --gmem 24G
#     done
# done


# # =================================================================================================
# # CROSS-VALIDATION PIXEL CSF (prepare regression methods)

# seed=0
# for fold in 0 1 2;
# do
#     python launcher.py --task  validate_pixel_csf --dataset mnms --fold $fold --seed $seed --backbone dynamic_resencunet_dropout --csf_pixel baseline \
#         --group $group $maybe_cluster
#     python launcher.py --task  validate_pixel_csf --dataset mnms --fold $fold --seed $seed --backbone dynamic_wideunet_dropout --csf_pixel baseline \
#         --group $group $maybe_cluster
#     python launcher.py --task  validate_pixel_csf --dataset kits23 --fold $fold --seed $seed --backbone dynamic_resencunet_dropout --csf_pixel baseline \
#         --group $group $maybe_cluster --overwrites datamodule.nnunet_plans_id=nnUNetResEncUNetLPlans datamodule.patch_size="[160,224,192]"
# done

# # THEN
# python prepare_auxdata.py --expt_group $group --start_fold 0 --backbone dynamic_resencunet_dropout $maybe_cluster
# python prepare_auxdata.py --expt_group $group --start_fold 0 --backbone dynamic_wideunet_dropout $maybe_cluster


# # =================================================================================================
# # STAGE 2 TRAINING

# seed=0
# for fold in 0 1 2; do
#     # MnMs
#     # python launcher.py --task train_image_csf --dataset mnms --fold $fold --seed $seed --backbone dynamic_resencunet_dropout --csf_image mahalanobis_gonzalez \
#     #     --group $group $maybe_cluster
#     # python launcher.py --task train_image_csf --dataset mnms --fold $fold --seed $seed --backbone dynamic_wideunet_dropout --csf_image mahalanobis_gonzalez \
#     #     --group $group $maybe_cluster
#     # python launcher.py --task train_image_csf --dataset mnms --fold $fold --seed $seed --backbone dynamic_resencunet_dropout --csf_pixel baseline --csf_image quality_regression \
#     #     --group $group $maybe_cluster
#     # python launcher.py --task train_image_csf --dataset mnms --fold $fold --seed $seed --backbone dynamic_wideunet_dropout --csf_pixel baseline --csf_image quality_regression \
#     #     --group $group $maybe_cluster
#     # python launcher.py --task train_image_csf --dataset mnms --fold $fold --seed $seed --backbone dynamic_resencunet_dropout --csf_pixel baseline --csf_aggregation predictive_entropy+heuristic \
#     #     --group $group $maybe_cluster --cpu
#     # python launcher.py --task train_image_csf --dataset mnms --fold $fold --seed $seed --backbone dynamic_wideunet_dropout --csf_pixel baseline --csf_aggregation predictive_entropy+heuristic \
#     #     --group $group $maybe_cluster --cpu
#     # KiTS
#     # python launcher.py --task train_image_csf --dataset kits23 --fold $fold --seed $seed --backbone dynamic_resencunet_dropout --csf_image mahalanobis_gonzalez \
#     #     --group $group $maybe_cluster
#     python launcher.py --task train_image_csf --dataset kits23 --fold $fold --seed $seed --backbone dynamic_resencunet_dropout --csf_pixel baseline --csf_image quality_regression \
#         --group $group $maybe_cluster --gmem 16G
#     python launcher.py --task train_image_csf --dataset kits23 --fold $fold --seed $seed --backbone dynamic_resencunet_dropout --csf_pixel baseline --csf_aggregation predictive_entropy+heuristic \
#         --group $group $maybe_cluster --cpu
# done


# # =================================================================================================
# # INFERENCE PIXEL CSF

# seed=0
# for fold in 0 1 2; do
#     # M&Ms
#     # python launcher.py --task test_pixel_csf --dataset mnms --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel baseline \
#     #     --group $group $maybe_cluster
#     # python launcher.py --task test_pixel_csf --dataset mnms --fold $fold --seed $seed --backbone dynamic_resencunet_dropout --csf_pixel baseline \
#     #     --group $group $maybe_cluster
#     # python launcher.py --task test_pixel_csf --dataset mnms --fold $fold --seed $seed --backbone dynamic_resencunet_dropout --csf_pixel mcdropout \
#     #     --group $group $maybe_cluster
#     # python launcher.py --task test_pixel_csf --dataset mnms --fold $fold --seed $seed --backbone dynamic_resencunet_dropout --csf_pixel deep_ensemble \
#     #     --group $group $maybe_cluster
#     # python launcher.py --task test_pixel_csf --dataset mnms --fold $fold --seed $seed --backbone dynamic_wideunet_dropout --csf_pixel baseline \
#     #     --group $group $maybe_cluster
#     # python launcher.py --task test_pixel_csf --dataset mnms --fold $fold --seed $seed --backbone dynamic_wideunet_dropout --csf_pixel mcdropout \
#     #     --group $group $maybe_cluster
#     # python launcher.py --task test_pixel_csf --dataset mnms --fold $fold --seed $seed --backbone dynamic_wideunet_dropout --csf_pixel deep_ensemble \
#     #     --group $group $maybe_cluster --gmem 16G
#     # KiTS
#     # python launcher.py --task test_pixel_csf --dataset kits23 --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel baseline \
#     #     --group $group $maybe_cluster
#     # python launcher.py --task test_pixel_csf --dataset kits23 --fold $fold --seed $seed --backbone dynamic_resencunet_dropout --csf_pixel baseline \
#     #     --group $group $maybe_cluster --overwrites datamodule.nnunet_plans_id=nnUNetResEncUNetLPlans datamodule.patch_size="[160,224,192]" --gmem 16G
#     # python launcher.py --task test_pixel_csf --dataset kits23 --fold $fold --seed $seed --backbone dynamic_resencunet_dropout --csf_pixel mcdropout \
#     #     --group $group $maybe_cluster --overwrites datamodule.nnunet_plans_id=nnUNetResEncUNetLPlans datamodule.patch_size="[160,224,192]" --gmem 16G
#     python launcher.py --task test_pixel_csf --dataset kits23 --fold $fold --seed $seed --backbone dynamic_resencunet_dropout --csf_pixel deep_ensemble \
#         --group $group $maybe_cluster --overwrites datamodule.nnunet_plans_id=nnUNetResEncUNetLPlans datamodule.patch_size="[160,224,192]" --gmem 16G
# done


# =================================================================================================
# FAILURE DETECTION TESTING

# seed=0
# for fold in 0 1 2; do
#     # MnMs
#     # python launcher.py --task test_fd --dataset mnms --fold $fold --seed $seed --backbone dynamic_resencunet_dropout --csf_pixel baseline --csf_aggregation all_simple \
#     #     --group $group --cluster --cpu
#     # # python launcher.py --task test_fd --dataset mnms --fold $fold --seed $seed --backbone dynamic_resencunet_dropout --csf_pixel mcdropout --csf_aggregation all_simple \
#     # #     --group $group --cluster --cpu
#     # # python launcher.py --task test_fd --dataset mnms --fold $fold --seed $seed --backbone dynamic_resencunet_dropout --csf_pixel deep_ensemble --csf_aggregation all_simple \
#     # #     --group $group --cluster --cpu
#     # # python launcher.py --task test_fd --dataset mnms --fold $fold --seed $seed --backbone dynamic_resencunet_dropout --csf_image mahalanobis_gonzalez \
#     # #     --group $group $maybe_cluster
#     # # python launcher.py --task test_fd --dataset mnms --fold $fold --seed $seed --backbone dynamic_resencunet_dropout --csf_pixel deep_ensemble --csf_image quality_regression \
#     # #     --group $group $maybe_cluster
#     # python launcher.py --task test_fd --dataset mnms --fold $fold --seed $seed --backbone dynamic_resencunet_dropout --csf_pixel baseline --csf_aggregation predictive_entropy+heuristic \
#     #     --group $group $maybe_cluster --cpu
#     # python launcher.py --task test_fd --dataset mnms --fold $fold --seed $seed --backbone dynamic_wideunet_dropout --csf_pixel baseline --csf_aggregation all_simple \
#     #     --group $group --cluster --cpu
#     # # python launcher.py --task test_fd --dataset mnms --fold $fold --seed $seed --backbone dynamic_wideunet_dropout --csf_pixel mcdropout --csf_aggregation all_simple \
#     # #     --group $group --cluster --cpu
#     # # python launcher.py --task test_fd --dataset mnms --fold $fold --seed $seed --backbone dynamic_wideunet_dropout --csf_pixel deep_ensemble --csf_aggregation all_simple \
#     # #     --group $group --cluster --cpu
#     # # python launcher.py --task test_fd --dataset mnms --fold $fold --seed $seed --backbone dynamic_wideunet_dropout --csf_image mahalanobis_gonzalez \
#     # #     --group $group $maybe_cluster
#     # # python launcher.py --task test_fd --dataset mnms --fold $fold --seed $seed --backbone dynamic_wideunet_dropout --csf_pixel deep_ensemble --csf_image quality_regression \
#     # #     --group $group $maybe_cluster
#     # python launcher.py --task test_fd --dataset mnms --fold $fold --seed $seed --backbone dynamic_wideunet_dropout --csf_pixel baseline --csf_aggregation predictive_entropy+heuristic \
#     #     --group $group $maybe_cluster --cpu

#     # KiTS
#     # python launcher.py --task test_fd --dataset kits23 --fold $fold --seed $seed --backbone dynamic_resencunet_dropout --csf_pixel baseline --csf_aggregation all_simple \
#     #     --group $group --cluster --cpu
#     # python launcher.py --task test_fd --dataset kits23 --fold $fold --seed $seed --backbone dynamic_resencunet_dropout --csf_pixel mcdropout --csf_aggregation all_simple \
#     #     --group $group --cluster --cpu
#     # python launcher.py --task test_fd --dataset kits23 --fold $fold --seed $seed --backbone dynamic_resencunet_dropout --csf_pixel deep_ensemble --csf_aggregation all_simple \
#     #     --group $group --cluster --cpu
#     # python launcher.py --task test_fd --dataset kits23 --fold $fold --seed $seed --backbone dynamic_resencunet_dropout --csf_image mahalanobis_gonzalez \
#     #     --group $group $maybe_cluster --gmem 16G
#     # python launcher.py --task test_fd --dataset kits23 --fold $fold --seed $seed --backbone dynamic_resencunet_dropout --csf_pixel deep_ensemble --csf_image quality_regression \
#     #     --group $group $maybe_cluster --gmem 16G
#     # python launcher.py --task test_fd --dataset kits23 --fold $fold --seed $seed --backbone dynamic_resencunet_dropout --csf_pixel baseline --csf_aggregation predictive_entropy+heuristic \
#     #     --group $group $maybe_cluster --cpu
# done
