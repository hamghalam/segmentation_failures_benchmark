#!/bin/bash

maybe_cluster="--cluster"

# =================================================================================================
# INVESTIGATING dataset size
group=revision_datasize_2408
dataset=mnms

# # SEGMENTATION
# base_epochs=2500
# for fold in 5 10 15 20; do
#     for seed in 0; do
#         # MnMs
#         if (( fold >= 20 )); then
#             check_every=80
#             epochs=$(( base_epochs * 16 ))
#         elif (( fold >= 15 )); then
#             check_every=40
#             epochs=$(( base_epochs * 8 ))
#         elif (( fold >= 10 )); then
#             check_every=20
#             epochs=$(( base_epochs * 4 ))
#         elif (( fold >= 5 )); then
#             check_every=10
#             epochs=$(( base_epochs * 2 ))
#         else
#             epochs=$base_epochs
#             check_every=5
#         fi
#         echo "epochs: $epochs"
#         python launcher.py --task train_seg --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --group $group $maybe_cluster \
#             --overwrites trainer.check_val_every_n_epoch=$check_every trainer.max_epochs=$epochs
#     done
# done


# # =================================================================================================
# # CROSS-VALIDATION PIXEL CSF (prepare regression methods)

# seed=0
# for fold in {5..24}; do
#     python launcher.py --task  validate_pixel_csf --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel baseline \
#         --group $group $maybe_cluster
# done

# # THEN
# for fold in 5 10 15 20; do
#     python prepare_auxdata.py --expt_group $group --start_fold $fold $maybe_cluster
# done



# # =================================================================================================
# # STAGE 2 TRAINING

# seed=0
# base_epochs=1000
# for fold in 10; do
#     # python launcher.py --task train_image_csf --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_image mahalanobis_gonzalez \
#     #     --group $group $maybe_cluster
#     if (( fold >= 20 )); then
#         check_every=80
#         epochs=$(( base_epochs * 16 ))
#     elif (( fold >= 15 )); then
#         check_every=40
#         epochs=$(( base_epochs * 8 ))
#     elif (( fold >= 10 )); then
#         check_every=20
#         epochs=$(( base_epochs * 4 ))
#     elif (( fold >= 5 )); then
#         check_every=10
#         epochs=$(( base_epochs * 2 ))
#     else
#         epochs=$base_epochs
#         check_every=5
#     fi
#     echo "epochs: $epochs"
#     python launcher.py --task train_image_csf --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel baseline --csf_image quality_regression \
#         --group $group $maybe_cluster --overwrites trainer.check_val_every_n_epoch=$check_every trainer.max_epochs=$epochs
#     # python launcher.py --task train_image_csf --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel baseline --csf_aggregation predictive_entropy+heuristic \
#     #     --group $group $maybe_cluster --cpu
# done


# # =================================================================================================
# # INFERENCE PIXEL CSF
# seed=0
# for fold in 5 10 15 20; do
#     python launcher.py --task test_pixel_csf --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel baseline \
#         --group $group $maybe_cluster
#     # python launcher.py --task test_pixel_csf --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel mcdropout \
#     #     --group $group $maybe_cluster
#     # python launcher.py --task test_pixel_csf --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel deep_ensemble \
#     #     --group $group $maybe_cluster
# done


# # =================================================================================================
# # FAILURE DETECTION TESTING
# seed=0
# for fold in 5 10 15 20; do
#     python launcher.py --task test_fd --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel baseline --csf_aggregation all_simple \
#         --group $group $maybe_cluster --cpu
#     python launcher.py --task test_fd --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel mcdropout --csf_aggregation all_simple \
#         --group $group $maybe_cluster --cpu
#     python launcher.py --task test_fd --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel deep_ensemble --csf_aggregation all_simple \
#         --group $group $maybe_cluster --cpu
#     python launcher.py --task test_fd --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_image mahalanobis_gonzalez \
#         --group $group $maybe_cluster
#     python launcher.py --task test_fd --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel deep_ensemble --csf_image quality_regression \
#         --group $group $maybe_cluster
#     python launcher.py --task test_fd --dataset $dataset --fold $fold --seed $seed --backbone dynamic_unet_dropout --csf_pixel baseline --csf_aggregation predictive_entropy+heuristic \
#         --group $group $maybe_cluster --cpu
# done
