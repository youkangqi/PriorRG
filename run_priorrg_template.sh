#!/usr/bin/env bash
set -euo pipefail

# ============================
# PriorRG command template
# ============================
# Usage examples:
#   MODE=stage1_train bash run_priorrg_template.sh
#   MODE=stage2_train STAGE1_CKPT=/abs/path/best_model.ckpt bash run_priorrg_template.sh
#   MODE=stage2_infer STAGE2_CKPT=/abs/path/best_model.ckpt bash run_priorrg_template.sh
#   MODE=single_sample_infer STAGE2_CKPT=/abs/path/best_model.ckpt bash run_priorrg_template.sh

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${MODE:-stage1_train}"

# ========== Required paths (edit these) ==========
CKPT_ZOO_DIR="${CKPT_ZOO_DIR:-/homeB/youkangqi/PriorRG/ckpt_zoo_dir}"
ANN_PATH="${ANN_PATH:-/homeB/youkangqi/PriorRG/ckpt_zoo_dir/priorrg_mimic_cxr_annotation.json}"
VIEW_POSITION_DICT="${VIEW_POSITION_DICT:-/homeB/youkangqi/PriorRG/ckpt_zoo_dir/view-positions-dict-mimic.json}"
IMAGES_DIR="${IMAGES_DIR:-${PROJECT_ROOT}/MIMIC-CXR/files}"
#STAGE1_CKPT="${STAGE1_CKPT:-/homeB/youkangqi/.cache/huggingface/hub/models--MK-runner--PriorRG/snapshots/34320325dffeb46217d18eca90b6bd5c0113aa99/checkpoints/mimic-cxr/pretraining/checkpoint/best_model.ckpt}"
#STAGE2_CKPT="${STAGE2_CKPT:-/homeB/youkangqi/.cache/huggingface/hub/models--MK-runner--PriorRG/snapshots/34320325dffeb46217d18eca90b6bd5c0113aa99/checkpoints/mimic-cxr/report-generation-gpt2/checkpoint/best_model.ckpt}"

# ========== Runtime ==========
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6,7}"
NUM_GPUS="${NUM_GPUS:-3}"
NUM_WORKERS="${NUM_WORKERS:-6}"
SEED="${SEED:-9233}"
VERSION="${VERSION:-repro}"
# Helps reduce CUDA memory fragmentation.
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ========== Model/training hyper-params ==========
MAX_LENGTH="${MAX_LENGTH:-100}"
ENCODER_MAX_LENGTH="${ENCODER_MAX_LENGTH:-300}"
TEMPORAL_FUSION_NUM_BLOCKS="${TEMPORAL_FUSION_NUM_BLOCKS:-3}"
PERCEIVER_NUM_BLOCKS="${PERCEIVER_NUM_BLOCKS:-3}"
NUM_LATENTS="${NUM_LATENTS:-128}"
NUM_BEAMS="${NUM_BEAMS:-3}"
STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-24}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-24}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-10}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-50}"

stage1_train() {
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" "${PYTHON_BIN}" "${PROJECT_ROOT}/main_github.py" \
    --data_name "mimic_cxr" \
    --version "${VERSION}" \
    --task "pretraining" \
    --phase "finetune" \
    --ann_path "${ANN_PATH}" \
    --view_position_dict "${VIEW_POSITION_DICT}" \
    --images_dir "${IMAGES_DIR}" \
    --max_length "${MAX_LENGTH}" \
    --encoder_max_length "${ENCODER_MAX_LENGTH}" \
    --num_workers "${NUM_WORKERS}" \
    --num_gpus "${NUM_GPUS}" \
    --seed "${SEED}" \
    --is_save_checkpoint "yes" \
    --ckpt_zoo_dir "${CKPT_ZOO_DIR}" \
    --temporal_fusion_num_blocks "${TEMPORAL_FUSION_NUM_BLOCKS}" \
    --perceiver_num_blocks "${PERCEIVER_NUM_BLOCKS}" \
    --num_latents "${NUM_LATENTS}" \
    --patience 10 \
    --pt_lr 5.0e-5 \
    --epochs "${STAGE1_EPOCHS}" \
    --batch_size "${STAGE1_BATCH_SIZE}"
}

stage1_infer() {
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" "${PYTHON_BIN}" "${PROJECT_ROOT}/main_github.py" \
    --data_name "mimic_cxr" \
    --version "${VERSION}" \
    --task "pretraining" \
    --phase "inference" \
    --ann_path "${ANN_PATH}" \
    --view_position_dict "${VIEW_POSITION_DICT}" \
    --images_dir "${IMAGES_DIR}" \
    --max_length "${MAX_LENGTH}" \
    --encoder_max_length "${ENCODER_MAX_LENGTH}" \
    --num_workers "${NUM_WORKERS}" \
    --num_gpus "${NUM_GPUS}" \
    --seed "${SEED}" \
    --is_save_checkpoint "no" \
    --ckpt_zoo_dir "${CKPT_ZOO_DIR}" \
    --test_ckpt_path "${STAGE1_CKPT}" \
    --temporal_fusion_num_blocks "${TEMPORAL_FUSION_NUM_BLOCKS}" \
    --perceiver_num_blocks "${PERCEIVER_NUM_BLOCKS}" \
    --num_latents "${NUM_LATENTS}" \
    --patience 10 \
    --pt_lr 5.0e-5 \
    --epochs "${STAGE1_EPOCHS}" \
    --batch_size "${STAGE1_BATCH_SIZE}"
}

stage2_train() {
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" "${PYTHON_BIN}" "${PROJECT_ROOT}/main_github.py" \
    --data_name "mimic_cxr" \
    --version "${VERSION}" \
    --task "report-generation-gpt2" \
    --phase "finetune" \
    --ann_path "${ANN_PATH}" \
    --view_position_dict "${VIEW_POSITION_DICT}" \
    --images_dir "${IMAGES_DIR}" \
    --max_length "${MAX_LENGTH}" \
    --encoder_max_length "${ENCODER_MAX_LENGTH}" \
    --num_workers "${NUM_WORKERS}" \
    --num_gpus "${NUM_GPUS}" \
    --seed "${SEED}" \
    --num_beams "${NUM_BEAMS}" \
    --is_save_checkpoint "yes" \
    --load "${STAGE1_CKPT}" \
    --ckpt_zoo_dir "${CKPT_ZOO_DIR}" \
    --temporal_fusion_num_blocks "${TEMPORAL_FUSION_NUM_BLOCKS}" \
    --perceiver_num_blocks "${PERCEIVER_NUM_BLOCKS}" \
    --num_latents "${NUM_LATENTS}" \
    --patience 5 \
    --pt_lr 5.0e-6 \
    --ft_lr 5.0e-5 \
    --monitor_metric "RCB" \
    --epochs "${STAGE2_EPOCHS}" \
    --batch_size "${STAGE2_BATCH_SIZE}"
}

stage2_infer() {
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" "${PYTHON_BIN}" "${PROJECT_ROOT}/main_github.py" \
    --data_name "mimic_cxr" \
    --version "${VERSION}" \
    --task "report-generation-gpt2" \
    --phase "inference" \
    --ann_path "${ANN_PATH}" \
    --view_position_dict "${VIEW_POSITION_DICT}" \
    --images_dir "${IMAGES_DIR}" \
    --max_length "${MAX_LENGTH}" \
    --encoder_max_length "${ENCODER_MAX_LENGTH}" \
    --num_workers "${NUM_WORKERS}" \
    --num_gpus "${NUM_GPUS}" \
    --seed "${SEED}" \
    --num_beams "${NUM_BEAMS}" \
    --is_save_checkpoint "no" \
    --test_ckpt_path "${STAGE2_CKPT}" \
    --ckpt_zoo_dir "${CKPT_ZOO_DIR}" \
    --temporal_fusion_num_blocks "${TEMPORAL_FUSION_NUM_BLOCKS}" \
    --perceiver_num_blocks "${PERCEIVER_NUM_BLOCKS}" \
    --num_latents "${NUM_LATENTS}" \
    --patience 5 \
    --pt_lr 5.0e-6 \
    --ft_lr 5.0e-5 \
    --monitor_metric "RCB" \
    --epochs "${STAGE2_EPOCHS}" \
    --batch_size "${STAGE2_BATCH_SIZE}"
}

single_sample_infer() {
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" "${PYTHON_BIN}" "${PROJECT_ROOT}/main_single_sample_github.py" \
    --task "report-generation-single-sample" \
    --phase "inference" \
    --ckpt_zoo_dir "${CKPT_ZOO_DIR}" \
    --view_position_dict "${VIEW_POSITION_DICT}" \
    --test_ckpt_path "${STAGE2_CKPT}" \
    --max_length "${MAX_LENGTH}" \
    --encoder_max_length "${ENCODER_MAX_LENGTH}" \
    --num_beams "${NUM_BEAMS}" \
    --num_gpus "${NUM_GPUS}" \
    --seed "${SEED}" \
    --temporal_fusion_num_blocks "${TEMPORAL_FUSION_NUM_BLOCKS}" \
    --perceiver_num_blocks "${PERCEIVER_NUM_BLOCKS}" \
    --num_latents "${NUM_LATENTS}"
}

case "${MODE}" in
  stage1_train) stage1_train ;;
  stage1_infer) stage1_infer ;;
  stage2_train) stage2_train ;;
  stage2_infer) stage2_infer ;;
  single_sample_infer) single_sample_infer ;;
  *)
    echo "Unknown MODE: ${MODE}"
    echo "Valid MODE values: stage1_train | stage1_infer | stage2_train | stage2_infer | single_sample_infer"
    exit 1
    ;;
esac
