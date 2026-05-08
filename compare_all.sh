#!/usr/bin/env bash
# =============================================================================
# UNet 遥感建筑提取 — 全组合方案对比脚本
# =============================================================================
#
# 对比范围：
#   - 模型变体：Res34 基线 × CBAM 4 种插入模式 × EfficientNet 5 种尺度
#   - 推理策略：no_overlap × overlap+uniform × overlap+gaussian × overlap+gauss
#     (stride=256)
#   - 合计：11 模型 × 4 推理 = 44 组结果
#
# 使用：
#   bash compare_all.sh              # 全部模型
#   bash compare_all.sh --quick       # 仅核心模型（res34 + cbam + enet-b0）
#   bash compare_all.sh --cbam-only   # 仅 Res34 + CBAM 变体
#   bash compare_all.sh --enet-only   # 仅 EfficientNet 变体
#   bash compare_all.sh --skip-train  # 跳过训练，仅做推理（checkpoint 已就绪）
#
# 输出：
#   comparison_results/{model}/  各模型实验输出（predict.png 等）
#   comparison_results/records.json   汇总记录
#   comparison_results/table.txt      对比表格
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_DIR="${REPO_DIR}/code"
WEIGHT_FILE="${CODE_DIR}/weight/resnet34-b627a593.pth"
RESULT_DIR="${REPO_DIR}/comparison_results"

# ResNet34 baseline
MODEL_RES34="res34"

# CBAM 变体（ResNet34 encoder + 不同 CBAM 插入位置）
MODELS_CBAM=(
    "res34_cbam_enc_dec"   # adaptive: decoder + skip
    "res34_cbam_enc"       # encoder only
    "res34_cbam_dec"       # decoder only
    "res34_cbam_skip"      # skip connections only
)

# EfficientNet 变体（b0–b4 + auto-adaptive）
MODELS_EFFICIENTNET=(
    "efficientnet_b0"
    "efficientnet_b1"
    "efficientnet_b2"
    "efficientnet_b3"
    "efficientnet_b4"
    "efficientnet_auto"
)

# 推理实验预设（与 step3_predict.py EXPERIMENT_PRESETS 对齐）
INFERENCE_EXPERIMENTS=(
    "baseline"
    "overlap_uniform"
    "overlap_gaussian"
    "overlap_stride256"
)

# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

MODE="all"
SKIP_TRAIN=false

for arg in "$@"; do
    case "$arg" in
        --quick)     MODE="quick" ;;
        --cbam-only) MODE="cbam" ;;
        --enet-only) MODE="enet" ;;
        --skip-train) SKIP_TRAIN=true ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: bash compare_all.sh [--quick|--cbam-only|--enet-only|--skip-train]"
            exit 1
            ;;
    esac
done

select_models() {
    case "$MODE" in
        quick)
            MODELS=("$MODEL_RES34" "${MODELS_CBAM[@]}" "efficientnet_b0")
            ;;
        cbam)
            MODELS=("$MODEL_RES34" "${MODELS_CBAM[@]}")
            ;;
        enet)
            MODELS=("$MODEL_RES34" "${MODELS_EFFICIENTNET[@]}")
            ;;
        all)
            MODELS=("$MODEL_RES34" "${MODELS_CBAM[@]}" "${MODELS_EFFICIENTNET[@]}")
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

log_section() {
    echo ""
    echo "============================================================"
    echo "  $*"
    echo "  $(timestamp)"
    echo "============================================================"
}

log_info() {
    echo "[$(timestamp)] $*"
}

log_warn() {
    echo "[$(timestamp)] WARNING: $*" >&2
}

log_error() {
    echo "[$(timestamp)] ERROR: $*" >&2
}

# ---------------------------------------------------------------------------
# Prerequisite checks
# ---------------------------------------------------------------------------

check_prerequisites() {
    log_section "Checking prerequisites"

    # Source images
    local missing=()
    for f in "img_trainval.png" "label_trainval.png" "img_test.png" "label_test.png"; do
        if [[ ! -f "${CODE_DIR}/${f}" ]]; then
            missing+=("${CODE_DIR}/${f}")
        fi
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing source files:"
        for f in "${missing[@]}"; do
            echo "  - $f"
        done
        exit 1
    fi
    log_info "Source images: OK"

    # Pretrained ResNet34 weight
    if [[ ! -f "$WEIGHT_FILE" ]]; then
        log_error "Missing pretrained weight: $WEIGHT_FILE"
        log_error "Download resnet34-b627a593.pth and place it under code/weight/"
        exit 1
    fi
    log_info "Pretrained weight: OK"

    # Python and key packages
    python3 -c "import torch; import torchvision; import numpy; import PIL; import skimage; import tqdm" 2>/dev/null || {
        log_error "Missing Python dependencies. Install: pip install -r requirements.txt"
        exit 1
    }
    log_info "Python dependencies: OK"

    # Detect device
    DEVICE=$(python3 -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")
    log_info "Torch device: ${DEVICE}"
}

# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

ensure_dataset() {
    log_section "Ensuring baseline dataset is ready"

    local ds_dir="${CODE_DIR}/dataset/train/image"
    if [[ -d "$ds_dir" ]] && [[ $(find "$ds_dir" -maxdepth 1 -name '*.png' | head -1) ]]; then
        log_info "Dataset already exists, skipping crop step."
    else
        log_info "Running step1_crop_dataset.py (baseline, stride=512)..."
        cd "$CODE_DIR"
        python3 step1_crop_dataset.py --train-stride 512
        cd "$REPO_DIR"
        log_info "Dataset created."
    fi
}

# ---------------------------------------------------------------------------
# Training (per model)
# ---------------------------------------------------------------------------

train_model() {
    local model_type="$1"
    local ckpt_name="UNet_${model_type}_best.pth"
    local ckpt_path="${CODE_DIR}/checkpoints/${ckpt_name}"

    log_section "Train: ${model_type}"

    if [[ -f "$ckpt_path" ]]; then
        log_info "Checkpoint exists: ${ckpt_path}  →  skip training."
        return 0
    fi

    log_info "Training ${model_type} on device=${DEVICE} ..."
    log_info "Checkpoint will be: ${ckpt_path}"

    local start_ts
    start_ts=$(date +%s)

    cd "$CODE_DIR"

    # Check registration before launching lengthy training
    local is_registered
    is_registered=$(python3 -c "
import sys; sys.path.insert(0,'.')
from step2_train import MODEL_REGISTRY
print('yes' if '${model_type}' in MODEL_REGISTRY else 'no')
" 2>/dev/null)
    if [[ "$is_registered" != "yes" ]]; then
        log_warn "${model_type} is not available in MODEL_REGISTRY (missing module?). Skip."
        cd "$REPO_DIR"
        return 1
    fi

    # ---- run training via embedded Python ---------------------------------
    MODEL_TYPE="$model_type" \
    DEVICE_STR="$DEVICE" \
    python3 << 'PYEOF'
import os, sys, logging

model_type = os.environ["MODEL_TYPE"]
device_str = os.environ["DEVICE_STR"]

sys.path.insert(0, ".")
from step2_train import CONFIG, train_net, build_model, resolve_config_paths, set_seed
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Only touch the keys we need to change; leave everything else at defaults
CONFIG["model_type"] = model_type
config = resolve_config_paths(CONFIG)
set_seed(config["seed"])

device = torch.device(device_str)
logging.info("Device: %s", device)

model = build_model(config)
n_params = sum(p.numel() for p in model.parameters())
logging.info("Model: %s  params: %s", model_type, f"{n_params:,}")

net = model.to(device)
train_net(net, device, config)
PYEOF
    local rc=$?
    cd "$REPO_DIR"

    local end_ts
    end_ts=$(date +%s)
    local elapsed=$(( end_ts - start_ts ))

    if [[ $rc -ne 0 ]]; then
        log_error "Training FAILED for ${model_type} (exit code ${rc}, elapsed ${elapsed}s)"
        return 1
    fi

    if [[ ! -f "$ckpt_path" ]]; then
        log_error "Training completed but checkpoint NOT FOUND: ${ckpt_path}"
        return 1
    fi

    log_info "Training SUCCESS: ${model_type}  (elapsed ${elapsed}s, $(du -h "$ckpt_path" | cut -f1))"
    return 0
}

# ---------------------------------------------------------------------------
# Inference suite (per model)
# ---------------------------------------------------------------------------

run_inference_suite() {
    local model_type="$1"
    local ckpt_name="UNet_${model_type}_best.pth"
    local ckpt_path="${CODE_DIR}/checkpoints/${ckpt_name}"
    local model_out_dir="${RESULT_DIR}/${model_type}"

    log_section "Inference: ${model_type}"

    if [[ ! -f "$ckpt_path" ]]; then
        log_error "Checkpoint missing: ${ckpt_path}. Run training first."
        return 1
    fi

    log_info "Checkpoint: ${ckpt_path}"
    log_info "Output dir: ${model_out_dir}"

    cd "$CODE_DIR"

    python3 step3_predict.py \
        --model-type "${model_type}" \
        --checkpoint "${ckpt_path}" \
        --output-dir "${model_out_dir}" \
        --run-suite \
        --use-history-threshold \
        --no-save-prob \
        --no-save-vis

    local rc=$?
    cd "$REPO_DIR"

    if [[ $rc -ne 0 ]]; then
        log_error "Inference FAILED for ${model_type}"
        return 1
    fi

    # Verify output
    local rec_file="${model_out_dir}/experiment_records.json"
    if [[ ! -f "$rec_file" ]]; then
        log_error "experiment_records.json not found: ${rec_file}"
        return 1
    fi

    log_info "Inference SUCCESS: ${model_type}"
    return 0
}

# ---------------------------------------------------------------------------
# Results collection & comparison table
# ---------------------------------------------------------------------------

collect_results() {
    log_section "Collecting all experiment records"

    # Compile a single flat JSON array from all models that produced output
    python3 << 'PYEOF'
import json, os, sys
from pathlib import Path


def _model_family(mt):
    """Classify model_type string into a family for grouping in recommendations."""
    if mt == "res34":
        return "res34"
    if mt.startswith("res34_cbam"):
        return "res34_cbam"
    if mt.startswith("efficientnet"):
        return "efficientnet"
    return "other"


result_dir = os.environ.get("RESULT_DIR", "comparison_results")
result_dir = Path(result_dir)
records = []

for model_dir in sorted(result_dir.iterdir()):
    if not model_dir.is_dir():
        continue
    rec_file = model_dir / "experiment_records.json"
    if not rec_file.is_file():
        continue

    with open(rec_file, "r") as f:
        model_records = json.load(f)

    for rec in model_records:
        rec["_model_type"] = model_dir.name
        rec["_model_family"] = _model_family(rec.get("model_type", model_dir.name))
        records.append(rec)

if not records:
    print("ERROR: No experiment records found.", file=sys.stderr)
    sys.exit(1)

# Save composite records
out_path = result_dir / "records.json"
with open(out_path, "w") as f:
    json.dump(records, f, indent=2, ensure_ascii=False)
print(f"Collected {len(records)} records → {out_path}")

# ---- Build markdown comparison table ----
# Sort by mean_iou descending
records.sort(key=lambda r: r.get("mean_iou", -1), reverse=True)

lines = []
lines.append("")
lines.append("## Full Comparison Table")
lines.append("")
lines.append("| # | Model | Inference | mIoU | Bld IoU | Bg IoU | FP | FN | Time(s) |")
lines.append("|---|-------|-----------|------|---------|--------|----|----|---------|")

best_idx = {}
for i, rec in enumerate(records, 1):
    miou  = rec.get("mean_iou", float("nan"))
    build = rec.get("iou_building", float("nan"))
    bg    = rec.get("iou_background", float("nan"))
    fp    = rec.get("fp_pixels", -1)
    fn    = rec.get("fn_pixels", -1)
    t     = rec.get("inference_time_s", -1)

    model_label = rec["_model_type"]
    exp_id      = rec.get("exp_id", "?")
    lines.append(
        f"| {i} | {model_label} | {exp_id} "
        f"| {miou:.4f} | {build:.4f} | {bg:.4f} "
        f"| {fp} | {fn} | {t:.0f} |"
    )

    # Track best per model family
    family = rec.get("_model_family", "other")
    prev = best_idx.get(family)
    if prev is None or miou > records[prev].get("mean_iou", -1):
        best_idx[family] = i - 1

table_text = "\n".join(lines)

# ---- Recommendations ----
reco_lines = []
reco_lines.append("")
reco_lines.append("## Recommendations")
reco_lines.append("")

# 1. Overall best
best = records[0]
reco_lines.append(f"### Best overall: `{best['_model_type']}` + `{best.get('exp_id','?')}`")
reco_lines.append(f"- mIoU = **{best['mean_iou']:.4f}**, Building IoU = **{best['iou_building']:.4f}**")
reco_lines.append(f"- FP = {best['fp_pixels']}, FN = {best['fn_pixels']}")
reco_lines.append("")

# 2. Best per family
reco_lines.append("### Best per model family")
family_names = {
    "res34": "ResNet34 baseline (no attention, no EfficientNet)",
    "res34_cbam": "ResNet34 + CBAM attention",
    "efficientnet": "EfficientNet encoder (no CBAM)",
}
for family in ["res34", "res34_cbam", "efficientnet"]:
    if family in best_idx:
        r = records[best_idx[family]]
        reco_lines.append(
            f"- **{family_names.get(family, family)}**: "
            f"`{r['_model_type']}` + `{r.get('exp_id','?')}` "
            f"→ mIoU={r['mean_iou']:.4f}, Bld={r['iou_building']:.4f}"
        )

reco_lines.append("")

# 3. Best inference strategy (averaged across all models)
reco_lines.append("### Best inference strategy (aggregate across models)")
inf_scores = {}
for rec in records:
    eid = rec.get("exp_id", "?")
    inf_scores.setdefault(eid, []).append(rec.get("mean_iou", 0))
for eid in sorted(inf_scores, key=lambda k: sum(inf_scores[k])/max(len(inf_scores[k]),1), reverse=True):
    avg = sum(inf_scores[eid]) / max(len(inf_scores[eid]), 1)
    reco_lines.append(f"- `{eid}`: avg mIoU = {avg:.4f} over {len(inf_scores[eid])} models")

reco_lines.append("")

# 4. Cost-effectiveness (IoU vs time)
reco_lines.append("### Cost-effectiveness (mIoU per inference-second)")
cost_rank = sorted(records, key=lambda r: r.get("mean_iou", 0) / max(r.get("inference_time_s", 1), 1), reverse=True)
for i, rec in enumerate(cost_rank[:5], 1):
    miou = rec.get("mean_iou", 0)
    t = max(rec.get("inference_time_s", 1), 1)
    reco_lines.append(
        f"{i}. `{rec['_model_type']}` + `{rec.get('exp_id','?')}` "
        f"→ mIoU={miou:.4f}, time={t:.0f}s, mIoU/s={miou/t:.5f}"
    )

reco_lines.append("")
reco_lines.append("---")
reco_lines.append(f"*Generated: {__import__('datetime').datetime.now().isoformat()}*")
reco_text = "\n".join(reco_lines)

# Write table
table_path = result_dir / "table.txt"
with open(table_path, "w") as f:
    f.write(table_text)
    f.write("\n")
    f.write(reco_text)

print(f"Comparison table → {table_path}")
print(table_text)
print(reco_text)
PYEOF
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    echo ""
    echo "================================================================================"
    echo "  UNet Remote Sensing Building Extraction — Comprehensive Comparison"
    echo "  Started: $(timestamp)"
    echo "  Mode:    ${MODE}"
    echo "  Skip train: ${SKIP_TRAIN}"
    echo "================================================================================"

    check_prerequisites

    select_models
    log_info "Models to test (${#MODELS[@]}):"
    for m in "${MODELS[@]}"; do
        echo "  - $m"
    done
    echo ""

    ensure_dataset

    # Prepare result directory
    mkdir -p "$RESULT_DIR"

    # ---- Phase 1: Train ----
    if [[ "$SKIP_TRAIN" == true ]]; then
        log_info "Skipping training phase (--skip-train)"
    else
        local train_ok=()
        local train_fail=()
        for model_type in "${MODELS[@]}"; do
            if train_model "$model_type"; then
                train_ok+=("$model_type")
            else
                train_fail+=("$model_type")
            fi
        done

        log_section "Training summary"
        log_info "Succeeded (${#train_ok[@]}): ${train_ok[*]:-none}"
        if [[ ${#train_fail[@]} -gt 0 ]]; then
            log_warn "Failed (${#train_fail[@]}): ${train_fail[*]}"
        fi
    fi

    # ---- Phase 2: Inference ----
    local infer_ok=()
    local infer_fail=()
    for model_type in "${MODELS[@]}"; do
        if run_inference_suite "$model_type"; then
            infer_ok+=("$model_type")
        else
            infer_fail+=("$model_type")
        fi
    done

    log_section "Inference summary"
    log_info "Succeeded (${#infer_ok[@]}): ${infer_ok[*]:-none}"
    if [[ ${#infer_fail[@]} -gt 0 ]]; then
        log_warn "Failed (${#infer_fail[@]}): ${infer_fail[*]}"
    fi

    # ---- Phase 3: Results ----
    RESULT_DIR="$RESULT_DIR" collect_results

    echo ""
    log_section "Done"
    log_info "Results: ${RESULT_DIR}/"
    log_info "  records.json   — all experiment records (structured)"
    log_info "  table.txt      — formatted comparison table + recommendations"
    log_info "  {model}/       — per-model prediction outputs"
}

main "$@"
