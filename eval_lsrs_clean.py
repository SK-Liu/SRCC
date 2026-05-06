"""Evaluation script for LSRS/SRCC-style CXR multi-label models.

Example:
    python eval_lsrs_clean.py \
        --checkpoint output/NIH/best_model.pth \
        --eval_sets nih_clean nih_clean14 openi padchest \
        --num_classes 14 \
        --train_data NIH \
        --resize 512 \
        --gpu 0
"""

import argparse
import os
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn

import gcn_ours
from data.cx14_dataloader_cut import construct_cx14_cut
from data.openi_dataloader_cut import construct_openi_cut
from data.padchest_dataloader_cut import construct_pc_cut
from metrics import _test_google_nih, _test_google_nih14, _test_openi, _test_pc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LSRS/SRCC checkpoints")

    # Model / data settings required by the model and dataloaders.
    parser.add_argument("--num_classes", "--c", dest="c", type=int, default=14)
    parser.add_argument("--train_data", type=str, default="NIH", choices=["NIH", "CXP"])
    parser.add_argument("--root_dir", type=str, default="dataset")
    parser.add_argument("--resize", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_fea", type=int, default=5)
    parser.add_argument("--trim_data", action="store_true")

    # Runtime settings.
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="", help="Path to a single checkpoint.")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="",
        help="Evaluate all .pth/.pt checkpoints in this directory when --checkpoint is not given.",
    )
    parser.add_argument(
        "--eval_sets",
        nargs="+",
        default=["nih_clean"],
        choices=["nih_clean", "nih_clean14", "openi", "padchest"],
        help="Test sets to evaluate.",
    )
    parser.add_argument("--strict", action="store_true", help="Use strict checkpoint loading.")

    # Compatibility flags used by existing dataloaders/model constructors.
    parser.add_argument("--add_noise", action="store_true")
    parser.add_argument("--noise_rate", type=float, default=0.4)
    parser.add_argument("--noise_type", type=str, default="symmetric")
    parser.add_argument("--dataset", type=str, default="CXP")

    return parser.parse_args()


def build_model(args: argparse.Namespace) -> nn.Module:
    model = gcn_ours.get_model(num_classes=args.c, args=args)
    model = nn.DataParallel(model).cuda()
    model.eval()
    return model


def load_checkpoint(model: nn.Module, checkpoint_path: str, strict: bool = False) -> Tuple[nn.Module, int, Dict]:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[CKPT] Loading: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")

    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict, strict=strict)

    epoch = checkpoint.get("epoch", 0) if isinstance(checkpoint, dict) else 0
    meta = checkpoint if isinstance(checkpoint, dict) else {}
    print(f"[CKPT] Loaded checkpoint from epoch {epoch}")
    return model, epoch, meta


def build_test_loaders(args: argparse.Namespace) -> Dict[str, object]:
    loaders = {}
    if "openi" in args.eval_sets:
        loaders["openi"] = construct_openi_cut(args, mode="test")
    if "padchest" in args.eval_sets:
        loaders["padchest"] = construct_pc_cut(args, mode="test")
    if "nih_clean" in args.eval_sets:
        loaders["nih_clean"], _ = construct_cx14_cut(args, mode="test", file_name="clean_test")
    if "nih_clean14" in args.eval_sets:
        loaders["nih_clean14"], _ = construct_cx14_cut(args, mode="test", file_name="clean_test14")
    return loaders


def evaluate_checkpoint(model: nn.Module, epoch: int, loaders: Dict[str, object], args: argparse.Namespace) -> Dict[str, float]:
    results = {}
    with torch.no_grad():
        if "openi" in loaders:
            results["openi_auc"] = _test_openi(epoch, model, loaders["openi"], args.c)
        if "padchest" in loaders:
            results["padchest_auc"] = _test_pc(epoch, model, loaders["padchest"], args.c)
        if "nih_clean" in loaders:
            pneu_auc, mass_nodule_auc = _test_google_nih(epoch, model, loaders["nih_clean"], args.c)
            results["nih_clean_pneumothorax_auc"] = pneu_auc
            results["nih_clean_mass_nodule_auc"] = mass_nodule_auc
        if "nih_clean14" in loaders:
            nih14_auc, nih14_f1 = _test_google_nih14(epoch, model, loaders["nih_clean14"], args.c)
            results["nih_clean14_auc"] = nih14_auc
            results["nih_clean14_f1"] = nih14_f1
    return results


def checkpoint_paths(args: argparse.Namespace) -> Iterable[str]:
    if args.checkpoint:
        return [args.checkpoint]
    if not args.checkpoint_dir:
        raise ValueError("Please provide either --checkpoint or --checkpoint_dir.")

    files = sorted(os.listdir(args.checkpoint_dir))
    return [
        os.path.join(args.checkpoint_dir, name)
        for name in files
        if name.endswith((".pth", ".pt"))
    ]


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the current model/dataloader setup.")

    loaders = build_test_loaders(args)
    if not loaders:
        raise RuntimeError("No test loader was constructed. Check --eval_sets.")

    for ckpt_path in checkpoint_paths(args):
        model = build_model(args)
        model, epoch, meta = load_checkpoint(model, ckpt_path, strict=args.strict)
        results = evaluate_checkpoint(model, epoch, loaders, args)

        print("\n===== Evaluation Results =====")
        print(f"Checkpoint: {ckpt_path}")
        if meta.get("mean_auc_openi") is not None:
            print(f"Stored mean_auc_openi: {meta.get('mean_auc_openi')}")
        for key, value in results.items():
            print(f"{key}: {value:.6f}")
        print("==============================\n")


if __name__ == "__main__":
    main()
