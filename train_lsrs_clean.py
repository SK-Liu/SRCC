"""Training script for noisy multi-label CXR classification.
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.cx14_dataloader_cut import construct_cx14_cut
from data.cx14_pdc_dataloader_cut import construct_cx14_pdc_cut
from data.cxp_dataloader_cut import construct_cxp_cut
from data.openi_dataloader_cut import construct_openi_cut
from data.padchest_dataloader_cut import construct_pc_cut
from metrics import _test_google_nih, _test_google_nih14, _test_openi, _test_pc
from eval_lsrs_clean import load_checkpoint
from utils.NSD import NSD
from utils.loss_gls import focal_loss_tail, loss_gls_logits
from utils.mid_loss import MID_LOSS, AsymmetricLossClassWiseSmooth
from utils.utils import load_word_vec

warnings.filterwarnings("ignore")

NIH_CLASS_COUNTS = torch.tensor(
    [7522, 1642, 7896, 13066, 3660, 4405, 812, 2531, 1284, 1365, 1202, 2092, 140, 50489],
    dtype=torch.float32,
)

CXP_CLASS_COUNTS = torch.tensor(
    [28076, 54086, 2971, 30646, 27506, 67344, 7415, 29220],
    dtype=torch.float32,
)

NIH_CLASS_SMOOTH = [
    0.05, 0.02, 0.02, 0.10, 0.02, 0.05, 0.05,
    0.02, 0.02, 0.08, 0.03, 0.05, 0.01, 0.03,
]

CXP_CLASS_SMOOTH = [0.03] * 8

NIH_TAIL_IDX = [12, 6, 10, 8]
NIH_MEDIUM_IDX = [9, 1, 11, 7, 4]
NIH_HEAD_IDX = [5, 0, 2, 3, 13]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSRS/SRCC on noisy multi-label datasets")

    parser.add_argument("--train_data", default="NIH", choices=["NIH", "CXP"], type=str)
    parser.add_argument("--root_dir", default="dataset", type=str)
    parser.add_argument("--num_classes", "--c", dest="num_classes", default=14, type=int)
    parser.add_argument("--resize", default=512, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--epochs", "--n_epoch", dest="epochs", default=40, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-8, type=float)
    parser.add_argument("--seed", default=3407, type=int)
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--output_dir", default="output/NIH", type=str)

    parser.add_argument("--warmup_epochs", "--epoch_update_start", dest="warmup_epochs", default=30, type=int)
    parser.add_argument("--pseudo_update_every", default=15, type=int)
    parser.add_argument("--nsd_topk", default=10, type=int)
    parser.add_argument("--compute_nsd", action="store_true")
    parser.add_argument("--knn_path", default="knn.npy", type=str)

    parser.add_argument("--bert_name", default="bluebert", type=str)
    parser.add_argument("--embed_len", default=1024, type=int)
    parser.add_argument("--num_fea", default=5, type=int)
    parser.add_argument("--beta_l", default=1.0, type=float)
    parser.add_argument("--smooth_rate", default=0.4, type=float)
    parser.add_argument("--add_noise", action="store_true")
    parser.add_argument("--noise_ratio", default=0.4, type=float)
    parser.add_argument("--noise_p", default=0.6, type=float)
    parser.add_argument("--trim_data", action="store_true", default=True)

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_file", default="", type=str)

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_train_constructor(args: argparse.Namespace):
    if args.add_noise:
        return construct_cx14_pdc_cut
    if args.train_data == "NIH":
        return construct_cx14_cut
    if args.train_data == "CXP":
        return construct_cxp_cut
    raise ValueError(f"Unsupported training dataset: {args.train_data}")


def get_class_counts(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    if args.train_data == "CXP":
        return CXP_CLASS_COUNTS.to(device)
    return NIH_CLASS_COUNTS.to(device)


def get_class_smooth(args: argparse.Namespace) -> List[float]:
    if args.train_data == "CXP":
        return CXP_CLASS_SMOOTH
    return NIH_CLASS_SMOOTH


def build_criterion(args: argparse.Namespace, device: torch.device) -> nn.Module:
    return AsymmetricLossClassWiseSmooth(
        gamma_neg=3,
        gamma_pos=0,
        clip=0.05,
        class_smooth=get_class_smooth(args),
    ).to(device)


def update_ema_model(model: nn.Module, ema_model: nn.Module, momentum: float = 0.99) -> None:
    with torch.no_grad():
        for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(momentum).add_(model_param.data, alpha=1.0 - momentum)


def forward_model(model: nn.Module, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    output = model(images, drop=True, return_attn=True, return_graph=True)
    return output["logits"], output["semantic_tokens"]


def warmup_step(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    mid_criterion: nn.Module,
    mid_weight: float = 0.05,
    bce_weight: float = 0.8,
) -> float:
    model.train()
    logits, semantic_tokens = forward_model(model, images)
    loss_cls = criterion(logits, labels)
    loss_mid = mid_criterion(semantic_tokens, labels)
    loss = bce_weight * loss_cls + mid_weight * loss_mid

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def build_class_group_masks(num_classes: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tail_mask = torch.zeros(num_classes, device=device, dtype=torch.bool)
    medium_mask = torch.zeros(num_classes, device=device, dtype=torch.bool)
    head_mask = torch.zeros(num_classes, device=device, dtype=torch.bool)

    tail_idx = [i for i in NIH_TAIL_IDX if i < num_classes]
    medium_idx = [i for i in NIH_MEDIUM_IDX if i < num_classes]
    head_idx = [i for i in NIH_HEAD_IDX if i < num_classes]

    tail_mask[tail_idx] = True
    medium_mask[medium_idx] = True
    head_mask[head_idx] = True
    return head_mask, medium_mask, tail_mask


def generate_corrected_labels(
    pred_prob: torch.Tensor,
    labels: torch.Tensor,
    knn_labels: torch.Tensor,
    class_counts: torch.Tensor,
    alpha_model: float = 0.1,
    beta_nsd: float = 0.5,
) -> torch.Tensor:
    batch_size, num_classes = labels.shape
    device = labels.device

    tail_threshold = 0.33 * class_counts.max().item()
    tail_mask = class_counts.to(device) < tail_threshold

    pos_th = torch.full((num_classes,), 0.8, device=device)
    neg_th = torch.full((num_classes,), 0.2, device=device)
    pos_th[tail_mask] = 0.75
    neg_th[tail_mask] = 0.25

    corrected_labels = torch.zeros(batch_size, num_classes, device=device)
    fused = alpha_model * pred_prob + beta_nsd * knn_labels + (1.0 - alpha_model - beta_nsd) * labels

    for i in range(batch_size):
        y_corr = labels[i].clone()
        y_corr[fused[i] >= pos_th] = fused[i][fused[i] >= pos_th]
        y_corr[fused[i] <= neg_th] = fused[i][fused[i] <= neg_th]

        if y_corr.sum() == 0:
            if labels[i].sum() > 0:
                y_corr = labels[i].clone()
            else:
                y_corr[torch.argmax(fused[i])] = 1.0

        corrected_labels[i] = y_corr

    return corrected_labels


def correction_step(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    previous_labels: torch.Tensor,
    knn_labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    mid_criterion: nn.Module,
    class_counts: torch.Tensor,
    epoch: int,
    args: argparse.Namespace,
) -> Tuple[float, torch.Tensor]:
    model.train()
    logits, semantic_tokens = forward_model(model, images)
    pred_prob = torch.sigmoid(logits)

    corrected_labels = generate_corrected_labels(
        pred_prob=pred_prob,
        labels=previous_labels,
        knn_labels=knn_labels,
        class_counts=class_counts,
    )

    _, medium_mask, tail_mask = build_class_group_masks(args.num_classes, images.device)
    head_mask = ~(medium_mask | tail_mask)

    fused = 0.1 * pred_prob + 0.5 * knn_labels + 0.4 * corrected_labels
    confidence = torch.abs(fused - 0.5)

    lambda_base = torch.zeros(args.num_classes, device=images.device)
    lambda_base[head_mask] = 0.8
    lambda_base[medium_mask] = 0.8
    lambda_base[tail_mask] = 1.0
    lambda_ic = lambda_base.unsqueeze(0).expand_as(corrected_labels)
    lambda_ic = lambda_ic * (1.0 - confidence.clamp(0, 0.5) / 0.5)

    loss_cls = criterion(logits, corrected_labels)
    loss_gls = loss_gls_logits(
        logits,
        y_tilde=corrected_labels,
        y_knn=knn_labels,
        smooth_rate=lambda_ic,
        beta_l=args.beta_l,
    )
    loss_focal = focal_loss_tail(pred_prob, corrected_labels, tail_mask=tail_mask)
    loss_mid = mid_criterion(semantic_tokens, corrected_labels)

    if epoch < 10:
        weights = dict(cls=1.0, gls=0.0, focal=0.0, mid=0.0)
    elif epoch < 15:
        weights = dict(cls=0.5, gls=0.4, focal=0.0, mid=0.1)
    else:
        weights = dict(cls=0.4, gls=0.4, focal=0.1, mid=0.1)

    loss = (
        weights["cls"] * loss_cls
        + weights["gls"] * loss_gls
        + weights["focal"] * loss_focal
        + weights["mid"] * loss_mid
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    return float(loss.item()), corrected_labels.detach()


def train_warmup_epoch(
    model: nn.Module,
    ema_model: nn.Module,
    train_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    mid_criterion: nn.Module,
    device: torch.device,
) -> float:
    total_loss = 0.0
    total_steps = 0

    for images, labels, _ in train_loader:
        images = images.to(device).float()
        labels = labels.to(device).float()
        loss = warmup_step(model, images, labels, optimizer, criterion, mid_criterion)
        update_ema_model(model, ema_model)
        total_loss += loss
        total_steps += 1

    return total_loss / max(total_steps, 1)


def train_correction_epoch(
    model: nn.Module,
    ema_model: nn.Module,
    train_loader: Iterable,
    corrected_labels_cache: List[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    mid_criterion: nn.Module,
    class_counts: torch.Tensor,
    epoch: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[float, List[torch.Tensor]]:
    total_loss = 0.0
    total_steps = 0
    new_cache = []

    for batch_idx, (images, labels, _, knn_labels) in enumerate(train_loader):
        images = images.to(device).float()
        labels = labels.to(device).float()
        knn_labels = knn_labels.to(device).float() / float(args.nsd_topk)

        if corrected_labels_cache:
            previous_labels = corrected_labels_cache[batch_idx].to(device).float()
        else:
            previous_labels = labels

        loss, batch_corrected = correction_step(
            model=model,
            images=images,
            labels=labels,
            previous_labels=previous_labels,
            knn_labels=knn_labels,
            optimizer=optimizer,
            criterion=criterion,
            mid_criterion=mid_criterion,
            class_counts=class_counts,
            epoch=epoch,
            args=args,
        )
        update_ema_model(model, ema_model)
        new_cache.append(batch_corrected.cpu())
        total_loss += loss
        total_steps += 1

    return total_loss / max(total_steps, 1), new_cache


def evaluate_all(epoch: int, model: nn.Module, args: argparse.Namespace) -> Dict[str, float]:
    test_loader_openi = construct_openi_cut(args, mode="test")
    test_loader_pc = construct_pc_cut(args, mode="test")
    test_loader_google_nih, _ = construct_cx14_cut(args, mode="test", file_name="clean_test")
    test_loader_google_nih14, _ = construct_cx14_cut(args, mode="test", file_name="clean_test14")

    mean_auc_openi = _test_openi(epoch, model, test_loader_openi, args.num_classes)
    mean_auc_pc = _test_pc(epoch, model, test_loader_pc, args.num_classes)
    mean_auc_pneux, auc_mass_nodule = _test_google_nih(epoch, model, test_loader_google_nih, args.num_classes)
    mean_auc_nih14, mean_f1 = _test_google_nih14(epoch, model, test_loader_google_nih14, args.num_classes)

    return {
        "openi": float(mean_auc_openi),
        "pc": float(mean_auc_pc),
        "pneux": float(mean_auc_pneux),
        "mass_nodule": float(auc_mass_nodule),
        "nih14": float(mean_auc_nih14),
        "f1": float(mean_f1),
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    output_dir: str,
) -> Path:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ckpt_name = (
        f"epoch{epoch:03d}_openi{metrics['openi']:.4f}_pc{metrics['pc']:.4f}_"
        f"pneux{metrics['pneux']:.4f}_mass{metrics['mass_nodule']:.4f}.pth"
    )
    ckpt_path = Path(output_dir) / ckpt_name
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
        },
        ckpt_path,
    )
    return ckpt_path


def append_metrics(log_path: Path, epoch: int, metrics: Dict[str, float]) -> None:
    if not log_path.exists():
        log_path.write_text("epoch,openi,pc,pneux,mass_nodule,nih14,f1\n")
    with log_path.open("a") as f:
        f.write(
            f"{epoch},{metrics['openi']:.6f},{metrics['pc']:.6f},{metrics['pneux']:.6f},"
            f"{metrics['mass_nodule']:.6f},{metrics['nih14']:.6f},{metrics['f1']:.6f}\n"
        )


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.c = args.num_classes

    construct_func = get_train_constructor(args)
    print("Loading datasets...")
    train_loader, data_num = construct_func(args, mode="train", file_name="train", stage="MID")
    static_train_loader, _ = construct_func(args, mode="test", file_name="train", stage="STATIC")
    train_loader_cls, _ = construct_func(args, mode="train", file_name="train", stage="CLS")

    print("Building model...")
    import gcn_ours

    wordvec_array = load_word_vec(args)
    model = gcn_ours.get_model(num_classes=args.num_classes, args=args)
    model = nn.DataParallel(model).to(device)

    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad_(False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)

    start_epoch = 0
    if args.resume:
        if not args.resume_file:
            raise ValueError("--resume_file must be specified when --resume is enabled.")
        model, start_epoch, optimizer, _ = load_checkpoint(model, optimizer, args.resume_file)
        ema_model = copy.deepcopy(model)
        ema_model.eval()

    criterion = build_criterion(args, device)
    mid_criterion = MID_LOSS(beta=0.3, wordvec_array=wordvec_array, args=args).to(device)
    class_counts = get_class_counts(args, device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "metrics.csv"

    best_score = -1.0
    corrected_labels_cache: List[torch.Tensor] = []

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        if epoch < args.warmup_epochs:
            train_loss = train_warmup_epoch(
                model=model,
                ema_model=ema_model,
                train_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                mid_criterion=mid_criterion,
                device=device,
            )
        else:
            should_update_nsd = (
                epoch == args.warmup_epochs
                or (epoch - args.warmup_epochs) % args.pseudo_update_every == 0
            )
            if should_update_nsd and args.compute_nsd:
                torch.cuda.empty_cache()
                nsd_runner = NSD(args, model, wordvec_array, static_train_loader)
                nsd_runner.run()

            if Path(args.knn_path).exists():
                train_loader_cls.dataset.knn = np.load(args.knn_path)
            elif args.compute_nsd:
                raise FileNotFoundError(f"KNN file not found: {args.knn_path}")

            train_loss, corrected_labels_cache = train_correction_epoch(
                model=model,
                ema_model=ema_model,
                train_loader=train_loader_cls,
                corrected_labels_cache=corrected_labels_cache,
                optimizer=optimizer,
                criterion=criterion,
                mid_criterion=mid_criterion,
                class_counts=class_counts,
                epoch=epoch,
                args=args,
                device=device,
            )

        print(f"Train loss: {train_loss:.6f}")
        metrics = evaluate_all(epoch, model, args)
        append_metrics(log_path, epoch, metrics)

        current_score = max(metrics["openi"], metrics["pc"], metrics["pneux"], metrics["mass_nodule"])
        if current_score > best_score:
            best_score = current_score
            ckpt_path = save_checkpoint(model, optimizer, epoch, metrics, args.output_dir)
            print(f"Saved checkpoint: {ckpt_path}")

        scheduler.step()
        lr = sorted({group["lr"] for group in optimizer.param_groups})
        print(f"Learning rate: {lr}")


if __name__ == "__main__":
    main()
