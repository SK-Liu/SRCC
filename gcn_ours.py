from __future__ import annotations

from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn
import torchvision


class DynamicGraphConvolution(nn.Module):
    """Attention-driven dynamic graph convolution over class nodes.

    Args:
        in_features: Input channel dimension.
        out_features: Output channel dimension.
        num_nodes: Number of class nodes.
        temperature: Temperature used in row-wise softmax for dynamic adjacency.
        self_loop_alpha: Weight of the identity self-loop added to dynamic adjacency.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_nodes: int,
        temperature: float = 0.3,
        self_loop_alpha: float = 0.1,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.self_loop_alpha = self_loop_alpha

        self.static_adj = nn.Sequential(
            nn.Conv1d(num_nodes, num_nodes, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.static_weight = nn.Sequential(
            nn.Conv1d(in_features, out_features, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv_global = nn.Conv1d(in_features, in_features, kernel_size=1)
        self.bn_global = nn.BatchNorm1d(in_features)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.conv_create_co_mat = nn.Conv1d(in_features * 2, num_nodes, kernel_size=1)
        self.dynamic_weight = nn.Conv1d(in_features, out_features, kernel_size=1)

    def forward_static_gcn(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the static branch.

        Args:
            x: Class-level features with shape ``[B, C, N]``.
        """
        out = self.static_adj(x.transpose(1, 2))
        out = self.static_weight(out.transpose(1, 2))
        return out

    def forward_construct_dynamic_graph(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct a sample-specific dynamic adjacency matrix.

        Args:
            x: Class-level features with shape ``[B, C, N]``.

        Returns:
            dynamic_adj: Normalized dynamic adjacency with shape ``[B, N, N]``.
            dynamic_adj_logits: Raw centered logits with shape ``[B, N, N]``.
        """
        x_global = self.gap(x)
        x_global = self.conv_global(x_global)
        x_global = self.bn_global(x_global)
        x_global = self.relu(x_global)
        x_global = x_global.expand(-1, -1, x.size(2))

        graph_input = torch.cat((x_global, x), dim=1)
        dynamic_adj_logits = self.conv_create_co_mat(graph_input)
        dynamic_adj_logits = dynamic_adj_logits - dynamic_adj_logits.mean(dim=-1, keepdim=True)

        dynamic_adj = torch.softmax(dynamic_adj_logits / self.temperature, dim=-1)
        eye = torch.eye(
            dynamic_adj.size(-1),
            device=dynamic_adj.device,
            dtype=dynamic_adj.dtype,
        ).unsqueeze(0)
        dynamic_adj = (1.0 - self.self_loop_alpha) * dynamic_adj + self.self_loop_alpha * eye
        return dynamic_adj, dynamic_adj_logits

    def forward_dynamic_gcn(self, x: torch.Tensor, dynamic_adj: torch.Tensor) -> torch.Tensor:
        out = torch.matmul(x, dynamic_adj)
        out = self.relu(out)
        out = self.dynamic_weight(out)
        out = self.relu(out)
        return out

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        static_out = self.forward_static_gcn(x)
        x = x + static_out
        dynamic_adj, dynamic_adj_logits = self.forward_construct_dynamic_graph(x)
        out = self.forward_dynamic_gcn(x, dynamic_adj)
        return {
            "dynamic_adj": dynamic_adj,
            "dynamic_adj_logits": dynamic_adj_logits,
            "out": out,
        }


class SemanticTokenizer(nn.Module):
    """Aggregate class-level representations into image-level semantic tokens."""

    def __init__(self, dim: int = 1024, num_tokens: int = 5) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_tokens, dim))
        self.key_proj = nn.Linear(dim, dim)
        self.val_proj = nn.Linear(dim, dim)

    def forward(self, class_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create semantic tokens from class-level features.

        Args:
            class_feats: Tensor with shape ``[B, C, N]``.

        Returns:
            tokens: Tensor with shape ``[B, C, K]``.
            attn: Attention weights with shape ``[B, K, N]``.
        """
        batch_size, channels, _ = class_feats.shape
        feats = class_feats.transpose(1, 2)  # [B, N, C]

        keys = self.key_proj(feats)
        values = self.val_proj(feats)
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)

        attn = torch.matmul(queries, keys.transpose(1, 2)) / (channels ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        tokens = torch.matmul(attn, values).transpose(1, 2)
        return tokens, attn


class ADD_GCN(nn.Module):
    """ResNet-50 based attention-driven dynamic GCN for multi-label classification."""

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        num_tokens: int = 5,
        feature_dim: int = 1024,
        dropout_p: float = 0.4,
        dgcn_temperature: float = 0.3,
        self_loop_alpha: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.dropout_p = dropout_p

        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        self.fc = nn.Conv2d(backbone.fc.in_features, num_classes, kernel_size=1, bias=False)
        self.conv_transform = nn.Conv2d(backbone.fc.in_features, feature_dim, kernel_size=1)
        self.gcn = DynamicGraphConvolution(
            feature_dim,
            feature_dim,
            num_classes,
            temperature=dgcn_temperature,
            self_loop_alpha=self_loop_alpha,
        )

        self.semantic_tokenizer = SemanticTokenizer(dim=feature_dim, num_tokens=num_tokens)
        self.mask_mat = nn.Parameter(torch.eye(num_classes), requires_grad=True)
        self.last_linear = nn.Conv1d(feature_dim, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_p)

    def forward_feature(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

    def forward_classification_branch(self, x: torch.Tensor) -> torch.Tensor:
        logits_map = self.fc(x)
        logits = logits_map.flatten(2).topk(1, dim=-1)[0].mean(dim=-1)
        return logits

    def forward_attention_branch(
        self,
        x: torch.Tensor,
        apply_dropout: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_maps = torch.sigmoid(self.fc(x))
        class_masks = attn_maps.flatten(2).transpose(1, 2)  # [B, HW, N]

        features = self.conv_transform(x)
        if apply_dropout:
            features = self.dropout(features)

        features = features.flatten(2)  # [B, C, HW]
        class_features = torch.matmul(features, class_masks)  # [B, C, N]
        return class_features, attn_maps

    def forward(
        self,
        x: torch.Tensor,
        drop: bool = False,
        return_attn: bool = False,
        return_graph: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        features = self.forward_feature(x)
        logits_cls = self.forward_classification_branch(features)

        class_features, attn_maps = self.forward_attention_branch(features, apply_dropout=drop)
        gcn_dict = self.gcn(class_features)
        dynamic_adj = gcn_dict["dynamic_adj"]
        dynamic_adj_logits = gcn_dict["dynamic_adj_logits"]
        gcn_features = gcn_dict["out"]

        fused_features = class_features + gcn_features
        semantic_tokens, token_attn = self.semantic_tokenizer(class_features)

        logits_gcn = self.last_linear(fused_features)
        logits_gcn = (logits_gcn * self.mask_mat.detach()).sum(dim=-1)
        logits = (logits_cls + logits_gcn) / 2.0

        if return_attn and return_graph:
            return {
                "logits": logits,
                "dynamic_adj": dynamic_adj,
                "dynamic_adj_logits": dynamic_adj_logits,
                "out": gcn_features,
                "v": class_features,
                "attn_maps": attn_maps,
                "semantic_tokens": semantic_tokens,
                "token_attn": token_attn,
            }

        if return_attn:
            return logits, dynamic_adj, class_features, attn_maps, semantic_tokens, token_attn

        return logits, dynamic_adj, semantic_tokens

    def get_config_optim(self, lr: float, lrp: float) -> list[dict[str, Any]]:
        backbone_params = set(map(id, self.features.parameters()))
        task_params = [p for p in self.parameters() if id(p) not in backbone_params]
        return [
            {"params": self.features.parameters(), "lr": lr * lrp},
            {"params": task_params, "lr": lr},
        ]


def _get_arg(args: Any, name: str, default: Any) -> Any:
    return getattr(args, name, default) if args is not None else default


def build_resnet50(pretrained: bool = True) -> nn.Module:
    """Build ResNet-50 with compatibility across torchvision versions."""
    try:
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        return torchvision.models.resnet50(weights=weights)
    except AttributeError:
        return torchvision.models.resnet50(pretrained=pretrained)


def get_model(num_classes: int, args: Any = None) -> ADD_GCN:
    """Factory function used by the training and evaluation scripts."""
    backbone = build_resnet50(pretrained=_get_arg(args, "pretrained", True))
    return ADD_GCN(
        backbone=backbone,
        num_classes=num_classes,
        num_tokens=_get_arg(args, "num_fea", 5),
        feature_dim=_get_arg(args, "feature_dim", 1024),
        dropout_p=_get_arg(args, "dropout_p", 0.4),
        dgcn_temperature=_get_arg(args, "dgcn_temperature", 0.3),
        self_loop_alpha=_get_arg(args, "self_loop_alpha", 0.1),
    )
