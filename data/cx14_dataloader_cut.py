import os

import numpy as np
from numpy.lib.type_check import _imag_dispatcher
import pandas as pd
import torch
from PIL import Image
from skimage import io
from sklearn.preprocessing import MultiLabelBinarizer
from torch.nn.modules import transformer
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
#import wandb
import csv
from loguru import logger

# from utils.gcloud import download_chestxray_unzip

Labels = {
    "Atelectasis": 0,
    "Cardiomegaly": 1,
    "Effusion": 2,
    "Infiltration": 3,
    "Mass": 4,
    "Nodule": 5,
    "Pneumonia": 6,
    "Pneumothorax": 7,
    "Consolidation": 8,
    "Edema": 9,
    "Emphysema": 10,
    "Fibrosis": 11,
    "Pleural_Thickening": 12,
    "Hernia": 13,
    "No Finding": 14,
}
mlb = MultiLabelBinarizer(classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])


class ChestDataset(Dataset):
    """
    Del sample when delete label
    """

    def __init__(
        self, root_dir, transforms, mode, file_name=None, stage="CLS", args=None
    ) -> None:
        self.stage = stage
        self.transform = transforms
        self.root_dir = os.path.join(root_dir, "chestxray")
        self.mode = mode
        self.pathologies = np.array(list(Labels))
        self.drop_list = []
        df_path = os.path.join(self.root_dir, "Data_Entry_2017_v2020.csv")
        self.csv = pd.read_csv(df_path, index_col=0)
        gt = self.csv.to_dict()["Finding Labels"]

        if file_name == "test":
            img_list = os.path.join(self.root_dir, "test_list.txt")
        elif file_name == "clean_test":
            self.clean_root = os.path.join(
                root_dir, "NIH_GOOGLE", "four_findings_expert_labels"
            )
            img_list = os.path.join(self.clean_root, "four_findings_expert_labels_test_labels.csv")
        elif file_name == "clean_test14":
            self.clean_root = os.path.join(
                root_dir, "NIH_GOOGLE", "all_findings_expert_labels"
            )
            img_list = os.path.join(self.clean_root, "all_findings_expert_labels_test_labels.csv")
        elif file_name == "train":
            img_list = os.path.join(self.root_dir, "train_val_list.txt")
        else:
            raise ValueError(f"Unknow file_name {file_name}")

        if file_name == "clean_test":
            gt = pd.read_csv(img_list, index_col=0)
            self.imgs = gt.index.to_numpy()
            gt = gt.iloc[:, -5:-1]
            # replace NO YES with 0 1
            gt = gt.replace("NO", 0)
            gt = gt.replace("YES", 1)
            clean_gt = gt.to_numpy()
            self.gt = np.zeros((clean_gt.shape[0], args.num_classes))
            # Nodule
            self.gt[:, 4] = clean_gt[:, 3]
            # Mass
            self.gt[:, 5] = clean_gt[:, 3]
            # Pheumothorax
            self.gt[:, 7] = clean_gt[:, 1]
        elif file_name == "clean_test14":
            PATHOLOGIES_15 = [
                "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
                "Mass", "Nodule", "Pneumonia", "Pneumothorax",#"Consolidation",
                "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening",
                "Hernia", "No Finding"
            ]
            P2I = {p: i for i, p in enumerate(PATHOLOGIES_15)}

            def _to01(df_or_series):
                """Support YES/NO, True/False, 0/1."""
                return df_or_series.replace({"YES": 1, "NO": 0, True: 1, False: 0}).astype(np.float32)

            # ---------------- inside your ChestDataset.__init__ ----------------
            if file_name == "clean_test14":
                df = pd.read_csv(img_list)

                # 1) image list
                # CSV header: "Image ID"
                self.imgs = df["Image ID"].to_numpy()

                # 2) unify column name
                # CSV header: "Pleural Thickening" (with space)
                df = df.rename(columns={"Pleural Thickening": "Pleural_Thickening"})

                # 3) build 13-disease GT (exclude Consolidation), then add No Finding
                disease_cols_13 = [
                    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
                    "Mass", "Nodule", "Pneumonia", "Pneumothorax",#"Consolidation",
                    "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening",
                    "Hernia",
                ]
                gt13 = _to01(df[disease_cols_13].copy()).to_numpy()  # (N, 13)

                # 4) derive No Finding from "Abnormal"
                # Abnormal=1 => not normal => No Finding=0
                # Abnormal=0 => normal => No Finding=1
                abnormal = _to01(df["Abnormal"])
                no_finding = (1.0 - abnormal.to_numpy()).reshape(-1, 1)  # (N, 1)

                # 5) final gt14 = [13 disease, No Finding]
                gt14 = np.concatenate([gt13, no_finding], axis=1).astype(np.float32)  # (N, 14)

                # 6) assign to dataset
                self.pathologies = np.array(PATHOLOGIES_15)
                self.gt = gt14  # shape (N, 14)
                for i,name in enumerate(self.pathologies):
                    print(name, np.sum(self.gt[:,i]))

                # Optional sanity checks (can remove later)
                #assert self.gt.shape[1] == 14
                # No Finding should not co-exist with Abnormal==1 (by construction)
        else:
            with open(img_list) as f:
                names = f.read().splitlines()
            self.imgs = np.asarray([x for x in names])
            gt = np.asarray([gt[i] for i in self.imgs])
            self.gt = np.zeros((gt.shape[0], 15))
            for idx, i in enumerate(gt):
                target = i.split("|")
                binary_result = mlb.fit_transform(
                    [[Labels[i] for i in target]]
                ).squeeze()
                self.gt[idx] = binary_result

            if args.trim_data:
                self.trim()
                # Assign no finding
                row_sum = np.sum(self.gt, axis=1)
                # self.gt[np.where(row_sum == 0), -1] = 1

                # Ensure no empty rows
                drop_idx = np.where(row_sum == 0)[0]
                self.gt = np.delete(self.gt, drop_idx, axis=0)
                self.imgs = np.delete(self.imgs, drop_idx, axis=0)

        self.train_img = self.imgs.copy()
        self.train_label = self.gt.copy()

        del self.imgs, self.gt

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, "data", self.train_img[index])
        gt = self.train_label[index]
        img = Image.fromarray(io.imread(img_path)).convert("RGB")
        img_t = self.transform(img)
        if self.mode == "train" and self.stage == "CLS":
            return img_t, gt, index, self.knn[index]
        else:
            return img_t, gt, index

    def __len__(self):
        return self.train_img.shape[0]

    def trim(self):
        cut_list = [
            # 'Mass',
            # 'Nodule',
            "Consolidation",
        ]

        for dp_class in cut_list:
            # Find column and row related to the cut list
            drop_idx_col = np.where(self.pathologies == dp_class)[0].item()
            drop_idx_row = np.where(self.gt[:, drop_idx_col] == 1.0)[0]
            self.drop_list.append(
                {
                    "dp_class": dp_class,
                    "drop_idx_col": drop_idx_col,
                    "drop_idx_row": drop_idx_row,
                }
            )
            if len(drop_idx_row) == 0:
                print(f"skip {dp_class}")
                continue
            self.pathologies = np.delete(self.pathologies, drop_idx_col)
            # Drop gt for row and col
            self.gt = np.delete(self.gt, drop_idx_row, axis=0)
            self.gt = np.delete(self.gt, drop_idx_col, axis=1)
            # Drop imag
            self.imgs = np.delete(self.imgs, drop_idx_row, axis=0)

        return

class AddGaussianNoise(object):
    def __init__(self, std=0.01, p=0.5):
        self.std = std
        self.p = p

    def __call__(self, tensor):
        if torch.rand(1).item() < self.p:
            noise = torch.randn_like(tensor) * self.std
            tensor = tensor + noise
            tensor = torch.clamp(tensor, 0.0, 1.0)
        return tensor
    
def construct_cx14_cut(args, mode, file_name=None, stage="CLS"):
    root_dir = args.root_dir
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if mode == "train" and stage == "MID":
        """ transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (args.resize, args.resize), scale=(0.2, 1)
                ),
                transforms.ColorJitter(
                    (0.9, 1.1), (0.9, 1.1), (0.9, 1.1), (-0.01, 0.01)
                ),
                transforms.RandomAffine(
                    degrees=10,
                    translate=(0.15, 0.15),
                    scale=(0.95, 1.05),
                    shear=(-10, 10, -10, 10),
                    fill=0,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ) """
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                (args.resize, args.resize),
                scale=(0.8, 1.0),
                ratio=(0.95, 1.05)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.15, contrast=0.15)
            ], p=0.8),
            transforms.RandomAffine(
                degrees=5,
                translate=(0.03, 0.03),
                scale=(0.95, 1.05),
                shear=(-10, 10, -10, 10),
                fill=0
            ),
            transforms.ToTensor(),
            AddGaussianNoise(std=0.01, p=0.3),

            transforms.Normalize(mean, std),
        ])
    elif mode == "train" and stage == "CLS":
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                (args.resize, args.resize), 
                scale=(0.8, 1.0),
                ratio=(0.95, 1.05)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.15, contrast=0.15)
            ], p=0.8),
            transforms.RandomAffine(
                degrees=5,
                translate=(0.03, 0.03),
                scale=(0.95, 1.05),
                shear=(-10, 10, -10, 10),
                fill=0
            ),
            transforms.ToTensor(),
            AddGaussianNoise(std=0.01, p=0.3),
            transforms.Normalize(mean, std),
            ])
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(args.resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    assert file_name is not None

    dataset = ChestDataset(
        root_dir, transform, mode, file_name=file_name, stage=stage, args=args
    )

    shuffle = True if mode == "train" else False
    drop_last = True if mode == "train" else False

    if stage == "CLS":
        batch_size = args.batch_size
    else:
        batch_size = args.batch_size

    logger.bind(stage="DATALOADER").warning(f"Stage = [{stage}] | Mode = [{mode}]")
    logger.bind(stage="DATALOADER").warning(
        f"[Configs] | batch_size={batch_size} | shuffle={shuffle} | drop_last={drop_last}"
    )
    logger.bind(stage="DATALOADER").info(transform)

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )
    return loader, len(dataset)

