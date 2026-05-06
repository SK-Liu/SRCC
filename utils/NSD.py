import numpy as np
from loguru import logger
import faiss
import wandb
import os
from utils.mid_loss import MID_LOSS
import torch

from tqdm import tqdm
class NSD:
    def __init__(self, args, model, wordvec_array, static_train_loader) -> None:
        self.args = args
        self.embed_len = args.embed_len
        self.num_classes = args.c
        self.device = self.args.device
        # self.train_loader = train_loader
        self.static_train_loader = static_train_loader
        self.wordvec_array = wordvec_array
        self.model = model 
        if args.train_data == "2007" or args.train_data == "2012":
            self.org_gt = self.static_train_loader.dataset.labels
        else:
            self.org_gt = self.static_train_loader.dataset.train_label
        self.train_size = self.org_gt.shape[0]
        # relabel_method -> 3
        self.relabel_methods = f"relabel_v{args.relabel_method}"
        self.relabel_func = getattr(self, self.relabel_methods)
        logger.bind(stage="NSD").warning(f"Using method v{args.relabel_method}")

        if self.args.load_mid_features:
            fp = self.args.mid_ckpt
            logger.bind(stage="NSD").warning("LOADING PD CKPT")
            self.pred_idxs = np.load(f"./ckpt/saved/{fp}/files/pred_rank.npy")
            self.va_matrix = np.load(f"./ckpt/saved/{fp}/files/va_matrix.npy")
        if self.args.load_sample_graph:
            logger.bind(stage="NSD").warning("LOADING SAMPLE GRAPH")
            fp = self.args.mid_ckpt
            self.sample_graph = np.load(f"./ckpt/saved/{fp}/files/sample_graph.npy")

    def run(
        self,
    ):
        if not self.args.load_mid_features:
            logger.bind(stage="NSD").critical(f"FETCHING ATTRIBUTES")
            self.pred_idxs, self.va_matrix = self.get_attributes()

        if not self.args.load_sample_graph:
            logger.bind(stage="NSD").critical(f"CONSTRUCTING SAMPLE GRAPH")
            self.sample_graph = self.graph_based_maker()

        self.knn_relabel()
        return

    """ def load_model(
        self,
    ):# this need revise
        if len(self.args.mid_ckpt) != 0:
            tmp = self.args.mid_ckpt
            dir = f"./ckpt/saved/{tmp}/files"
        else:
            dir = wandb.run.dir
        fp = os.path.join(dir, "model_best_mid.pth")
        self.model.load_state_dict(torch.load(fp)["net"])
        self.model.to(self.device) """

    def get_attributes(
        self,
    ):
        # train_loader = construct_cx14(args, args.root_dir, mode="god", file_name="train")
        mid_criteria = MID_LOSS(
            wordvec_array=self.wordvec_array, beta=0.3, args=self.args
        ).to(self.device)
        total_loss = 0.0
        preds_regular = []
        targets = []
        self.wordvec_array = self.wordvec_array.squeeze().transpose(0, 1)
        self.model.eval()

        tbar = tqdm(
            self.static_train_loader,
            total=int(len(self.static_train_loader)),
            ncols=100,
            position=0,
            leave=True,
        )

        with torch.no_grad():
            for batch_idx, (inputs, labels, idx) in enumerate(tbar):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs, _, v = self.model(inputs)
                loss = mid_criteria(v, labels)
                total_loss += loss.item()
                # unsqueeze方法用于增加一个维度。这里的1表示在第二个维度（索引从0开始）上增加一个大小为1的维度。
                # 比如，如果outputs的形状是[batch_size, features]，使用unsqueeze(1)后，形状会变成[batch_size, 1, features]。
                # 这个操作通常用于满足某些特定的操作或函数输入的需求，比如需要将数据批次以序列的形式处理时。
                preds_regular.append(v.unsqueeze(1).cpu().detach())
                targets.append(labels.cpu().detach())
                wandb.log({"relabel_loss": loss})

        # np.save("./pred_pd.npy", torch.cat(preds_regular).numpy())

        train_loss = total_loss / (batch_idx + 1)
        logger.bind(stage="NSD").info(f"Train Loss {train_loss}")
        # torch.cat是PyTorch中的一个函数，用于将给定张量序列沿着某个维度拼接起来。
        # 默认情况下，它是沿着第一个维度（dim=0）进行拼接的，但你也可以通过指定dim参数来选择其他维度。
        # 在这个例子中，preds_regular列表中的张量被沿着第一个维度拼接起来，
        # 这意味着如果每个张量的形状是[batch_size_i, ...]（其中...表示其他可能的维度），那么拼接后的张量形状将是[sum(batch_size_i), ...]
        va_matrix = torch.cat(preds_regular).numpy()
        pred_idxs, dists = get_knns(self.wordvec_array.cpu().detach(), va_matrix)
        
        np.save(
            os.path.join(wandb.run.dir, "pred_rank.npy"),
            pred_idxs,
        )

        np.save(
            os.path.join(wandb.run.dir, "va_matrix.npy"),
            va_matrix,
        )
        return pred_idxs, va_matrix

    def graph_based_maker(
        self,
    ):
        num_fea = self.args.num_fea

        # pd_reshape = va_matrix
        print(f"number of pd :{num_fea}")
        # 重塑后的数组的形状将变为(self.va_matrix.shape[0], num_fea, self.embed_len)。
        pd_reshape = self.va_matrix.reshape(
            self.va_matrix.shape[0], self.embed_len, num_fea
        ).transpose(0, 2, 1)
        # (self.va_matrix.shape[0] * num_fea, self.embed_len)
        pd_reshape = pd_reshape.reshape(-1, self.embed_len)
        # pd_reshape

        topk = 200
        index = faiss.IndexFlatL2(
            pd_reshape.shape[-1]
        )  # OR IP for cosine similarity. But need to normalize before clustering
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index.add(pd_reshape)

        D, I = gpu_index.search(pd_reshape, topk)
        # Attribute-level Graph
        idx = I.reshape(self.train_size, num_fea, topk)

        # args = edict({"device": 0})
        # self.wordvec_array = load_word_vec(args)
        # self.wordvec_array = self.wordvec_array.squeeze().transpose(0, 1)

        sample_graph = []
        # Sample-level Graph
        for sample_idx in tqdm(range(self.train_size), ncols=100):
            lb_corr = [
                np.unique(idx[sample_idx, i, :] // num_fea, return_index=True)
                for i in range(num_fea)
            ]
            sort_index0 = None
            sort_index0 = [
                lb_corr[i][0][np.argsort(lb_corr[i][1])] for i in range(num_fea)
            ]
            sort_index0 = [i.reshape(1, -1) for i in sort_index0]
            idx_score = []
            uniq_sample = np.unique(np.concatenate(sort_index0, axis=1))
            for item in uniq_sample:
                tmp_list = []
                for row in sort_index0:
                    tmp_list.extend(np.where(row == item)[1])
                idx_score.append(tmp_list)

            argmax_idx = np.argsort([sum(i) for i in idx_score])
            # Top 20 sample
            sorted_sample = uniq_sample[argmax_idx][1:15]
            sample_graph.append(sorted_sample)
        for i, graph in enumerate(sample_graph):  
            print(f"Array {i} shape: {graph.shape}")
        sample_graph = np.stack(sample_graph, axis=1).transpose(1, 0)

        np.save(os.path.join(wandb.run.dir, "sample_graph.npy"), sample_graph)
        return sample_graph
        # sample_graph = np.load("./clustering_test/sample_graph.npy")

    def knn_relabel(
        self,
    ):
        #这行代码修剪了self.sample_graph，只保留每个样本的前self.args.nsd_topk个邻居或特征。
        self.sample_graph = self.sample_graph[:, : self.args.nsd_topk]
        logger.bind(stage="NSD").critical(f"TOP-{self.args.nsd_topk} LABEL AGGREGATION")

        total_new_label = []
        # pred = np.zeros((self.num_classes,))
        self.CLEAN_LIST = []

        # 添加到这个clean list的选择中 num_classes -> 14
        # 循环寻找干净的标签
        min_k = 1      # 排除全阴
        max_k = min(6, self.num_classes - 1)
        for top_num in range(min_k, max_k + 1):
        #for top_num in range(1, 7):
            for i in tqdm(range(self.train_size), ncols=100):
                curret_pred = np.zeros((self.num_classes,))
                # 从排位1开始找，找到rank 7，哪一种情况与标签相符 都会被加入到干净的标签中。
                # self.pred_idxs 是对每个样本中标签置信度进行排序，比如 前两位为10，11，那么该标签属于第10和11类的概率就高
                pos_idx = self.pred_idxs[i][:top_num]
                curret_pred[pos_idx] = 1

                if (self.org_gt[i] == curret_pred).all():
                    # pred += curret_pred
                    self.CLEAN_LIST.append(i)
        self.CLEAN_LIST = sorted(self.CLEAN_LIST)
        logger.bind(stage="NSD").info(
            f"CLEAN SET CONTAIN {len(self.CLEAN_LIST)} SAMPLES"
        )

        for sample_idx in tqdm(range(self.train_size), ncols=100):
            row_new_label = self.relabel_func(sample_idx)
            total_new_label.append(row_new_label)

        new_gt = np.stack(total_new_label)
        np.save(os.path.join(wandb.run.dir, "knn_pdc.npy"), new_gt)
        return new_gt
    
    def relabel_v3(self, sample_idx):
        row_new_label = None
        # 如果样本索引在self.CLEAN_LIST中，则复制其原始标签，并乘以self.args.nsd_topk（这里可能是一个错误或特定用途的缩放因子）。
        if sample_idx in self.CLEAN_LIST:
            row_new_label = self.org_gt[sample_idx].copy() * self.args.nsd_topk # *? args.nsd_topk->10 为啥*10？
        # 否则，对于非干净样本，从其self.sample_graph中的邻居收集标签，对它们进行求和，以生成新的标签。
        else:
            topk_mat = np.stack(
                [self.org_gt[sp_idx] for sp_idx in self.sample_graph[sample_idx]] #?
            )
            row_new_label = np.sum(topk_mat, axis=0)
        return row_new_label

def get_dist(gallery, vecs, k=10, for_map=False):
    if for_map:
        mat = np.reshape(gallery, (gallery.shape[0], vecs.shape[1], -1))

        K_c = mat.shape[2]
        dot_prod = [np.expand_dims(np.matmul(vecs, mat[:, :, i].transpose(1, 0)), axis=2) for i in
                    range(K_c)]
    else:
        mat = np.reshape(vecs, (vecs.shape[0], gallery.shape[1], -1))
        K_c = mat.shape[2]
        dot_prod = [np.expand_dims(np.matmul(mat[:, :, i], gallery.transpose(1, 0)), axis=2) for i in
                    range(K_c)]
    dist = -np.max(np.concatenate(dot_prod, axis=2), axis=-1)

    if for_map:
        return dist

    ids = np.argsort(dist, axis=1)
    top_k_ids = ids[:, :k]
    top_k_dist = np.take_along_axis(dist, ids, axis=1)[:, :k]
    return top_k_ids, top_k_dist


def get_knns(gallery, vecs, k=-1, for_map=False):
    if k == -1:
        k = gallery.shape[0]  # Using all possible samples
    return get_dist(gallery, vecs, k=k, for_map=for_map)