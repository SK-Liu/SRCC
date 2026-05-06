from sklearn.metrics import hamming_loss, label_ranking_loss, coverage_error, label_ranking_average_precision_score,accuracy_score
import numpy as np
import torch
import math
import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

def test(net, loader, criterion=torch.nn.BCELoss(), return_map=False):
    running_loss = 0

    net.eval()
    target = []
    pred_list = []
    output=[]

    ap_meter=AveragePrecisionMeter()
    for i, (X, y, idx) in enumerate(loader):
        # Pass to gpu or cpu
        X, y = X.cuda().float(), y.cuda().float()

        with torch.no_grad():
            out_1, _, _ = net(X, drop = False)
          
            ap_meter.add(out_1.cpu().detach(), y.cpu())
            
            out = torch.sigmoid(out_1)
            
            y[y==0]=1
            y[y==-1]=0
            loss = criterion(out, y)

            # vals, indices = out.topk(k=2, dim=1, largest=True, sorted=True)
            # pred=torch.zeros_like(pred)
            # for i in range(2):
                # pred[range(len(pred)),indices[:,i]]=1
            # pred_list.append(pred.cpu().detach().numpy())
            
            pred_list.append((out>0.5).cpu().detach().numpy())
            output.append(out.cpu().detach().numpy())
            target.append(y.cpu().detach().numpy())
            

        # Calculate stats
        running_loss += loss.item()
    
    target = np.concatenate(target)
    output = np.concatenate(output)
    preds = np.concatenate(pred_list) 
    
    learn_loss=running_loss / len(loader)

    hloss = hamming_loss(target,preds)
    rloss = label_ranking_loss(target, output)
    cover = compute_cover(target,output)
    avgpre = label_ranking_average_precision_score(target, output)
    
    top_label=np.argmax(output,axis=-1)
    oneerror=sum(1-target[range(len(loader.dataset)),top_label])/len(loader.dataset)
    
    acc=accuracy_score(target,preds)

    map = 100 * ap_meter.value().mean().cpu().detach().numpy()
    OP, OR, OF1, CP, CR, CF1 = ap_meter.overall()
    #OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = ap_meter.overall_topk(3)    

    if(return_map):
        return (learn_loss, hloss, rloss, cover, avgpre, oneerror, acc),(map, OP, OR, OF1, CP, CR, CF1)
    else:
        return learn_loss, hloss, rloss, cover, avgpre, oneerror, acc

def _test(test_loader, model, class_num, clean_nih=False, clean_nih14=False, thr= 0.1):
        """ if net2 is not None:
            net2.eval() """
        model.eval()
        all_preds, all_gts = [], []
        total_loss = 0.0
        
        for batch_idx, (inputs, labels, item) in enumerate(test_loader):
            with torch.no_grad():
                inputs, labels = inputs.cuda().float(), labels.cuda().float()
                outputs, _ , _= model(inputs, drop = False)
                preds = torch.sigmoid(outputs)
                all_preds.append(preds)
                all_gts.append(labels)

        all_preds = torch.cat(all_preds).cpu().numpy()
        all_gts = torch.cat(all_gts).cpu().numpy()

        if clean_nih:
            pred_pneux = all_preds[:, [7]]
            pred_mass_nodule = np.concatenate(
                (all_preds[:, [4]], all_preds[:, [5]]), axis=1
            )
            assert (all_gts[:, 4] - all_gts[:, 5]).sum() == 0.0
            auc_pneux = roc_auc_score(all_gts[:, 7], pred_pneux[:, 0])
            auc_mass_nodule = roc_auc_score(all_gts[:, 4], pred_mass_nodule.mean(-1))
            return auc_pneux, auc_mass_nodule
        elif clean_nih14:
            print(all_preds.shape, all_gts.shape, class_num)
            all_auc = [
            roc_auc_score(all_gts[:, i], all_preds[:, i])
            for i in range(class_num - 1)
            ]
            print(all_auc)
        all_auc = [
            roc_auc_score(all_gts[:, i], all_preds[:, i])
            for i in range(class_num - 1)
        ]
        print(all_auc)
        all_map = [
            average_precision_score(all_gts[:, i], all_preds[:, i])
            for i in range(class_num - 1)
        ]

        # f1
        best_thr = np.zeros(class_num-1)

        for c in range(class_num-1):
            best_f1 = 0
            for t in np.arange(0.05,0.95,0.05):
                pred = (all_preds[:,c] >= t).astype(int)
                f1 = f1_score(all_gts[:,c], pred)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thr[c] = t
        idxs = range(class_num -1)
        #pred_bin = (all_preds[:, list(idxs)] >= thr).astype(np.int32)
        pred_bin = (all_preds[:, list(idxs)] >= best_thr).astype(np.int32)
        all_f1 = []
        for j, i in enumerate(idxs):
            y = all_gts[:, i].astype(np.int32)
            all_f1.append(f1_score(y, pred_bin[:, j], zero_division=0))

        if clean_nih14:
            print(all_preds.shape, all_gts.shape, class_num)
            print("AUC:", all_auc)
            print("F1 :", all_f1)
        return all_auc, total_loss / (batch_idx + 1), all_map, all_f1

def _test_openi(epoch, net, test_loader_openi, class_num):
        #logger.bind(stage="EVAL").info("************** EVAL ON OPENI **************")
        print("************** EVAL ON OPENI **************")
        all_auc, test_loss, all_map, _ = _test(test_loader_openi, net, class_num)
        mean_auc = np.asarray(all_auc).mean()
        mean_map = np.asarray(all_map).mean()
        print("Eval/MeanAUC OPENI", mean_auc, "mAP OPENI", mean_map, "epoch", epoch)
        return mean_auc


def _test_pc(epoch, net, test_loader_padchest, class_num):
    #logger.bind(stage="EVAL").info("************** EVAL ON PADCHEST **************")
    print("************** EVAL ON PADCHEST **************")
    all_auc, test_loss, all_map, _ = _test(test_loader_padchest, net, class_num)
    mean_auc = np.asarray(all_auc).mean()
    mean_map = np.asarray(all_map).mean()
    print("Eval/MeanAUC PADCHEST", mean_auc, "mAP PADCHEST", mean_map, "epoch", epoch)
    #log_csv(self.epoch, all_auc, mean_auc, mean_map, "padchest_auc")
    return mean_auc

def _test_google_nih(epoch, net, test_loader_padchest, class_num):
    #logger.bind(stage="EVAL").info("************** EVAL ON PADCHEST **************")
    print("************** EVAL ON PADCHEST **************")
    auc_pneux, auc_mass_nodule = _test(test_loader_padchest, net, class_num, clean_nih=True)
    
    print("Eval/MeanAUC google NIH auc_pneux", auc_pneux, "auc_mass_nodule", auc_mass_nodule, "epoch", epoch)
    #log_csv(self.epoch, all_auc, mean_auc, mean_map, "padchest_auc")
    return auc_pneux, auc_mass_nodule

def _test_google_nih14(epoch, net, test_loader_padchest, class_num):
    all_auc, test_loss, all_map, all_f1 = _test(test_loader_padchest, net, class_num, clean_nih=False, clean_nih14=True)
    
    mean_auc = np.asarray(all_auc).mean()
    mean_f1  = np.nanmean(np.asarray(all_f1,  dtype=np.float32))

    print("Eval/MeanAUC google NIH14", mean_auc, "epoch", epoch)
    print("Eval/MeanF1  google NIH14", mean_f1,  "epoch", epoch)
    return mean_auc, mean_f1

def compute_cover(labels, outputs):
    n_labels = labels.shape[1]
    loss = coverage_error(labels, outputs)

    return (loss-1)/n_labels


    
class AveragePrecisionMeter(object):
    """
    The APMeter measures the average precision per class.
    The APMeter is designed to operate on `NxK` Tensors `output` and
    `target`, and optionally a `Nx1` Tensor weight where (1) the `output`
    contains model output scores for `N` examples and `K` classes that ought to
    be higher when the model is more convinced that the example should be
    positively labeled, and smaller when the model believes the example should
    be negatively labeled (for instance, the output of a sigmoid function); (2)
    the `target` contains only values 0 (for negative examples) and 1
    (for positive examples); and (3) the `weight` ( > 0) represents weight for
    each sample.
    """

    def __init__(self, difficult_examples=True):
        super(AveragePrecisionMeter, self).__init__()
        self.reset()
        self.difficult_examples = difficult_examples

    def reset(self):
        """Resets the meter with empty member variables"""
        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.targets = torch.LongTensor(torch.LongStorage())

    def add(self, output, target):
        """
        Args:
            output (Tensor): NxK tensor that for each of the N examples
                indicates the probability of the example belonging to each of
                the K classes, according to the model. The probabilities should
                sum to one over all classes
            target (Tensor): binary NxK tensort that encodes which of the K
                classes are associated with the N-th input
                    (eg: a row [0, 1, 0, 1] indicates that the example is
                         associated with classes 2 and 4)
            weight (optional, Tensor): Nx1 tensor representing the weight for
                each example (each weight > 0)
        """
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column \
                per class)'
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1), \
                'dimensions for output should match previously added examples.'

        # make sure storage is of sufficient size
        if self.scores.storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.storage().size() * 1.5)
            self.scores.storage().resize_(int(new_size + output.numel()))
            self.targets.storage().resize_(int(new_size + output.numel()))

        # store scores and targets
        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output)
        self.targets.narrow(0, offset, target.size(0)).copy_(target)

    def value(self):
        """Returns the model's average precision for each class
        Return:
            ap (FloatTensor): 1xK tensor, with avg precision for each class k
        """

        if self.scores.numel() == 0:
            return 0
        ap = torch.zeros(self.scores.size(1))
        rg = torch.arange(1, self.scores.size(0)).float()
        # compute average precision for each class
        for k in range(self.scores.size(1)):
            # sort scores
            scores = self.scores[:, k]
            targets = self.targets[:, k]
            # compute average precision
            ap[k] = AveragePrecisionMeter.average_precision(scores, targets, self.difficult_examples)
        return ap

    @staticmethod
    def average_precision(output, target, difficult_examples=True):

        # sort examples
        sorted, indices = torch.sort(output, dim=0, descending=True)

        # Computes prec@i
        pos_count = 0.
        total_count = 0.
        precision_at_i = 0.
        for i in indices:
            label = target[i]
            if difficult_examples and label == 0:
                continue
            if label == 1:
                pos_count += 1
            total_count += 1
            if label == 1:
                precision_at_i += pos_count / total_count
        precision_at_i /= pos_count
        return precision_at_i

    def overall(self):
        if self.scores.numel() == 0:
            return 0
        scores = self.scores.cpu().numpy()
        targets = self.targets.cpu().numpy()
        targets[targets == -1] = 0
        return self.evaluation(scores, targets)

    def overall_topk(self, k):
        targets = self.targets.cpu().numpy()
        targets[targets == -1] = 0
        n, c = self.scores.size()
        scores = np.zeros((n, c)) - 1
        index = self.scores.topk(k, 1, True, True)[1].cpu().numpy()
        tmp = self.scores.cpu().numpy()
        for i in range(n):
            for ind in index[i]:
                scores[i, ind] = 1 if tmp[i, ind] >= 0 else -1
        return self.evaluation(scores, targets)


    def evaluation(self, scores_, targets_):
        n, n_class = scores_.shape
        Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
        for k in range(n_class):
            scores = scores_[:, k]
            targets = targets_[:, k]
            targets[targets == -1] = 0
            Ng[k] = np.sum(targets == 1)
            Np[k] = np.sum(scores >= 0)
            Nc[k] = np.sum(targets * (scores >= 0))
        Np[Np == 0] = 1
        OP = np.sum(Nc) / np.sum(Np)
        OR = np.sum(Nc) / np.sum(Ng)
        OF1 = (2 * OP * OR) / (OP + OR)

        CP = np.sum(Nc / Np) / n_class
        CR = np.sum(Nc / Ng) / n_class
        CF1 = (2 * CP * CR) / (CP + CR)
        return OP * 100, OR * 100, OF1 * 100, CP * 100, CR * 100, CF1 * 100

def _forward_ensemble3(model1, model2, model3, inputs):
    out1, _, _ = model1(inputs, drop=False)
    out2, _, _ = model2(inputs, drop=False)
    out3, _, _ = model3(inputs, drop=False)

    # logits average
    out = (out1 + out2 + out3) / 3.0
    return out

def _test_ensemble3(test_loader, model1, model2, model3, class_num, clean_nih=False, clean_nih14=False, thr=0.1):
    model1.eval()
    model2.eval()
    model3.eval()

    all_preds, all_gts = [], []
    total_loss = 0.0

    for batch_idx, (inputs, labels, item) in enumerate(test_loader):
        with torch.no_grad():
            inputs, labels = inputs.cuda().float(), labels.cuda().float()

            outputs = _forward_ensemble3(model1, model2, model3, inputs)
            preds = torch.sigmoid(outputs)

            all_preds.append(preds)
            all_gts.append(labels)

    all_preds = torch.cat(all_preds).cpu().numpy()
    all_gts = torch.cat(all_gts).cpu().numpy()

    if clean_nih:
        pred_pneux = all_preds[:, [7]]
        pred_mass_nodule = np.concatenate(
            (all_preds[:, [4]], all_preds[:, [5]]), axis=1
        )
        assert (all_gts[:, 4] - all_gts[:, 5]).sum() == 0.0
        auc_pneux = roc_auc_score(all_gts[:, 7], pred_pneux[:, 0])
        auc_mass_nodule = roc_auc_score(all_gts[:, 4], pred_mass_nodule.mean(-1))
        return auc_pneux, auc_mass_nodule

    elif clean_nih14:
        print(all_preds.shape, all_gts.shape, class_num)
        all_auc = [
            roc_auc_score(all_gts[:, i], all_preds[:, i])
            for i in range(class_num - 1)
        ]
        print(all_auc)

    all_auc = [
        roc_auc_score(all_gts[:, i], all_preds[:, i])
        for i in range(class_num - 1)
    ]
    print(all_auc)

    all_map = [
        average_precision_score(all_gts[:, i], all_preds[:, i])
        for i in range(class_num - 1)
    ]

    # best f1 threshold per class
    best_thr = np.zeros(class_num - 1)

    for c in range(class_num - 1):
        best_f1 = 0
        for t in np.arange(0.05, 0.95, 0.05):
            pred = (all_preds[:, c] >= t).astype(int)
            f1 = f1_score(all_gts[:, c], pred)
            if f1 > best_f1:
                best_f1 = f1
                best_thr[c] = t

    idxs = range(class_num - 1)
    pred_bin = (all_preds[:, list(idxs)] >= best_thr).astype(np.int32)

    all_f1 = []
    for j, i in enumerate(idxs):
        y = all_gts[:, i].astype(np.int32)
        all_f1.append(f1_score(y, pred_bin[:, j], zero_division=0))

    if clean_nih14:
        print(all_preds.shape, all_gts.shape, class_num)
        print("AUC:", all_auc)
        print("F1 :", all_f1)

    return all_auc, total_loss / (batch_idx + 1), all_map, all_f1

def _test_openi_ensemble3(epoch, model1, model2, model3, test_loader_openi, class_num):
    print("************** EVAL ON OPENI (ENSEMBLE-3) **************")
    all_auc, test_loss, all_map, _ = _test_ensemble3(
        test_loader_openi, model1, model2, model3, class_num
    )
    mean_auc = np.asarray(all_auc).mean()
    mean_map = np.asarray(all_map).mean()
    print("Eval/MeanAUC OPENI", mean_auc, "mAP OPENI", mean_map, "epoch", epoch)
    return mean_auc


def _test_pc_ensemble3(epoch, model1, model2, model3, test_loader_padchest, class_num):
    print("************** EVAL ON PADCHEST (ENSEMBLE-3) **************")
    all_auc, test_loss, all_map, _ = _test_ensemble3(
        test_loader_padchest, model1, model2, model3, class_num
    )
    mean_auc = np.asarray(all_auc).mean()
    mean_map = np.asarray(all_map).mean()
    print("Eval/MeanAUC PADCHEST", mean_auc, "mAP PADCHEST", mean_map, "epoch", epoch)
    return mean_auc


def _test_google_nih_ensemble3(epoch, model1, model2, model3, test_loader_google_nih, class_num):
    print("************** EVAL ON GOOGLE NIH (ENSEMBLE-3) **************")
    auc_pneux, auc_mass_nodule = _test_ensemble3(
        test_loader_google_nih, model1, model2, model3, class_num, clean_nih=True
    )
    print("Eval/MeanAUC google NIH auc_pneux", auc_pneux, "auc_mass_nodule", auc_mass_nodule, "epoch", epoch)
    return auc_pneux, auc_mass_nodule


def _test_google_nih14_ensemble3(epoch, model1, model2, model3, test_loader_google_nih14, class_num):
    all_auc, test_loss, all_map, all_f1 = _test_ensemble3(
        test_loader_google_nih14, model1, model2, model3, class_num, clean_nih=False, clean_nih14=True
    )

    mean_auc = np.asarray(all_auc).mean()
    mean_f1 = np.nanmean(np.asarray(all_f1, dtype=np.float32))

    print("Eval/MeanAUC google NIH14", mean_auc, "epoch", epoch)
    print("Eval/MeanF1  google NIH14", mean_f1, "epoch", epoch)
    return mean_auc, mean_f1