import os
from argparse import ArgumentParser
import torch

torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "*"
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics import AUROC, Specificity, Recall, F1Score, Accuracy
import pandas as pd

from utils import Kinetics400, load_yaml
from model import VTN
from utils.Tree_Loss import TreeLoss

# Arguments (kept consistent with train.py validation where possible)
parser = ArgumentParser()
parser.add_argument("--val-annotations", type=str, default="*",
                    help="Dataset labels path for evaluation")
parser.add_argument("--val-root-dir", type=str, default="*",
                    help="Dataset files root-dir for evaluation")
parser.add_argument("--classes", type=int, default=4, help="Number of mid-level classes")
parser.add_argument("--config", type=str, default='configs/full-vtn.yaml', help="Config file")

parser.add_argument("--dataset", choices=['ucf', 'smth', 'kinetics'], default='kinetics')
parser.add_argument("--weight-path", type=str, default="*",
                    help='Path for weights and outputs')
parser.add_argument("--checkpoint1", type=str, default="*",
                    help='Checkpoint for coarse head evaluation')
parser.add_argument("--checkpoint2", type=str, default="*",
                    help='Checkpoint for mid head evaluation')

parser.add_argument("--batch-size", type=int, default=2, help="Batch size for evaluation")
parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers")

# Hierarchy configuration (copied from train.py for consistency)
coarse_offset, num_coarse = 0, 2
mid_offset, num_mid = 2, 4
total_nodes, levels = mid_offset + num_mid, 2
mid_to_coarse_list = [0, 0, 1, 1]
# two-level hierarchy: mid -> coarse (global indices)
trees = [
    [2, 0],  # mid 0 -> coarse 0
    [3, 0],  # mid 1 -> coarse 0
    [4, 1],  # mid 2 -> coarse 1
    [5, 1]   # mid 3 -> coarse 1
]


def get_hierarchy_tensors(device):
    """获取层次结构张量，避免重复创建"""
    mid_to_coarse_tensor = torch.tensor(mid_to_coarse_list, device=device, dtype=torch.long)
    tree_loss = TreeLoss(trees, total_nodes, levels, device=device)
    return mid_to_coarse_tensor, tree_loss


args = parser.parse_args()
cfg = load_yaml(args.config)

# Build two models for decoupled evaluation of heads
model_coarse = VTN(**vars(cfg)).cuda()
preprocess = model_coarse.preprocess
train_preprocess = model_coarse.train_preprocess
model_mid = VTN(**vars(cfg)).cuda()

# Resolve checkpoint paths and load
ckpt_path1 = args.checkpoint1 if os.path.isabs(args.checkpoint1) else os.path.join(args.weight_path, args.checkpoint1)
if not os.path.isfile(ckpt_path1):
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path1}")
state1 = torch.load(ckpt_path1, map_location='cpu')
model_coarse.load_state_dict(state1)
model_coarse.eval()

ckpt_path2 = args.checkpoint2 if os.path.isabs(args.checkpoint2) else os.path.join(args.weight_path, args.checkpoint2)
if not os.path.isfile(ckpt_path2):
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path2}")
state2 = torch.load(ckpt_path2, map_location='cpu')
model_mid.load_state_dict(state2)
model_mid.eval()

# Dataset
if args.dataset == 'kinetics':
    # train_set = Kinetics400(args.annotations, args.root_dir, preprocess=train_preprocess, frames=cfg.frames)
    # train_set, val_set = random_split(dataset, [len(dataset) - int(len(dataset) * args.validation_split), int(len(dataset) * args.validation_split)], generator=torch.Generator().manual_seed(12345))
    val_set = Kinetics400(args.val_annotations, args.val_root_dir, preprocess=preprocess, frames=cfg.frames)

val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=8, persistent_workers=False)

# Metrics (align with train.py validation)
num_classes_mid = 4
num_classes_coarse = 2

Acc = Accuracy(task="multiclass", num_classes=num_classes_mid, average="weighted")
Auc = AUROC(task="multiclass", num_classes=num_classes_mid, average='weighted')
Spe = Specificity(task="multiclass", num_classes=num_classes_mid, average='weighted')
Sen = Recall(task="multiclass", num_classes=num_classes_mid, average='weighted')
F1 = F1Score(task="multiclass", num_classes=num_classes_mid, average='weighted')

mid_correct = 0
coarse_correct = 0
mid_total = 0
coarse_total = 0

# Prepare per-class probability collectors
data_mid = {"predict_label": [], "real_label": [], "patient_prefix": []}
for i in range(num_classes_mid):
    data_mid[f"pre_class{i}"] = []

data_coarse = {"predict_label": [], "real_label": [], "patient_prefix": []}
for i in range(num_classes_coarse):
    data_coarse[f"pre_class{i}"] = []

with torch.no_grad():
    for batch in tqdm(val_loader, desc=f"Testing"):
        src, target, patient_prefixes = batch
        src = torch.autograd.Variable(src).cuda()
        target = torch.autograd.Variable(target).cuda()

        # Hierarchy tensors and forward
        mid_to_coarse_tensor, _ = get_hierarchy_tensors(src.device)

        # Forward through separate models for each head
        y_coarse_from_coarse, _ = model_coarse(src)
        _, y_mid_from_mid = model_mid(src)

        label_id_hard = target if target.dim() == 1 else target.argmax(dim=1)
        gt_mid = label_id_hard
        gt_coarse = mid_to_coarse_tensor[label_id_hard]
        mid_probs = torch.softmax(y_mid_from_mid, dim=1)
        preds_mid = mid_probs.argmax(dim=1)

        coarse_probs = torch.softmax(y_coarse_from_coarse, dim=1)
        preds_coarse = coarse_probs.argmax(dim=1)

        mid_correct += (preds_mid == gt_mid).cpu().sum().item()
        coarse_correct += (preds_coarse == gt_coarse).cpu().sum().item()
        mid_total += gt_mid.size(0)
        coarse_total += gt_coarse.size(0)

        Acc.update(preds_mid.cpu(), label_id_hard.cpu())
        Auc.update(mid_probs.cpu(), label_id_hard.cpu())
        Spe.update(preds_mid.cpu(), label_id_hard.cpu())
        Sen.update(preds_mid.cpu(), label_id_hard.cpu())
        F1.update(preds_mid.cpu(), label_id_hard.cpu())

        # Collect data with patient prefixes
        data_mid["real_label"].extend(gt_mid.detach().cpu().tolist())
        data_mid["predict_label"].extend(preds_mid.detach().cpu().tolist())
        data_mid["patient_prefix"].extend(patient_prefixes)
        probs_per_class_mid = mid_probs.detach().cpu()
        for i in range(num_classes_mid):
            data_mid[f"pre_class{i}"].extend(probs_per_class_mid[:, i].tolist())

        data_coarse["real_label"].extend(gt_coarse.detach().cpu().tolist())
        data_coarse["predict_label"].extend(preds_coarse.detach().cpu().tolist())
        data_coarse["patient_prefix"].extend(patient_prefixes)
        probs_per_class_coarse = coarse_probs.detach().cpu()
        for i in range(num_classes_coarse):
            data_coarse[f"pre_class{i}"].extend(probs_per_class_coarse[:, i].tolist())

# Compute metrics
Acc_val = Acc.compute().item() * 100
auc_val = Auc.compute().item() * 100
spe_val = Spe.compute().item() * 100
sen_val = Sen.compute().item() * 100
f1_val = F1.compute().item() * 100

mid_acc = mid_correct / max(1, mid_total)
coarse_acc = coarse_correct / max(1, coarse_total)

print("mid_acc:", mid_acc, "coarse_acc:", coarse_acc)
print("Test ACC:", Acc_val, "Test AUC:", auc_val,
      "Test SPE:", spe_val, "Test SEN:", sen_val,
      "Test F1:", f1_val)

# Export Excel files
os.makedirs(args.weight_path, exist_ok=True)
out_test_mid = os.path.join(args.weight_path, 'Test_mid_norule.xlsx')
out_test_coarse = os.path.join(args.weight_path, 'Test_coarse_norule.xlsx')

pd.DataFrame(data_mid).to_excel(out_test_mid, index=False)
pd.DataFrame(data_coarse).to_excel(out_test_coarse, index=False)
print(f"Saved: {out_test_mid}\nSaved: {out_test_coarse}")