from .blocks import *
import torch
from torch import nn
import torch.nn.functional as F

"""
Veriye ulaşma şekli

from spidata.data.registery import Registery
from spidata.struct.datamodule import SpiDataModule
from spidata.tools.transformations import SpiTransforms

# Birden fazla veri paketi (sekans verisi)
datamodule = SpiDataModule(
    datapacks=[
        Registery.ot25_1, 
        Registery.ot25_2,
        Registery.ot25_3, 
        Registery.ot25_4,
        ],  # birden fazla sekans
    train_ratio=0.8,
    batch_size=8,
    train_transform=SpiTransforms.default_training
)

print(f"Toplam train batch: {len(datamodule.trainloader)}")
print(f"Toplam val batch  : {len(datamodule.valloader)}")

# Train döngüsü — new=True yeni bir sekansın başladığını gösterir
for batch in datamodule.trainloader:
    if batch["new"]:
        print("Yeni sekans başladı, state sıfırlanabilir.")

    images       = batch["image"]        # (B, 3, 512, 512)
    translations = batch["translations"] # (B, 3)
    objects      = batch["objects"]      # liste

    # ... model adımı buraya ...
    break

"""



class DetCriterion(nn.Module):
    def __init__(self, num_classes=4, alpha=0.25, gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def _make_grid(H, W, device):
        ys, xs = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij",
        )
        return (xs + 0.5) / W, (ys + 0.5) / H

    @staticmethod
    def _giou(pred, target):
        x1 = torch.min(pred[:, 0], target[:, 0])
        y1 = torch.min(pred[:, 1], target[:, 1])
        x2 = torch.max(pred[:, 2], target[:, 2])
        y2 = torch.max(pred[:, 3], target[:, 3])

        a_pred = (pred[:, 2] - pred[:, 0]).clamp(min=0) * (pred[:, 3] - pred[:, 1]).clamp(min=0)
        a_tgt = (target[:, 2] - target[:, 0]).clamp(min=0) * (target[:, 3] - target[:, 1]).clamp(min=0)

        inter = (torch.min(pred[:, 2], target[:, 2]) - torch.max(pred[:, 0], target[:, 0])).clamp(min=0) * \
                (torch.min(pred[:, 3], target[:, 3]) - torch.max(pred[:, 1], target[:, 1])).clamp(min=0)
        union = a_pred + a_tgt - inter
        iou = inter / (union + 1e-7)
        area_c = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        giou = iou - (area_c - union) / (area_c + 1e-7)
        return 1 - giou

    def forward(self, det_outs, gt_boxes, gt_labels):
        device = det_outs[0][0].device
        cls_loss_sum = torch.tensor(0.0, device=device)
        reg_loss_sum = torch.tensor(0.0, device=device)
        ctr_loss_sum = torch.tensor(0.0, device=device)

        for b_idx in range(det_outs[0][0].shape[0]):
            gt = gt_boxes[b_idx]
            lbl = gt_labels[b_idx]
            num_gts = gt.shape[0]

            for cls, reg, ctr in det_outs:
                _, C, H, W = cls.shape
                cx, cy = self._make_grid(H, W, device)
                cx_f, cy_f = cx.flatten(), cy.flatten()
                HW = H * W

                cls_pred = cls[b_idx].flatten(1).permute(1, 0)
                reg_pred = reg[b_idx].flatten(1).permute(1, 0)
                ctr_pred = ctr[b_idx].flatten(1).permute(1, 0)

                if num_gts == 0:
                    loss = F.binary_cross_entropy_with_logits(
                        cls_pred, torch.zeros_like(cls_pred), reduction='mean'
                    )
                    cls_loss_sum = cls_loss_sum + loss
                    continue

                l = cx_f[None] - gt[:, 0:1]
                t = cy_f[None] - gt[:, 1:2]
                r = gt[:, 2:3] - cx_f[None]
                b = gt[:, 3:4] - cy_f[None]

                inside = (l > 0) & (t > 0) & (r > 0) & (b > 0)

                gt_areas = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
                best_gt = torch.full((HW,), -1, device=device, dtype=torch.long)
                best_area = torch.full((HW,), float('inf'), device=device)

                for gt_idx in range(num_gts):
                    pos = inside[gt_idx]
                    smaller = pos & (gt_areas[gt_idx] < best_area)
                    best_gt[smaller] = gt_idx
                    best_area[smaller] = gt_areas[gt_idx]

                pos_mask = best_gt >= 0
                num_pos = pos_mask.sum()

                cls_target = torch.zeros(HW, self.num_classes, device=device)
                reg_target = torch.zeros(HW, 4, device=device)
                ctr_target = torch.zeros(HW, device=device)

                if num_pos > 0:
                    pos_idx = torch.where(pos_mask)[0]
                    pos_gt = best_gt[pos_idx]

                    l_pos = l[pos_gt, pos_idx]
                    t_pos = t[pos_gt, pos_idx]
                    r_pos = r[pos_gt, pos_idx]
                    b_pos = b[pos_gt, pos_idx]

                    reg_target[pos_idx] = torch.stack([l_pos, t_pos, r_pos, b_pos], dim=1)
                    ctr_target[pos_idx] = torch.sqrt(
                        (torch.min(l_pos, r_pos) / (torch.max(l_pos, r_pos) + 1e-7)) *
                        (torch.min(t_pos, b_pos) / (torch.max(t_pos, b_pos) + 1e-7))
                    )
                    cls_target[pos_idx] = 0
                    cls_target[pos_idx, lbl[pos_gt]] = 1.0

                cls_loss = F.binary_cross_entropy_with_logits(cls_pred, cls_target, reduction='none')
                p = torch.sigmoid(cls_pred)
                p_t = p * cls_target + (1 - p) * (1 - cls_target)
                focal_weight = (1 - p_t) ** self.gamma
                if self.alpha >= 0:
                    alpha_t = self.alpha * cls_target + (1 - self.alpha) * (1 - cls_target)
                    focal_weight = focal_weight * alpha_t
                cls_loss = (cls_loss * focal_weight).sum() / (HW * self.num_classes)
                cls_loss_sum = cls_loss_sum + cls_loss

                if num_pos > 0:
                    pred_ltrb = reg_pred[pos_idx]
                    pred_x1y1x2y2 = torch.stack([
                        cx_f[pos_idx] - pred_ltrb[:, 0],
                        cy_f[pos_idx] - pred_ltrb[:, 1],
                        cx_f[pos_idx] + pred_ltrb[:, 2],
                        cy_f[pos_idx] + pred_ltrb[:, 3],
                    ], dim=1)
                    tgt_x1y1x2y2 = torch.stack([
                        cx_f[pos_idx] - reg_target[pos_idx, 0],
                        cy_f[pos_idx] - reg_target[pos_idx, 1],
                        cx_f[pos_idx] + reg_target[pos_idx, 2],
                        cy_f[pos_idx] + reg_target[pos_idx, 3],
                    ], dim=1)
                    reg_loss = self._giou(pred_x1y1x2y2, tgt_x1y1x2y2).sum() / num_pos
                    reg_loss_sum = reg_loss_sum + reg_loss

                    ctr_loss = F.binary_cross_entropy_with_logits(
                        ctr_pred[pos_idx, 0], ctr_target[pos_idx], reduction='mean'
                    )
                    ctr_loss_sum = ctr_loss_sum + ctr_loss

        total = cls_loss_sum + reg_loss_sum + ctr_loss_sum
        return {
            "loss": total,
            "cls_loss": cls_loss_sum,
            "reg_loss": reg_loss_sum,
            "ctr_loss": ctr_loss_sum,
        }


class PosCriterion(nn.Module):
    def __init__(self, loss_type="mse"):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, pred, target):
        loss = F.mse_loss(pred, target, reduction="mean")
        return {"loss": loss, "pos_loss": loss}


def train_step(model, batch, crit_det, crit_pos, opt=None, temporal=False):
    if temporal:
        x_t, x_t1, gt_boxes, gt_labels, gt_pos = batch
        det, pos = model.forward_pair(x_t, x_t1)
    else:
        x, gt_boxes, gt_labels, gt_pos = batch
        det, pos = model(x)

    loss_det = crit_det(det, gt_boxes, gt_labels)
    loss_pos = crit_pos(pos, gt_pos)

    total = loss_det["loss"] + loss_pos["loss"]
    losses = {
        "loss": total,
        "det_loss": loss_det["loss"],
        "cls_loss": loss_det["cls_loss"],
        "reg_loss": loss_det["reg_loss"],
        "ctr_loss": loss_det["ctr_loss"],
        "pos_loss": loss_pos["pos_loss"],
    }

    if opt is not None:
        opt.zero_grad()
        total.backward()
        opt.step()

    return losses

# from spiral.struct.nn.trainer import DetCriterion, PosCriterion, train_step

# model = SpiMulti(temporal=True, num_classes=4)
# crit_det = DetCriterion(num_classes=4, strides=(1, 2, 4))
# crit_pos = PosCriterion()
# opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

# # Single-frame
# batch = (x, gt_boxes, gt_labels, gt_pos, img_size)
# losses = train_step(model, batch, crit_det, crit_pos, opt, temporal=False)

# # Temporal pair
# batch = (x_t, x_t1, gt_boxes, gt_labels, gt_pos, img_size)
# losses = train_step(model, batch, crit_det, crit_pos, opt, temporal=True)

# # losses = {loss, det_loss, cls_loss, reg_loss, ctr_loss, pos_loss}
