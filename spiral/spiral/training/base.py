from .blocks import *
import torch
import torchvision
from torch import nn
import torch.nn.functional as F

class Backbone(nn.Module):
    def __init__(self, c1=3):
        super().__init__()
        self.c3 = C3K2(c1, 64)
        self.down34 = Conv(64, 128, k=3, s=2)
        self.c4 = C3K2(128, 128)
        self.down45 = Conv(128, 256, k=3, s=2)
        self.c5 = C3K2(256, 256)

    def forward(self, x):
        c3 = self.c3(x)
        c4 = self.c4(self.down34(c3))
        c5 = self.c5(self.down45(c4))
        return c3, c4, c5
    
class Neck(nn.Module):
    def __init__(self, ch3=64, ch4=128, ch5=256, hidden=128):
        super().__init__()
        self.l3 = nn.Conv2d(ch3, hidden, 1)
        self.l4 = nn.Conv2d(ch4, hidden, 1)
        self.l5 = nn.Conv2d(ch5, hidden, 1)
        self.fpn3 = Conv(hidden, hidden)
        self.fpn4 = Conv(hidden, hidden)
        self.fpn5 = Conv(hidden, hidden)
        self.down34 = Conv(hidden, hidden, k=3, s=2)
        self.down45 = Conv(hidden, hidden, k=3, s=2)
        self.pan4 = Conv(hidden, hidden)
        self.pan5 = Conv(hidden, hidden)

    def forward(self, c3, c4, c5):
        p5 = self.l5(c5)
        p4 = self.l4(c4) + F.interpolate(p5, scale_factor=2, mode="nearest")
        p4 = self.fpn4(p4)
        p3 = self.l3(c3) + F.interpolate(p4, scale_factor=2, mode="nearest")
        p3 = self.fpn3(p3)
        p5 = self.fpn5(p5)

        n4 = self.down34(p3) + p4
        n4 = self.pan4(n4)
        n5 = self.down45(n4) + p5
        n5 = self.pan5(n5)
        return p3, n4, n5
    




class TLayerNeck(nn.Module):
    def __init__(self, dim=128, num_heads=8):
        super().__init__()
        self.ca3 = CrossAttention(dim, num_heads)
        self.ca4 = CrossAttention(dim, num_heads)
        self.ca5 = CrossAttention(dim, num_heads)

    def forward(self, p1, p2):
        p3, n4, n5 = p1
        p3_t1, n4_t1, n5_t1 = p2
        p3 = self.ca3(p3, p3_t1)
        n4 = self.ca4(n4, n4_t1)
        n5 = self.ca5(n5, n5_t1)
        return p3, n4, n5
    
class OdHead(nn.Module):
    def __init__(self, hidden=128, num_classes=4, n_layers=2):
        super().__init__()
        self.stem = nn.Sequential(*[Conv(hidden, hidden, k=3, s=1) for _ in range(n_layers)])
        self.cls_out = nn.Conv2d(hidden, num_classes, 1)
        self.reg_out = nn.Conv2d(hidden, 4, 1)
        self.ctr_out = nn.Conv2d(hidden, 1, 1)

    def forward(self, p3, n4, n5):
        outs = []
        for x in [p3, n4, n5]:
            feat = self.stem(x)
            outs.append((self.cls_out(feat), self.reg_out(feat), self.ctr_out(feat)))
        return outs

class DetectDecoder:
    def __init__(self, score_thresh=0.5, nms_iou=0.45, max_det=100, fmt="xyxy"):
        self.score_thresh = score_thresh
        self.nms_iou = nms_iou
        self.max_det = max_det
        self.fmt = fmt

    @staticmethod
    def _make_grid(H, W, device):
        ys, xs = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij",
        )
        return (xs + 0.5) / W, (ys + 0.5) / H

    def decode(self, od_head_outs):
        device = od_head_outs[0][0].device
        dets = []
        for cls, reg, ctr in od_head_outs:
            B, C, H, W = cls.shape
            cx, cy = self._make_grid(H, W, device)
            cx, cy = cx[None, None, ...], cy[None, None, ...]
            l, t, r, b = reg[:, 0:1], reg[:, 1:2], reg[:, 2:3], reg[:, 3:4]
            x1 = cx - l
            y1 = cy - t
            x2 = cx + r
            y2 = cy + b
            score = cls.sigmoid() * ctr.sigmoid()
            for b_idx in range(B):
                for c_idx in range(C):
                    mask = score[b_idx, c_idx] >= self.score_thresh
                    if not mask.any():
                        continue
                    box_x1 = x1[b_idx, 0][mask]
                    box_y1 = y1[b_idx, 0][mask]
                    box_x2 = x2[b_idx, 0][mask]
                    box_y2 = y2[b_idx, 0][mask]
                    if self.fmt == "xywh":
                        boxes = torch.stack([
                            (box_x1 + box_x2) / 2,
                            (box_y1 + box_y2) / 2,
                            box_x2 - box_x1,
                            box_y2 - box_y1,
                        ], dim=1)
                    else:
                        boxes = torch.stack([box_x1, box_y1, box_x2, box_y2], dim=1)
                    scores = score[b_idx, c_idx][mask]
                    keep = torchvision.ops.nms(boxes, scores, self.nms_iou)
                    for idx in keep[:self.max_det]:
                        dets.append((
                            boxes[idx, 0].item(),
                            boxes[idx, 1].item(),
                            boxes[idx, 2].item(),
                            boxes[idx, 3].item(),
                            scores[idx].item(),
                            c_idx,
                        ))
        return dets

class PosHead(nn.Module):
    def __init__(self, hidden=128, n_layers=2):
        super().__init__()
        self.branch = PosBranch(hidden, n_layers)
        self.fc = nn.Linear(hidden * 3, 3)

    def forward(self, p3, n4, n5):
        xyz = self.fc(torch.cat([
            self.branch(p3),
            self.branch(n4),
            self.branch(n5),
        ], dim=1))
        return xyz
    
class SpiMultiModel(nn.Module):
    def __init__(self, temporal=True, hidden=128, num_classes=4, decode_cfg=None):
        super().__init__()
        self.backbone = Backbone()
        self.neck = Neck(hidden=hidden)
        self.od_head = OdHead(hidden=hidden, num_classes=num_classes)
        self.pos_head = PosHead(hidden=hidden)
        self.temporal = temporal
        if temporal:
            self.tneck = TLayerNeck(dim=hidden)
            self.prev = None
        self._decoder = DetectDecoder(**(decode_cfg or {})) if decode_cfg else None

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        p3, n4, n5 = self.neck(c3, c4, c5)
        if self.temporal and self.prev is not None:
            p3, n4, n5 = self.tneck(self.prev, (p3, n4, n5))
        if self.temporal:
            self.prev = (p3, n4, n5)
        det = self.od_head(p3, n4, n5)
        pos = self.pos_head(p3, n4, n5)
        return det, pos

    def forward_pair(self, x_t, x_t1):
        c3_t, c4_t, c5_t = self.backbone(x_t)
        p3_t, n4_t, n5_t = self.neck(c3_t, c4_t, c5_t)
        c3, c4, c5 = self.backbone(x_t1)
        p3, n4, n5 = self.neck(c3, c4, c5)
        if self.tneck is not None:
            p3, n4, n5 = self.tneck((p3_t, n4_t, n5_t), (p3, n4, n5))
        det = self.od_head(p3, n4, n5)
        pos = self.pos_head(p3, n4, n5)
        return det, pos

    def decode(self, det_outs):
        return self._decoder.decode(det_outs) if self._decoder else []