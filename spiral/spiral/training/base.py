from .blocks import *
import torch
import torchvision
from torch import nn
import torch.nn.functional as F

class Backbone(nn.Module):
    """
    Erken downsampling icin ayarlanabilir stem blogu (FocusStem veya DeepStem) kullanan backbone.

    512x512x3 girdi, C3K2 bloklarina girmeden once stem ile 128x128 boyutuna indirilir.
    Bu sayede erken katmanlardaki FLOP ve bellek tuketimi onemli olcude azalir.

    stem_type:
        'focus' -> Space-to-Depth (bilgi kaybi yok, daha az FLOP)
        'deep'  -> Ardisik 3x3 evrişimler (daha genis receptive field)
    """
    def __init__(self, c1=3, base_channels=32, stem_type="focus"):
        super().__init__()
        c2 = base_channels
        # Stem: 512x512x3 -> 128x128xc2 (4x downsampling)
        if stem_type == "focus":
            self.stem = FocusStem(in_channels=c1, out_channels=c2, downscale=4)
        elif stem_type == "deep":
            self.stem = DeepStem(in_channels=c1, out_channels=c2)
        else:
            raise ValueError(f"Bilinmeyen stem_type: {stem_type}. 'focus' veya 'deep' kullanin.")

        # C3K2 bloklari artik 128x128 uzerinde calisir (512x512 yerine)
        self.c3 = C3K2(c2, c2)
        self.down34 = Conv(c2, c2 * 2, k=3, s=2)
        self.c4 = C3K2(c2 * 2, c2 * 2)
        self.down45 = Conv(c2 * 2, c2 * 4, k=3, s=2)
        self.c5 = C3K2(c2 * 4, c2 * 4)

    def forward(self, x):
        # x: 512x512 -> stem -> 128x128
        s = self.stem(x)
        c3 = self.c3(s)
        c4 = self.c4(self.down34(c3))
        c5 = self.c5(self.down45(c4))
        return c3, c4, c5

class Neck(nn.Module):
    def __init__(self, ch3=32, ch4=64, ch5=128, hidden=64):
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
    """OdHead çıktılarını alıp skor threshold, NMS ve format dönüşümü ile
    nihai detection listesine çeviren post-processing sınıfı.

    Dönen her detection: (x1, y1, x2, y2, score, class_idx)
    """

    def __init__(self, score_thresh=0.5, nms_iou=0.45, max_det=100, fmt="xyxy"):
        self.score_thresh = score_thresh
        self.nms_iou = nms_iou
        self.max_det = max_det
        self.fmt = fmt

    @staticmethod
    def _make_grid(H, W, device):
        """Feature map boyutunda normalize edilmiş merkez noktaları grid'i oluşturur.
        Her hücrenin merkezini [0,1] aralığında döndürür."""
        ys, xs = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij",
        )
        return (xs + 0.5) / W, (ys + 0.5) / H

    def decode(self, od_head_outs):
        """OdHead'ten gelen (cls, reg, ctr) üçlülerini bounding box'lara çevirir.

        Args:
            od_head_outs: Her ölçek için (cls, reg, ctr) içeren liste.
                          cls: (B, C, H, W), reg: (B, 4, H, W), ctr: (B, 1, H, W)

        Returns:
            (x1, y1, x2, y2, score, class_idx) demetlerinden oluşan liste.
        """
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
    def __init__(self, hidden=64, num_classes=4, decode_cfg=None, od=True, pos=True, stem_type="focus"):
        super().__init__()
        self.backbone = Backbone(base_channels=hidden // 2, stem_type=stem_type)
        self.neck = Neck(ch3=hidden // 2, ch4=hidden, ch5=hidden * 2, hidden=hidden)
        if od:
            self.od_head = OdHead(hidden=hidden, num_classes=num_classes)
        if pos:
            self.pos_head = PosHead(hidden=hidden)
            self.tneck = TLayerNeck(dim=hidden)
        self.prev = None
        if od and decode_cfg:
            self._decoder = DetectDecoder(**decode_cfg)
        self.has_od = od
        self.has_pos = pos

    def _shared(self, x):
        c3, c4, c5 = self.backbone(x)
        return self.neck(c3, c4, c5)

    def forward_od(self, x):
        neck_out = self._shared(x)
        return self.od_head(*neck_out)

    def forward_pos(self, x, prev=None):
        neck_out = self._shared(x)
        if prev is not None:
            temporal_out = self.tneck(prev, neck_out)
        else:
            temporal_out = neck_out
        self.prev = neck_out
        return self.pos_head(*temporal_out)

    def forward(self, x, prev=None):
        neck_out = self._shared(x)

        # pos_head varsa temporal durumu hesapla
        if self.has_pos:
            if prev is not None:
                temporal_out = self.tneck(prev, neck_out)
            else:
                temporal_out = neck_out
            self.prev = neck_out
            pos = self.pos_head(*temporal_out)
        else:
            pos = None

        det = self.od_head(*neck_out) if self.has_od else None

        return det, pos

    def decode(self, det_outs):
        return self._decoder.decode(det_outs) if hasattr(self, "_decoder") else []