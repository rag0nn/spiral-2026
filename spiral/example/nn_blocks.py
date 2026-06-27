from spiral.training.blocks import *
from spiral.training.base import *
from spiral.training.trainer import DetCriterion, PosCriterion, train_step
from spiral.utils import setup_logging

import torch
import torchinfo
import logging

setup_logging()

def show(model, inp_size):
    model.eval()
    x = torch.randn(*inp_size)
    out = model(x)
    logging.info(torchinfo.summary(model, input_size=inp_size))
    logging.info(f"Input:  {inp_size}")
    if isinstance(out, torch.Tensor):
        logging.info(f"Output: {out.shape}")
    else:
        for o in out:
            logging.info(f"Output: {o.shape}")    


if __name__ == "__main__":
    # == Seperable Conv ( Dephh + Point) ==========================
    # logging.info("\n=== SeparableConv ===")
    # model = SeparableConv(3, 64, 3)
    # shape = (1, 3, 64, 64)
    # show(model, shape)
    
    # # == C3K2 Conv Block ==========================
    # logging.info("\n=== C3K2 ===")
    # model = C3K2(64, 64, n=2)
    # shape = (1, 64, 32, 32)
    # show(model, shape)
    
    # logging.info("\n=== Backbone ===")
    # model = Backbone(3)
    # shape = (1, 3, 512, 512)
    # show(model, shape)
    
    # logging.info("\n=== Neck ===")
    # model = Neck()
    # shape = (
    #     torch.randn(1, 64, 512, 512),
    #     torch.randn(1, 128, 256, 256),
    #     torch.randn(1, 256, 128, 128),
    # )

    # outputs = model(*shape)

    # logging.info(torchinfo.summary(model))
    # for i, o in enumerate(outputs):
    #     logging.info(f"Output {i}: {o.shape}")
    
    # logging.info("\n=== SpiMulti (temporal=False) ===")
    # model = SpiMulti(temporal=False)
    # inp = torch.randn(1, 3, 512, 512)
    # outputs = model(inp)

    # logging.info(torchinfo.summary(model))
    # for i, o in enumerate(outputs):
    #     logging.info(f"Output {i}: {o.shape}")

    # logging.info("\n=== OdHead ===")
    # model = OdHead(hidden=128, num_classes=80)
    # inp = (torch.randn(1, 128, 32, 32),
    #        torch.randn(1, 128, 16, 16),
    #        torch.randn(1, 128, 8, 8))
    # outs = model(*inp)
    # for i, (cls, reg, ctr) in enumerate(outs):
    #     logging.info(f"Scale {i}: cls={cls.shape} reg={reg.shape} ctr={ctr.shape}")

    logging.info("\n=== SpiMulti (full pipeline) ===")
    model = SpiMultiModel(temporal=False, num_classes=4)
    inp = torch.randn(1, 3, 512, 512)
    det, pos = model(inp)
    for i, (cls, reg, ctr) in enumerate(det):
        logging.info(f"Det scale {i}: cls={cls.shape} reg={reg.shape} ctr={ctr.shape}")
    logging.info(f"Pos: {pos.shape}")

    logging.info("\n=== DetectDecoder (xyxy) ===")
    decoder = DetectDecoder(score_thresh=0.5, fmt="xyxy")
    boxes = decoder.decode(det)
    logging.info(f"Boxes xyxy: {len(boxes)}")
    if boxes:
        logging.info(f"First: {boxes[0]}")

    logging.info("\n=== DetectDecoder (xywh) ===")
    decoder = DetectDecoder(score_thresh=0.5, fmt="xywh")
    boxes = decoder.decode(det)
    logging.info(f"Boxes xywh: {len(boxes)}")
    if boxes:
        logging.info(f"First: {boxes[0]}")

    logging.info("\n=== Training (single-frame) ===")
    model = SpiMultiModel(temporal=False, num_classes=4)
    crit_det = DetCriterion(num_classes=4)
    crit_pos = PosCriterion()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    x = torch.randn(2, 3, 128, 128)
    batch = (
        x,
        [torch.tensor([[0.078, 0.078, 0.391, 0.391],
                        [0.469, 0.469, 0.703, 0.703]]),
         torch.tensor([[0.156, 0.156, 0.312, 0.312]])],
        [torch.tensor([1, 2]), torch.tensor([3])],
        torch.randn(2, 3),
    )
    losses = train_step(model, batch, crit_det, crit_pos, opt, temporal=False)
    for k, v in losses.items():
        logging.info(f"  {k}: {v.item():.4f}")

    logging.info("\n=== Training (temporal pair) ===")
    model = SpiMultiModel(temporal=True, num_classes=4)
    crit_det = DetCriterion(num_classes=4)
    crit_pos = PosCriterion()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    x_t = torch.randn(2, 3, 128, 128)
    x_t1 = torch.randn(2, 3, 128, 128)
    batch = (
        x_t, x_t1,
        [torch.tensor([[0.078, 0.078, 0.391, 0.391],
                        [0.469, 0.469, 0.703, 0.703]]),
         torch.tensor([[0.156, 0.156, 0.312, 0.312]])],
        [torch.tensor([1, 2]), torch.tensor([3])],
        torch.randn(2, 3),
    )
    losses = train_step(model, batch, crit_det, crit_pos, opt, temporal=True)
    for k, v in losses.items():
        logging.info(f"  {k}: {v.item():.4f}")