import time
import torch
import torchinfo
import logging

from spiral.training.base import SpiMultiModel, DetectDecoder
from spiral.utils import setup_logging

setup_logging(force=True)

INPUT_SHAPE = (1, 3, 640,640)
NUM_CLASSES = 4
WARMUP_RUNS = 5
MEASURE_RUNS = 20

def _make_input(device="cpu"):
    return torch.randn(*INPUT_SHAPE).to(device)

def _measure_inference(model, device="cpu", use_decode=False, decoder=None):
    """
    Modelin tek bir forward (ve opsiyonel decode) adiminin ortalama süresini ölçer.
    GPU varsa CUDA event ile, yoksa perf_counter ile ölçüm yapilir.
    """
    model.eval()
    x = _make_input(device)
    use_cuda = device == "cuda" and torch.cuda.is_available()

    with torch.no_grad():
        # Isınma turları (GPU caching vs. hariç tutmak için)
        for _ in range(WARMUP_RUNS):
            out = model(x)

        # Ölçüm
        if use_cuda:
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_evt.record()
            for _ in range(MEASURE_RUNS):
                det, pos = model(x)
                if use_decode and decoder is not None and det is not None:
                    decoder.decode(det)
            end_evt.record()
            torch.cuda.synchronize()
            elapsed_ms = start_evt.elapsed_time(end_evt) / MEASURE_RUNS
        else:
            t0 = time.perf_counter()
            for _ in range(MEASURE_RUNS):
                det, pos = model(x)
                if use_decode and decoder is not None and det is not None:
                    decoder.decode(det)
            elapsed_ms = (time.perf_counter() - t0) / MEASURE_RUNS * 1000

    return elapsed_ms

def od_summary(device="cpu"):
    """
    Sadece Object Detection (od=True, pos=False) modelinin:
    - torchinfo summary
    - inference time ölçümü (decode dahil)
    """
    logging.info("=== OD-Only Model Summary ===")
    model = SpiMultiModel(num_classes=NUM_CLASSES, od=True, pos=False, stem_type="focus")
    model.to(device).eval()

    torchinfo.summary(
        model,
        input_size=INPUT_SHAPE,
        device=device,
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
        depth=4,
    )

    # Örnek forward çıktıları
    with torch.no_grad():
        det, _ = model(_make_input(device))
    logging.info("OD forward çıktıları:")
    for i, (cls, reg, ctr) in enumerate(det):
        logging.info(f"  Scale {i}: cls={tuple(cls.shape)}  reg={tuple(reg.shape)}  ctr={tuple(ctr.shape)}")

    # Decoder ile inference time
    decoder = DetectDecoder(score_thresh=0.3, fmt="xyxy")
    elapsed = _measure_inference(model, device, use_decode=True, decoder=decoder)
    logging.info(f"OD Inference Time (decode dahil, {MEASURE_RUNS} run ort.): {elapsed:.2f} ms\n")

def pos_summary(device="cpu"):
    """
    Sadece Pose/Translation (od=False, pos=True) modelinin:
    - torchinfo summary
    - inference time ölçümü
    """
    logging.info("=== Pos-Only Model Summary ===")
    model = SpiMultiModel(num_classes=NUM_CLASSES, od=False, pos=True, stem_type="focus")
    model.to(device).eval()

    torchinfo.summary(
        model,
        input_size=INPUT_SHAPE,
        device=device,
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
        depth=4,
    )

    # Örnek forward çıktısı
    with torch.no_grad():
        _, pos = model(_make_input(device))
    logging.info(f"Pos forward çıktısı: {tuple(pos.shape)}")

    elapsed = _measure_inference(model, device)
    logging.info(f"Pos Inference Time ({MEASURE_RUNS} run ort.): {elapsed:.2f} ms\n")

def full_summary(device="cpu"):
    """
    Tam model (od=True, pos=True) summary ve inference time.
    OD çıktıları decode ile de gösterilir.
    """
    logging.info("=== Full Model Summary (OD + Pos) ===")
    model = SpiMultiModel(num_classes=NUM_CLASSES, od=True, pos=True, stem_type="focus")
    model.to(device).eval()

    torchinfo.summary(
        model,
        input_size=INPUT_SHAPE,
        device=device,
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
        depth=4,
    )

    # Örnek forward çıktıları
    decoder = DetectDecoder(score_thresh=0.3, fmt="xyxy")
    with torch.no_grad():
        det, pos = model(_make_input(device))
    logging.info("Full forward çıktıları:")
    for i, (cls, reg, ctr) in enumerate(det):
        logging.info(f"  Det Scale {i}: cls={tuple(cls.shape)}  reg={tuple(reg.shape)}  ctr={tuple(ctr.shape)}")
    logging.info(f"  Pos: {tuple(pos.shape)}")
    boxes = decoder.decode(det)
    logging.info(f"  Decode sonucu (score_thresh=0.3): {len(boxes)} kutu")

    elapsed = _measure_inference(model, device, use_decode=True, decoder=decoder)
    logging.info(f"Full Inference Time (decode dahil, {MEASURE_RUNS} run ort.): {elapsed:.2f} ms\n")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Kullanilan cihaz: {device}\n")

    # od_summary(device)
    # pos_summary(device)
    full_summary(device)