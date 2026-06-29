from spiral.training.base import SpiMultiModel
from spiral.training.base import DetectDecoder
from spidata.tools.transformations import SpiTransforms
from spidata.tools.tiling import TileGenerator
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
from spiral.utils import setup_logging
import logging

setup_logging(
    level=logging.INFO,
    force=True
)

decoder = DetectDecoder(score_thresh=0.00, fmt="xyxy")
tile_gen = TileGenerator(tile_size=640, pad_if_needed=False)
def get_model(shared_path, odbranch_path,device = "cuda", ):

    model = SpiMultiModel(num_classes=4, od=True, pos=False, stem_type="focus")
    model.to(device)
    
    logging.info("Model oluşturuldu")

    shared_dict = torch.load(str(shared_path), map_location=device)
    odbranch_dict = torch.load(str(odbranch_path), map_location=device)

    try:
        model.backbone.load_state_dict(shared_dict["backbone"])
        logging.info("Model backbone ağırlıkları yüklendi")
    except:
        logging.warning(
            f"Keys: {shared_dict.keys()}"
        )
        raise ValueError("Model backbone ağırlıkları yüklenemedi")
    try:
        model.neck.load_state_dict(shared_dict["neck"])
        logging.info("Model neck ağırlıkları yüklendi")
    except:
        raise ValueError("Model neck ağırlıkları yüklenemedi")
    try:
        model.od_head.load_state_dict(odbranch_dict["od_head"])
        logging.info("Model od head ağırlıkları yüklendi")
    except:
        raise ValueError("Model od head ağırlıkları yüklenemedi")
    model.eval()
    return model

def prepare_input(img):
    img = cv2.resize(img,(1920,1280))
    tiles= []
    tile_positions = []
    cnt = 0
    for tile_data in tile_gen.generate(img):
        tile = tile_data["tile"]
        tile_pos = tile_data["position"]
        tile_positions.append(tile_pos)
        logging.info(f"-[{cnt}]-> Tile shape: {tile.shape}")
        
        tranform = SpiTransforms.default_inference
        tile = tranform(sample={"image" : tile,
                                "translations": None,"objects": None})["image"]
        tiles.append(tile)
        logging.info(f"-[{cnt}]->Tile Final shape: {tile.shape}")
        cnt += 1
        
    logging.info(f"Length of Tiles: {len(tiles)}")
    return tiles, tile_positions


def predict(model, images):
    if not isinstance(images, torch.Tensor):
        images = torch.from_numpy(np.stack(images)).float()

    if images.ndim == 4 and images.shape[-1] == 3:
        images = images.permute(0, 3, 1, 2)

    device = next(model.parameters()).device
    results = []

    with torch.no_grad():
        for image in tqdm(images, desc="Prediction.."):
            # Batch boyutu ekle
            input_image = image.unsqueeze(0).to(device)

            det, pos = model(input_image)
            boxes = decoder.decode(det)
            results.append(boxes)

    return results
        
def visualize(images, results):
    tiles = []
    
    for image, boxes in zip(images, results):
        # Görselleştirme için CPU NumPy'ye çevir
        vis = image.copy()

        h, w = vis.shape[:2]

        print(len(boxes))
        for x1, y1, x2, y2, score, class_idx in boxes:
            xmin = max(0, min(w - 1, int(x1 * w)))
            ymin = max(0, min(h - 1, int(y1 * h)))
            xmax = max(0, min(w - 1, int(x2 * w)))
            ymax = max(0, min(h - 1, int(y2 * h)))
            
            vis = np.ascontiguousarray(vis)
            cv2.rectangle(vis, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            print(x1, y1, x2, y2, score, class_idx)

        tiles.append(vis)
    
    return tiles


# == Execution ==========================
model = get_model(
    odbranch_path="/home/enes/Desktop/spiral_ws/spiral/spiral/training/weights/model_20260627_214712/best_odbranch.pth",
    shared_path="/home/enes/Desktop/spiral_ws/spiral/spiral/training/weights/model_20260627_214712/best_shared.pth",
)

def image_prediction():
    image = Image.open("/home/enes/Desktop/spiral_ws/spidata/spidata/data/2021/TP21_ORNEK1_O1/frames/frame_00000.jpg")
    if not image:
        raise ValueError("İmage yükelenemedi,")
    image = np.array(image)
    logging.info(f"Image original shape: {image.shape}")
    
    tiles, tile_positions = prepare_input(image)

    preds = predict(model, tiles)
    
    annotateds = visualize(tiles,preds)
    collected_image = tile_gen.collect(annotateds, tile_positions)
    plt.imshow(collected_image)
    plt.axis("off")
    plt.show()


image_prediction()