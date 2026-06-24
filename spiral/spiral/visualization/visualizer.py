from ..struct.packets import *
import cv2
import numpy as np

LABEL_TEXT = {
  0 : "Tasit",
  1 : "Insan",
  2 : "UAP",
  3 : "UAI",
  4 : "BLNMYR"
}
LABEL_COLOR_BGR = {
  0 : [191,44,123],
  1 : [0,109,255],
  2 : [255,242,40],
  3 : [40,40,255],
  4 : [0,0,255] 
}
LANDING_STATUS_COLOR_BGR = {
  -1: [255,255,255],
  0: [0,0,255],
  1: [0,255,100]
}
MOTION_STATUS_COLOR_BGR = {
  -1: [130,53,67],
  0: [59,217,217],
  1: [159, 224, 79]
}

class Visualizer:
   
    def __init__(self):
        pass
    
    def visualize_objs(self, result:Result) -> np.ndarray:
        """
        Return:
            annotated image (np.darray):  RGB görsel
        """
        
        anno = result.packet.original_image.copy()
        anno = cv2.cvtColor(anno, cv2.COLOR_BGR2RGB)
        W,H,C = result.packet.original_shape_HWC
        for obj in result.od_objects:
            x1 = obj.top_left_x
            y1 = obj.top_left_y
            x2 = obj.bottom_right_x
            y2 = obj.bottom_right_y
        
            # sınırları güvenli hale getir
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            
            # alpha blended arkaplan (per-ROI overlay)
            alpha = 0.3
            white_patch = np.full_like(anno[y1:y2, x1:x2], LABEL_COLOR_BGR[obj.cls_], dtype=np.uint8)
            anno[y1:y2, x1:x2] = cv2.addWeighted(
                white_patch, alpha,
                anno[y1:y2, x1:x2], 1 - alpha,
                0
            )
            
            # label
            lbl= f"[#{obj.id_}] {LABEL_TEXT[obj.cls_]} {obj.conf}"
            label_rect_gap = max(1, int(min(H, W) / 100))
            font_scale = max(0.35, min(W, H) / 1200)
            font_thickness = max(1, int(min(W, H) / 200))
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_width, text_height), baseline = cv2.getTextSize(lbl, font, font_scale, font_thickness)
            
            # çerçeve
            cv2.rectangle(
                        anno, 
                        (x1, y1),
                        (x2, y2),
                        LANDING_STATUS_COLOR_BGR[obj.landing_status],
                        max(1, int(min(H, W) / 500)))
        
            # label rectangle (bbox üstünde)
            label_rect_top = y1 - text_height - label_rect_gap
            if label_rect_top >= 0:
                cv2.rectangle(
                    anno,
                    (x1, label_rect_top),
                    (x1 + text_width + label_rect_gap, y1),
                    MOTION_STATUS_COLOR_BGR[obj.motion_status],
                    -1
                )
                # label text
                cv2.putText(
                    anno,
                    lbl,
                    (x1 + int(label_rect_gap / 2), y1 - int(label_rect_gap / 2)),
                    font,
                    font_scale,
                    (255,255,255),
                    font_thickness
                )
        anno = cv2.cvtColor(anno, cv2.COLOR_BGR2RGB)
        return anno