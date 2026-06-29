from spidata.tools.conversion import xywhc_to_xyxyc, XML_to_TXT
from pathlib import Path
from tqdm import tqdm
import shutil

# for pth in tqdm(arr):
#     pth = pth
#     dst_path:Path = pth.parent / "lbls_xyxyc"
#     xywhc_to_xyxyc(pth, dst_path)
    
# arr = [pth for pth in Path("/home/enes/Desktop/datasets").iterdir()]
# dst_path = Path("/home/enes/Desktop/target")
# dst_path.mkdir(exist_ok=True)
    
# def squueze():
#     for pth in tqdm(arr):
#         images_path = pth / "detect/images/train"
#         labels_path = pth / "detect/labels/lbls_xyxyc"
#         t_im_path = dst_path / pth.name / "frames"
#         t_lbl_path = dst_path / pth.name / "labels"

#         t_im_path.mkdir(exist_ok=True,parents=True)
#         t_lbl_path.mkdir(exist_ok=True,parents=True)
        
#         for im_path in tqdm(images_path.iterdir()):
#             target_path = t_im_path / im_path.name
#             shutil.move(im_path, target_path)
        
#         for lbl_path in tqdm(labels_path.iterdir()):
#             target_path = t_lbl_path / lbl_path.name
#             shutil.move(lbl_path, target_path)
         
# src_path = Path("/home/enes/Desktop/spiral_ws/spidata/spidata/data/2025/OT2/THYZ_2025_Oturum_2_etiket/THYZ_2025_Oturum_2/Annotations/Annotations")     
# dst = Path("/home/enes/Desktop/spiral_ws/spidata/spidata/data/2025/OT2")
# XML_to_TXT(src_path, dst)