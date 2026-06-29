from pathlib import Path
import xml.etree.ElementTree as ET
from tqdm import tqdm

# def XML_to_TXT(path, dst_folder_path):
#     CLASS_MAP = {
#         "taşıt": 0,
#         "tasit": 0,
#         "Taşıt" : 0,
#         "insan": 1,
#         "ınsan": 1,
#         "İnsan" : 1,
#         "uap": 2,
#         "UAP" : 2,
#         "uai": 3,
#         "UAI" : 3,
#         "UAİ" : 3,
#     }
#     dst = Path(dst_folder_path)
#     dst.mkdir(parents=True, exist_ok=True)

#     src = Path(path)
#     xml_files = [src] if src.is_file() else list(src.glob("*.xml"))

#     for xml_path in tqdm(xml_files):
#         tree = ET.parse(xml_path)
#         root = tree.getroot()

#         width = float(root.findtext("size/width"))
#         height = float(root.findtext("size/height"))

#         txt_path = dst / f"{xml_path.stem}.txt"
#         with open(txt_path, "w") as f:
#             for obj in root.findall("object"):
#                 name = obj.findtext("name")
#                 class_id = CLASS_MAP.get(name.strip())
#                 if class_id is None:
#                     raise ValueError(f"Bilinmeyen class name {name} {name.lower()} {type(name)} {xml_path}")

#                 xmin = float(obj.findtext("bndbox/xmin"))
#                 ymin = float(obj.findtext("bndbox/ymin"))
#                 xmax = float(obj.findtext("bndbox/xmax"))
#                 ymax = float(obj.findtext("bndbox/ymax"))

#                 xmin_n = xmin / width
#                 ymin_n = ymin / height
#                 xmax_n = xmax / width
#                 ymax_n = ymax / height

#                 f.write(f"{class_id} {xmin_n:.6f} {ymin_n:.6f} {xmax_n:.6f} {ymax_n:.6f}\n")

def XML_to_TXT(path, dst_folder_path):
    CLASS_MAP = {
        "taşıt": 0,
        "tasit": 0,
        "Taşıt": 0,
        "insan": 1,
        "ınsan": 1,
        "İnsan": 1,
        "uap": 2,
        "UAP": 2,
        "uai": 3,
        "UAI": 3,
        "UAİ": 3,
    }

    dst = Path(dst_folder_path)
    dst.mkdir(parents=True, exist_ok=True)

    src = Path(path)
    xml_files = [src] if src.is_file() else list(src.glob("*.xml"))

    for xml_path in tqdm(xml_files):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        width = float(root.findtext("size/width"))
        height = float(root.findtext("size/height"))

        txt_path = dst / f"{xml_path.stem}.txt"

        with open(txt_path, "w") as f:
            for obj in root.findall("object"):
                name = obj.findtext("name")
                class_id = CLASS_MAP.get(name.strip())

                if class_id is None:
                    raise ValueError(f"Bilinmeyen class name {name} {xml_path}")

                xmin = float(obj.findtext("bndbox/xmin"))
                ymin = float(obj.findtext("bndbox/ymin"))
                xmax = float(obj.findtext("bndbox/xmax"))
                ymax = float(obj.findtext("bndbox/ymax"))

                # ---- xywh (pixel) ----
                cx = (xmin + xmax) / 2
                cy = (ymin + ymax) / 2
                w = (xmax - xmin)
                h = (ymax - ymin)

                # ---- normalize ----
                cx_n = cx / width
                cy_n = cy / height
                w_n = w / width
                h_n = h / height

                f.write(f"{class_id} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}\n")
    
def xywhc_to_xyxyc(path: Path, dst: Path):
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    src = Path(path)
    txt_files = [src] if src.is_file() else list(src.glob("*.txt"))

    for txt_path in txt_files:
        lines_out = []
        with open(txt_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls_id = parts[0]
                cx, cy, w, h = map(float, parts[1:5])
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
                lines_out.append(f"{cls_id} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f}\n")

        out_path = dst / txt_path.name
        with open(out_path, "w") as f:
            f.writelines(lines_out)
        


def xyxyc_to_xywhc(path: Path, dst: Path):
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    src = Path(path)
    txt_files = [src] if src.is_file() else list(src.glob("*.txt"))

    for txt_path in txt_files:
        lines_out = []

        with open(txt_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 5:
                    continue

                cls_id = parts[0]
                x1, y1, x2, y2 = map(float, parts[1:5])

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1

                lines_out.append(
                    f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
                )

        out_path = dst / txt_path.name
        with open(out_path, "w") as f:
            f.writelines(lines_out)