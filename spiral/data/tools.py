from dataclasses import dataclass
from pathlib import Path
import logging



@dataclass
class DataPack:
    frames_path:Path
    xml_labels_path:Path
    translations_path:Path
    txt_labels_path: Path | None = None
    
    def create_txt_folder(self):
        self.txt_labels_path = self.frames_path.parent / "txt_labels"
        self.txt_labels_path.mkdir(exist_ok=True)
        logging.info("Labels folder oluşturuldu.")
        
        
def XML_to_TXT(path, dst_path):
    # object name "insan" -> 0
    # object name "tasit" -> 1
    # object name "uap" -> 2
    # object name "uai" -> 3
    # txt file name -> frame_name
    # bbox coors -> normalized_with_frame_size[xmin,ymin,xmax,ymax]
    # total -> frame_xxxx.txt -> 0 0.2 0.4 0.5 0.55\n2 0.4 0.4 0.6 0.9
    pass

    