from dataclasses import dataclass
from pathlib import Path
import logging
from ..tools.conversion import XML_to_TXT

@dataclass
class DataPack:
    frames_path:Path
    xml_labels_path:Path
    translations_path:Path | None = None
    txt_labels_path: Path | None = None
            
    def __post_init__(self):
        txt_labels_path = self.frames_path.parent / "txt_labels"
        if txt_labels_path.exists():
            logging.info("Veri txt labelları içeriyor.")
            self.txt_labels_path = txt_labels_path
            
    def create_txt_folder_from_xml(self):
        if self.txt_labels_path is None:
            self.txt_labels_path = self.frames_path.parent / "txt_labels"
            self.txt_labels_path.mkdir(exist_ok=True)
            XML_to_TXT(self.xml_labels_path,self.txt_labels_path)
            logging.info("Labels folder oluşturuldu.")