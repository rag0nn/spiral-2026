from ..struct.pack import DataPack
from pathlib import Path

BASE_DIR = Path(__file__).parent

class Registery:
    
    ot1 = DataPack(
        frames_path=BASE_DIR / "2025/OT1/THYZ_2025_Oturum_1",
        xml_labels_path=BASE_DIR / "2025/OT1/THYZ_2025_Oturum_1_etiket/THYZ_2025_Oturum_1/Annotations/Annotations",
        translations_path=BASE_DIR / "2025/OT1/THYZ_2025_Oturum_1_Translation.csv",
    )