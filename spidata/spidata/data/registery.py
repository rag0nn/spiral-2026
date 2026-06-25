from ..struct.pack import DataPack
from pathlib import Path

BASE_DIR = Path(__file__).parent

class Registery:
    
    ot25_1 = DataPack(
        frames_path=BASE_DIR / "2025/OT1/THYZ_2025_Oturum_1",
        xml_labels_path=BASE_DIR / "2025/OT1/THYZ_2025_Oturum_1_etiket/THYZ_2025_Oturum_1/Annotations/Annotations",
        translations_path=BASE_DIR / "2025/OT1/THYZ_2025_Oturum_1_Translation.csv",
    )
    
    ot25_2 = DataPack(
        frames_path=BASE_DIR / "2025/OT2/THYZ_2025_Oturum_2",
        xml_labels_path=BASE_DIR / "2025/OT2/THYZ_2025_Oturum_2_etiket/THYZ_2025_Oturum_2/Annotations/Annotations",
        translations_path=BASE_DIR / "2025/OT2/THYZ_2025_Oturum_2_Translation.csv",
    )
    
    ot25_3 = DataPack(
        frames_path=BASE_DIR / "2025/OT3/THYZ_2025_Oturum_3",
        xml_labels_path=BASE_DIR / "2025/OT3/THYZ_2025_Oturum_3_etiket/THYZ_2025_Oturum_3/Annotations/Annotations",
        translations_path=BASE_DIR / "2025/OT3/THYZ_2025_Oturum_3_Translation.csv",
    )
    
    ot25_4 = DataPack(
        frames_path=BASE_DIR / "2025/OT4/THYZ_2025_Oturum_4",
        xml_labels_path=BASE_DIR / "2025/OT4/THYZ_2025_Oturum_4_etiket/THYZ_2025_Oturum_4/Annotations/Annotations",
        translations_path=BASE_DIR / "2025/OT4/THYZ_2025_Oturum_4_Translation.csv",
    )