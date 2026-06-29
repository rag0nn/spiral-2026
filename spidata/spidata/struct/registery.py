from .pack import DataPack
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent / "data"

class Registery:
    
    # == Teknofest 2025 ==========================
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
    
    # == Open Sources ==========================
    os_afo = DataPack(
        frames_path=BASE_DIR / "OPENSRC/AFO/frames",
        txt_labels_path=BASE_DIR / "OPENSRC/AFO/txt_labels",
        translations_path=None
    )
    
    os_lacmus = DataPack(
        frames_path=BASE_DIR / "OPENSRC/lacmus/frames",
        txt_labels_path=BASE_DIR / "OPENSRC/lacmus/txt_labels",
        translations_path=None
    )
    
    os_semantic_drone = DataPack(
        frames_path=BASE_DIR / "OPENSRC/semantic_drone/frames",
        txt_labels_path=BASE_DIR / "OPENSRC/semantic_drone/txt_labels",
        translations_path=None
    )
    
    os_SOAP = DataPack(
        frames_path=BASE_DIR / "OPENSRC/SOAP/frames",
        txt_labels_path=BASE_DIR / "OPENSRC/SOAP/txt_labels",
        translations_path=None
    )
    
    # == Teknofest 2024 ==========================
    ot24_6 = DataPack(
        frames_path=BASE_DIR / "2024/TP24_O6_0-390/frames",
        txt_labels_path=BASE_DIR / "2024/TP24_O6_0-390/txt_labels",
        translations_path=None
    )

    # == Teknofest 2022 ==========================
    ot22_2_1 = DataPack(
        frames_path=BASE_DIR / "2022/T22_O2_1/frames",
        txt_labels_path=BASE_DIR / "2022/T22_O2_1/txt_labels",
        translations_path=None
    )
    
    ot22_2_2 = DataPack(
        frames_path=BASE_DIR / "2022/T22_O2_2/frames",
        txt_labels_path=BASE_DIR / "2022/T22_O2_2/txt_labels",
        translations_path=None
    )
    
    ot22_2_3 = DataPack(
        frames_path=BASE_DIR / "2022/T22_O2_3/frames",
        txt_labels_path=BASE_DIR / "2022/T22_O2_3/txt_labels",
        translations_path=None
    )
    
    ot22_2_4 = DataPack(
        frames_path=BASE_DIR / "2022/T22_02_4/frames",
        txt_labels_path=BASE_DIR / "2022/TP22_02_4/txt_labels",
        translations_path=None
    )
    
    ot22_2_5 = DataPack(
        frames_path=BASE_DIR / "2022/T22_O2_5/frames",
        txt_labels_path=BASE_DIR / "2022/TP22_O2_5/txt_labels",
        translations_path=None
    )
    
    ot22_1_1 = DataPack(
        frames_path=BASE_DIR / "2022/TP22_O1_1_1/frames",
        txt_labels_path=BASE_DIR / "2022/TP22_O1_1_1/txt_labels",
        translations_path=None
    )
    
    
    ot22_1_2 = DataPack(
        frames_path=BASE_DIR / "2022/TP22_O1_2/frames",
        txt_labels_path=BASE_DIR / "2022/TP22_O1_2/txt_labels",
        translations_path=None
    )
    
    ot22_3_1 = DataPack(
        frames_path=BASE_DIR / "2022/TP22_O3_1/frames",
        txt_labels_path=BASE_DIR / "2022/TP22_O3_1/txt_labels",
        translations_path=None
    )
    
    ot22_3_2 = DataPack(
        frames_path=BASE_DIR / "2022/TP22_O3_2/frames",
        txt_labels_path=BASE_DIR / "2022/TP22_O3_2/txt_labels",
        translations_path=None
    )
    
    ot22_3_3 = DataPack(
        frames_path=BASE_DIR / "2022/TP22_O3_3/frames",
        txt_labels_path=BASE_DIR / "2022/TP22_O3_3/txt_labels",
        translations_path=None
    )
    
    ot22_4_1_1 = DataPack(
        frames_path=BASE_DIR / "2022/TP22_O4_1_1/frames",
        txt_labels_path=BASE_DIR / "2022/TP22_O4_1_1/txt_labels",
        translations_path=None
    )

    ot22_4_1_2 = DataPack(
        frames_path=BASE_DIR / "2022/TP22_O4_1_2_AREAFUL/frames",
        txt_labels_path=BASE_DIR / "2022/TP22_O4_1_2_AREAFUL/txt_labels",
        translations_path=None
    )
    
    ot22_4_2 = DataPack(
        frames_path=BASE_DIR / "2022/TP22_O4_2/frames",
        txt_labels_path=BASE_DIR / "2022/TP22_O4_2/txt_labels",
        translations_path=None
    )
    # == Teknofest 2021 ==========================
    ot21_1 = DataPack(
        frames_path=BASE_DIR / "2021/T21_O1/frames",
        txt_labels_path=BASE_DIR / "2021/T21_O1/txt_labels",
        translations_path=None
    )
    
    ot21_p = DataPack(
        frames_path=BASE_DIR / "2021/TP21_ORNEK1_O1/frames",
        txt_labels_path=BASE_DIR / "2021/TP21_ORNEK1_O1/txt_labels",
        translations_path=None
    )
    
    @classmethod
    def get_all(cls) -> list[DataPack]:
        return [
            # == Teknofest 2025 ==
            cls.ot25_1,
            cls.ot25_2,
            cls.ot25_3,
            cls.ot25_4,

            # == Open Sources ==
            cls.os_afo,
            cls.os_lacmus,
            cls.os_semantic_drone,
            cls.os_SOAP,

            # == Teknofest 2024 ==
            cls.ot24_6,

            # == Teknofest 2022 ==
            cls.ot22_2_1,
            cls.ot22_2_2,
            cls.ot22_2_3,
            cls.ot22_2_4,
            cls.ot22_2_5,
            cls.ot22_1_1,
            cls.ot22_1_2,
            cls.ot22_3_1,
            cls.ot22_3_2,
            cls.ot22_3_3,
            cls.ot22_4_1_1,
            cls.ot22_4_1_2,
            cls.ot22_4_2,

            # == Teknofest 2021 ==
            cls.ot21_1,
            cls.ot21_p,
        ]