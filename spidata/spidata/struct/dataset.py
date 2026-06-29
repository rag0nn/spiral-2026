import os
import cv2
import pandas as pd
import numpy as np
import random
from pathlib import Path
from torch.utils.data import Dataset

class SpiDataset(Dataset):
    def __init__(self, datapack, objects=None, filter_missing_translations=True, transform=None):
        """
        Spidata için PyTorch Dataset sınıfı.
        
        Parametreler:
            datapack (DataPack): Görüntü kareleri (frames), translasyonlar vb. yolları içeren veri paketi.
            objects (dict, isteğe bağlı): Kare adını (örneğin 'frame_000000') nesne demetleri/listeleri
                                          içeren bir listeye veya numpy dizisine eşleyen önceden yüklenmiş nesneler sözlüğü.
                                          Çalışma zamanında doğrudan txt veya xml dosyalarından okunmayacağı için buradan aktarılabilir.
            filter_missing_translations (bool): True ise translasyon bilgisi eksik olan kareleri filtreler.
            transform (callable, isteğe bağlı): Örnek üzerinde uygulanacak isteğe bağlı dönüşüm işlemi.
        """
        self.datapack = datapack
        self.objects = objects if objects is not None else {}
        self.filter_missing_translations = filter_missing_translations
        self.transform = transform

        # .txt etiketlerinin varligindan emin ol, yoksa XML'den üret
        if hasattr(self.datapack, "create_txt_folder_from_xml"):
            self.datapack.create_txt_folder_from_xml()

        # Translasyonları yükle
        self.translations = {}
        if datapack.translations_path and Path(datapack.translations_path).exists():
            df = pd.read_csv(datapack.translations_path)
            for _, row in df.iterrows():
                frame_name = str(row['frame_numbers']).strip()
                # Eger CSV'de uzanti varsa temizle (örn. frame_000000.webp -> frame_000000)
                frame_name = os.path.splitext(frame_name)[0]
                x = float(row['translation_x'])
                y = float(row['translation_y'])
                z = float(row['translation_z'])
                self.translations[frame_name] = np.array((x, y, z), dtype=np.float32)

        # Kareleri yükle
        self.frame_paths = []
        if datapack.frames_path and Path(datapack.frames_path).exists():
            # OT1 dizinindeki webp resimlerini ve diger standart formatlari destekle
            extensions = ('*.webp', '*.png', '*.jpg', '*.jpeg')
            files = []
            for ext in extensions:
                files.extend(Path(datapack.frames_path).glob(ext))
            
            # Belirli bir sira korumak için dosyalari sirala
            files = sorted(files, key=lambda p: p.name)
            
            for fp in files:
                frame_name = fp.stem  # örn. 'frame_000000'
                has_trans = frame_name in self.translations and not np.isnan(self.translations[frame_name]).any()
                
                if self.filter_missing_translations and not has_trans:
                    continue
                    
                self.frame_paths.append((frame_name, fp))

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        frame_name, fp = self.frame_paths[idx]
        
        # Resmi yükle
        image = cv2.imread(str(fp))
        if image is not None:
            # RGB formatina dönüstür (PyTorch veri kümeleri için standart)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # Resim yükleme basarisiz olursa yedek olarak bos resim olustur
            image = np.zeros((224, 224, 3), dtype=np.uint8)
            raise ValueError(f"Bu path'ta bir dosya bulunamadı: {fp}")

        # Translasyon bilgisini al
        translation = self.translations.get(frame_name)
        if translation is None:
            translation = np.array((np.nan, np.nan, np.nan), dtype=np.float32)

        # Ilgili kare için nesneleri aynı isimdeki .txt dosyasından oku
        objects_list = []
        txt_labels_path = self.datapack.txt_labels_path
        if txt_labels_path is None:
            txt_labels_path = Path(self.datapack.frames_path).parent / "txt_labels"

        if txt_labels_path and Path(txt_labels_path).exists():
            txt_path = Path(txt_labels_path) / f"{frame_name}.txt"
            if txt_path.exists():
                with open(txt_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) == 5:
                            cls_id = int(parts[0])
                            x = float(parts[1])
                            y = float(parts[2])
                            w = float(parts[3])
                            h = float(parts[4])
                            objects_list.append((cls_id, x, y, w, h))

        if len(objects_list) > 0:
            # Liste veya demet yapisini numpy demet dizisine dönüstür
            tuple_list = [tuple(obj) for obj in objects_list]
            objects_arr = np.array(tuple_list, dtype=object)
        else:
            objects_arr = np.array([], dtype=object)

        sample = {
            "image": image,
            "translations": translation,
            "objects": objects_arr
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_random_sample(self):
        if len(self) == 0:
            raise ValueError("Veri kümesi bos.")
        idx = random.randint(0, len(self) - 1)
        return self[idx]