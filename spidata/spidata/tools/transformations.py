import random
import numpy as np
import albumentations as A

class SpiTransformsWrapper:
    def __init__(self, albumentations_pipeline):
        """
        Albumentations dönüşümlerini ve translasyon/etiket güncellemelerini sarmalayan sınıf.
        """
        self.pipeline = albumentations_pipeline

    def __call__(self, sample):
        image = sample["image"]
        translations = sample["translations"].copy()
        objects = sample["objects"]

        # Albumentations için etiketleri ve sınır kutularını hazırla ([cx, cy, w, h] normalized)
        bboxes = []
        class_labels = []
        for obj in objects:
            bboxes.append([float(obj[1]), float(obj[2]), float(obj[3]), float(obj[4])])
            class_labels.append(int(obj[0]))

        # Dönüşümleri ReplayCompose ile uygula
        transformed = self.pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
        
        new_image = transformed["image"]
        new_bboxes = transformed["bboxes"]
        new_class_labels = transformed["class_labels"]
        
        # Dönüştürülmüş nesneleri yeniden oluştur ([cx, cy, w, h] normalized)
        new_objects_list = []
        for i in range(len(new_bboxes)):
            new_objects_list.append((
                new_class_labels[i],
                new_bboxes[i][0], new_bboxes[i][1], new_bboxes[i][2], new_bboxes[i][3]
            ))
            
        if len(new_objects_list) > 0:
            new_objects = np.array(new_objects_list, dtype=object)
        else:
            new_objects = np.array([], dtype=object)

        # Translasyon değerlerini uygulanan dönüşümlere göre güncelle
        replay = transformed.get("replay")
        if replay and replay.get("applied", False):
            for t in replay.get("transforms", []):
                if not t.get("applied", False):
                    continue
                
                classname = t.get("__class_fullname__")
                params = t.get("params", {})
                
                # Yatay çevirme durumu
                if classname == "HorizontalFlip":
                    translations[0] = -translations[0]
                
                # Dikey çevirme durumu
                elif classname == "VerticalFlip":
                    translations[1] = -translations[1]
                
                # Döndürme veya Affine dönüşüm durumları
                elif classname in ("Rotate", "ShiftScaleRotate", "Affine") and "matrix" in params:
                    matrix = params["matrix"]
                    if matrix is not None and isinstance(matrix, np.ndarray):
                        M = matrix[:2, :2]
                        col0 = M[:, 0]
                        col1 = M[:, 1]
                        scale_x = np.linalg.norm(col0)
                        scale_y = np.linalg.norm(col1)
                        if scale_x > 1e-8:
                            col0 = col0 / scale_x
                        if scale_y > 1e-8:
                            col1 = col1 / scale_y
                        R_pure = np.stack([col0, col1], axis=1)
                        
                        # Translasyon x ve y değerlerini döndür
                        xy = np.array([translations[0], translations[1]], dtype=np.float32)
                        new_xy = R_pure @ xy
                        translations[0] = new_xy[0]
                        translations[1] = new_xy[1]

        return {
            "image": new_image,
            "translations": translations,
            "objects": new_objects
        }

class SpiInferenceTransform:
    def __init__(self, size=(640,640)):
        """
        Çıkarım (inference) aşamasında sadece görseli boyutlandıran sınıf.
        Diğer değerlere (translasyon, nesneler vb.) kesinlikle dokunulmaz.
        """
        self.transform = A.Resize(height=size[0], width=size[1])

    def __call__(self, sample):
        image = sample["image"]
        transformed = self.transform(image=image)
        new_image = transformed["image"]
        return {
            "image": new_image,
            "translations": sample["translations"],
            "objects": sample["objects"]
        }

class RandomCropOrResize(A.DualTransform):
    def __init__(self, size=640, p=1.0):
        super().__init__(p=p)
        self.size = size

    def __call__(self, force_apply=False, **data):
        image = data["image"]
        h, w = image.shape[:2]

        if h >= self.size and w >= self.size:
            t = A.RandomCrop(self.size, self.size)
        else:
            t = A.Resize(self.size, self.size)

        return t(force_apply=True, **data)

class SpiTransforms:
    # Sınır kutusu ayarları
    _bbox_params = A.BboxParams(format='yolo', label_fields=['class_labels'])

    # Varsayılan eğitim dönüşümleri
    default_training = SpiTransformsWrapper(
        A.ReplayCompose([
            # A.RandomCrop(height=640, width=640),
            RandomCropOrResize(640),
            # A.RandomResizedCrop(height=640, width=640, scale=(0.5, 1.0), ratio=(0.8, 1.2), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=30, p=0.5),
            A.Affine(scale=(0.8, 1.2), translate_percent=(-0.1, 0.1), shear=(-10, 10), rotate=(0, 0), p=0.5),
            A.Perspective(scale=(0.05, 0.2), p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.2),
            A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 1.0), p=0.2),
            A.Equalize(p=0.1),
            A.AutoContrast(p=0.1),
        ], bbox_params=_bbox_params)
    )

    # Varsayılan çıkarım dönüşümleri
    default_inference = SpiInferenceTransform((640, 640))