import numpy as np


class TileGenerator:
    """
    Büyük görüntüleri (2K/4K) sabit boyutlu alt tile'lara böler.
    Her tile için bounding box'ları kırpıp tile koordinat sistemine
    normalize eder; tile ile örtüşmeyen bbox'ları atar.

    Parametreler:
        tile_size (int): Üretilecek kare tile kenar uzunluğu (örn. 640).
        overlap (int): Komşu tile'lar arasındaki piksel örtüşmesi (overlap < tile_size olmalı).
        pad_if_needed (bool): True ise görüntü tile boyutundan küçükse sıfırla doldurur.
    """

    def __init__(self, tile_size: int = 640, overlap: int = 0, pad_if_needed: bool = True):
        if overlap >= tile_size:
            raise ValueError(f"overlap ({overlap}) must be less than tile_size ({tile_size})")
        self.tile_size = tile_size
        self.overlap = overlap
        self.pad_if_needed = pad_if_needed

    def generate(self, image: np.ndarray, objects: np.ndarray = None) -> list[dict]:
        """
        Görüntüyü tile'lara böler ve her tile için objects verisini kırpar.

        Parametreler:
            image (np.ndarray): RGB görüntü, shape (H, W, C).
            objects (np.ndarray, opsiyonel): (N, 5) array, her satır
                (class_id, x1, y1, x2, y2) — normalized [0,1].

        Dönüş:
            tiles_data (list[dict]): Her öğe:
                - "tile" (np.ndarray): (tile_size, tile_size, C)
                - "objects" (np.ndarray): Tile'a ait kırpılmış bbox'lar
                - "position" (tuple[int, int]): (x, y) sol-üst köşe
        """
        H, W = image.shape[:2]
        ts = self.tile_size
        step = ts - self.overlap

        if self.pad_if_needed:
            image = self._pad(image)
            H, W = image.shape[:2]

        tiles_data = []
        for y in range(0, H - ts + 1, step):
            for x in range(0, W - ts + 1, step):
                tile = image[y: y + ts, x: x + ts]
                tile_objs = self._crop_objects(objects, W, H, ts, x, y)
                tiles_data.append({
                    "tile": tile,
                    "objects": tile_objs,
                    "position": (x, y),
                })

        if len(tiles_data) == 0:
            tile = image[:ts, :ts]
            tile_objs = self._crop_objects(objects, W, H, ts, 0, 0)
            tiles_data.append({
                "tile": tile,
                "objects": tile_objs,
                "position": (0, 0),
            })

        return tiles_data

    def _crop_objects(self, objects: np.ndarray, img_w: int, img_h: int,
                      ts: int, tx: int, ty: int) -> np.ndarray:
        """
        Nesneleri tile bölgesine kırpar.
        Tile içinde kalmayan bbox'lar atılır, kısmen kalanlar kırpılır.
        """
        if objects is None or len(objects) == 0:
            return np.array([], dtype=object)

        tx2 = min(tx + ts, img_w)
        ty2 = min(ty + ts, img_h)

        clipped = []
        for obj in objects:
            cls_id = int(obj[0])
            x1, y1, x2, y2 = float(obj[1]), float(obj[2]), float(obj[3]), float(obj[4])

            ax1, ay1 = x1 * img_w, y1 * img_h
            ax2, ay2 = x2 * img_w, y2 * img_h

            ix1 = max(ax1, tx)
            iy1 = max(ay1, ty)
            ix2 = min(ax2, tx2)
            iy2 = min(ay2, ty2)

            if ix2 > ix1 and iy2 > iy1:
                nx1 = (ix1 - tx) / ts
                ny1 = (iy1 - ty) / ts
                nx2 = (ix2 - tx) / ts
                ny2 = (iy2 - ty) / ts
                clipped.append((cls_id, nx1, ny1, nx2, ny2))

        if len(clipped) > 0:
            return np.array(clipped, dtype=object)

        return np.array([], dtype=object)

    def _pad(self, image: np.ndarray) -> np.ndarray:
        """Görüntüyü tile boyutunun katına tamamlar (sıfır padding)."""
        H, W = image.shape[:2]
        ts = self.tile_size
        step = ts - self.overlap

        cols = max(1, int(np.ceil((W - ts) / step)) + 1) if W >= ts else 1
        rows = max(1, int(np.ceil((H - ts) / step)) + 1) if H >= ts else 1

        target_W = (cols - 1) * step + ts
        target_H = (rows - 1) * step + ts

        pad_w = max(0, target_W - W)
        pad_h = max(0, target_H - H)

        if pad_w == 0 and pad_h == 0:
            return image

        if image.ndim == 3:
            padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
        else:
            padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode="constant")

        return padded

    def collect(self, images: list[np.ndarray], positions: list[tuple[int, int]]) -> np.ndarray:
        if len(images) == 0:
            raise ValueError("No images to collect")

        H, W = 0, 0
        for img, (x, y) in zip(images, positions):
            H = max(H, y + img.shape[0])
            W = max(W, x + img.shape[1])

        dtype = images[0].dtype
        nch = images[0].shape[2] if images[0].ndim == 3 else 1

        shape = (H, W) if nch == 1 else (H, W, nch)
        acc = np.zeros(shape, dtype=np.float64)
        w = np.zeros(shape, dtype=np.float64)

        for img, (x, y) in zip(images, positions):
            h, w_ = img.shape[0], img.shape[1]
            acc[y:y+h, x:x+w_] += img.astype(np.float64)
            w[y:y+h, x:x+w_] += 1.0

        mask = w > 0
        out = np.divide(acc, w, where=mask, out=np.zeros_like(acc))
        out = np.clip(out, 0, 255).astype(dtype)
        return out