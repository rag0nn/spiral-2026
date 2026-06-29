import cv2
import numpy as np

def draw_sample(sample, class_names=None, window_name="sample", show=True):
    """
    Bir sample dict'ini görselleştirir.
    Görüntü üzerine normalize YOLO xywh sınır kutularını ve class etiketlerini çizer,
    sol üst köşeye translasyon değerlerini yazar.

    Parametreler:
        sample (dict): 'image' (HxWx3 numpy), 'objects' (numpy array), 'translations' (numpy array) içeren dict.
        class_names (list, isteğe bağlı): Sınıf indeksine karşılık gelen isim listesi.
        window_name (str): OpenCV pencere başlığı.
        show (bool): True ise pencere açılır, False ise yalnızca çizilmiş görüntü döner.

    Dönüş:
        numpy.ndarray: Çizimler yapılmış BGR görüntü.
    """
    image = sample["image"].copy()
    objects = sample.get("objects", np.array([]))
    translations = sample.get("translations", None)

    h, w = image.shape[:2]

    # RGB -> BGR (OpenCV için)
    canvas = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Sınır kutularını çiz
    for obj in objects:
        cls_id = int(obj[0])
        cx = float(obj[1]) * w
        cy = float(obj[2]) * h
        bw = float(obj[3]) * w
        bh = float(obj[4]) * h

        xmin = int(cx - bw / 2)
        ymin = int(cy - bh / 2)
        xmax = int(cx + bw / 2)
        ymax = int(cy + bh / 2)

        color = _class_color(cls_id)
        cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), color, 2)
        label = class_names[cls_id] if class_names and cls_id < len(class_names) else str(cls_id)
        _draw_label(canvas, label, (xmin, ymin), color)

    # Translasyon değerlerini sol üst köşeye yaz
    if translations is not None and len(translations) >= 3:
        tx, ty, tz = float(translations[0]), float(translations[1]), float(translations[2])
        lines = [
            f"tx: {tx:.4f}",
            f"ty: {ty:.4f}",
            f"tz: {tz:.4f}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(canvas, line, (10, 22 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, line, (10, 22 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 1, cv2.LINE_AA)

    if show:
        cv2.imshow(window_name, canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return canvas

def _class_color(cls_id):
    # Sınıf indeksine göre belirli bir renk üret
    palette = [
        (0, 200, 100),
        (255, 80, 80),
        (80, 120, 255),
        (255, 200, 0),
        (180, 0, 255),
        (0, 220, 220),
        (255, 140, 0),
        (0, 100, 255),
    ]
    return palette[cls_id % len(palette)]

def _draw_label(canvas, text, pos, color):
    # Etiket arka planı ve yazı
    x, y = pos
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    y0 = max(y - 4, th + 4)
    cv2.rectangle(canvas, (x, y0 - th - 4), (x + tw + 4, y0 + 2), color, -1)
    cv2.putText(canvas, text, (x + 2, y0 - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
