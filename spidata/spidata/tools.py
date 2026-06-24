from PIL import Image
import numpy as np

def get_random_sample():
    # TODO tamamıyla random şekidle sample load ekle
    
    image_path = "/home/enes/Desktop/spiral_ws/spidata/spidata/data/2025/OT1/THYZ_2025_Oturum_1/frame_000601.webp"
    txt_path = "/home/enes/Desktop/spiral_ws/spidata/spidata/data/2025/OT1/txt_labels/frame_000601.txt"
    
    image = np.array( Image.open(image_path))

    if image is None:
        raise FileNotFoundError()
    
    f = open(txt_path,"r")
    lines = f.readlines()
    f.close()
    return image, lines