from spidata.data.registery import Registery
from spidata.struct.dataset import SpiDataset
from spidata.tools.transformations import SpiTransforms
from spidata.tools.visualize import draw_sample

import cv2
# Veri paketi
datapack = Registery.ot25_2

# Dataset oluştur
dataset = SpiDataset(
    datapack=datapack,
    filter_missing_translations=False,
    transform=SpiTransforms.default_training
)

print(f"Toplam örnek sayısı: {len(dataset)}")

# Tek örnek çek
sample = dataset[0]
print(f"Görüntü boyutu : {sample['image'].shape}")
print(f"Translasyon    : {sample['translations']}")
print(f"Nesneler       : {sample['objects']}")

# Rastgele örnek çek
random_sample = dataset.get_random_sample()
print(f"Rastgele görüntü boyutu: {random_sample['image'].shape}")

def show_random_sample():
    for i in range(5):
        sample = dataset.get_random_sample()
        canvas = draw_sample(sample, show=False)
        out = cv2.resize(canvas,(1200,700))
        cv2.imshow("put",out)
        cv2.waitKey(0)
        
def fully_look():
    idx = 0
    while True:
        sample = dataset[idx]
        canvas = draw_sample(sample, show=False)
        out = cv2.resize(canvas,(1200,700))
        cv2.imshow("put",out)
        key = cv2.waitKey(0)
        if key == ord("d"):
            idx += 1
        elif key == ord("a"):
            idx -= 1
        elif key == ord("j"):
            idx += 5
        elif key == ord("n"):
            idx -= 5
        else:
            break

        
# show_random_sample()
fully_look()