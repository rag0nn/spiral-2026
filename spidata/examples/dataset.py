from spidata.data.registery import Registery
from spidata.struct.dataset import SpiDataset
from spidata.tools.transformations import SpiTransforms
from spidata.tools.visualize import draw_sample

# Veri paketi
datapack = Registery.ot25_4

# Dataset oluştur
dataset = SpiDataset(
    datapack=datapack,
    filter_missing_translations=True,
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
    sample = dataset.get_random_sample()
    draw_sample(sample, show=True)

    # img = draw_sample(sample, show=True)
show_random_sample()
    