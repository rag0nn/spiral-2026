from spidata.data.registery import Registery
from spidata.struct.dataloader import SpiDataLoader

# Veri paketi
datapack = Registery.ot25_1

# DataLoader oluştur (ilk %80 train, kalan %20 val)
dataloader = SpiDataLoader(
    datapack=datapack,
    train_ratio=0.8,
    batch_size=8
)

print(f"Train örnek sayısı : {len(dataloader.train_dataset)}")
print(f"Val örnek sayısı   : {len(dataloader.val_dataset)}")

# Train batch'i üzerinde döngü
for batch in dataloader.trainloader:
    images       = batch["image"]        # (B, 3, 512, 512)
    translations = batch["translations"] # (B, 3)
    objects      = batch["objects"]      # liste, her eleman o karedeki kutular
    print(f"Train batch görüntü boyutu: {images.shape}")
    break

# Val batch'i üzerinde döngü
for batch in dataloader.valloader:
    print(f"Val batch görüntü boyutu: {batch['image'].shape}")
    break
