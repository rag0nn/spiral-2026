from spidata.struct.registery import Registery
from spidata.struct.datamodule import SpiDataModule
from spidata.tools.transformations import SpiTransforms

# Birden fazla veri paketi (sekans verisi)
datamodule = SpiDataModule(
    datapacks=[
        Registery.ot25_1, 
        Registery.ot25_2,
        Registery.ot25_3, 
        Registery.ot25_4,
        ],  # birden fazla sekans
    train_ratio=0.8,
    batch_size=8,
    train_transform=SpiTransforms.default_inference
)

print(f"Toplam train batch: {len(datamodule.trainloader)}")
print(f"Toplam val batch  : {len(datamodule.valloader)}")

# Train döngüsü — new=True yeni bir sekansın başladığını gösterir
for batch in datamodule.trainloader:
    if batch["new"]:
        print("Yeni sekans başladı, state sıfırlanabilir.")

    images       = batch["image"]        # (B, 3, 640, 640)
    translations = batch["translations"] # (B, 3)
    objects      = batch["objects"]      # liste

    # ... model adımı buraya ...
    print(images.shape)
    print(translations.shape)
    print(len(objects))
    break
