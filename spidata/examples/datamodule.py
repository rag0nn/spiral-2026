from spidata.data.registery import Registery
from spidata.struct.datamodule import SpiDataModule

# Birden fazla veri paketi (sekans verisi)
datamodule = SpiDataModule(
    datapacks=[Registery.ot25_1, Registery.ot25_1],  # birden fazla sekans
    train_ratio=0.8,
    batch_size=8
)

print(f"Toplam train batch: {len(datamodule.trainloader)}")
print(f"Toplam val batch  : {len(datamodule.valloader)}")

# Train döngüsü — new=True yeni bir sekansın başladığını gösterir
for batch in datamodule.trainloader:
    if batch["new"]:
        print("Yeni sekans başladı, state sıfırlanabilir.")

    images       = batch["image"]        # (B, 3, 512, 512)
    translations = batch["translations"] # (B, 3)
    objects      = batch["objects"]      # liste

    # ... model adımı buraya ...
    break
