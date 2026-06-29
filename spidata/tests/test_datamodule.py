import torch
from spidata.struct.registery import Registery
from spidata.struct.datamodule import SpiDataModule

def test_spi_datamodule():
    print("--- SpiDataModule Testi Başlatılıyor ---")

    datamodule = SpiDataModule(
        datapacks=[Registery.ot25_1, Registery.ot25_1],
        train_ratio=0.8,
        batch_size=4
    )

    print(f"\nToplam train batch sayısı: {len(datamodule.train_loader)}")
    print(f"Toplam val batch sayısı: {len(datamodule.val_loader)}")

    first_pack_train_batches = len(datamodule.data_loaders[0].train_loader)
    print(f"Birinci datapack train batch sayısı: {first_pack_train_batches}")

    # Geçiş noktasını ve 'new' bayraklarını kontrol et, ikinci 'new=True' gelince dur
    print("\n--- Trainloader 'new' Bayrağı Kontrolü ---")
    new_seen_at = []
    for i, batch in enumerate(datamodule.trainloader):
        if batch["new"]:
            new_seen_at.append(i)
            print(f"  Batch {i:3d} | new=True  | image={batch['image'].shape}")
        elif i < 3:
            print(f"  Batch {i:3d} | new=False | image={batch['image'].shape}")

        # İkinci 'new=True' gelip birkaç batch daha geçince çık
        if len(new_seen_at) == 2 and i > new_seen_at[1] + 2:
            break

    assert new_seen_at[0] == 0, "Birinci batch 'new=True' olmalıdır!"
    assert new_seen_at[1] == first_pack_train_batches, \
        f"İkinci 'new=True' {first_pack_train_batches}. batch'te olmalı, {new_seen_at[1]}'de görüldü!"

    print(f"\n'new=True' sinyalleri batch indeksleri: {new_seen_at}")
    print("SpiDataModule testleri başarıyla tamamlandı!")

if __name__ == "__main__":
    test_spi_datamodule()
