import os
from pathlib import Path
import numpy as np
import torch
from spidata.struct.registery import Registery
from spidata.struct.dataset import SpiDataset
from spidata.tools.transformations import SpiTransforms
from spidata.struct.dataloader import SpiDataLoader

def test_spi_dataset():
    # 1. Kayitli veri paketini al
    datapack = Registery.ot25_1

    # 2. SpiDataset nesnesini olustur (transformsuz)
    dataset = SpiDataset(
        datapack=datapack,
        filter_missing_translations=True
    )
    
    # 3. Veri kümesi uzunlugunu dogrula
    print(f"Veri kümesi uzunlugu: {len(dataset)}")
    assert len(dataset) > 0, "Veri kümesi bos olmamali."

    # 4. Belirli bir indeksteki örnegi getir (__getitem__)
    sample = dataset[0]
    print("\n--- Örnek dataset[0] Testi (Dönüsümsüz) ---")
    print(f"Görüntü tipi: {type(sample['image'])}")
    print(f"Görüntü sekli: {sample['image'].shape}")
    print(f"Translasyon degeri: {sample['translations']}")
    print(f"Nesneler: {sample['objects']}")

    # Dogrulamalar
    assert isinstance(sample['image'], np.ndarray), "Görüntü bir numpy dizisi olmalidir"
    assert isinstance(sample['translations'], np.ndarray), "Translasyonlar bir numpy dizisi olmalidir"
    assert isinstance(sample['objects'], np.ndarray), "Nesneler bir numpy dizisi olmalidir"

    # 5. Egitim dönüsümlü SpiDataset nesnesini olustur
    dataset_training = SpiDataset(
        datapack=datapack,
        filter_missing_translations=True,
        transform=SpiTransforms.default_training
    )

    # 6. Egitim dönüsümlü örnek getir
    sample_train = dataset_training[0]
    print("\n--- Örnek dataset[0] Testi (Eğitim Dönüşümlü) ---")
    print(f"Eğitim Görüntü tipi: {type(sample_train['image'])}")
    print(f"Eğitim Görüntü sekli: {sample_train['image'].shape}")
    print(f"Eğitim Translasyon degeri: {sample_train['translations']}")
    print(f"Eğitim Nesneler: {sample_train['objects']}")
    assert sample_train['image'].shape == (640, 640, 3), "Eğitim görüntüsü boyutu (640, 640, 3) olmalidir"

    # 7. Çıkarım (Inference) dönüsümlü SpiDataset nesnesini olustur
    dataset_inference = SpiDataset(
        datapack=datapack,
        filter_missing_translations=True,
        transform=SpiTransforms.default_inference
    )

    # 8. Çıkarım dönüsümlü örnek getir ve kontrol et
    sample_inf = dataset_inference[0]
    print("\n--- Örnek dataset[0] Testi (Çıkarım Dönüşümlü) ---")
    print(f"Çıkarım Görüntü sekli: {sample_inf['image'].shape}")
    print(f"Çıkarım Translasyon degeri: {sample_inf['translations']}")
    print(f"Çıkarım Nesneler: {sample_inf['objects']}")

    # Görselin boyutu 512x512 olmali, ancak translasyon ve nesneler degismemeli (ham veriyle ayni kalmali)
    assert sample_inf['image'].shape == (640, 640, 3), "Çıkarım görüntüsü boyutu (640, 640, 3) olmalidir"
    assert np.allclose(sample_inf['translations'], sample['translations'], equal_nan=True), "Çıkarım translasyonu degismemeliydi!"
    
    # Nesnelerin ayni kaldigini dogrula
    assert len(sample_inf['objects']) == len(sample['objects']), "Çıkarım nesne sayisi degismemeliydi!"
    for i in range(len(sample['objects'])):
        assert np.allclose(sample_inf['objects'][i], sample['objects'][i]), "Çıkarım nesne koordinatlari degismemeliydi!"

    # 9. Rastgele bir örnek getir
    random_sample = dataset_training.get_random_sample()
    print("\n--- get_random_sample() Testi (Eğitim Dönüşümlü) ---")
    print(f"Rastgele Görüntü sekli: {random_sample['image'].shape}")
    print(f"Rastgele Translasyon degeri: {random_sample['translations']}")
    print(f"Rastgele Nesneler: {random_sample['objects']}")

def test_spi_dataloader():
    print("\n--- SpiDataLoader Testi Başlatılıyor ---")
    datapack = Registery.ot25_1
    
    # DataLoader oluştur (0.8 train, 0.2 val split oranı ile)
    batch_size = 4
    train_ratio = 0.8
    
    dataloader = SpiDataLoader(
        datapack=datapack,
        train_ratio=train_ratio,
        batch_size=batch_size
    )
    
    total_expected = len(SpiDataset(datapack=datapack, filter_missing_translations=True))
    expected_train = int(total_expected * train_ratio)
    expected_val = total_expected - expected_train
    
    print(f"Toplam beklenen veri: {total_expected}")
    print(f"Eğitim altküme boyutu: {len(dataloader.train_dataset)} (Beklenen: {expected_train})")
    print(f"Doğrulama altküme boyutu: {len(dataloader.val_dataset)} (Beklenen: {expected_val})")
    
    assert len(dataloader.train_dataset) == expected_train, "Eğitim altküme boyutu uyuşmuyor!"
    assert len(dataloader.val_dataset) == expected_val, "Doğrulama altküme boyutu uyuşmuyor!"
    
    # Bir batch okuma testi (trainloader)
    batch_train = next(iter(dataloader.trainloader))
    print("\n--- Trainloader Batch Örneği ---")
    print(f"Batch Görsel tipi: {type(batch_train['image'])}")
    print(f"Batch Görsel şekli: {batch_train['image'].shape}")  # (4, 3, 640, 640) olmalı
    print(f"Batch Translasyon tipi: {type(batch_train['translations'])}")
    print(f"Batch Translasyon şekli: {batch_train['translations'].shape}")  # (4, 3) olmalı
    print(f"Batch Nesneler liste uzunluğu: {len(batch_train['objects'])}")
    
    assert isinstance(batch_train['image'], torch.Tensor), "Batch görseli PyTorch Tensor olmalıdır"
    assert batch_train['image'].shape == (batch_size, 3, 640, 640), f"Görsel batch boyutu uyuşmuyor: {batch_train['image'].shape}"
    assert batch_train['translations'].shape == (batch_size, 3), f"Translasyon batch boyutu uyuşmuyor: {batch_train['translations'].shape}"
    
    # Bir batch okuma testi (valloader)
    batch_val = next(iter(dataloader.valloader))
    print("\n--- Valloader Batch Örneği ---")
    print(f"Val Batch Görsel şekli: {batch_val['image'].shape}")
    print(f"Val Batch Translasyon şekli: {batch_val['translations'].shape}")
    print(f"Val Batch Nesneler liste uzunluğu: {len(batch_val['objects'])}")
    
    assert batch_val['image'].shape == (batch_size, 3, 640, 640), "Doğrulama görsel batch boyutu uyuşmuyor!"
    
    print("\nSpiDataLoader testleri başarıyla tamamlandı!")

if __name__ == "__main__":
    test_spi_dataset()
    test_spi_dataloader()
