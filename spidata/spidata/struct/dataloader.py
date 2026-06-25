import torch
from torch.utils.data import DataLoader
from spidata.struct.dataset import SpiDataset

def spi_collate_fn(batch):
    """
    Farklı karelerin farklı sayıda nesne (sınır kutusu) içermesi durumunda,
    PyTorch varsayılan collate fonksiyonunun hata vermesini engellemek için
    kullanılan özel collate fonksiyonu.
    """
    images = []
    translations = []
    objects = []
    
    for sample in batch:
        images.append(sample["image"])
        translations.append(sample["translations"])
        objects.append(sample["objects"])
    
    # Görüntüleri PyTorch standardına (C, H, W) getirip yığın haline getirelim
    images_t = []
    for img in images:
        t_img = torch.from_numpy(img)
        if len(t_img.shape) == 3:
            t_img = t_img.permute(2, 0, 1)
        images_t.append(t_img)
        
    images_stacked = torch.stack(images_t)
    translations_stacked = torch.stack([torch.from_numpy(t) for t in translations])
    
    return {
        "image": images_stacked,
        "translations": translations_stacked,
        "objects": objects  # Liste olarak bırakılır
    }

class SpiDataLoader:
    def __init__(self, datapack, train_ratio=0.8, batch_size=4, 
                 num_workers=0, pin_memory=False, filter_missing_translations=True, 
                 train_transform=None, val_transform=None, collate_fn=None):
        """
        SpiDataset'leri yükleyen ve train/val bölen DataLoader sınıfı.
        
        Parametreler:
            datapack (DataPack): Yollar ve veri setini barındıran nesne.
            train_ratio (float): Eğitim verisi oranı (örn. 0.8). Geri kalanı doğrulama olur.
            batch_size (int): Batch boyutu.
            num_workers (int): Alt süreç (worker) sayısı.
            pin_memory (bool): Bellek sabitleme seçeneği.
            filter_missing_translations (bool): Translasyonu eksik olanları filtreleme seçeneği.
            train_transform (callable, isteğe bağlı): Eğitim dönüşümü.
            val_transform (callable, isteğe bağlı): Doğrulama dönüşümü.
            collate_fn (callable, isteğe bağlı): Özel harmanlama fonksiyonu.
        """
        # Varsayılan dönüşümleri ayarla
        if train_transform is None:
            from spidata.tools.transformations import SpiTransforms
            train_transform = SpiTransforms.default_training
        if val_transform is None:
            from spidata.tools.transformations import SpiTransforms
            val_transform = SpiTransforms.default_inference
            
        if collate_fn is None:
            collate_fn = spi_collate_fn

        # Eğitim için veri kümesini yükle (eğitim dönüşümleri ile)
        train_ds = SpiDataset(
            datapack=datapack,
            filter_missing_translations=filter_missing_translations,
            transform=train_transform
        )
        
        # Doğrulama için veri kümesini yükle (çıkarım dönüşümleri ile)
        val_ds = SpiDataset(
            datapack=datapack,
            filter_missing_translations=filter_missing_translations,
            transform=val_transform
        )

        total_len = len(train_ds)
        split_idx = int(total_len * train_ratio)

        # Sıralı split: ilk train_ratio kadarını train, kalanını val yap
        train_ds.frame_paths = train_ds.frame_paths[:split_idx]
        val_ds.frame_paths = val_ds.frame_paths[split_idx:]

        self.train_dataset = train_ds
        self.val_dataset = val_ds

        # Eğitim veri yükleyicisi
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn
        )

        # Doğrulama veri yükleyicisi
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn
        )

        # Kullanıcı tercihine göre doğrudan erişim takma adları (alias)
        self.trainloader = self.train_loader
        self.valloader = self.val_loader
