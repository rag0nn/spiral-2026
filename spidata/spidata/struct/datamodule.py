from spidata.struct.dataloader import SpiDataLoader

class SpiSequentialDataLoader:
    def __init__(self, loaders):
        """
        Birden fazla DataLoader'ı sırayla dönen sarmalayıcı sınıf.
        """
        self.loaders = loaders

    def __len__(self):
        # Toplam batch sayısı, alt loader'ların batch sayılarının toplamıdır
        return sum(len(loader) for loader in self.loaders)

    def __iter__(self):
        # Alt loader'ları sırayla dön
        for loader in self.loaders:
            is_new = True
            for batch in loader:
                # Yeni sekansın ilk batch'i için 'new' anahtarını True, diğerleri için False yap
                batch["new"] = is_new
                is_new = False
                yield batch

class SpiDataModule:
    def __init__(self, datapacks, train_ratio=0.8, batch_size=4, 
                 num_workers=0, pin_memory=False, filter_missing_translations=True, 
                 train_transform=None, val_transform=None, collate_fn=None):
        """
        Birden fazla datapack'i yöneten, her biri için sıralı loader'ları kuran
        ve bu loader'lardan sırayla veri çekilmesini sağlayan veri modülü.
        
        Parametreler:
            datapacks (list of DataPack): Yüklenecek veri paketlerinin listesi.
            train_ratio (float): Eğitim verisi oranı. Geri kalanı doğrulama olur.
            batch_size (int): Batch boyutu.
            num_workers (int): Alt süreç sayısı.
            pin_memory (bool): Bellek sabitleme seçeneği.
            filter_missing_translations (bool): Translasyonu eksik olanları filtreleme seçeneği.
            train_transform (callable, isteğe bağlı): Eğitim dönüşümü.
            val_transform (callable, isteğe bağlı): Doğrulama dönüşümü.
            collate_fn (callable, isteğe bağlı): Özel harmanlama fonksiyonu.
        """
        self.datapacks = datapacks
        self.data_loaders = []

        # Her bir datapack için SpiDataLoader oluştur
        for dp in datapacks:
            loader = SpiDataLoader(
                datapack=dp,
                train_ratio=train_ratio,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                filter_missing_translations=filter_missing_translations,
                train_transform=train_transform,
                val_transform=val_transform,
                collate_fn=collate_fn
            )
            self.data_loaders.append(loader)

        # Eğitim ve doğrulama alt loader'larını sıralı sarmalayıcıyla birleştir
        train_sub_loaders = [loader.train_loader for loader in self.data_loaders]
        val_sub_loaders = [loader.val_loader for loader in self.data_loaders]

        self.train_loader = SpiSequentialDataLoader(train_sub_loaders)
        self.val_loader = SpiSequentialDataLoader(val_sub_loaders)

        # Kullanıcı tercihine göre doğrudan erişim takma adları (alias)
        self.trainloader = self.train_loader
        self.valloader = self.val_loader
