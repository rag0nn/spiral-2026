import torch
import logging
from pathlib import Path

# spidata kütüphanesi modülleri
from spidata.data.registery import Registery
from spidata.struct.datamodule import SpiDataModule
from spidata.tools.transformations import SpiTransforms

# Kendi egitim modüllerimiz
from spiral.training.base import SpiMultiModel
from spiral.training.loss import SpiLoss
from spiral.training.trainer import SpiTrainModules, SpiTrainer
from spiral.utils import setup_logging

def run_training_example(mode="multi_task", epochs=5):
    """
    Sifirdan model egitimini baslatan ornek fonksiyon
    """
    setup_logging(level=logging.INFO, force=True)
    logging.info(f"Sifirdan egitim baslatiliyor. Mod: {mode}")

    # Veri modülü kurulumu
    datamodule = SpiDataModule(
        datapacks=[
            Registery.ot25_1,
            Registery.ot25_2
        ],
        train_ratio=0.8,
        batch_size=2,
        train_transform=SpiTransforms.default_training,
        val_transform=SpiTransforms.default_inference
    )

    # Cihaz secimi
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model kurulumu (varsayilan hidden=64)
    temporal_flag = True if mode == "multi_task" else False
    model = SpiMultiModel(temporal=temporal_flag, num_classes=4)
    model.to(device)

    # Loss kurulumu
    loss_fn = SpiLoss(num_classes=4, mode=mode, lambda_pos=1.0)

    # Optimizer ve Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # Modul sarmalayicisi
    modules = SpiTrainModules(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        mode=mode
    )

    # Trainer kurulumu (save_dir=None verildigi icin weights/model_timestamp klasorune kaydeder)
    trainer = SpiTrainer(
        modules=modules,
        train_loader=datamodule.train_loader,
        val_loader=datamodule.val_loader,
        loss_fn=loss_fn,
        device=device,
        save_dir=None,
        patience=5,
        total_epochs=epochs
    )

    # Egitimi baslat (CTRL+C ile kesilebilir)
    trainer.start()

def resume_training_example(checkpoint_path: str, epochs=10):
    """
    Kaydedilmis bir checkpoint dosyasini (.pth) yukleyip egitime devam eden fonksiyon
    """
    setup_logging(level=logging.INFO, force=True)
    logging.info(f"Egitime kalinan yerden devam ediliyor. Checkpoint: {checkpoint_path}")

    # Veri modülü kurulumu
    datamodule = SpiDataModule(
        datapacks=[
            Registery.ot25_1,
            # Registery.ot25_2
        ],
        train_ratio=0.8,
        batch_size=2,
        train_transform=SpiTransforms.default_training,
        val_transform=SpiTransforms.default_inference
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Modeli gecici olarak varsayilan baslat (checkpoint yuklenecek)
    model = SpiMultiModel(temporal=True, num_classes=4)
    model.to(device)

    loss_fn = SpiLoss(num_classes=4, mode="multi_task", lambda_pos=1.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    modules = SpiTrainModules(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        mode="multi_task"
    )

    # Mevcut checkpoint'in bulundugu klasorde kaydetmeye devam etsin
    save_dir = str(Path(checkpoint_path).parent)

    # Trainer kurulumu
    trainer = SpiTrainer(
        modules=modules,
        train_loader=datamodule.train_loader,
        val_loader=datamodule.val_loader,
        loss_fn=loss_fn,
        device=device,
        save_dir=save_dir,
        patience=5,
        total_epochs=epochs
    )

    # Checkpoint durumunu yukle
    trainer.load_checkpoint(checkpoint_path)

    # Egitime devam et
    trainer.start()

if __name__ == "__main__":
    # Test amacli sifirdan egitim calistirmak icin:
    run_training_example(mode="det_only", epochs=2)

    # Eger durdurulmus bir egitimi devam ettirmek istiyorsaniz:
    # resume_training_example(checkpoint_path="weights/model_20260627_123456/last.pth", epochs=10)
