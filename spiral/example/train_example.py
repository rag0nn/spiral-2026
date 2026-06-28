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

def run_od_training_example(epochs=3):
    """
    Sadece Object Detection (Nesne Tespiti) egitimi yapan ornek fonksiyon.
    Sadece od_head + shared (backbone + neck) olusturulur, pos_head/tneck hic baslatilmaz.
    """
    setup_logging(level=logging.INFO, force=True)
    logging.info("Sadece Object Detection egitimi baslatiliyor...")

    datamodule = SpiDataModule(
        datapacks=[Registery.ot25_1, 
                   Registery.ot25_2,
                   Registery.ot25_3,
                   Registery.ot25_4,
                   ],
        train_ratio=0.8,
        batch_size=32,
        train_transform=SpiTransforms.default_training,
        val_transform=SpiTransforms.default_inference,
        filter_missing_translations=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # od=True, pos=False -> pos_head ve tneck hic olusturulmaz
    # stem_type="focus" -> FocusStem ile 512x512 girdi 128x128'e indirilir
    model = SpiMultiModel(num_classes=4, od=True, pos=False, stem_type="focus")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    loss_fn = SpiLoss(num_classes=4, mode="det_only", lambda_pos=1.0)

    modules = SpiTrainModules(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        mode="det_only"
    )

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

    trainer.start()


def run_pos_training_example(epochs=3):
    """
    Sadece Pose/Translation (Konum Tahmini) egitimi yapan ornek fonksiyon.
    Sadece pos_head + tneck + shared (backbone + neck) olusturulur, od_head hic baslatilmaz.
    """
    setup_logging(level=logging.INFO, force=True)
    logging.info("Sadece Pose/Translation egitimi baslatiliyor...")

    datamodule = SpiDataModule(
        datapacks=[Registery.ot25_1, Registery.ot25_2],
        train_ratio=0.8,
        batch_size=2,
        train_transform=SpiTransforms.default_training,
        val_transform=SpiTransforms.default_inference
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # od=False, pos=True -> od_head hic olusturulmaz, tneck + pos_head aktif
    # stem_type="focus" -> FocusStem ile 512x512 girdi 128x128'e indirilir
    model = SpiMultiModel(num_classes=4, od=False, pos=True, stem_type="focus")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    loss_fn = SpiLoss(num_classes=4, mode="multi_task", lambda_pos=1.0)

    modules = SpiTrainModules(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        mode="multi_task"
    )

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

    trainer.start()


def resume_training_example(checkpoint_path: str, epochs=10):
    """
    Kaydedilmis parcali checkpoint durumlarini yukleyip egitime devam eden fonksiyon
    """
    setup_logging(level=logging.INFO, force=True)
    logging.info(f"Egitime kalinan yerden devam ediliyor. Checkpoint: {checkpoint_path}")

    datamodule = SpiDataModule(
        datapacks=[
            Registery.ot25_1,
            # Registery.ot25_2,
            # Registery.ot25_3,
            # Registery.ot25_4,      
            ],
        train_ratio=0.8,
        batch_size=16,
        train_transform=SpiTransforms.default_training,
        val_transform=SpiTransforms.default_inference,
        filter_missing_translations=False,
        shuffle=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resume: varsayilan olarak iki head de aktif ve focus stem kullanilir
    model = SpiMultiModel(num_classes=4, od=True, pos=False, stem_type="focus")
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

    # Checkpoint klasorunu kurtarip egitime oradan devam etmesini sagliyoruz
    save_dir = str(Path(checkpoint_path).parent)

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

    # Parcali checkpoint yukleyici cagrisi
    trainer.load_checkpoint(checkpoint_path)

    trainer.start()

if __name__ == "__main__":
    # Test amacli OD egitimi calistirmak icin:
    # run_od_training_example(epochs=1)

    # Test amacli Pose egitimi calistirmak icin:
    # run_pos_training_example(epochs=2)

    # Egitimi resume etmek icin (herhangi bir parcanin path'ini veya klasorunu vermeniz yeterlidir):
    # resume_training_example(checkpoint_path="weights/model_20260627_123456/last_trainer_state.pth", epochs=10)
    
    # ==============================
    # run_od_training_example(epochs=100)
    resume_training_example(checkpoint_path="/home/enes/Desktop/spiral_ws/spiral/spiral/training/weights/model_20260627_214712/last_trainer_state.pth", epochs=26)
