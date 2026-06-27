from dataclasses import dataclass, field, fields
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
import json
import logging
from pathlib import Path

@dataclass
class SpiEpochMetric:
    """
    Her epoch icin olculen kayip degerleri
    """
    loss: float
    det_loss: float
    cls_loss: float
    reg_loss: float
    ctr_loss: float
    pos_loss: float

    def __str__(self):
        return (f"Loss: {self.loss:.4f} (Det: {self.det_loss:.4f}, "
                f"Cls: {self.cls_loss:.4f}, Reg: {self.reg_loss:.4f}, "
                f"Ctr: {self.ctr_loss:.4f}, Pos: {self.pos_loss:.4f})")

    @classmethod
    def get_field_list(cls) -> List[str]:
        return [f.name for f in fields(cls)]

@dataclass
class SpiTrainingHistory:
    """
    Egitim boyunca biriken epoch metriklerinin gecmisi
    """
    epochs: List[int] = field(default_factory=list)
    train_history: List[SpiEpochMetric] = field(default_factory=list)
    val_history: List[SpiEpochMetric] = field(default_factory=list)

    def add(self, epoch: int, train_metric: SpiEpochMetric, val_metric: SpiEpochMetric):
        self.epochs.append(epoch)
        self.train_history.append(train_metric)
        self.val_history.append(val_metric)

    def latest_train_score(self) -> Optional[SpiEpochMetric]:
        return self.train_history[-1] if self.train_history else None

    def latest_val_score(self) -> Optional[SpiEpochMetric]:
        return self.val_history[-1] if self.val_history else None

class SpiReporting:
    """
    Egitim sonuclarini gorsellestirme ve kaydetme yardimci sinifi
    """
    @staticmethod
    def plot_training_history(history: SpiTrainingHistory, save_path: str):
        epochs = history.epochs
        if not epochs:
            logging.warning("Cizilecek egitim gecmisi bulunamadi.")
            return

        # Grafikleri ciz
        plt.figure(figsize=(15, 10))

        # 1. Toplam Kayip
        plt.subplot(2, 2, 1)
        plt.plot(epochs, [m.loss for m in history.train_history], label="Train")
        plt.plot(epochs, [m.loss for m in history.val_history], label="Val")
        plt.title("Toplam Kayip (Total Loss)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. Object Detection Kaybi
        plt.subplot(2, 2, 2)
        plt.plot(epochs, [m.det_loss for m in history.train_history], label="Train")
        plt.plot(epochs, [m.det_loss for m in history.val_history], label="Val")
        plt.title("Detection Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 3. Pos (Translation) Kaybi
        plt.subplot(2, 2, 3)
        plt.plot(epochs, [m.pos_loss for m in history.train_history], label="Train")
        plt.plot(epochs, [m.pos_loss for m in history.val_history], label="Val")
        plt.title("Translation (Pos) Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 4. Det Alt Kayiplari (Cls, Reg, Ctr) - Sadece Train
        plt.subplot(2, 2, 4)
        plt.plot(epochs, [m.cls_loss for m in history.train_history], label="Cls Loss")
        plt.plot(epochs, [m.reg_loss for m in history.train_history], label="Reg Loss")
        plt.plot(epochs, [m.ctr_loss for m in history.train_history], label="Ctr Loss")
        plt.title("Detection Alt Kayiplari (Train)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.suptitle("Spiral Model Egitim Gecmisi", fontsize=16, fontweight="bold")
        plt.tight_layout()
        
        # Dosyayi kaydet
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_file), dpi=100)
        logging.info(f"Egitim gecmisi grafigi kaydedildi: {save_path}")
        plt.close()

    @staticmethod
    def save_metrics_to_json(history: SpiTrainingHistory, save_path: str):
        data = {
            "epochs": history.epochs,
            "train": [
                {
                    "loss": m.loss,
                    "det_loss": m.det_loss,
                    "cls_loss": m.cls_loss,
                    "reg_loss": m.reg_loss,
                    "ctr_loss": m.ctr_loss,
                    "pos_loss": m.pos_loss
                } for m in history.train_history
            ],
            "val": [
                {
                    "loss": m.loss,
                    "det_loss": m.det_loss,
                    "cls_loss": m.cls_loss,
                    "reg_loss": m.reg_loss,
                    "ctr_loss": m.ctr_loss,
                    "pos_loss": m.pos_loss
                } for m in history.val_history
            ]
        }
        
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        with open(save_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        logging.info(f"Metrik verileri JSON formatinda kaydedildi: {save_path}")
