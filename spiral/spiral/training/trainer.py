import os
import torch
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from dataclasses import dataclass
from typing import Dict, Any, Optional

from .loss import SpiLoss
from .reporting import SpiEpochMetric, SpiTrainingHistory, SpiReporting
from .base import SpiMultiModel
from spiral.utils import get_timestamp

BASE_WEIGTS_DIR = Path(__file__).parent / "weights"

@dataclass
class SpiTrainModules:
    """
    Model, Optimizer, Scheduler, Scaler ve egitim modunu sarmalayan veri sinifi
    """
    model: SpiMultiModel
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
    scaler: GradScaler
    mode: str = "multi_task" # det_only veya multi_task

    def get_split_state_dicts(self) -> Dict[str, Any]:
        """
        Model agirliklarini ve egitim durumunu ayri parcalar halinde dondurur.
        has_od/has_pos flag'lerine gore sadece var olan headlerin state_dict'i alinir.
        """
        shared_state = {
            "backbone": self.model.backbone.state_dict(),
            "neck": self.model.neck.state_dict()
        }

        # od_head sadece has_od=True ise vardir
        odbranch_state = {
            "od_head": self.model.od_head.state_dict() if self.model.has_od else None
        }

        # pos_head ve tneck sadece has_pos=True ise vardir
        posbranch_state = {
            "pos_head": self.model.pos_head.state_dict() if self.model.has_pos else None,
            "tneck": self.model.tneck.state_dict() if self.model.has_pos else None
        }

        trainer_state = {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "mode": self.mode
        }

        return {
            "shared": shared_state,
            "odbranch": odbranch_state,
            "posbranch": posbranch_state,
            "trainer": trainer_state
        }

    def load_split_state_dicts(
        self, 
        shared_dict: Dict[str, Any], 
        odbranch_dict: Dict[str, Any], 
        posbranch_dict: Dict[str, Any], 
        trainer_dict: Dict[str, Any]
    ):
        """
        Ayri yuklenen parcaları ilgili modullere geri yukler
        """
        if shared_dict:
            self.model.backbone.load_state_dict(shared_dict["backbone"])
            self.model.neck.load_state_dict(shared_dict["neck"])

        # Sadece model has_od=True ise od_head yukle
        if odbranch_dict and odbranch_dict.get("od_head") is not None and self.model.has_od:
            self.model.od_head.load_state_dict(odbranch_dict["od_head"])

        # Sadece model has_pos=True ise pos_head ve tneck yukle
        if posbranch_dict and self.model.has_pos:
            if posbranch_dict.get("pos_head") is not None:
                self.model.pos_head.load_state_dict(posbranch_dict["pos_head"])
            if posbranch_dict.get("tneck") is not None:
                self.model.tneck.load_state_dict(posbranch_dict["tneck"])

        if trainer_dict:
            try:
                self.optimizer.load_state_dict(trainer_dict["optimizer_state_dict"])
            except ValueError:
                logging.warning("Optimizer state dict uyumsuz (model mimarisi degismis olabilir). Optimizer sifirdan baslatiliyor.")
            if self.scheduler and trainer_dict.get("scheduler_state_dict"):
                self.scheduler.load_state_dict(trainer_dict["scheduler_state_dict"])
            if self.scaler and trainer_dict.get("scaler_state_dict"):
                self.scaler.load_state_dict(trainer_dict["scaler_state_dict"])
            if "mode" in trainer_dict:
                self.mode = trainer_dict["mode"]
        logging.info("Modul durumlari basariyla yuklendi.")

class SpiTrainer:
    """
    SpiMultiModel egitimini koordine eden ana egitici sinif
    """
    def __init__(
        self,
        modules: SpiTrainModules,
        train_loader,
        val_loader,
        loss_fn: SpiLoss,
        device: torch.device,
        save_dir: Optional[str] = None,
        patience: int = 15,
        total_epochs: int = 50
    ):
        self.modules = modules
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.device = device
        self.patience = patience
        self.total_epochs = total_epochs

        # save_dir verilmezse weights altında timestamp klasörü oluştur
        if save_dir is None:
            self.save_dir = BASE_WEIGTS_DIR / f"model_{get_timestamp()}"
        else:
            self.save_dir = Path(save_dir)

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.history = SpiTrainingHistory()
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def start(self):
        logging.info(f"Egitim basliyor... Toplam Epoch: {self.total_epochs}, Cihaz: {self.device}")
        logging.info(f"Egitim Modu: {self.modules.mode}")
        logging.info(f"Model Kayit Klasoru: {self.save_dir}")

        start_epoch = len(self.history.epochs) + 1

        try:
            for epoch in range(start_epoch, self.total_epochs + 1):
                logging.info(f"\nEpoch {epoch}/{self.total_epochs}")
                
                # Epoch egitimi ve dogrulamasi
                train_metric = self._train_epoch()
                val_metric = self._val_epoch()

                self.history.add(epoch, train_metric, val_metric)

                # Epoch özeti (kısa, detaylı özet sonda _print_summary'de gösterilir)
                logging.info(
                    f"Epoch {epoch:02d} sonucu | "
                    f"Train: {train_metric.loss:.4f}  Val: {val_metric.loss:.4f}"
                )

                # En iyi model kontrolu
                val_loss = val_metric.loss
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self._save_checkpoint("best")
                    logging.info(f"En iyi model parcalari kaydedildi (best_*.pth). Val Loss: {val_loss:.4f}")
                else:
                    self.patience_counter += 1
                    logging.info(f"Gelisim yok. Patience Counter: {self.patience_counter}/{self.patience}")

                # Scheduler adimi
                if self.modules.scheduler is not None:
                    self.modules.scheduler.step()

                # Son model kontrolu
                self._save_checkpoint("last")

                # Grafikleri ve metrikleri guncelle
                SpiReporting.plot_training_history(self.history, str(self.save_dir / "history_plot.png"))
                SpiReporting.save_metrics_to_json(self.history, str(self.save_dir / "metrics.json"))

                # Early stopping kontrolu
                if self.patience_counter >= self.patience:
                    logging.info("Early stopping tetiklendi. Egitim sonlandirildi.")
                    break

                # GPU bellek temizligi
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except KeyboardInterrupt:
            logging.info("\n[INTERRUPT] Egitim kullanici tarafindan CTRL+C ile durduruldu!")
            logging.info("Son durum degerlendiriliyor, son validasyon adimi calistiriliyor...")
            
            # Son validasyon
            val_metric = self._val_epoch()
            logging.info(f"Son Validasyon Sonucu -> {val_metric}")
            
            # Raporlari guncelle ve son check'leri kaydet
            SpiReporting.plot_training_history(self.history, str(self.save_dir / "history_plot.png"))
            SpiReporting.save_metrics_to_json(self.history, str(self.save_dir / "metrics.json"))
            self._save_checkpoint("last")
            logging.info("Son durum son model parcalari olarak kaydedildi (last_*.pth).")
            
        finally:
            # Egitim sonu metrik raporunu terminale bas
            self._print_summary()

    def load_checkpoint(self, checkpoint_path: str):
        """
        Kaydedilmis model checkpoint durumlarini parcali yukler
        """
        path = Path(checkpoint_path)
        if path.is_dir():
            folder = path
            prefix = "last"
        else:
            folder = path.parent
            # "last_shared.pth" veya "best_trainer_state.pth" durumundan prefix ayikla
            prefix = path.name.split("_")[0]

        shared_path = folder / f"{prefix}_shared.pth"
        odbranch_path = folder / f"{prefix}_odbranch.pth"
        posbranch_path = folder / f"{prefix}_posbranch.pth"
        trainer_path = folder / f"{prefix}_trainer_state.pth"

        # Dosyalarin varligini kontrol et
        if not shared_path.exists() or not odbranch_path.exists() or not posbranch_path.exists() or not trainer_path.exists():
            raise FileNotFoundError(f"Checkpoint parcalari eksik. Klasor: {folder}, Prefix: {prefix}")

        shared_dict = torch.load(str(shared_path), map_location=self.device)
        odbranch_dict = torch.load(str(odbranch_path), map_location=self.device)
        posbranch_dict = torch.load(str(posbranch_path), map_location=self.device)

        # SpiTrainingHistory ve SpiEpochMetric özel sınıflar olduğu için safe globals'a eklenmesi gerekiyor
        with torch.serialization.safe_globals([SpiTrainingHistory, SpiEpochMetric]):
            trainer_dict = torch.load(str(trainer_path), map_location=self.device)

        self.modules.load_split_state_dicts(shared_dict, odbranch_dict, posbranch_dict, trainer_dict)

        if "best_val_loss" in trainer_dict:
            self.best_val_loss = trainer_dict["best_val_loss"]
        if "history" in trainer_dict:
            self.history = trainer_dict["history"]
        if "patience_counter" in trainer_dict:
            self.patience_counter = trainer_dict["patience_counter"]

        logging.info(f"Checkpoint basariyla yuklendi. Kalinan Epoch: {len(self.history.epochs)}")

    def _train_epoch(self) -> SpiEpochMetric:
        self.modules.model.train()
        
        # Mod det_only ise temporal durumunu kapali tut
        if self.modules.mode == "det_only":
            self.modules.model.temporal = False
            self.modules.model.prev = None
        else:
            self.modules.model.temporal = True

        running_loss = 0.0
        running_det = 0.0
        running_cls = 0.0
        running_reg = 0.0
        running_ctr = 0.0
        running_pos = 0.0
        total_samples = 0

        pbar = tqdm(self.train_loader, desc="Egitim", leave=False)
        for batch in pbar:
            images, translations, gt_boxes, gt_labels = self._process_batch(batch)
            batch_size = images.size(0)
            total_samples += batch_size

            # Optimizer gradyan sıfırlama
            self.modules.optimizer.zero_grad(set_to_none=True)

            # Autocast ile egitim adimi
            use_amp = self.device.type == "cuda"
            with autocast(device_type=self.device.type, enabled=use_amp):
                det_outs, pos_out = self.modules.model(images)
                
                loss_dict = self.loss_fn(
                    det_outs, 
                    pos_out, 
                    gt_boxes, 
                    gt_labels, 
                    translations
                )
                total_loss = loss_dict["loss"]

            # Geri yayilim ve guncelleme
            if self.modules.scaler is not None:
                self.modules.scaler.scale(total_loss).backward()
                self.modules.scaler.unscale_(self.modules.optimizer)
                torch.nn.utils.clip_grad_norm_(self.modules.model.parameters(), max_norm=5.0)
                self.modules.scaler.step(self.modules.optimizer)
                self.modules.scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.modules.model.parameters(), max_norm=5.0)
                self.modules.optimizer.step()

            # Kayiplari biriktir
            running_loss += total_loss.item() * batch_size
            running_det += loss_dict["det_loss"].item() * batch_size
            running_cls += loss_dict["cls_loss"].item() * batch_size
            running_reg += loss_dict["reg_loss"].item() * batch_size
            running_ctr += loss_dict["ctr_loss"].item() * batch_size
            running_pos += loss_dict["pos_loss"].item() * batch_size

            pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})

        return SpiEpochMetric(
            loss=running_loss / total_samples,
            det_loss=running_det / total_samples,
            cls_loss=running_cls / total_samples,
            reg_loss=running_reg / total_samples,
            ctr_loss=running_ctr / total_samples,
            pos_loss=running_pos / total_samples
        )

    def _val_epoch(self) -> SpiEpochMetric:
        self.modules.model.eval()
        
        # Mod det_only ise temporal durumunu kapali tut
        if self.modules.mode == "det_only":
            self.modules.model.temporal = False
            self.modules.model.prev = None
        else:
            self.modules.model.temporal = True

        running_loss = 0.0
        running_det = 0.0
        running_cls = 0.0
        running_reg = 0.0
        running_ctr = 0.0
        running_pos = 0.0
        total_samples = 0

        use_amp = self.device.type == "cuda"
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Dogrulama", leave=False)
            for batch in pbar:
                images, translations, gt_boxes, gt_labels = self._process_batch(batch)
                batch_size = images.size(0)
                total_samples += batch_size

                with autocast(device_type=self.device.type, enabled=use_amp):
                    det_outs, pos_out = self.modules.model(images)
                    
                    loss_dict = self.loss_fn(
                        det_outs, 
                        pos_out, 
                        gt_boxes, 
                        gt_labels, 
                        translations
                    )
                    total_loss = loss_dict["loss"]

                running_loss += total_loss.item() * batch_size
                running_det += loss_dict["det_loss"].item() * batch_size
                running_cls += loss_dict["cls_loss"].item() * batch_size
                running_reg += loss_dict["reg_loss"].item() * batch_size
                running_ctr += loss_dict["ctr_loss"].item() * batch_size
                running_pos += loss_dict["pos_loss"].item() * batch_size

                pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})

        return SpiEpochMetric(
            loss=running_loss / total_samples,
            det_loss=running_det / total_samples,
            cls_loss=running_cls / total_samples,
            reg_loss=running_reg / total_samples,
            ctr_loss=running_ctr / total_samples,
            pos_loss=running_pos / total_samples
        )

    def _process_batch(self, batch):
        images = batch["image"].to(self.device).float() / 255.0
        translations = batch["translations"].to(self.device)

        # Yeni sekans basladiginda model state sıfırlanır
        if batch.get("new", False) and self.modules.mode == "multi_task":
            self.modules.model.prev = None

        # objects listesini modelin kabul ettigi tensor formatina donustur
        gt_boxes = []
        gt_labels = []
        for obj_arr in batch["objects"]:
            if len(obj_arr) > 0:
                # obj_arr: [[cls_id, xmin, ymin, xmax, ymax], ...]
                boxes = torch.from_numpy(obj_arr[:, 1:5].astype(np.float32)).to(self.device)
                labels = torch.from_numpy(obj_arr[:, 0].astype(np.int64)).to(self.device)
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32, device=self.device)
                labels = torch.zeros((0,), dtype=torch.int64, device=self.device)
            gt_boxes.append(boxes)
            gt_labels.append(labels)

        return images, translations, gt_boxes, gt_labels

    def _save_checkpoint(self, prefix: str):
        try:
            split_states = self.modules.get_split_state_dicts()
            
            # Shared: backbone + neck
            torch.save(split_states["shared"], str(self.save_dir / f"{prefix}_shared.pth"))
            # Odbranch: od_head
            torch.save(split_states["odbranch"], str(self.save_dir / f"{prefix}_odbranch.pth"))
            # Posbranch: tneck + pos_head
            torch.save(split_states["posbranch"], str(self.save_dir / f"{prefix}_posbranch.pth"))

            # Trainer durumlarini da ekle
            trainer_state = split_states["trainer"]
            trainer_state["best_val_loss"] = self.best_val_loss
            trainer_state["history"] = self.history
            trainer_state["patience_counter"] = self.patience_counter
            torch.save(trainer_state, str(self.save_dir / f"{prefix}_trainer_state.pth"))

        except Exception as e:
            logging.error(f"Checkpoint kaydedilemedi ({prefix}): {e}")

    def _print_summary(self):
        if not self.history.epochs:
            logging.info("Yazdirilacak egitim gecmisi bulunamadi.")
            return

        logging.info("\n=== EGITIM OZET RAPORU ===")
        logging.info(f"Kayit Klasoru: {self.save_dir}")
        logging.info(f"Toplam Tamamlanan Epoch: {len(self.history.epochs)}")

        # En iyi val loss ve epoch
        best_epoch_idx = np.argmin([m.loss for m in self.history.val_history])
        best_epoch = self.history.epochs[best_epoch_idx]
        best_val = self.history.val_history[best_epoch_idx]
        logging.info(f"En Iyi Val Loss: {best_val.loss:.4f} (Epoch {best_epoch})")

        logging.info("\nEpoch Detaylari:")
        for epoch, t_m, v_m in zip(self.history.epochs, self.history.train_history, self.history.val_history):
            logging.info(
                f"Epoch {epoch:02d} | "
                f"Train Loss: {t_m.loss:.4f} (Det: {t_m.det_loss:.4f}, Pos: {t_m.pos_loss:.4f}) | "
                f"Val Loss: {v_m.loss:.4f} (Det: {v_m.det_loss:.4f}, Pos: {v_m.pos_loss:.4f})"
            )
