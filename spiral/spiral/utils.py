# import time
from functools import wraps
import time
import yaml
from pathlib import Path
from rich.logging import RichHandler
import logging
from datetime import datetime

def time_monitor(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start = time.perf_counter()

        result = func(self, *args, **kwargs)

        elapsed = time.perf_counter() - start  # seconds

        verbose = kwargs.get("verbose", False)

        if verbose:
            logging.info(
                f"--- [{self.__class__.__name__}.{func.__name__}] took {elapsed:.3f} s"
            )
        return result
    return wrapper


class Dicty:
    """

    Kalıtılan class'ta dict dönüşümlerini otomatik ele aldırmaya yarayan class.

    Ayrıca dict'ten object'e geçişi direkt bu class üzerinden yapmaya yarayan 
    registery methoduna da sahiptir:
    ```
        class A(DictyClass):

        a = DictyClass.fromDict({})

        gibi.
    ```

    Inheritance Example:
    ```
        from pathlib import Path  

        class TrainCfg(DictyClass):

            total_epoch: int
            elapsed_epoch: int
            batch_size: int
            save_path: Path
            patience: int
            lr: float = 0.01

            # serialize edilmesin
            cache: dict = field(
                default_factory=dict,
                metadata={"serialize": False}
            )
    ```
    """
    registry = {}

    __exclude__ = set()

    def __init_subclass__(cls):
        super().__init_subclass__()
        Dicty.registry[cls.__name__] = cls

    def to_dict(self):

        result = {
            "__class__": self.__class__.__name__
        }

        for key, value in self.__dict__.items():

            if key in self.__exclude__:
                continue

            if isinstance(value, Dicty):
                result[key] = value.to_dict()
            else:
                result[key] = value

        return result

    @classmethod
    def from_dict(cls, data):

        class_name = data["__class__"]
        target_cls = cls.registry[class_name]

        obj = target_cls.__new__(target_cls)

        for key, value in data.items():

            if key == "__class__":
                continue

            if isinstance(value, dict) and "__class__" in value:
                value = cls.from_dict(value)

            setattr(obj, key, value)

        return obj

    def save_yaml(self, path):

        """
        Save this object to a YAML file at `path`.
        """

        data = self.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    @classmethod
    def from_yaml_file(cls, path):

        """
        Load an object from a YAML file at `path`.
        """

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data is None:
            raise ValueError(f"YAML file {path} is empty or invalid")

        return cls.from_dict(data)

def get_timestamp()->str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')  

def setup_logging(
    level=logging.INFO,
    log_path: str | Path | None = None,
    force: bool = False
    ):
    root = logging.getLogger()

    # tekrar eklenmesini engelle
    if force:
        root.handlers.clear()
    elif root.handlers:
        return root

    root.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s | %(message)s",
        datefmt="%H:%M:%S"
    )

    # Terminal çıktısı
    rich_handler = RichHandler(
        rich_tracebacks=True,
        show_time=False,
        show_level=True,
        show_path=True,
        markup=True
    )
    rich_handler.setFormatter(formatter)
    root.addHandler(rich_handler)

    # Dosyaya kayıt
    if log_path is not None:
        log_path = Path(log_path)
        file_handler = logging.FileHandler(
            str(log_path),
            encoding="utf-8"
        )

        file_handler.setFormatter(formatter)

        root.addHandler(file_handler)

    return root