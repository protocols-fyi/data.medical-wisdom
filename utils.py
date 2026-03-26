"""Project-local runtime utilities."""

from datetime import datetime
import logging
from pathlib import Path
from zoneinfo import ZoneInfo


NEW_ZEALAND_TZ = ZoneInfo("Pacific/Auckland")


def init_logging(*, level: int = logging.INFO) -> Path:
    run_dir = Path("run").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f"{datetime.now(NEW_ZEALAND_TZ).strftime('%Y%m%d-%H%M%S')}.log"

    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    root_logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    formatter.converter = (
        lambda timestamp: datetime.fromtimestamp(timestamp, NEW_ZEALAND_TZ).timetuple()
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)
    return log_path
