"""WebShop Benchmark"""

from .data import download_train_data
from .eval import WebShopEvaluator  # nosec

__all__ = ["WebShopEvaluator", "download_train_data"]
