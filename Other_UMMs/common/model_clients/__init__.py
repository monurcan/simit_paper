"""Backend-name -> UMMClient factory, so scripts can take a --model flag
instead of hardcoding a client import. Every script that previously did
`from common.model_clients.sensenova_client import SensenovaClient` and
constructed it directly still works unchanged -- this is purely additive."""

from .base import UMMClient

BACKENDS = ("sensenova", "lance")


def get_client(backend: str = "sensenova", **kwargs) -> UMMClient:
    if backend == "sensenova":
        from .sensenova_client import SensenovaClient
        return SensenovaClient(**kwargs)
    if backend == "lance":
        from .lance_client import LanceClient
        return LanceClient(**kwargs)
    raise ValueError(f"unknown model backend '{backend}', expected one of {BACKENDS}")
