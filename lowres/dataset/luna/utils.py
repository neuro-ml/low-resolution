from pathlib import Path
from typing import Sequence, Any

import numpy as np

from bev import Repository
from bev.interface import Version
from bev.utils import PathLike, HashNotFound

from connectome import CacheToDisk as Disk
from connectome.cache import is_stable
from connectome.serializers import NumpySerializer, JsonSerializer, DictSerializer
from connectome.storage.storage import QueryError


_NO_ARG = object()

REPOSITORY = Repository.from_here('../../../assets')
LATEST_COMMIT = REPOSITORY.latest_version()


@is_stable
def glob(*parts: PathLike, version: Version = None) -> Sequence[Path]:
    return REPOSITORY.glob(*parts, version=version)


@is_stable
def read(loader, *parts: PathLike, version: Version, default=_NO_ARG, **kwargs) -> Any:
    try:
        key = REPOSITORY.get_key(*parts, version=version)
        return REPOSITORY.storage.load(loader, key, **kwargs)
    except (QueryError, HashNotFound):
        if default == _NO_ARG:
            raise
        return default


def _default_serializer(serializers):
    if serializers is None:
        arrays = NumpySerializer({np.bool_: 1, np.int_: 1})
        serializers = [
            JsonSerializer(),
            DictSerializer(serializer=arrays),
            arrays,
        ]
    return serializers


class CacheToDisk(Disk):
    def __init__(self, names, serializers=None, **kwargs):
        super().__init__(
            REPOSITORY.cache, REPOSITORY.storage,
            serializer=_default_serializer(serializers), names=names, **kwargs
        )
