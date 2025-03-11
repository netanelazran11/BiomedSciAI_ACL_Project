import types
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tokenizers import EncodeInput

# TODO: fix this after updating huggingface tokenizers version
EncodedInput = list[int]


@dataclass
class MultiFieldInstance(Mapping):
    data: Mapping[str, EncodeInput]
    metadata: Mapping[str, Any] | None = None

    def __getitem__(self, key):
        return self.data[key]

    def __iter__(self):
        return iter(self.data.items())

    def __len__(self):
        return len(self.data)

    def keys(self):
        return list(self.data.keys())

    def values(self):
        return list(self.data.values())

    @property
    def seq_length(self):
        return len(next(iter(self.data.values())))


@dataclass(frozen=True)
class MultiFieldEncodedInstance(Mapping):
    data: Mapping[str, EncodedInput]
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self):
        object.__setattr__(self, "data", types.MappingProxyType(self.data))
        if self.metadata:
            object.__setattr__(self, "metadata", types.MappingProxyType(self.metadata))

    def __getitem__(self, key):
        return self.data[key]

    def __iter__(self):
        return iter(self.data.items())

    def __len__(self):
        return len(self.data)

    # keys returns list of data keys
    def keys(self):
        return list(self.data.keys())

    def values(self):
        return list(self.data.values())
