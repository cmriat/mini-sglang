from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from minisgl.core import SamplingParams

from .utils import deserialize_type, serialize_type


# External types registered here are recognized by the IPC decoder.
# Example: register_msg_type(SFTRequest) allows SFTRequest to be
# sent over ZMQ and deserialized on the scheduler side.
_extra_types: Dict[str, type] = {}


def register_msg_type(cls: type) -> None:
    """Make a type known to the IPC decoder so it can be sent/received over ZMQ."""
    _extra_types[cls.__name__] = cls


@dataclass
class BaseBackendMsg:
    def encoder(self) -> Dict:
        return serialize_type(self)

    @staticmethod
    def decoder(json: Dict) -> "BaseBackendMsg":
        return deserialize_type({**globals(), **_extra_types}, json)


@dataclass
class BatchBackendMsg(BaseBackendMsg):
    data: List[BaseBackendMsg]


@dataclass
class ExitMsg(BaseBackendMsg):
    pass


@dataclass
class UserMsg(BaseBackendMsg):
    uid: int
    input_ids: torch.Tensor  # CPU 1D int32 tensor
    sampling_params: SamplingParams


@dataclass
class AbortBackendMsg(BaseBackendMsg):
    uid: int
