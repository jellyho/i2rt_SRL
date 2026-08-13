"""msgpack with NumPy support, byte-compatible with openpi's ``openpi_client.msgpack_numpy``.

This is deliberately **not** the pip ``msgpack-numpy`` package. The two encode arrays
differently — pip uses ``{b'nd': True, b'type': ..., b'data': ...}``, openpi uses
``{b'__ndarray__': True, b'dtype': ..., b'data': ...}`` — and neither side raises when it
meets the other's format: the array simply arrives as a plain dict. A policy server would
then fail deep inside a transform, or worse, silently receive garbage, while the websocket
connection and the metadata handshake both look perfectly healthy.

openpi is the standard here, so we speak its encoding. Adapted from openpi, which in turn
adapted it from https://github.com/lebedov/msgpack-numpy — the reason neither uses that
library directly is that it falls back to pickle for object arrays.
"""

from __future__ import annotations

import functools

import msgpack
import numpy as np


def pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)

Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)
