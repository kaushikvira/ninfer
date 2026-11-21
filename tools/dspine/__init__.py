"""CPU-side DSpine draft tooling (P2 stage 1: draft forward).

The engine binds a DSpine draft through ``src/models/common/dspine`` and runs the
forward on the device.  The device is production-reserved for the whole of P2
(plan STATUS `⚠` P2 GPU reality), so the same binding geometry and the same draft
forward are exercised on the host: ``artifact.py`` reads + dequantizes the draft
region of a real ``.ninfer`` v3 file, and ``cpu_forward.py`` reproduces the
forward's arithmetic in NumPy at the op storage precisions.

Nothing here writes to ``/home/kv/models``.
"""

from __future__ import annotations
