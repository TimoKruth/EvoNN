"""Evolution pipeline orchestration modules.

Public helpers live in sibling modules such as ``archive``, ``coordinator``,
``evaluate``, and ``reproduce``. This package initializer intentionally avoids
re-exporting them to keep imports side-effect free and preserve the existing
module-level API.
"""

from __future__ import annotations

__all__: tuple[str, ...] = ()
