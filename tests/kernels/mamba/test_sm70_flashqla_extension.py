# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the SM70 FlashQLA extension loader."""

from __future__ import annotations

from types import ModuleType

import pytest

from flash_qla.ops.gated_delta_rule.chunk.sm70 import fused_fwd


@pytest.fixture(autouse=True)
def reset_extension_load_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fused_fwd, "_EXT", None)
    monkeypatch.setattr(fused_fwd, "_EXT_LOAD_ERROR", None)
    monkeypatch.setattr(fused_fwd.torch.cuda, "is_available", lambda: True)


def test_sm70_flashqla_prefers_bundled_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundled = ModuleType("flash_qla_sm70_gdn_strided")
    monkeypatch.setattr(fused_fwd.importlib, "import_module", lambda name: bundled)
    monkeypatch.setattr(
        fused_fwd,
        "load",
        lambda **kwargs: pytest.fail("JIT fallback must not run"),
    )

    assert fused_fwd._load_ext() is bundled


def test_sm70_flashqla_jit_fallback_uses_fixed_arches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = ModuleType("flash_qla_sm70_gdn_strided")
    load_kwargs = {}

    def missing_bundled(name: str):
        error = ModuleNotFoundError(name=name)
        raise error

    def fake_load(**kwargs):
        load_kwargs.update(kwargs)
        return loaded

    monkeypatch.setattr(fused_fwd.importlib, "import_module", missing_bundled)
    monkeypatch.setattr(fused_fwd, "load", fake_load)

    assert fused_fwd._load_ext() is loaded
    assert load_kwargs["extra_cuda_cflags"] == [
        "-O3",
        "-gencode=arch=compute_70,code=sm_70",
        "-gencode=arch=compute_75,code=sm_75",
    ]


def test_sm70_flashqla_retains_bundled_loader_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial_error = OSError("undefined symbol")
    import_calls = 0

    def failing_import(name: str):
        nonlocal import_calls
        import_calls += 1
        raise initial_error

    monkeypatch.setattr(fused_fwd.importlib, "import_module", failing_import)

    with pytest.raises(RuntimeError, match="failed to load") as first_error:
        fused_fwd._load_ext()
    assert first_error.value.__cause__ is initial_error

    with pytest.raises(RuntimeError, match="previously failed") as retry_error:
        fused_fwd._load_ext()
    assert retry_error.value.__cause__ is initial_error
    assert import_calls == 1
