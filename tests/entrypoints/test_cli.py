# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import subprocess
import sys

from vllm.version import __version__


def test_cli_version_uses_package_version():
    result = subprocess.run(
        [sys.executable, "-m", "vllm.entrypoints.cli.main", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == __version__
