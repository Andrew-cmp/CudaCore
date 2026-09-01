"""Compatibility entry for the original environment-variable NCU workflow.

New commands should use run_ncu.sh, which invokes test.py --profile directly.
"""

import os

from test import main


if __name__ == "__main__":
    kernel = os.environ.get("HGEMM_NCU_KERNEL", "hgemm_v0")
    size = os.environ.get("HGEMM_NCU_SIZE", "4096")
    main(
        [
            "--profile",
            "--kernel",
            kernel,
            "--size",
            size,
            "--warmup",
            "0",
            "--repeat",
            "1",
        ]
    )
