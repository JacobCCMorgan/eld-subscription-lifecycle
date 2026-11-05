"""`sublife` command line: run the full pipeline."""
from __future__ import annotations

import sys

from .run import main as run_main


def main(argv=None):
    return run_main(argv)


if __name__ == "__main__":
    sys.exit(main())
