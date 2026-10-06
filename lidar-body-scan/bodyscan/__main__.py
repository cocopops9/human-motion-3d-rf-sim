"""Entry point: python -m bodyscan <command> [options]."""

from bodyscan.commands import main

if __name__ == "__main__":
    raise SystemExit(main())
