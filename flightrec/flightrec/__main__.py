# File: __main__.py
# Location: flightrec/__main__.py
# Purpose: Module entry so `python -m flightrec ...` works.
# Dependencies: flightrec.cli

"""Entry point for ``python -m flightrec``."""

from flightrec.cli import main

raise SystemExit(main())
