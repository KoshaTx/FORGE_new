"""Entry point for `python -m cli`, kept separate so importing the package has no effect."""

from cli import main

raise SystemExit(main())
