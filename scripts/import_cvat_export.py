"""SUPERSEDED -- use import_annotations.py

We moved from CVAT Online to labelme (the CVAT free tier caps how many tasks
you can create, and we need five). import_annotations.py handles labelme JSON
by default and still accepts CVAT zips with --format cvat.

    python scripts\\import_annotations.py --exports ... --map ... --out ... --format labelme

This file is safe to delete.
"""
import sys

print(__doc__)
sys.exit(1)
