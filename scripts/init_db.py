"""Run this once before starting the API for the first time (and again any
time you add a new model), against a running Postgres instance:

    python scripts/init_db.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distriquery.db.session import DATABASE_URL, create_all_tables  # noqa: E402


def main():
    print(f"Creating tables at {DATABASE_URL} ...")
    create_all_tables()
    print("Done.")


if __name__ == "__main__":
    main()