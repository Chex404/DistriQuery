"""Create a new tenant and print its API key.

Usage:
    python scripts/create_tenant.py "Acme Corp"

Run this once per tenant you want to test with. The printed API key is
what you pass as the X-API-Key header on every /documents and /query
request for that tenant.
"""

import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distriquery.db.models import Tenant  # noqa: E402
from distriquery.db.session import SessionLocal  # noqa: E402


def main():
    if len(sys.argv) != 2:
        print('Usage: python scripts/create_tenant.py "<tenant name>"')
        sys.exit(1)

    name = sys.argv[1]
    api_key = secrets.token_hex(16)

    db = SessionLocal()
    try:
        tenant = Tenant(name=name, api_key=api_key)
        db.add(tenant)
        db.commit()
        db.refresh(tenant)
        print(f"Created tenant '{tenant.name}' (id={tenant.id})")
        print(f"API key: {tenant.api_key}")
    finally:
        db.close()


if __name__ == "__main__":
    main()