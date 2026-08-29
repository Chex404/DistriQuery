"""API key authentication.

Deliberately simple: one random API key per tenant, checked against the
tenants table on every request via the X-API-Key header. No JWTs, sessions,
or OAuth — those would be genuine overengineering for a project whose
actual research question is about ingestion reliability and retrieval
quality, not auth protocol design.

There's no signup endpoint on purpose — tenants are provisioned by an
operator via scripts/create_tenant.py, not self-service.
"""

from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session

from distriquery.db.models import Tenant
from distriquery.db.session import get_db


def get_current_tenant(
    x_api_key: str = Header(None, description="Your tenant API key"),
    db: Session = Depends(get_db),
) -> Tenant:
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")

    tenant = db.query(Tenant).filter(Tenant.api_key == x_api_key).one_or_none()
    if tenant is None:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return tenant