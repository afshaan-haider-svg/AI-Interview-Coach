"""API routes for local/demo user management."""

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import User
from app.interview.schemas import DemoUserResponse

router = APIRouter(prefix="/users", tags=["Users"])

DEMO_USER_EMAIL = "demo_candidate@interviewcoach.local"
DEMO_USER_NAME = "Demo Candidate"


def _get_or_create_demo_user(db: Session) -> User:
    user = db.query(User).filter(User.email == DEMO_USER_EMAIL).first()
    if not user:
        user = User(name=DEMO_USER_NAME, email=DEMO_USER_EMAIL)
        db.add(user)
        db.commit()
        db.refresh(user)
    return user


@router.post(
    "/demo",
    response_model=DemoUserResponse,
    status_code=status.HTTP_200_OK,
    summary="Get or initialize default demo candidate account",
    description="Retrieves the persistent local demo user, initializing it if it does not yet exist.",
)
def create_demo_user(db: Session = Depends(get_db)) -> DemoUserResponse:
    """Returns or provisions the default demo user."""
    user = _get_or_create_demo_user(db)
    return DemoUserResponse(id=user.id, name=user.name, email=user.email)


@router.get(
    "/demo",
    response_model=DemoUserResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve default demo candidate account",
    description="Retrieves the persistent local demo user without requiring manual login credentials.",
)
def get_demo_user(db: Session = Depends(get_db)) -> DemoUserResponse:
    """Returns the default demo user."""
    user = _get_or_create_demo_user(db)
    return DemoUserResponse(id=user.id, name=user.name, email=user.email)
