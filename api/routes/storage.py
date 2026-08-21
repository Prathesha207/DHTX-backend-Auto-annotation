from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel

from database import database
from crud import crud

router = APIRouter(tags=["Storage"])

class StorageResponse(BaseModel):
    id: int
    root_path: str
    is_active: bool

class StorageUpdate(BaseModel):
    root_path: str

@router.get("/storage", response_model=StorageResponse)
async def get_storage(db: Session = Depends(database.get_db)):
    """
    Get the currently active storage location.
    """
    active_storage = crud.get_active_storage(db)
    if not active_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active storage setting found."
        )
    return active_storage

@router.put("/storage", response_model=StorageResponse)
async def update_storage(
    settings: StorageUpdate, 
    db: Session = Depends(database.get_db)
):
    """
    Update the active storage location. 
    This is where all new video uploads and inference results will be saved.
    """
    new_storage = crud.set_active_storage(db, new_root_path=settings.root_path)
    return new_storage
