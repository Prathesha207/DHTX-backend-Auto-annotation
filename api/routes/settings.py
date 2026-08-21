from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, model_validator

from database import database
from crud import crud

router = APIRouter(tags=["Settings"])

class ModelSettingResponse(BaseModel):
    m1_total: int
    m1_pass: int
    m2_total: int
    m2_pass: int

    class Config:
        from_attributes = True

class ModelSettingUpdate(BaseModel):
    m1_total: int
    m1_pass: int
    m2_total: int
    m2_pass: int

    @model_validator(mode='after')
    def check_pass_frames(self) -> 'ModelSettingUpdate':
        if not (1 <= self.m1_pass <= self.m1_total):
            raise ValueError('m1_pass must be between 1 and m1_total')
        if not (1 <= self.m2_pass <= self.m2_total):
            raise ValueError('m2_pass must be between 1 and m2_total')
        return self

@router.get("/settings/model", response_model=ModelSettingResponse)
async def get_model_setting(db: Session = Depends(database.get_db)):
    """
    Get the currently active model settings for Model 1 and Model 2.
    """
    setting = crud.get_active_model_setting(db)
    return setting

from fastapi import HTTPException

@router.put("/settings/model", response_model=ModelSettingResponse)
async def update_model_setting(
    settings: ModelSettingUpdate, 
    db: Session = Depends(database.get_db)
):
    """
    Update the active model settings.
    """
    try:
        new_setting = crud.update_model_setting(
            db,
            m1_total=settings.m1_total,
            m1_pass=settings.m1_pass,
            m2_total=settings.m2_total,
            m2_pass=settings.m2_pass
        )
        return new_setting
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
