from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.dependencies import get_db
from app.crud.inference_config import get_config, update_config, _DEFAULTS
from app.schemas.inference_config import InferenceConfigResponse, InferenceConfigUpdate


router = APIRouter(
    prefix="/settings",
    tags=["Settings"],
)


@router.get(
    "/inference",
    response_model=InferenceConfigResponse,
)
def read_inference_config(db: Session = Depends(get_db)):
    return get_config(db)


@router.put(
    "/inference",
    response_model=InferenceConfigResponse,
)
def update_inference_config(
    body: InferenceConfigUpdate,
    db: Session = Depends(get_db),
):
    current = get_config(db)
    updates = body.model_dump(exclude_unset=True)

    if not updates:
        return current

    # Cross-validate against existing DB values when only one side of a
    # pass/count pair is present in this request.
    m1_count = updates.get("model1_frame_count", current.model1_frame_count)
    m1_pass = updates.get("model1_pass_frames", current.model1_pass_frames)
    if m1_pass > m1_count:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"model1_pass_frames ({m1_pass}) must be "
                f"<= model1_frame_count ({m1_count})"
            ),
        )

    m2_count = updates.get("model2_frame_count", current.model2_frame_count)
    m2_pass = updates.get("model2_pass_frames", current.model2_pass_frames)
    if m2_pass > m2_count:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"model2_pass_frames ({m2_pass}) must be "
                f"<= model2_frame_count ({m2_count})"
            ),
        )

    return update_config(db, **updates)

@router.post(
    "/inference/restore",
    response_model=InferenceConfigResponse,
)
def restore_inference_config(db: Session = Depends(get_db)):
    """Restores the database configuration to backend defaults."""
    return update_config(db, **_DEFAULTS)
