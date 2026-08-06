from pydantic import BaseModel, ConfigDict, model_validator, Field
from typing import Optional


class InferenceConfigResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    model1_frame_count: int
    model1_pass_frames: int
    model2_start_skip_frame: int
    model2_frame_count: int
    model2_pass_frames: int
    socket_absent_frames: int
    socket_loss_abort_frames: int
    enable_debug_logging: bool
    enable_perf_logging: bool
    default_output_dir: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class InferenceConfigUpdate(BaseModel):
    model1_frame_count: Optional[int] = Field(None, gt=0)
    model1_pass_frames: Optional[int] = Field(None, gt=0)
    model2_start_skip_frame: Optional[int] = Field(None, ge=0)
    model2_frame_count: Optional[int] = Field(None, gt=0)
    model2_pass_frames: Optional[int] = Field(None, gt=0)
    socket_absent_frames: Optional[int] = Field(None, gt=0)
    socket_loss_abort_frames: Optional[int] = Field(None, gt=0)
    enable_debug_logging: Optional[bool] = None
    enable_perf_logging: Optional[bool] = None
    default_output_dir: Optional[str] = None

    @model_validator(mode="after")
    def check_pass_lte_count(self):
        """
        Enforce: pass_frames <= frame_count for both Model1 and Model2.
        When only one of the pair is being updated, the validator allows it
        through — the route layer must cross-check against the current DB
        values.
        """
        if (
            self.model1_pass_frames is not None
            and self.model1_frame_count is not None
            and self.model1_pass_frames > self.model1_frame_count
        ):
            raise ValueError(
                "model1_pass_frames must be <= model1_frame_count"
            )
        if (
            self.model2_pass_frames is not None
            and self.model2_frame_count is not None
            and self.model2_pass_frames > self.model2_frame_count
        ):
            raise ValueError(
                "model2_pass_frames must be <= model2_frame_count"
            )
        return self
