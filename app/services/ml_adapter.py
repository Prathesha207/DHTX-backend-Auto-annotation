from app.services.inference_settings import settings

if settings.ml_pipeline == "new":
    import models.inference_video_full_detection_new as engine
else:
    # pyrefly: ignore [missing-import]
    import models.inference_video_full_detection as engine

# Re-export EVERYTHING dynamically to avoid ImportError
import sys
module = sys.modules[__name__]
for attr in dir(engine):
    if not attr.startswith('__'):
        setattr(module, attr, getattr(engine, attr))
