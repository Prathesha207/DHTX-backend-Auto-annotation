# # quick_test.py
# from video_metadata import probe_video
# print(probe_video("D:\dhtx-videos\session_2026-06-10_05-45-37-224720_ANOMALY_inf_1781019865459_cycle206.mp4"))

from app.services.inference_settings import settings

print(settings.frame_skip)
print(settings.warmup_frames)