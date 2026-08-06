import sys
from pathlib import Path
from app.plugins.base_plugin import BaseMLPlugin

from models.inference_video_full_detection import process_video_cycles

class DHTXPlugin(BaseMLPlugin):
    def initialize(self, config: dict):
        # The models should be passed in via config or loaded here.
        # Since MLRunner (the controller) loads the models into ModelManager, we can just pass them.
        self.seg_net = config.get("seg_net")
        self.yolo_socket = config.get("yolo_socket")
        self.yolo_pose = config.get("yolo_pose")
        
        self.video_path = config["video_path"]
        self.output_dir = config["output_dir"]
        
        from models.renderers.renderer_factory import get_renderer
        renderer_type = config.get("renderer_type", "production")
        self.renderer = get_renderer(renderer_type)
        
        fps_src = config.get("fps_src", 30.0)
        src_w = config.get("src_w", 1920)
        src_h = config.get("src_h", 1080)

        # Initialize the generator
        self.gen = process_video_cycles(
            video_path=self.video_path,
            resolved_output_dir=self.output_dir,
            seg_net=self.seg_net,
            yolo_socket=self.yolo_socket,
            yolo_pose=self.yolo_pose,
            print_summary=False,
            enable_debug=False,
            yield_mode=True,
            renderer=self.renderer,
            fps_src=fps_src,
            src_w=src_w,
            src_h=src_h,
            config=config
        )
        # Prime the generator to wait for the first frame
        next(self.gen)
        
    def process_frame(self, frame, show_roi=False):
        try:
            # Send (frame, show_roi) to the generator
            ml_result = self.gen.send((frame, show_roi))
            
            # The generator yields the result, then loops back to 'payload = yield'
            # We need to prime it for the next send by calling next() again.
            next(self.gen)
            
            return ml_result
        except StopIteration:
            return None

    def shutdown(self, aborted: bool = True):
        try:
            self.gen.send(None if aborted else "EOF")
        except StopIteration:
            pass
        finally:
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
