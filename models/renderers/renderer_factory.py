from models.renderers.base_renderer import BaseRenderer, NullRenderer
from models.renderers.current_renderer import CurrentRenderer
from models.renderers.production_renderer import ProductionRenderer

RENDERERS = {
    "current": CurrentRenderer,
    "production": ProductionRenderer,
    "null": NullRenderer,
}

def get_renderer(renderer_type: str) -> BaseRenderer:
    """
    Instantiates and returns the appropriate renderer based on the given type.
    Falls back to CurrentRenderer if the type is unknown.
    """
    renderer_cls = RENDERERS.get(renderer_type.lower(), CurrentRenderer)
    return renderer_cls()
