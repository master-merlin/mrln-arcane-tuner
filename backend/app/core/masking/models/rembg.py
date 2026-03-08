from PIL import Image
from app.core.masking.models.base import MaskingModel
import structlog

logger = structlog.get_logger(__name__)

try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

class RemBGModel(MaskingModel):
    def __init__(self, service):
        self.service = service
        self.sessions = {}

    @property
    def model_id(self) -> str:
        return "rembg"

    def load(self) -> bool:
        if not REMBG_AVAILABLE:
            raise ImportError("rembg library is not installed.")
        return True

    def unload(self):
        self.sessions = {}

    def generate(self, image: Image.Image, params: dict) -> Image.Image:
        if not REMBG_AVAILABLE:
            raise ImportError("rembg not installed.")
        
        model_name = params.get("model_name", "u2net")
        alpha_matting = params.get("alpha_matting", False)
        af_threshold = int(params.get("alpha_matting_foreground_threshold", 240))
        ab_threshold = int(params.get("alpha_matting_background_threshold", 10))
        ae_size = int(params.get("alpha_matting_erode_size", 10))
        pp_mask = params.get("post_process_mask", False)
        
        
        logger.info("using_rembg_model", model_name=model_name, alpha_matting=alpha_matting)
        
        if model_name not in self.sessions:
            self.sessions[model_name] = new_session(model_name)
            
        try:
            output = remove(
                image, 
                session=self.sessions[model_name],
                alpha_matting=alpha_matting,
                alpha_matting_foreground_threshold=af_threshold,
                alpha_matting_background_threshold=ab_threshold,
                alpha_matting_erode_size=ae_size,
                post_process_mask=pp_mask
            )
            
            if output.mode == 'RGBA':
                return output.split()[3].convert("L")
            else:
                return output.convert("L")
                
        except Exception as e:
            logger.error("rembg_error", error=str(e))
            raise e
