from PIL import Image
from app.core.masking.models.base import MaskingModel
import structlog

logger = structlog.get_logger(__name__)

# `except Exception`, deliberately, NOT `except ImportError` (ARCHITECTURE D1:
# nothing imported at startup may raise, ever). On 2026-09-03 the cu128 image
# died on its FIRST launch with
#     RuntimeError: cannot cache function '_make_tree': no locator available
#     for .../pymatting/util/kdtree.py
# raised while importing rembg as a non-root user: numba tried to write its JIT
# cache next to the source file inside site-packages and had nowhere to fall
# back to. A RuntimeError is not an ImportError, so this guard did not catch
# it, the exception escaped `import app.main`, and uvicorn never started. An
# optional dependency that is merely ABSENT and one that is present but cannot
# initialise are the same thing to this app -- masking degrades -- so the guard
# has to be about the outcome, not about one exception class.
# The reason is kept rather than swallowed: `load()` puts it in front of the
# user, so this stays a diagnosis instead of a silent "unavailable".
try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
    REMBG_UNAVAILABLE_REASON = ""
except Exception as exc:  # noqa: BLE001 — see the note above
    REMBG_AVAILABLE = False
    REMBG_UNAVAILABLE_REASON = f"{type(exc).__name__}: {exc}"

class RemBGModel(MaskingModel):
    def __init__(self, service):
        self.service = service
        self.sessions = {}

    @property
    def model_id(self) -> str:
        return "rembg"

    def load(self) -> bool:
        if not REMBG_AVAILABLE:
            raise ImportError(f"rembg is unavailable — {REMBG_UNAVAILABLE_REASON}")
        return True

    def unload(self):
        self.sessions = {}

    def generate(self, image: Image.Image, params: dict) -> Image.Image:
        if not REMBG_AVAILABLE:
            raise ImportError(f"rembg is unavailable — {REMBG_UNAVAILABLE_REASON}")
        
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
