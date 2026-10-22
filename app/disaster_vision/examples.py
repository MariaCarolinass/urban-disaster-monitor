"""Load example videos only when requested."""

from huggingface_hub import hf_hub_download

from .config import EXAMPLE_VIDEO_CACHE_DIR


def load_example_video(filename):
    EXAMPLE_VIDEO_CACHE_DIR.mkdir(exist_ok=True)
    return hf_hub_download(
        repo_id="carolinasoares/urban-disaster-examples",
        filename=filename,
        cache_dir=EXAMPLE_VIDEO_CACHE_DIR,
    )
