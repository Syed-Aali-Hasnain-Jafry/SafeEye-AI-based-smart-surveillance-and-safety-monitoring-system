
from pathlib import Path
from huggingface_hub import hf_hub_download

BASE = Path(__file__).resolve().parent.parent
model_dir = BASE / "models"
model_dir.mkdir(exist_ok=True)

filename = (
    "osnet_x1_0_msmt17_combineall_256x128_amsgrad_"
    "ep150_stp60_lr0.0015_b64_fb10_softmax_"
    "labelsmooth_flip_jitter.pth"
)

path = hf_hub_download(
    repo_id="kaiyangzhou/osnet",
    filename=filename,
    local_dir=str(model_dir)
)

print("OSNet weights downloaded:")
print(path)
