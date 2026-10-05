set -euo pipefail
cd source
$PY -m pytest mini_dreamer4/tests/test_units.py -q -p no:cacheprovider
cd ..
echo "--- network"
for h in api.wandb.ai github.com; do timeout 10 curl -sS -o /dev/null -w "$h http=%{http_code}\n" https://$h || echo "$h UNREACHABLE"; done
echo "--- pusht"
$PY - <<'PY'
import numpy as np, zarr, torch
from mini_dreamer4.data import load_pusht_zarr
import os
root = zarr.open(os.environ["PUSHT"], mode="r")
img = np.asarray(root["data"]["img"][:2000])
print("img dtype", img.dtype, "min/max", img.min(), img.max(), "integer-valued:", bool((img == img.round()).all()))
eps = load_pusht_zarr(os.environ["PUSHT"])
a = np.concatenate([e["actions"] for e in eps])
print("episodes", len(eps), "frames", sum(len(e["video"]) for e in eps), "video", eps[0]["video"].shape, eps[0]["video"].dtype,
      "action min/max", a.min(0), a.max(0))
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name())
PY
echo "--- timing: synthetic 64px + lpips"
$PY -m mini_dreamer4.train tokenizer --data synthetic --synthetic-episodes 40 --image-size 64 --batch-size 32 --lpips-weight 0.2 \
    --steps 200 --log-every 100 --out runs/preflight/tok64
$PY -m mini_dreamer4.train dynamics --data synthetic --synthetic-episodes 40 --tokenizer runs/preflight/tok64/tokenizer.pt --seq-len 16 \
    --batch-size 32 --depth 8 --time-every 2 --steps 200 --log-every 100 --bootstrap-warmup 100 --out runs/preflight/dyn64
echo "--- timing: pusht 96px + lpips"
$PY -m mini_dreamer4.train tokenizer --data $PUSHT --image-size 96 --batch-size 32 --lpips-weight 0.2 \
    --steps 200 --log-every 100 --out runs/preflight/tok96
echo PREFLIGHT_OK
