"""Fix checkpoints that were saved with wrong DC counts after transfer training."""
import os
import sys
import torch

DIRS = {
    4: "models/gtarl_4dc",
    6: "models/gtarl_6dc",
    8: "models/gtarl_8dc",
}

for expected_dcs, model_dir in DIRS.items():
    path = os.path.join(model_dir, "gtarl_agents.pt")
    if not os.path.exists(path):
        print(f"  {expected_dcs} DCs: no checkpoint at {path}")
        continue

    ckpt = torch.load(path, weights_only=False, map_location="cpu")
    agents = ckpt.get("agents", {})
    if not agents:
        print(f"  {expected_dcs} DCs: no agents in checkpoint")
        continue

    saved_dcs = next(iter(agents.values())).get("num_dcs", -1)
    if saved_dcs == expected_dcs:
        print(f"  {expected_dcs} DCs: OK (already correct)")
        continue

    print(f"  {expected_dcs} DCs: BAD — has {saved_dcs} DCs, deleting bad checkpoint")
    os.remove(path)
    print(f"  Deleted {path}")
    print(f"  Re-run: python transfer_train.py --dc-counts {expected_dcs} --steps 100")

print("\nDone.")