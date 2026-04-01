"""
Transfer-train GTARL agents for multiple DC counts from a base 12-DC model.

Usage:
    python transfer_train.py                          # Train all DC counts
    python transfer_train.py --dc-counts 4 8          # Train specific counts
    python transfer_train.py --steps 200              # More fine-tuning steps
    python transfer_train.py --source models/gtarl    # Custom source dir

Each DC count gets its own model directory:
    models/gtarl_4dc/gtarl_agents.pt
    models/gtarl_6dc/gtarl_agents.pt
    models/gtarl_8dc/gtarl_agents.pt
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(
        description="Transfer-train GTARL agents for different DC counts")
    parser.add_argument('--dc-counts', type=int, nargs='+', default=[4, 6, 8],
                        help="DC counts to transfer-train (default: 4 6 8)")
    parser.add_argument('--source', type=str, default='models/gtarl',
                        help="Source model directory (base 12-DC model)")
    parser.add_argument('--source-dcs', type=int, default=12,
                        help="DC count the source model was trained on")
    parser.add_argument('--steps', type=int, default=100,
                        help="Offline fine-tuning steps after transfer (default: 100)")
    parser.add_argument('--target-util', type=float, default=0.95,
                        help="Target utilization for autoscaling during training")
    parser.add_argument('--output-base', type=str, default='models/gtarl',
                        help="Base path for output dirs (appends _{n}dc)")
    args = parser.parse_args()

    source_path = os.path.join(args.source, "gtarl_agents.pt")
    if not os.path.exists(source_path):
        print(f"[ERROR] Base model not found: {source_path}")
        print(f"        Train it first with:")
        print(f"        python simulator_LLM.py --framework parliament "
              f"--target-util {args.target_util} --num-dcs {args.source_dcs} "
              f"--offline-train 500 --model-dir {args.source}")
        sys.exit(1)

    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║  GTARL Transfer Training                        ║")
    print(f"╠══════════════════════════════════════════════════╣")
    print(f"║  Source : {source_path:<39} ║")
    print(f"║  DC counts : {str(args.dc_counts):<36} ║")
    print(f"║  Steps     : {args.steps:<36} ║")
    print(f"║  Util      : {args.target_util:<36} ║")
    print(f"╚══════════════════════════════════════════════════╝")

    results = {}
    total_start = time.time()

    for i, num_dcs in enumerate(args.dc_counts):
        if num_dcs == args.source_dcs:
            print(f"\n[{i+1}/{len(args.dc_counts)}] {num_dcs} DCs — same as source, skipping")
            results[num_dcs] = "skipped (same as source)"
            continue

        target_dir = f"{args.output_base}_{num_dcs}dc"
        target_path = os.path.join(target_dir, "gtarl_agents.pt")

        if os.path.exists(target_path):
            print(f"\n[{i+1}/{len(args.dc_counts)}] {num_dcs} DCs — "
                  f"model exists at {target_path}")
            overwrite = input("  Overwrite? [y/N]: ").strip().lower()
            if overwrite != 'y':
                results[num_dcs] = "skipped (exists)"
                continue

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(args.dc_counts)}] Transfer training: "
              f"{args.source_dcs} DCs → {num_dcs} DCs")
        print(f"  Output: {target_dir}")
        print(f"  Steps:  {args.steps}")
        print(f"{'='*60}")

        cmd = [
            sys.executable, "-u", "simulator_LLM.py",
            "--framework",       "parliament",
            "--num-dcs",         str(num_dcs),
            "--target-util",     str(args.target_util),
            "--transfer-from",   source_path,
            "--offline-train",   str(args.steps),
            "--model-dir",       target_dir,
            "--autoscale-mode",  "global_peak",
        ]

        step_start = time.time()
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
            for line in process.stdout:
                print(line, end="")
            process.wait()

            elapsed = (time.time() - step_start) / 60.0
            if process.returncode == 0:
                # Verify the saved checkpoint has the correct DC count
                try:
                    import torch
                    ckpt_path = os.path.join(target_dir, "gtarl_agents.pt")
                    if os.path.exists(ckpt_path):
                        ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
                        agents = ckpt.get("agents", {})
                        if agents:
                            saved_dcs = next(iter(agents.values())).get("num_dcs", -1)
                            if saved_dcs != num_dcs:
                                print(f"  [WARN] Checkpoint saved with {saved_dcs} DCs "
                                      f"instead of {num_dcs} — fixing...")
                                # Fix: update num_dcs in each agent entry
                                for ag_name in agents:
                                    agents[ag_name]["num_dcs"] = num_dcs
                                torch.save(ckpt, ckpt_path)
                                print(f"  [FIXED] Checkpoint updated to {num_dcs} DCs")
                            else:
                                print(f"  [VERIFIED] Checkpoint has correct {num_dcs} DCs")
                        del ckpt
                except Exception as ve:
                    print(f"  [WARN] Could not verify checkpoint: {ve}")

                results[num_dcs] = f"done in {elapsed:.1f} min"
                print(f"\n[✓] {num_dcs} DCs complete — {elapsed:.1f} min")
            else:
                results[num_dcs] = f"failed (exit={process.returncode})"
                print(f"\n[✗] {num_dcs} DCs failed — exit code {process.returncode}")

        except Exception as exc:
            results[num_dcs] = f"error: {exc}"
            print(f"\n[✗] {num_dcs} DCs error: {exc}")

    # Summary
    total_time = (time.time() - total_start) / 60.0
    print(f"\n{'='*60}")
    print(f"  Transfer Training Summary ({total_time:.1f} min total)")
    print(f"{'='*60}")
    for num_dcs in args.dc_counts:
        status = results.get(num_dcs, "not run")
        model_dir = f"{args.output_base}_{num_dcs}dc" if num_dcs != args.source_dcs else args.source
        exists = "✓" if os.path.exists(os.path.join(model_dir, "gtarl_agents.pt")) else "✗"
        print(f"  {num_dcs:>2} DCs: [{exists}] {model_dir:<30}  {status}")
    print(f"{'='*60}")
    print(f"\nTo run inference:")
    print(f"  python simulator_LLM.py --framework parliament --target-util 0.95 \\")
    print(f"      --num-dcs <N> --load-model --model-dir models/gtarl_<N>dc")


if __name__ == "__main__":
    main()