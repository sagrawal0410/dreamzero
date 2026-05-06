#!/usr/bin/env python3
"""Stage 4 / Runner — orchestrate the closed-loop sweep.

For every (method, budget, variant, diffusion_steps, scene) configuration:

  1. Spawn `server_allocation.py` in a subprocess on a free local port.
  2. Wait for the WebSocket server to be reachable (probe loop).
  3. Invoke a configurable client command (default: `eval_utils/run_sim_eval.py`)
     that runs N episodes per scene against the server and writes per-episode
     metrics to its own `runs/<date>/<time>/` directory.
  4. Tear the server down.
  5. Tag and append the resulting metrics into a single `all_episodes.csv`
     under `--output_dir`.

The runner is sim-env agnostic. It can use:

  * the existing `eval_utils/run_sim_eval.py` (Isaac Lab + sim-evals required), or
  * any other client invoked via `--client_cmd` that follows the policy
    protocol (we just substitute `{host}`/`{port}` placeholders), or
  * the bundled `--smoke` mode that drives the server with `test_client_AR.py`
    so you can verify the pipeline without sim-evals installed.

Per-episode metric extraction
-----------------------------

The runner expects the client to write a CSV named `episodes.csv` inside its
working directory with columns at minimum: ``episode, success, progress,
duration_s``. Optional columns: ``contact_failures, time_to_first_contact_s,
final_placement_error, control_freq_hz``. If the client does not emit
``episodes.csv``, the runner derives one minimal row per scene from the
server's diagnostics JSONL (if `--save_diagnostics` was enabled at startup).

Run examples
------------

Full sweep (≈1 h per scene per config; budget the wall time accordingly):

    python scripts/stage4/run_closed_loop.py \\
        --checkpoint /workspace/checkpoints/DreamZero-DROID \\
        --methods none,uniform,center,gripper,online_hybrid \\
        --budgets 100,50,25 \\
        --variants A_hard_retention \\
        --diffusion_steps 8,4 \\
        --scenes 1,2,3 \\
        --episodes 3 \\
        --output_dir runs/stage4_closedloop

Smoke mode (no sim-evals; just exercises the server <-> client pipeline):

    python scripts/stage4/run_closed_loop.py \\
        --checkpoint /workspace/checkpoints/DreamZero-DROID \\
        --methods none,online_hybrid --budgets 50 --variants A_hard_retention \\
        --scenes 1 --episodes 1 --smoke --output_dir runs/stage4_smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shlex
import socket
import subprocess
import sys
import threading
import time
from contextlib import closing
from itertools import product
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stage4.runner")


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent


# ============================================================================
# Subprocess helpers
# ============================================================================


def _free_port(start: int = 7100) -> int:
    """Return a TCP port that is currently free (binds to 0 if start is busy)."""
    for p in range(start, start + 200):
        try:
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("", p))
                return p
        except OSError:
            continue
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


def _wait_for_port(host: str, port: int, timeout: float = 600.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.settimeout(2.0)
            try:
                s.connect((host, port))
                return True
            except Exception:
                time.sleep(2.0)
    return False


def _stream_subprocess_output(proc: subprocess.Popen, prefix: str,
                              log_path: Path) -> threading.Thread:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def _pump() -> None:
        with log_path.open("a") as f:
            for line in proc.stdout:  # type: ignore[union-attr]
                f.write(line)
                f.flush()
                logger.debug("[%s] %s", prefix, line.rstrip())
    t = threading.Thread(target=_pump, daemon=True)
    t.start()
    return t


# ============================================================================
# Server lifecycle
# ============================================================================


def start_server(args: argparse.Namespace, method: str, budget: float, variant: str,
                 diffusion_steps: int | None, port: int,
                 server_log: Path, diagnostics_dir: Path) -> subprocess.Popen:
    cmd: list[str] = [
        sys.executable, str(SCRIPT_DIR / "server_allocation.py"),
        "--checkpoint", args.checkpoint,
        "--method", method,
        "--budget", str(budget),
        "--variant", variant,
        "--port", str(port),
        "--host", "127.0.0.1",
        "--image_height", str(args.image_height),
        "--image_width", str(args.image_width),
        "--seed", str(args.seed),
    ]
    if diffusion_steps and diffusion_steps > 0:
        cmd += ["--diffusion_steps", str(diffusion_steps)]
    if args.importance_map_path:
        cmd += ["--importance_map_path", args.importance_map_path]
    if args.save_diagnostics:
        cmd += ["--save_diagnostics", "--diagnostics_dir", str(diagnostics_dir)]
    env = os.environ.copy()
    env.setdefault("HF_HOME", os.environ.get("HF_HOME", "/workspace/hf_cache"))
    logger.info("starting server: %s", " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env, cwd=str(REPO_ROOT),
    )
    _stream_subprocess_output(proc, prefix="server", log_path=server_log)
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=15.0)
    except subprocess.TimeoutExpired:
        proc.kill()


# ============================================================================
# Client lifecycle
# ============================================================================


def _format_client_cmd(template: str, host: str, port: int, scene: int,
                       episodes: int, output_dir: Path) -> list[str]:
    rendered = (
        template.replace("{host}", host)
        .replace("{port}", str(port))
        .replace("{scene}", str(scene))
        .replace("{episodes}", str(episodes))
        .replace("{output_dir}", str(output_dir))
    )
    return shlex.split(rendered)


def run_client(args: argparse.Namespace, host: str, port: int, scene: int,
               episodes: int, run_dir: Path, log_path: Path) -> int:
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        cmd = [
            sys.executable, str(REPO_ROOT / "test_client_AR.py"),
            "--host", host, "--port", str(port),
            "--num-chunks", str(max(2, episodes)),
        ]
    else:
        if args.client_cmd:
            cmd = _format_client_cmd(args.client_cmd, host, port, scene, episodes, run_dir)
        else:
            # Fallback: existing run_sim_eval.py (sim-evals required).
            cmd = [
                sys.executable, str(REPO_ROOT / "eval_utils" / "run_sim_eval.py"),
                "--host", host, "--port", str(port),
                "--scene", str(scene), "--episodes", str(episodes),
                "--headless",
            ]
    logger.info("client cmd: %s", " ".join(shlex.quote(c) for c in cmd))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        proc = subprocess.run(
            cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(run_dir),
            text=True,
        )
    return int(proc.returncode)


# ============================================================================
# Metrics extraction
# ============================================================================


def _parse_episodes_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append({k: _coerce(v) for k, v in r.items()})
    return rows


def _coerce(v: Any) -> Any:
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            return int(s)
        except Exception:
            try:
                return float(s)
            except Exception:
                return s
    return v


def _episodes_from_diagnostics(diag_dir: Path) -> list[dict[str, Any]]:
    """Best-effort metric synthesis from server diagnostics JSONL when the
    client doesn't emit episodes.csv. Records one row per session_id with
    a 'no-success-info' flag so the analyzer can ignore success columns."""
    rows: list[dict[str, Any]] = []
    if not diag_dir.exists():
        return rows
    for sub in diag_dir.iterdir():
        f = sub / "diag.jsonl"
        if not f.exists():
            continue
        steps = 0
        latencies: list[float] = []
        try:
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                d = json.loads(line)
                steps += 1
                latencies.append(float(d.get("elapsed_s", 0.0)))
        except Exception:
            continue
        rows.append({
            "session_id": sub.name,
            "episode": -1,
            "success": None,
            "progress": None,
            "duration_s": float(sum(latencies)) if latencies else None,
            "control_freq_hz": (steps / sum(latencies)) if latencies else None,
            "n_steps": steps,
            "from_diagnostics": True,
        })
    return rows


# ============================================================================
# Driver
# ============================================================================


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--methods", default="none,uniform,center,gripper,online_hybrid",
                   help="Comma-separated method list.")
    p.add_argument("--budgets", default="100,50,25",
                   help="Comma-separated budget percentages.")
    p.add_argument("--variants", default="A_hard_retention",
                   help="Comma-separated variants.")
    p.add_argument("--diffusion_steps", default="",
                   help="Comma-separated DiT step counts, or empty for the checkpoint default.")
    p.add_argument("--scenes", default="1,2,3",
                   help="Comma-separated DROID sim-eval scene ids.")
    p.add_argument("--episodes", type=int, default=3,
                   help="Episodes per (config, scene).")
    p.add_argument("--image_height", type=int, default=180)
    p.add_argument("--image_width", type=int, default=320)
    p.add_argument("--save_diagnostics", action="store_true",
                   help="Forward to server: persist per-step rollout JSONL.")
    p.add_argument("--importance_map_path", default=None,
                   help="Forward to server: precomputed Stage-1 map for action_causal_oracle.")
    p.add_argument("--smoke", action="store_true",
                   help="Skip sim-evals and use test_client_AR.py for a pipeline smoke test.")
    p.add_argument("--client_cmd", default=None,
                   help="Custom client command template; supports {host} {port} {scene} "
                        "{episodes} {output_dir} substitutions.")
    p.add_argument("--server_warmup_s", type=float, default=600.0,
                   help="Max seconds to wait for the server port to come up.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip configs whose summary file already exists.")
    args = p.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    budgets = [float(b) for b in args.budgets.split(",") if b.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    diff_steps_list: list[int | None] = []
    if args.diffusion_steps.strip():
        diff_steps_list = [int(x) for x in args.diffusion_steps.split(",") if x.strip()]
    else:
        diff_steps_list = [None]
    scenes = [int(s) for s in args.scenes.split(",") if s.strip()]

    all_rows: list[dict[str, Any]] = []
    config_index = 0
    for method, budget, variant, diff_steps in product(methods, budgets, variants, diff_steps_list):
        config_index += 1
        cfg_id = f"{method}_b{int(round(budget))}_{variant}_d{diff_steps if diff_steps else 'def'}"
        cfg_dir = out_dir / cfg_id
        if args.skip_existing and (cfg_dir / "summary.json").exists():
            logger.info("[%d] skip existing config %s", config_index, cfg_id)
            try:
                with (cfg_dir / "all_episodes.csv").open("r") as f:
                    rd = csv.DictReader(f)
                    for row in rd:
                        all_rows.append({k: _coerce(v) for k, v in row.items()})
            except Exception:
                pass
            continue

        cfg_dir.mkdir(parents=True, exist_ok=True)
        diag_dir = cfg_dir / "diag"
        port = _free_port()
        server_proc = start_server(
            args, method=method, budget=budget, variant=variant,
            diffusion_steps=diff_steps, port=port,
            server_log=cfg_dir / "server.log", diagnostics_dir=diag_dir,
        )
        try:
            ok = _wait_for_port("127.0.0.1", port, timeout=args.server_warmup_s)
            if not ok:
                logger.error("Server did not come up within %.0fs (port=%d). See %s",
                             args.server_warmup_s, port, cfg_dir / "server.log")
                continue

            cfg_rows: list[dict[str, Any]] = []
            for scene in scenes:
                run_dir = cfg_dir / f"scene_{scene}"
                rc = run_client(args, host="127.0.0.1", port=port, scene=scene,
                                episodes=args.episodes, run_dir=run_dir,
                                log_path=run_dir / "client.log")
                # Try CSV first, then diagnostics fallback.
                ep_rows = _parse_episodes_csv(run_dir / "episodes.csv")
                if not ep_rows:
                    ep_rows = _episodes_from_diagnostics(diag_dir)
                for r in ep_rows:
                    r.update({
                        "config_id": cfg_id, "method": method, "budget_pct": budget,
                        "variant": variant, "diffusion_steps": diff_steps,
                        "scene": scene, "client_returncode": rc,
                    })
                    cfg_rows.append(r)
                # also write per-config CSV
            with (cfg_dir / "all_episodes.csv").open("w") as f:
                if cfg_rows:
                    fieldnames = list(cfg_rows[0].keys())
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    w.writeheader()
                    for r in cfg_rows:
                        w.writerow(r)
            (cfg_dir / "summary.json").write_text(json.dumps({
                "config_id": cfg_id, "method": method, "budget_pct": budget,
                "variant": variant, "diffusion_steps": diff_steps,
                "n_episodes": len(cfg_rows),
            }, indent=2))
            all_rows.extend(cfg_rows)
        finally:
            stop_server(server_proc)

    # Aggregate
    out_csv = out_dir / "all_episodes.csv"
    if all_rows:
        fieldnames: list[str] = []
        for r in all_rows:
            for k in r.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
        with out_csv.open("w") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in all_rows:
                w.writerow(r)
    (out_dir / "run_manifest.json").write_text(json.dumps({
        "checkpoint": args.checkpoint,
        "methods": methods, "budgets": budgets, "variants": variants,
        "diffusion_steps": [d if d is not None else "default" for d in diff_steps_list],
        "scenes": scenes, "episodes_per_config": args.episodes,
        "smoke": bool(args.smoke), "client_cmd": args.client_cmd,
        "n_configs": config_index, "n_rows": len(all_rows),
    }, indent=2))
    logger.info("Done. all_episodes.csv -> %s  (%d rows)", out_csv, len(all_rows))


if __name__ == "__main__":
    main()
