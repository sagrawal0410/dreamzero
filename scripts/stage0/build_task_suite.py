#!/usr/bin/env python3
"""Stage 0 / E0.0 — Dataset and Task-Suite Setup for DreamZero-DROID.

Builds a small but representative evaluation suite that later stages reuse.
Reads a LeRobot-format DROID dataset, classifies each episode into task
groups (A: easy / B: contact-sensitive / C: distractor / D: global-context)
based on the language instruction, samples K episodes per group and T
stratified timesteps per episode, and writes a single `manifest.json`
that downstream stages consume.

Example:

    python scripts/stage0/build_task_suite.py \
        --dataset_root ./data/droid_lerobot \
        --task_groups scripts/stage0/configs/task_groups.yaml \
        --num_episodes_per_group 8 \
        --num_timesteps_per_episode 5 \
        --output_dir runs/stage0_suite

Outputs (under --output_dir):

    manifest.json            : ordered list of evaluation examples
    manifest_by_group.json   : same examples grouped by task group
    summary.json             : counts, group sizes, instruction histograms
    preview/<example_id>.png : one preview frame per example (sanity check)

Each manifest entry contains everything the E0.1 baseline script needs to
re-load the timestep without rescanning the dataset:

    example_id, task_group, task_name, instruction,
    episode_index, episode_path, episode_length, frame_index,
    role (initial/approach/pre_contact/contact/post_contact),
    video_paths   : {camera_key: absolute_mp4_path},
    state_columns : list of parquet columns used as proprio state,
    action_columns: list of parquet columns used as action,
    state_at_t    : flat numpy state vector at frame_index,
    gt_action_chunk: (action_horizon, A) ground-truth action chunk if available,
    seed          : suite-build seed (for reproducibility)

This script does NOT load the model and does NOT call any GPU ops, so it
can be run on a small CPU box before kicking off baseline eval.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stage0.build_task_suite")


# ----------------------------------------------------------------------------
# Defaults that match the DROID LeRobot layout used in this repo
# ----------------------------------------------------------------------------

DEFAULT_VIDEO_KEYS = (
    "video.exterior_image_1_left",
    "video.exterior_image_2_left",
    "video.wrist_image_left",
)
DEFAULT_STATE_COLUMNS = ("observation.state",)
DEFAULT_ACTION_COLUMNS = ("action",)
DEFAULT_ACTION_HORIZON = 24


# ----------------------------------------------------------------------------
# Dataset discovery
# ----------------------------------------------------------------------------


@dataclass
class EpisodeInfo:
    episode_index: int
    parquet_path: Path
    length: int
    instruction: str
    video_paths: dict[str, str] = field(default_factory=dict)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    text = path.read_text()
    if text.startswith("version https://git-lfs.github.com/spec/"):
        raise RuntimeError(
            f"{path} is a git-lfs POINTER file (~150 bytes), not the real "
            f"metadata. Inside the cloned repo run:\n"
            f"    git config --unset lfs.skip-smudge   # if needed\n"
            f"    git lfs install\n"
            f"    git lfs pull --include 'meta/*'\n"
            f"Then re-run this script."
        )
    bad = 0
    for i, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            bad += 1
            if bad <= 3:
                logger.warning("%s:%d not valid JSON, skipping (%s)", path.name, i, e)
    if bad:
        logger.warning("%s: skipped %d unparseable lines", path.name, bad)
    return rows


def _load_tasks_map(meta_dir: Path) -> dict[int, str]:
    """Map task_index -> task description from `meta/tasks.jsonl`."""
    tasks = _read_jsonl(meta_dir / "tasks.jsonl")
    out: dict[int, str] = {}
    for row in tasks:
        idx = int(row.get("task_index", row.get("task_idx", -1)))
        text = str(row.get("task", row.get("task_description", row.get("instruction", ""))))
        if idx >= 0:
            out[idx] = text
    return out


def _load_episodes_meta(meta_dir: Path, task_map: dict[int, str]) -> dict[int, dict[str, Any]]:
    """Map episode_index -> {length, instruction} from `meta/episodes.jsonl`."""
    rows = _read_jsonl(meta_dir / "episodes.jsonl")
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        ep_idx = int(row.get("episode_index", row.get("episode_idx", -1)))
        if ep_idx < 0:
            continue
        length = int(row.get("length", 0))
        # Task may live as `tasks: [task_index]` or as a literal string under
        # `task` / `instruction` / `language_instruction`. Cover all of them.
        instruction = ""
        if "tasks" in row and isinstance(row["tasks"], list) and row["tasks"]:
            first = row["tasks"][0]
            if isinstance(first, int):
                instruction = task_map.get(first, "")
            elif isinstance(first, str):
                instruction = first
        for k in ("task", "instruction", "language_instruction", "annotation.task"):
            if not instruction and isinstance(row.get(k), str):
                instruction = row[k]
        out[ep_idx] = {"length": length, "instruction": instruction}
    return out


def _on_disk_camera_folder(video_key: str) -> str:
    """Translate a policy modality key (`video.X`) to the on-disk camera folder
    name (`observation.images.X`) used by the DreamZero-DROID-Data conversion.

    Falls back to the raw key for datasets that already use it as the folder
    name on disk.
    """
    if video_key.startswith("video."):
        return "observation.images." + video_key[len("video."):]
    return video_key


def _build_video_index(dataset_root: Path, video_keys: tuple[str, ...]) -> dict[tuple[str, int], str]:
    """Walk videos/ ONCE and return {(policy_video_key, episode_index): mp4_path}.

    This replaces O(num_episodes × cameras) recursive globs with a single
    rglob over videos/, which on a network volume is the difference between
    minutes-to-hours and a few seconds.
    """
    videos_root = dataset_root / "videos"
    index: dict[tuple[str, int], str] = {}
    if not videos_root.exists():
        return index
    folder_to_key = {_on_disk_camera_folder(k): k for k in video_keys}
    pat = re.compile(r"episode_(\d+)\.mp4$")
    for mp4 in videos_root.rglob("episode_*.mp4"):
        m = pat.search(mp4.name)
        if not m:
            continue
        cam_folder = mp4.parent.name
        policy_key = folder_to_key.get(cam_folder)
        if policy_key is None:
            continue
        index[(policy_key, int(m.group(1)))] = str(mp4.resolve())
    return index


def _video_paths_for_episode(index: dict[tuple[str, int], str],
                             video_keys: tuple[str, ...],
                             episode_index: int) -> dict[str, str]:
    out: dict[str, str] = {}
    for k in video_keys:
        p = index.get((k, episode_index))
        if p is not None:
            out[k] = p
    return out


def _discover_episodes(dataset_root: Path, video_keys: tuple[str, ...],
                       quick: bool = False) -> list[EpisodeInfo]:
    """Walk the LeRobot dataset and return one EpisodeInfo per episode.

    Video paths are NEVER discovered here — they are filled in later from the
    one-shot `_build_video_index(...)` walk. This keeps discovery to:

      - parquet glob (one filesystem call)
      - meta/{episodes,tasks}.jsonl read
      - optional parquet `num_rows` per episode (slow on network FS; skipped
        when `quick=True`).

    `quick=True` is used by --dry_run so the entire pre-flight runs in seconds.
    """
    meta_dir = dataset_root / "meta"
    task_map = _load_tasks_map(meta_dir)
    ep_meta = _load_episodes_meta(meta_dir, task_map)

    parquets = sorted(glob.glob(str(dataset_root / "data" / "**" / "episode_*.parquet"), recursive=True))
    if not parquets:
        raise FileNotFoundError(f"No episode_*.parquet found under {dataset_root}/data")

    if quick:
        logger.info(
            "Fast discovery: skipping parquet length reads "
            "(--dry_run only — full run reads them lazily for chosen episodes)."
        )

    episodes: list[EpisodeInfo] = []
    missing_length = 0
    missing_instruction = 0
    for pq in parquets:
        m = re.search(r"episode_(\d+)\.parquet$", pq)
        if not m:
            continue
        ep_idx = int(m.group(1))
        meta = ep_meta.get(ep_idx, {})
        length = int(meta.get("length") or 0)
        if length == 0 and not quick:
            try:
                import pyarrow.parquet as pq_reader
                length = pq_reader.read_table(pq, columns=[]).num_rows
            except Exception as e:
                logger.warning("Unable to read length for episode %d (%s); skipping. (%s)", ep_idx, pq, e)
                continue
        if length == 0:
            missing_length += 1
        instruction = str(meta.get("instruction") or "")
        if not instruction:
            missing_instruction += 1
        episodes.append(
            EpisodeInfo(
                episode_index=ep_idx,
                parquet_path=Path(pq),
                length=length,
                instruction=instruction,
                video_paths={},   # filled later from the video index
            )
        )

    if not episodes:
        raise RuntimeError(f"No usable episodes discovered in {dataset_root}.")
    if missing_length:
        logger.info("  episodes with no `length` in meta/episodes.jsonl: %d "
                    "(dry-run keeps them; full run would re-read parquet rows)", missing_length)
    if missing_instruction:
        logger.info("  episodes with no language instruction in meta: %d "
                    "(these will land in the `unmatched` group)", missing_instruction)
    return episodes


# ----------------------------------------------------------------------------
# Task-group classification
# ----------------------------------------------------------------------------


def _classify_episode(instruction: str, groups: dict[str, dict[str, Any]]) -> str | None:
    text = instruction.lower()
    for gname, gcfg in groups.items():
        for kw in gcfg.get("keywords", []) or []:
            if kw.lower() in text:
                return gname
    return None


# ----------------------------------------------------------------------------
# Timestep sampling and ground-truth extraction
# ----------------------------------------------------------------------------


def _stratified_indices(length: int, fractions: list[float]) -> list[int]:
    out = []
    for f in fractions:
        f = max(0.0, min(0.999, float(f)))
        idx = int(round(f * (length - 1)))
        out.append(idx)
    return out


def _default_fractions(n: int) -> list[float]:
    """Linearly stratified fractions in [0.05, 0.92] for `n` timesteps."""
    if n <= 0:
        return []
    if n == 1:
        return [0.5]
    return [0.05 + (0.92 - 0.05) * i / (n - 1) for i in range(n)]


def _default_roles(n: int) -> list[str]:
    """Human-readable role labels matching the stratified fractions above."""
    base = [
        "initial",
        "early_approach",
        "mid_approach",
        "approach",
        "pre_contact",
        "contact",
        "post_contact",
        "transport_start",
        "transport_mid",
        "transport_late",
        "near_target",
        "completion",
        "settle",
        "release",
        "retreat",
    ]
    if n <= len(base):
        return list(base[:n])
    return [f"phase_{i:02d}" for i in range(n)]


def _read_state_action(parquet_path: Path,
                       frame_indices: list[int],
                       state_columns: tuple[str, ...],
                       action_columns: tuple[str, ...],
                       action_horizon: int) -> dict[int, dict[str, Any]]:
    """Return {frame_index: {'state': flat np, 'gt_action_chunk': (H, A) np}}."""
    import pyarrow.parquet as pq

    table = pq.read_table(str(parquet_path))
    n = table.num_rows
    cols = set(table.column_names)

    def fetch(col: str, row: int) -> np.ndarray:
        if col not in cols:
            return np.array([], dtype=np.float64)
        val = table.column(col)[row].as_py()
        return np.array(val, dtype=np.float64).reshape(-1)

    out: dict[int, dict[str, Any]] = {}
    for t in frame_indices:
        if t < 0 or t >= n:
            continue
        state_parts = [fetch(c, t) for c in state_columns]
        state_vec = np.concatenate(state_parts) if state_parts else np.array([], dtype=np.float64)

        chunk_rows = []
        for h in range(action_horizon):
            r = min(t + h, n - 1)
            action_parts = [fetch(c, r) for c in action_columns]
            chunk_rows.append(np.concatenate(action_parts) if action_parts else np.array([], dtype=np.float64))
        gt_chunk = np.stack(chunk_rows, axis=0) if chunk_rows else np.zeros((action_horizon, 0))

        out[t] = {"state": state_vec.tolist(), "gt_action_chunk": gt_chunk.tolist()}
    return out


# ----------------------------------------------------------------------------
# Optional preview frames (sanity)
# ----------------------------------------------------------------------------


def _save_preview(video_paths: dict[str, str], frame_index: int, out_path: Path) -> None:
    try:
        import cv2  # local import; preview is optional
    except Exception:
        return
    if not video_paths:
        return
    panels = []
    for _, mp4 in video_paths.items():
        cap = cv2.VideoCapture(mp4)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            continue
        panels.append(frame)
    if not panels:
        return
    h = min(p.shape[0] for p in panels)
    panels = [cv2.resize(p, (int(p.shape[1] * h / p.shape[0]), h)) for p in panels]
    combined = np.concatenate(panels, axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), combined)


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------


def build_suite(args: argparse.Namespace) -> None:
    cfg_path = Path(args.task_groups)
    with cfg_path.open("r") as f:
        full_cfg = yaml.safe_load(f) or {}
    sampling_cfg = full_cfg.pop("sampling", {}) or {}
    groups = full_cfg

    # Hierarchy is Group -> unique Task (instruction) -> Episode -> Timestep.
    # Sampling parameters (CLI overrides YAML; YAML overrides hard-coded defaults).
    num_tasks = int(
        args.num_tasks_per_group
        if args.num_tasks_per_group is not None
        else sampling_cfg.get("num_tasks_per_group", 25)
    )
    num_eps_per_task = int(
        args.num_episodes_per_task
        if args.num_episodes_per_task is not None
        else sampling_cfg.get("num_episodes_per_task", 10)
    )
    # Legacy: if the user passed --num_episodes_per_group, we still honour it
    # by setting an upper bound on (num_tasks * num_eps_per_task) per group.
    legacy_eps_per_group = (
        int(args.num_episodes_per_group)
        if args.num_episodes_per_group is not None
        else sampling_cfg.get("num_episodes_per_group")
    )

    num_ts = int(
        args.num_timesteps_per_episode
        if args.num_timesteps_per_episode is not None
        else sampling_cfg.get("num_timesteps_per_episode", 12)
    )
    fractions_cfg = sampling_cfg.get("timestep_fractions") or []
    role_names_cfg = sampling_cfg.get("timestep_role_names") or []
    fractions = list(fractions_cfg)[:num_ts] if len(fractions_cfg) >= num_ts else _default_fractions(num_ts)
    role_names = list(role_names_cfg)[:num_ts] if len(role_names_cfg) >= num_ts else _default_roles(num_ts)
    fractions = fractions[:num_ts]
    role_names = role_names[:num_ts]

    include_unmatched = bool(sampling_cfg.get("include_unmatched", False))
    seed = int(args.random_seed if args.random_seed is not None else sampling_cfg.get("random_seed", 0))

    rng = random.Random(seed)
    np.random.seed(seed)

    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Discovering episodes under %s ...", dataset_root)
    episodes = _discover_episodes(dataset_root, DEFAULT_VIDEO_KEYS, quick=args.dry_run)
    logger.info("Found %d episodes.", len(episodes))

    grouped: dict[str, list[EpisodeInfo]] = {gname: [] for gname in groups}
    grouped["unmatched"] = []
    instruction_hist: dict[str, int] = {}
    for ep in episodes:
        instr = ep.instruction or ""
        instruction_hist[instr] = instruction_hist.get(instr, 0) + 1
        gname = _classify_episode(instr, groups)
        grouped.setdefault(gname or "unmatched", []).append(ep)

    # Per-group: count distinct tasks (unique instruction strings).
    tasks_per_group: dict[str, dict[str, list[EpisodeInfo]]] = {}
    for gname, eps in grouped.items():
        by_instr: dict[str, list[EpisodeInfo]] = {}
        for ep in eps:
            by_instr.setdefault(ep.instruction or "", []).append(ep)
        tasks_per_group[gname] = by_instr

    logger.info(
        "Plan: up to %d tasks/group × %d episodes/task × %d timesteps/episode",
        num_tasks, num_eps_per_task, num_ts,
    )
    logger.info("Group breakdown (eligible episodes / unique tasks):")
    for gname in list(grouped.keys()):
        eps = grouped[gname]
        n_tasks = len(tasks_per_group[gname])
        logger.info("  %-22s episodes=%-7d unique_tasks=%-5d", gname, len(eps), n_tasks)

    if args.dry_run:
        logger.info("--dry_run: not writing manifest.")
        return

    # ------------------------------------------------------------------
    # Build a one-shot videos/ index. We do this AFTER classification
    # (which doesn't need videos) so a misconfigured task_groups.yaml
    # short-circuits via --dry_run before paying for the rglob.
    # ------------------------------------------------------------------
    logger.info("Building video index (one rglob over videos/) ...")
    video_index = _build_video_index(dataset_root, DEFAULT_VIDEO_KEYS)
    logger.info("  indexed %d (camera, episode) pairs covering %d unique episodes",
                len(video_index), len({e for _, e in video_index.keys()}))

    # ------------------------------------------------------------------
    # Sample manifest: Group -> Task -> Episode -> Timestep
    # ------------------------------------------------------------------
    chosen_groups = list(groups.keys())
    if include_unmatched:
        chosen_groups.append("unmatched")

    manifest: list[dict[str, Any]] = []
    by_group: dict[str, list[dict[str, Any]]] = {}
    structure: dict[str, dict[str, Any]] = {}     # diagnostic per-group counts
    next_id = 0
    skipped_no_videos = 0
    skipped_no_length = 0

    for gname in chosen_groups:
        by_instr = dict(tasks_per_group.get(gname, {}))
        # Skip empty-instruction bucket inside a group only if the group is matched
        # (matched groups should never have a "" instruction key after classify).
        tasks = [t for t in by_instr.keys() if t]
        # Allow empty-instruction tasks to flow through only for `unmatched`.
        if gname == "unmatched":
            tasks = list(by_instr.keys())

        rng.shuffle(tasks)
        chosen_tasks = tasks[:num_tasks]
        if len(tasks) < num_tasks:
            logger.warning(
                "Group %s: only %d unique tasks available (requested %d).",
                gname, len(tasks), num_tasks,
            )

        by_group[gname] = []
        per_task_counts: dict[str, int] = {}
        chosen_episodes_count = 0

        for task_idx, task_text in enumerate(chosen_tasks):
            task_eps = list(by_instr.get(task_text, []))
            rng.shuffle(task_eps)
            chosen_eps = task_eps[:num_eps_per_task]
            if legacy_eps_per_group is not None and chosen_episodes_count >= legacy_eps_per_group:
                break

            for ep_idx_in_task, ep in enumerate(chosen_eps):
                if legacy_eps_per_group is not None and chosen_episodes_count >= legacy_eps_per_group:
                    break

                # Lazy parquet length read for chosen episodes whose meta
                # didn't carry it. Cheap because we only read num_rows.
                if ep.length <= 0:
                    try:
                        import pyarrow.parquet as pq_reader
                        ep.length = int(pq_reader.read_table(str(ep.parquet_path), columns=[]).num_rows)
                    except Exception as e:
                        logger.warning("Skipping ep %d: cannot read length (%s)", ep.episode_index, e)
                        skipped_no_length += 1
                        continue
                if ep.length <= 0:
                    skipped_no_length += 1
                    continue

                # Look up videos in the prebuilt index. Skip episode if any
                # required camera is missing — downstream scripts assume all
                # three streams are present.
                video_paths = _video_paths_for_episode(
                    video_index, DEFAULT_VIDEO_KEYS, ep.episode_index
                )
                if len(video_paths) < len(DEFAULT_VIDEO_KEYS):
                    skipped_no_videos += 1
                    continue

                t_indices = _stratified_indices(ep.length, fractions)
                sa = _read_state_action(
                    ep.parquet_path,
                    t_indices,
                    DEFAULT_STATE_COLUMNS,
                    DEFAULT_ACTION_COLUMNS,
                    args.action_horizon,
                )
                # Stable short hash for the task instruction (8 hex chars)
                import hashlib
                task_hash = hashlib.sha1((task_text or "").encode("utf-8")).hexdigest()[:8]
                wrote_any = False
                for t, role in zip(t_indices, role_names):
                    if t not in sa:
                        continue
                    example_id = (
                        f"{gname}__task{task_idx:03d}_{task_hash}__"
                        f"ep{ep.episode_index:06d}__t{t:05d}__{role}"
                    )
                    next_id += 1
                    entry = {
                        "example_id": example_id,
                        "global_index": next_id - 1,
                        "task_group": gname,
                        "task_name": (groups.get(gname, {}) or {}).get("description", gname),
                        "task_index_in_group": task_idx,
                        "task_hash": task_hash,
                        "instruction": ep.instruction,
                        "episode_index": ep.episode_index,
                        "episode_path": str(ep.parquet_path),
                        "episode_length": ep.length,
                        "episode_index_in_task": ep_idx_in_task,
                        "frame_index": int(t),
                        "role": role,
                        "video_paths": video_paths,
                        "state_columns": list(DEFAULT_STATE_COLUMNS),
                        "action_columns": list(DEFAULT_ACTION_COLUMNS),
                        "state_at_t": sa[t]["state"],
                        "gt_action_chunk": sa[t]["gt_action_chunk"],
                        "action_horizon": args.action_horizon,
                        "suite_seed": seed,
                    }
                    manifest.append(entry)
                    by_group[gname].append(entry)
                    wrote_any = True

                    if args.save_preview:
                        _save_preview(
                            video_paths,
                            t,
                            output_dir / "preview" / f"{example_id}.png",
                        )
                if wrote_any:
                    chosen_episodes_count += 1
                    per_task_counts[task_text] = per_task_counts.get(task_text, 0) + 1

        structure[gname] = {
            "num_unique_tasks_available": len(by_instr),
            "num_unique_tasks_chosen": len(chosen_tasks),
            "num_episodes_total": chosen_episodes_count,
            "episodes_per_task": per_task_counts,
        }
        logger.info(
            "  %-22s tasks=%d episodes=%d examples=%d",
            gname,
            len(chosen_tasks),
            chosen_episodes_count,
            len(by_group[gname]),
        )

    if skipped_no_videos:
        logger.warning(
            "Skipped %d episodes that had no matching videos in the on-disk index. "
            "(Are all three cameras present under videos/<chunk>/observation.images.*/?)",
            skipped_no_videos,
        )
    if skipped_no_length:
        logger.warning("Skipped %d episodes whose length could not be determined.", skipped_no_length)

    summary = {
        "dataset_root": str(dataset_root),
        "task_groups_config": str(cfg_path),
        "num_examples": len(manifest),
        "num_groups": len(chosen_groups),
        "examples_per_group": {g: len(by_group.get(g, [])) for g in chosen_groups},
        "structure_per_group": structure,
        "num_tasks_per_group_target": num_tasks,
        "num_episodes_per_task_target": num_eps_per_task,
        "num_timesteps_per_episode": num_ts,
        "legacy_num_episodes_per_group_cap": legacy_eps_per_group,
        "timestep_fractions": fractions,
        "timestep_role_names": role_names,
        "action_horizon": args.action_horizon,
        "include_unmatched": include_unmatched,
        "random_seed": seed,
        "skipped_no_videos": skipped_no_videos,
        "skipped_no_length": skipped_no_length,
        "instruction_histogram_top20": dict(
            sorted(instruction_hist.items(), key=lambda kv: kv[1], reverse=True)[:20]
        ),
    }

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output_dir / "manifest_by_group.json").write_text(json.dumps(by_group, indent=2))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Wrote %d manifest entries to %s", len(manifest), output_dir / "manifest.json")
    logger.info("Group sizes: %s", summary["examples_per_group"])


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)
    p.add_argument("--dataset_root", required=True,
                   help="Path to LeRobot DROID dataset (contains data/, videos/, meta/).")
    p.add_argument("--task_groups", required=True,
                   help="YAML file with task-group keyword definitions and sampling defaults.")
    p.add_argument("--output_dir", required=True,
                   help="Directory to write manifest.json / summary.json / preview frames into.")
    p.add_argument("--num_tasks_per_group", type=int, default=None,
                   help="Distinct task instructions to sample per group (default: from YAML, fallback 25).")
    p.add_argument("--num_episodes_per_task", type=int, default=None,
                   help="Episodes (trajectories) to sample per unique task (default: from YAML, fallback 10).")
    p.add_argument("--num_timesteps_per_episode", type=int, default=None,
                   help="Timesteps to sample per episode (default: from YAML, fallback 12).")
    p.add_argument("--num_episodes_per_group", type=int, default=None,
                   help="LEGACY: hard cap on total episodes per group "
                        "(if set, num_tasks_per_group × num_episodes_per_task is truncated).")
    p.add_argument("--action_horizon", type=int, default=DEFAULT_ACTION_HORIZON,
                   help="Number of future action steps to record as ground-truth chunk.")
    p.add_argument("--random_seed", type=int, default=None,
                   help="Override the YAML random seed for reproducible sampling.")
    p.add_argument("--save_preview", action="store_true",
                   help="Save a preview frame per example under <output_dir>/preview/.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print group / task counts only; do not write manifest.")
    args = p.parse_args()
    try:
        build_suite(args)
    except Exception as e:
        logger.error("Failed to build suite: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
