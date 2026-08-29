'''
    Copyright (c) 2026 Salomé A. Sepúlveda-Fontaine
    SPDX-License-Identifier: MIT
'''


#!/usr/bin/env python3
"""
Local/Colab YAML runner for Torch eGTM experiments.

This script is the local/Colab counterpart of the DANTZIG/Slurm runner.
It does not submit jobs to Slurm. Instead, it runs one YAML file or all YAML
files in a folder by calling run_experiment() from main_torch.py.

Typical usage:

    # Run a single YAML
    python code/run_colab_or_local.py \
        --config code/yamls_CPU/2_gen_by_MAD_1230x5000/TF_1_gen_proc_sorted5000.yaml \
        --main-file code/main_torch.py \
        --output-dir results/torch_results_CPU

    # Run all YAMLs in a folder, sequentially
    python code/run_colab_or_local.py \
        --config-dir code/yamls_CPU/2_gen_by_MAD_1230x5000 \
        --main-file code/main_torch.py \
        --output-dir results/torch_results_CPU

    # Run all YAMLs in a folder, with 4 parallel workers
    python code/run_colab_or_local.py \
        --config-dir code/yamls_CPU/2_gen_by_MAD_1230x5000 \
        --main-file code/main_torch.py \
        --output-dir results/torch_results_CPU \
        --workers 4

Optional behavior:
    --mark-done
        Rename each successful YAML to <name>.done. This is useful for large
        batches, but is disabled by default so repository examples remain unchanged.
"""



import argparse
import importlib
import importlib.util
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import yaml


def load_yaml_preview(yaml_path: Path) -> dict:
    """Load a YAML file only to print a compact execution preview."""
    with yaml_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    for key, value in list(cfg.items()):
        if isinstance(value, list) and len(value) == 1:
            cfg[key] = value[0]

    return cfg


def load_run_experiment(main_file: Path):
    """Import run_experiment(config_path, output_dir) from main_torch.py."""
    main_file = main_file.resolve()
    code_dir = main_file.parent

    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

    importlib.invalidate_caches()

    spec = importlib.util.spec_from_file_location("main_torch_module", str(main_file))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from: {main_file}")

    main_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main_module)

    if not hasattr(main_module, "run_experiment"):
        raise AttributeError(f"{main_file} does not define run_experiment().")

    return main_module.run_experiment


def print_yaml_preview(yaml_path: Path) -> None:
    cfg = load_yaml_preview(yaml_path)

    print(f"START YAML: {yaml_path.name}")
    print(f"YAML PATH: {yaml_path}")
    print(f"csv_path: {cfg.get('csv_path')}")
    print(f"num_rows: {cfg.get('num_rows')}")
    print(f"num_columns: {cfg.get('num_columns')}")
    print(f"standarization: {cfg.get('standarization')}")
    print(f"ce_regularization: {cfg.get('ce_regularization', False)}")

    param_grid = cfg.get("param_grid", {}) or {}
    if param_grid:
        print("param_grid:")
        for param_name, param_value in param_grid.items():
            print(f"  {param_name}: {param_value}")
    print()


def run_single_config(
    config: Path,
    main_file: Path,
    output_dir: Path,
    mark_done: bool = False,
) -> None:
    """Run one YAML config by calling main_torch.run_experiment()."""
    config = config.resolve()
    main_file = main_file.resolve()
    output_dir = output_dir.resolve()

    if not config.is_file():
        raise FileNotFoundError(f"YAML config not found: {config}")

    if not main_file.is_file():
        raise FileNotFoundError(f"main_torch.py not found: {main_file}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print_yaml_preview(config)

    run_experiment = load_run_experiment(main_file)
    run_experiment(config_path=str(config), output_dir=str(output_dir))

    if mark_done:
        renamed_config = config.with_name(f"{config.name}.done")
        if renamed_config.exists():
            raise FileExistsError(f"Cannot mark as done; destination exists: {renamed_config}")
        config.rename(renamed_config)
        print(f"DONE YAML: {config.name}")
        print(f"Renamed to: {renamed_config.name}")
    else:
        print(f"DONE YAML: {config.name}")


def iter_yaml_files(config_dir: Path, recursive: bool = False) -> list[Path]:
    pattern = "**/*.yaml" if recursive else "*.yaml"
    yaml_files = sorted(config_dir.glob(pattern))
    return [p for p in yaml_files if p.is_file()]


def run_config_dir(
    config_dir: Path,
    main_file: Path,
    output_dir: Path,
    log_dir: Path,
    workers: int = 1,
    recursive: bool = False,
    mark_done: bool = False,
) -> None:
    """Run all YAML files in a folder, optionally in parallel subprocesses."""
    config_dir = config_dir.resolve()
    main_file = main_file.resolve()
    output_dir = output_dir.resolve()
    log_dir = log_dir.resolve()

    if not config_dir.is_dir():
        raise NotADirectoryError(f"Config directory not found: {config_dir}")

    if workers < 1:
        raise ValueError("--workers must be >= 1")

    yaml_files = iter_yaml_files(config_dir, recursive=recursive)

    print(f"CONFIG DIR: {config_dir}")
    print(f"YAML FILES: {len(yaml_files)}")
    print(f"WORKERS: {workers}")
    print(f"OUTPUT DIR: {output_dir}")
    print(f"LOG DIR: {log_dir}")
    print(f"PYTHON: {sys.executable}")
    print()

    if not yaml_files:
        print("No pending *.yaml files found. Nothing to do.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    pending = list(yaml_files)
    running = []

    def launch(yaml_file: Path):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{stamp}__{yaml_file.stem}.log"

        print(f"START YAML: {yaml_file.name}")
        print(f"  log: {log_file}")

        cmd = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--config",
            str(yaml_file),
            "--main-file",
            str(main_file),
            "--output-dir",
            str(output_dir),
        ]

        if mark_done:
            cmd.append("--mark-done")

        log_handle = log_file.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return proc, yaml_file, log_file, log_handle

    while pending or running:
        while pending and len(running) < workers:
            running.append(launch(pending.pop(0)))

        still_running = []
        for proc, yaml_file, log_file, log_handle in running:
            ret = proc.poll()
            if ret is None:
                still_running.append((proc, yaml_file, log_file, log_handle))
                continue

            log_handle.close()

            if ret != 0:
                raise RuntimeError(
                    f"YAML failed: {yaml_file}\n"
                    f"Exit code: {ret}\n"
                    f"See log: {log_file}"
                )

            print(f"DONE YAML: {yaml_file.name}")

        running = still_running
        time.sleep(2)

    print("All YAML files finished successfully.")


def default_main_file() -> Path:
    """Prefer main_torch.py next to this runner; otherwise use ./main_torch.py."""
    script_dir_candidate = Path(__file__).resolve().parent / "main_torch.py"
    if script_dir_candidate.exists():
        return script_dir_candidate
    return Path.cwd() / "main_torch.py"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Torch eGTM YAML experiments locally or in Google Colab."
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", type=Path, help="Path to one YAML config file.")
    mode.add_argument("--config-dir", type=Path, help="Folder containing YAML config files.")

    parser.add_argument(
        "--main-file",
        type=Path,
        default=default_main_file(),
        help="Path to main_torch.py. Default: main_torch.py next to this runner, or ./main_torch.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/results_torch"),
        help="Directory where result CSV files will be written.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs/yaml_runs_local"),
        help="Directory for per-YAML logs when using --config-dir.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of YAML files to run in parallel when using --config-dir.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively for YAML files inside --config-dir.",
    )
    parser.add_argument(
        "--mark-done",
        action="store_true",
        help="Rename successful YAML files to <name>.done. Disabled by default.",
    )

    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    if args.config is not None:
        run_single_config(
            config=args.config,
            main_file=args.main_file,
            output_dir=args.output_dir,
            mark_done=args.mark_done,
        )
        return

    run_config_dir(
        config_dir=args.config_dir,
        main_file=args.main_file,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        workers=args.workers,
        recursive=args.recursive,
        mark_done=args.mark_done,
    )


if __name__ == "__main__":
    main()
