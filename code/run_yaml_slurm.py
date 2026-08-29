'''
    Copyright (c) 2026 Salomé A. Sepúlveda-Fontaine
    SPDX-License-Identifier: MIT
'''


import argparse
import importlib
import importlib.util
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml


def load_yaml_preview(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for k, v in cfg.items():
        if isinstance(v, list) and len(v) == 1:
            cfg[k] = v[0]

    return cfg


def load_run_experiment(main_file):
    main_file = Path(main_file).resolve()
    code_dir = main_file.parent

    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

    importlib.invalidate_caches()

    spec = importlib.util.spec_from_file_location("main_module", str(main_file))
    main_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main_module)

    return main_module.run_experiment


def run_single_yaml(yaml_file, main_file, output_dir):
    yaml_file = Path(yaml_file).resolve()

    print(f"START YAML: {yaml_file.name}")
    print(f"YAML PATH: {yaml_file}")

    cfg = load_yaml_preview(yaml_file)

    print(f"csv_path: {cfg.get('csv_path')}")
    print(f"num_rows: {cfg.get('num_rows')}")
    print(f"num_columns: {cfg.get('num_columns')}")
    print(f"standarization: {cfg.get('standarization')}")
    print(f"ce_regularization: {cfg.get('ce_regularization', False)}")


    for p_name, p_val in cfg.get("param_grid", {}).items():
        print(f"  {p_name}: {p_val}")

    run_experiment = load_run_experiment(main_file)

    run_experiment(
        config_path=str(yaml_file),
        output_dir=str(output_dir),
    )

    renamed_yaml = yaml_file.with_name(f"{yaml_file.name}.done")
    yaml_file.rename(renamed_yaml)

    print(f"DONE YAML: {yaml_file.name}")
    print(f"Renamed to: {renamed_yaml.name}")


def run_folder(folder, main_file, output_dir, processed_dir, log_dir, workers):
    folder = Path(folder).resolve()
    output_dir = Path(output_dir).resolve()
    processed_dir = Path(processed_dir).resolve()
    log_dir = Path(log_dir).resolve()

    yaml_files = sorted(folder.glob("*.yaml"))

    print(f"FOLDER: {folder}")
    print(f"PENDING YAML FILES: {len(yaml_files)}")
    print(f"WORKERS: {workers}")
    print(f"OUTPUT_DIR: {output_dir}")
    print(f"PROCESSED_DIR: {processed_dir}")
    print(f"LOG_DIR: {log_dir}")
    print(f"PYTHON: {sys.executable}")
    print()

    if not yaml_files:
        print("No pending *.yaml files found. Nothing to do.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    running = []
    pending = list(yaml_files)

    def launch(yaml_file):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{stamp}__{folder.name}__{yaml_file.stem}.log"

        print(f"START YAML: {yaml_file.name}")
        print(f"  log: {log_file}")

        log_handle = open(log_file, "w", encoding="utf-8")

        cmd = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--single-yaml",
            str(yaml_file),
            "--main-file",
            str(main_file),
            "--output-dir",
            str(output_dir),
        ]

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

    remaining_yaml_files = sorted(folder.glob("*.yaml"))

    if not remaining_yaml_files:
        destination_folder = processed_dir / folder.name

        if destination_folder.exists():
            raise FileExistsError(
                f"Destination folder already exists in Processed: {destination_folder}"
            )

        shutil.move(str(folder), str(destination_folder))
        print(f"Folder fully processed and moved to: {destination_folder}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--folder")
    parser.add_argument("--main-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--processed-dir")
    parser.add_argument("--log-dir")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--single-yaml")

    args = parser.parse_args()

    if args.single_yaml:
        run_single_yaml(
            yaml_file=args.single_yaml,
            main_file=args.main_file,
            output_dir=args.output_dir,
        )
        return

    if not args.folder:
        raise ValueError("--folder is required unless --single-yaml is used")

    if not args.processed_dir:
        raise ValueError("--processed-dir is required")

    if not args.log_dir:
        raise ValueError("--log-dir is required")

    run_folder(
        folder=args.folder,
        main_file=args.main_file,
        output_dir=args.output_dir,
        processed_dir=args.processed_dir,
        log_dir=args.log_dir,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
