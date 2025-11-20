#!/usr/bin/env python3
"""
Script to sequentially run training with all data efficiency strategy configs.

Usage:
    python train_all_strategies.py
    # or
    uv run python train_all_strategies.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List


def check_uv_installed() -> bool:
    """Check if uv is installed."""
    try:
        subprocess.run(["uv", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def run_command(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=check)
    return result


def main():
    """Main function to run all strategy configs sequentially."""
    print("=" * 60)
    print("Data Efficiency Strategies Training Script")
    print("=" * 60)

    # 1. Check if uv is installed
    print("\nStep 1: Checking environment...")
    if not check_uv_installed():
        print("Error: uv is not installed. Install uv: https://github.com/astral-sh/uv")
        sys.exit(1)
    print("✓ uv is installed")

    # 2. Sync environment
    print("\nStep 2: Syncing environment...")
    try:
        run_command(["uv", "sync"])
        print("✓ Environment synced")
    except subprocess.CalledProcessError as e:
        print(f"Error syncing environment: {e}")
        sys.exit(1)

    # 3. Check if dataset exists
    print("\nStep 3: Checking dataset...")
    data_dir = Path("./data")
    if (data_dir / "train").exists() and (data_dir / "validation").exists() and (data_dir / "test").exists():
        print("✓ Dataset already downloaded (skipping)")
    else:
        print("Downloading dataset...")
        try:
            run_command(["uv", "run", "download_dataset"])
            print("✓ Dataset downloaded")
        except subprocess.CalledProcessError as e:
            print(f"Error downloading dataset: {e}")
            sys.exit(1)

    # 4. Load environment variables (if .env exists)
    print("\nStep 4: Loading environment variables...")
    env_file = Path(".env")
    if env_file.exists():
        print("✓ .env file found (environment variables will be loaded by the training script)")
    else:
        print("⚠ .env file not found. Continuing without it.")
        print("  For ClearML usage, create .env with CLEARML_API_ACCESS_KEY and CLEARML_API_SECRET_KEY")

    # 5. Define config files to run
    configs = [
        "configs/train_config_edfs_lite.json",
        "configs/train_config_entropy_diversity.json",
        "configs/train_config_k_center.json",
        "configs/train_config_lexical_diversity.json",
        "configs/train_config_qdit_lite.json",
    ]

    # 6. Validate configs exist
    print("\nStep 5: Validating config files...")
    valid_configs = []
    for config_file in configs:
        config_path = Path(config_file)
        if config_path.exists():
            # Validate JSON
            try:
                with open(config_path, "r") as f:
                    json.load(f)
                valid_configs.append(config_file)
                print(f"✓ {config_file}")
            except json.JSONDecodeError as e:
                print(f"⚠ Error: {config_file} is not valid JSON: {e}")
        else:
            print(f"⚠ Warning: {config_file} not found. Skipping...")

    if not valid_configs:
        print("Error: No valid config files found!")
        sys.exit(1)

    # 7. Run training for each config
    print("\nStep 6: Running training for all strategies...")
    print("=" * 60)

    total = len(valid_configs)
    completed = 0
    failed = []

    for idx, config_file in enumerate(valid_configs, 1):
        # Extract strategy name from config file
        strategy_name = Path(config_file).stem.replace("train_config_", "")

        print("\n" + "-" * 60)
        print(f"[{idx}/{total}] Running: {strategy_name}")
        print(f"Config: {config_file}")
        print("-" * 60)

        try:
            # Run training
            run_command(["uv", "run", "run", "--config", config_file])
            completed += 1
            print(f"\n✓ Completed: {strategy_name}")
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Failed: {strategy_name}")
            failed.append((strategy_name, config_file, e))
            # Ask if we should continue
            response = input("\nContinue with next config? (y/n): ").strip().lower()
            if response != "y":
                print("Stopping execution.")
                break

    # 8. Print summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Total configs: {total}")
    print(f"Completed: {completed}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed configs:")
        for strategy_name, config_file, error in failed:
            print(f"  - {strategy_name} ({config_file})")

    print("\n" + "=" * 60)
    print("All training runs completed!")
    print("=" * 60)
    print("\nCheckpoints saved in: ./checkpoints/")
    print("Logs saved in: ./runs/")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

