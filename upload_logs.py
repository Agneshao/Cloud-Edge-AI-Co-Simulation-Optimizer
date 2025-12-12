#!/usr/bin/env python3
"""Upload local CSV logs to EdgeTwin SmartBucket via raindrop CLI."""

import os
import subprocess
import sys

LOG_DIR = "data/sample_logs"
BUCKET_NAME = "logs"

def run_command(cmd: list[str]) -> tuple[int, str, str]:
    """Run a shell command and return exit code, stdout, stderr."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr

def main():
    # Find all CSV files
    if not os.path.exists(LOG_DIR):
        print(f"Error: Directory {LOG_DIR} not found")
        sys.exit(1)

    csv_files = [f for f in os.listdir(LOG_DIR) if f.endswith('.csv')]

    if not csv_files:
        print(f"No CSV files found in {LOG_DIR}")
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV files to upload\n")

    # Upload each file
    success_count = 0
    for filename in csv_files:
        filepath = os.path.join(LOG_DIR, filename)
        dest_key = f"logs/{filename}"

        print(f"Uploading: {filename}")
        file_size = os.path.getsize(filepath)
        print(f"  Size: {file_size:,} bytes")

        cmd = [
            "raindrop", "object", "put",
            filepath,
            dest_key,
            "-b", BUCKET_NAME,
            "-t", "text/csv"
        ]

        exit_code, stdout, stderr = run_command(cmd)

        if exit_code == 0:
            print(f"  ✓ Uploaded successfully")
            success_count += 1
        else:
            print(f"  ✗ Upload failed")
            if stderr:
                print(f"    Error: {stderr.strip()}")
            if stdout:
                print(f"    Output: {stdout.strip()}")

        print()

    print("=" * 50)
    print(f"Uploaded {success_count}/{len(csv_files)} files")

    # List objects in bucket
    print("\nVerifying uploads...")
    exit_code, stdout, stderr = run_command(["raindrop", "object", "list", "-b", BUCKET_NAME])

    if exit_code == 0:
        print(stdout)
    else:
        print(f"Failed to list bucket contents: {stderr}")

    print("\nDone!")

if __name__ == "__main__":
    main()
