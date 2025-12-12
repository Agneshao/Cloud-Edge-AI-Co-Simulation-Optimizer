#!/usr/bin/env python3
"""
Jetson Logs MCP Server for EdgeTwin

Provides tools to list and analyze Jetson inference logs stored as CSV files.
"""

import os
import json
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Initialize MCP server
server = Server("jetson_logs_server")

# Path to log files (relative to project root)
LOGS_DIR = Path(__file__).parent.parent.parent / "data" / "sample_logs"


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Return the list of available tools."""
    return [
        Tool(
            name="list_logs",
            description="List all available Jetson inference log files in the data directory",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="load_log",
            description="Load a specific Jetson inference log and return performance statistics including latency, drop rate, and temperature metrics",
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the CSV log file to load (e.g., 'inference_log_20251201_211136.csv')"
                    }
                },
                "required": ["filename"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""

    if name == "list_logs":
        return await handle_list_logs()
    elif name == "load_log":
        return await handle_load_log(arguments.get("filename", ""))
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def handle_list_logs() -> list[TextContent]:
    """List all CSV log files in the logs directory."""

    if not LOGS_DIR.exists():
        return [TextContent(
            type="text",
            text=json.dumps({
                "error": f"Logs directory not found: {LOGS_DIR}",
                "files": []
            }, indent=2)
        )]

    csv_files = sorted(LOGS_DIR.glob("*.csv"))

    files_info = []
    for f in csv_files:
        stat = f.stat()
        files_info.append({
            "filename": f.name,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified": stat.st_mtime
        })

    result = {
        "logs_directory": str(LOGS_DIR),
        "file_count": len(files_info),
        "files": files_info
    }

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_load_log(filename: str) -> list[TextContent]:
    """Load a CSV log file and compute performance statistics."""

    if not filename:
        return [TextContent(
            type="text",
            text=json.dumps({"error": "filename parameter is required"}, indent=2)
        )]

    filepath = LOGS_DIR / filename

    if not filepath.exists():
        return [TextContent(
            type="text",
            text=json.dumps({
                "error": f"File not found: {filename}",
                "available_files": [f.name for f in LOGS_DIR.glob("*.csv")]
            }, indent=2)
        )]

    try:
        # Load the CSV
        df = pd.read_csv(filepath)

        # Compute statistics
        stats = compute_log_stats(df, filename)

        return [TextContent(type="text", text=json.dumps(stats, indent=2))]

    except Exception as e:
        return [TextContent(
            type="text",
            text=json.dumps({"error": f"Failed to load log: {str(e)}"}, indent=2)
        )]


def compute_log_stats(df: pd.DataFrame, filename: str) -> dict[str, Any]:
    """Compute comprehensive statistics from a Jetson inference log DataFrame."""

    total_frames = len(df)

    # Latency metrics (end-to-end is the full pipeline latency)
    e2e_latency = df["end_to_end_ms"] if "end_to_end_ms" in df.columns else df.get("trt_latency_ms", pd.Series([0]))
    trt_latency = df.get("trt_latency_ms", pd.Series([0]))

    avg_latency_ms = float(e2e_latency.mean())
    p50_latency_ms = float(e2e_latency.quantile(0.50))
    p95_latency_ms = float(e2e_latency.quantile(0.95))
    p99_latency_ms = float(e2e_latency.quantile(0.99))
    max_latency_ms = float(e2e_latency.max())
    min_latency_ms = float(e2e_latency.min())

    # Jitter (latency variability)
    latency_std_ms = float(e2e_latency.std())

    # TensorRT-specific latency
    avg_trt_latency_ms = float(trt_latency.mean())

    # Frame drops
    if "frame_drop" in df.columns:
        total_drops = int(df["frame_drop"].sum())
        # frame_drop column seems to be cumulative, so get the max
        max_drop_value = int(df["frame_drop"].max())
        drop_rate = max_drop_value / total_frames if total_frames > 0 else 0
    else:
        total_drops = 0
        drop_rate = 0.0

    # Temperature metrics
    if "gpu_temp_C" in df.columns:
        avg_temp_c = float(df["gpu_temp_C"].mean())
        max_temp_c = float(df["gpu_temp_C"].max())
        min_temp_c = float(df["gpu_temp_C"].min())
    else:
        avg_temp_c = max_temp_c = min_temp_c = 0.0

    # Power metrics
    if "power_mW" in df.columns:
        avg_power_mw = float(df["power_mW"].mean())
        max_power_mw = float(df["power_mW"].max())
    else:
        avg_power_mw = max_power_mw = 0.0

    # GPU utilization
    if "gpu_util_percent" in df.columns:
        avg_gpu_util = float(df["gpu_util_percent"].mean())
        max_gpu_util = float(df["gpu_util_percent"].max())
    else:
        avg_gpu_util = max_gpu_util = 0.0

    # Queue delay (indicates backpressure)
    if "queue_delay_ms" in df.columns:
        avg_queue_delay_ms = float(df["queue_delay_ms"].mean())
        max_queue_delay_ms = float(df["queue_delay_ms"].max())
    else:
        avg_queue_delay_ms = max_queue_delay_ms = 0.0

    # Extract deployment config info from first row
    config_info = {}
    if len(df) > 0:
        row = df.iloc[0]
        config_info = {
            "engine_name": str(row.get("engine_name", "unknown")),
            "engine_precision": str(row.get("engine_precision", "unknown")),
            "engine_batch": int(row.get("engine_batch", 1)),
            "engine_shape": str(row.get("engine_shape", "unknown")),
            "jetson_mode": str(row.get("jetson_mode", "unknown")),
            "platform": str(row.get("platform", "unknown")),
            "tensorrt_version": str(row.get("tensorrt_version", "unknown")),
            "cuda_version": str(row.get("cuda_version", "unknown"))
        }

    # Compute throughput estimate (frames per second)
    if avg_latency_ms > 0:
        estimated_fps = 1000.0 / avg_latency_ms
    else:
        estimated_fps = 0.0

    return {
        "file": filename,
        "total_frames": total_frames,

        "latency": {
            "avg_latency_ms": round(avg_latency_ms, 3),
            "p50_latency_ms": round(p50_latency_ms, 3),
            "p95_latency_ms": round(p95_latency_ms, 3),
            "p99_latency_ms": round(p99_latency_ms, 3),
            "max_latency_ms": round(max_latency_ms, 3),
            "min_latency_ms": round(min_latency_ms, 3),
            "jitter_std_ms": round(latency_std_ms, 3),
            "avg_trt_inference_ms": round(avg_trt_latency_ms, 3)
        },

        "throughput": {
            "estimated_fps": round(estimated_fps, 2)
        },

        "reliability": {
            "total_frame_drops": total_drops,
            "drop_rate": round(drop_rate, 4),
            "avg_queue_delay_ms": round(avg_queue_delay_ms, 3),
            "max_queue_delay_ms": round(max_queue_delay_ms, 3)
        },

        "thermal": {
            "avg_temp_c": round(avg_temp_c, 2),
            "max_temp_c": round(max_temp_c, 2),
            "min_temp_c": round(min_temp_c, 2)
        },

        "power": {
            "avg_power_mw": round(avg_power_mw, 2),
            "max_power_mw": round(max_power_mw, 2),
            "avg_power_w": round(avg_power_mw / 1000, 2)
        },

        "gpu": {
            "avg_util_percent": round(avg_gpu_util, 2),
            "max_util_percent": round(max_gpu_util, 2)
        },

        "deployment_config": config_info
    }


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
