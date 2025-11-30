"""CLI commands for data conversion and management."""

import argparse
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.utils.data_utils import convert_sample_data_to_edgetwin, load_profile_results


def convert_data(args):
    """Convert sample data CSV to EdgeTwin format."""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1
    
    output_path = args.output or "data/jetbenchdb/profiles_local.csv"
    
    print(f"Converting {input_path} to EdgeTwin format...")
    print(f"Output: {output_path}")
    
    try:
        df = convert_sample_data_to_edgetwin(
            csv_path=str(input_path),
            output_path=output_path,
            sku_mapping=args.sku_mapping
        )
        
        print(f"✓ Converted {len(df)} rows successfully")
        print(f"✓ Saved to {output_path}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def add_data_parser(subparsers):
    """Add data-related subcommands to CLI."""
    data_parser = subparsers.add_parser(
        "convert-data",
        help="Convert sample profiling CSV to EdgeTwin format"
    )
    data_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file (your sample data format)"
    )
    data_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: data/jetbenchdb/profiles_local.csv)"
    )
    data_parser.add_argument(
        "--sku-mapping",
        type=str,
        default=None,
        help="JSON string for SKU mapping (e.g., '{\"15W\": \"orin_nx\"}')"
    )
    data_parser.set_defaults(func=convert_data)

