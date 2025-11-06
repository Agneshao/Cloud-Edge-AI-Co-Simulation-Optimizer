"""Report generation using Jinja2 templates."""

from typing import Dict, Any, List, Optional
from pathlib import Path
import json


class ReportGenerator:
    """Generate HTML reports from profiling and optimization results."""
    
    def __init__(self, output_dir: str = "artifacts/reports"):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(
        self,
        profile_results: Dict[str, Any],
        predictions: Dict[str, Any],
        optimized_config: Optional[Dict[str, Any]] = None,
        output_file: Optional[str] = None
    ) -> str:
        """
        Generate HTML report.
        
        Args:
            profile_results: Profile data from Jetson
            predictions: Prediction results
            optimized_config: Optimized configuration (if available)
            output_file: Output filename (defaults to timestamped)
        
        Returns:
            Path to generated report
        """
        if output_file is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"edgetwin_report_{timestamp}.html"
        
        output_path = self.output_dir / output_file
        
        # Simple HTML template (can be enhanced with Jinja2 later)
        html_content = self._generate_html(
            profile_results,
            predictions,
            optimized_config
        )
        
        output_path.write_text(html_content, encoding="utf-8")
        
        return str(output_path)
    
    def _generate_html(
        self,
        profile_results: Dict[str, Any],
        predictions: Dict[str, Any],
        optimized_config: Optional[Dict[str, Any]]
    ) -> str:
        """Generate HTML content."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>EdgeTwin Performance Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 5px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .metric {{ font-weight: bold; color: #27ae60; }}
    </style>
</head>
<body>
    <h1>EdgeTwin Performance Report</h1>
    
    <h2>Profile Results</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Latency (ms)</td><td class="metric">{profile_results.get('latency_ms', 'N/A')}</td></tr>
        <tr><td>Power (W)</td><td class="metric">{profile_results.get('power_w', 'N/A')}</td></tr>
        <tr><td>Memory (MB)</td><td class="metric">{profile_results.get('memory_mb', 'N/A')}</td></tr>
        <tr><td>FPS</td><td class="metric">{profile_results.get('fps', 'N/A')}</td></tr>
    </table>
    
    <h2>Predictions</h2>
    <table>
        <tr><th>Metric</th><th>Predicted Value</th></tr>
        <tr><td>Latency (ms)</td><td class="metric">{predictions.get('latency_ms', 'N/A')}</td></tr>
        <tr><td>Power (W)</td><td class="metric">{predictions.get('power_w', 'N/A')}</td></tr>
        <tr><td>Time to Throttle (s)</td><td class="metric">{predictions.get('time_to_throttle_s', 'N/A')}</td></tr>
    </table>
"""
        
        if optimized_config:
            html += f"""
    <h2>Optimized Configuration</h2>
    <table>
        <tr><th>Parameter</th><th>Value</th></tr>
        <tr><td>Precision</td><td class="metric">{optimized_config.get('precision', 'N/A')}</td></tr>
        <tr><td>Resolution</td><td class="metric">{optimized_config.get('resolution', 'N/A')}</td></tr>
        <tr><td>Batch Size</td><td class="metric">{optimized_config.get('batch_size', 'N/A')}</td></tr>
        <tr><td>Frame Skip</td><td class="metric">{optimized_config.get('frame_skip', 'N/A')}</td></tr>
    </table>
"""
        
        html += """
</body>
</html>
"""
        return html

