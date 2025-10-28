#!/usr/bin/env python3
"""
Generate HTML Benchmark Report for vLLM Serving
"""

import json
import re
from pathlib import Path
from datetime import datetime


def parse_json_benchmark(file_path):
    """Parse JSON benchmark file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except:
        return None


def parse_txt_benchmark(file_path):
    """Parse text benchmark file and extract metrics"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        metrics = {}

        # Extract key metrics using regex
        patterns = {
            'successful_requests': r'Successful requests:\s+(\d+)',
            'request_rate': r'Request rate configured.*:\s+([\d.]+)',
            'duration': r'Benchmark duration.*:\s+([\d.]+)',
            'total_input_tokens': r'Total input tokens:\s+(\d+)',
            'total_generated_tokens': r'Total generated tokens:\s+(\d+)',
            'request_throughput': r'Request throughput.*:\s+([\d.]+)',
            'output_throughput': r'Output token throughput.*:\s+([\d.]+)',
            'total_throughput': r'Total Token throughput.*:\s+([\d.]+)',
            'mean_ttft': r'Mean TTFT.*:\s+([\d.]+)',
            'median_ttft': r'Median TTFT.*:\s+([\d.]+)',
            'p99_ttft': r'P99 TTFT.*:\s+([\d.]+)',
            'mean_tpot': r'Mean TPOT.*:\s+([\d.]+)',
            'median_tpot': r'Median TPOT.*:\s+([\d.]+)',
            'p99_tpot': r'P99 TPOT.*:\s+([\d.]+)',
            'correct_rate': r'correct_rate\(%\)\s+([\d.]+)'
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                metrics[key] = float(match.group(1))

        return metrics if metrics else None
    except:
        return None


def collect_all_benchmarks(results_dir):
    """Collect all benchmark data"""
    results_path = Path(results_dir)
    timestamp_dirs = [d for d in results_path.iterdir() if d.is_dir()]

    if not timestamp_dirs:
        return {}

    latest_dir = max(timestamp_dirs, key=lambda d: d.name)
    benchmarks = {}

    for category_dir in latest_dir.iterdir():
        if not category_dir.is_dir():
            continue

        category = category_dir.name
        benchmarks[category] = []

        # Parse JSON files
        for json_file in category_dir.glob("*.json"):
            data = parse_json_benchmark(json_file)
            if data:
                data['file'] = json_file.name
                data['type'] = 'json'
                benchmarks[category].append(data)

        # Parse TXT files
        for txt_file in category_dir.glob("*.txt"):
            data = parse_txt_benchmark(txt_file)
            if data:
                data['file'] = txt_file.name
                data['type'] = 'txt'
                benchmarks[category].append(data)

    return benchmarks


def generate_html_report(benchmarks, output_file):
    """Generate HTML report with charts"""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Collect all metrics for comparison
    all_ttft = []
    all_tpot = []
    all_throughput = []
    labels = []

    for category, results in benchmarks.items():
        for result in results:
            # Use just the category name for cleaner labels
            label = category.replace('_', ' ').title()
            labels.append(label)

            # Extract metrics
            if result.get('type') == 'json':
                all_ttft.append(result.get('mean_ttft_ms', 0))
                all_tpot.append(result.get('mean_tpot_ms', 0))
                all_throughput.append(result.get('output_throughput', 0))
            else:
                all_ttft.append(result.get('mean_ttft', 0))
                all_tpot.append(result.get('mean_tpot', 0))
                all_throughput.append(result.get('output_throughput', 0))

    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>vLLM Benchmark Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}

        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}

        header h1 {{ font-size: 2.5em; margin-bottom: 10px; font-weight: 700; }}
        header p {{ font-size: 1.1em; opacity: 0.9; }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            padding: 40px;
            background: #f8f9fa;
        }}

        .stat-card {{
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }}

        .stat-card:hover {{ transform: translateY(-5px); }}

        .stat-card h3 {{
            color: #667eea;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }}

        .stat-card .value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #2d3748;
            margin-bottom: 5px;
        }}

        .stat-card .label {{ color: #718096; font-size: 0.9em; }}

        .content {{ padding: 40px; }}

        .benchmark-section {{
            margin-bottom: 50px;
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}

        .benchmark-section h2 {{
            font-size: 1.8em;
            color: #2d3748;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}

        .chart-container {{
            background: white;
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
        }}

        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}

        .metrics-table th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}

        .metrics-table td {{
            padding: 12px;
            border-bottom: 1px solid #e2e8f0;
        }}

        .metrics-table tr:hover {{ background: #f7fafc; }}

        footer {{
            background: #2d3748;
            color: white;
            text-align: center;
            padding: 20px;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🚀 vLLM Benchmark Report</h1>
            <p>Generated on {timestamp}</p>
        </header>

        <div class="stats-grid">
            <div class="stat-card">
                <h3>Benchmark Categories</h3>
                <div class="value">{len(benchmarks)}</div>
                <div class="label">Test Types</div>
            </div>
            <div class="stat-card">
                <h3>Total Tests</h3>
                <div class="value">{sum(len(v) for v in benchmarks.values())}</div>
                <div class="label">Benchmark Runs</div>
            </div>
            <div class="stat-card">
                <h3>Avg Throughput</h3>
                <div class="value">{sum(all_throughput)/len(all_throughput) if all_throughput else 0:.1f}</div>
                <div class="label">tokens/sec</div>
            </div>
            <div class="stat-card">
                <h3>Avg TTFT</h3>
                <div class="value">{sum(all_ttft)/len(all_ttft) if all_ttft else 0:.0f}</div>
                <div class="label">ms</div>
            </div>
        </div>

        <div class="content">
            <div class="benchmark-section">
                <h2>📊 Performance Comparison</h2>
                <div class="chart-container" id="ttft-chart"></div>
                <div class="chart-container" id="throughput-chart"></div>
                <div class="chart-container" id="tpot-chart"></div>
            </div>
'''

    # Add detailed sections for each category
    for category, results in benchmarks.items():
        if not results:
            continue

        html += f'''
            <div class="benchmark-section">
                <h2>📋 {category.replace('_', ' ').title()}</h2>
                <table class="metrics-table">
                    <thead>
                        <tr>
                            <th>File</th>
                            <th>Requests</th>
                            <th>TTFT (ms)</th>
                            <th>TPOT (ms)</th>
                            <th>Throughput (tok/s)</th>
                            <th>Duration (s)</th>
                        </tr>
                    </thead>
                    <tbody>
'''

        for result in results:
            file_name = result.get('file', 'unknown')

            if result.get('type') == 'json':
                requests = result.get('completed', 0)
                ttft = result.get('mean_ttft_ms', 0)
                tpot = result.get('mean_tpot_ms', 0)
                throughput = result.get('output_throughput', 0)
                duration = result.get('duration', 0)
            else:
                requests = result.get('successful_requests', 0)
                ttft = result.get('mean_ttft', 0)
                tpot = result.get('mean_tpot', 0)
                throughput = result.get('output_throughput', 0)
                duration = result.get('duration', 0)

            html += f'''
                        <tr>
                            <td><strong>{file_name[:50]}</strong></td>
                            <td>{requests:.0f}</td>
                            <td>{ttft:.2f}</td>
                            <td>{tpot:.2f}</td>
                            <td>{throughput:.2f}</td>
                            <td>{duration:.2f}</td>
                        </tr>
'''

        html += '''
                    </tbody>
                </table>
            </div>
'''

    # Add Plotly charts
    html += f'''
        </div>

        <footer>
            <p>vLLM Benchmark Report | Model: Qwen2.5-VL-7B-Instruct-AWQ</p>
        </footer>
    </div>

    <script>
        // TTFT Chart
        var ttftData = [{{
            x: {json.dumps(labels)},
            y: {json.dumps(all_ttft)},
            type: 'bar',
            marker: {{
                color: '#667eea',
                line: {{ color: '#764ba2', width: 2 }}
            }},
            name: 'Mean TTFT'
        }}];

        var ttftLayout = {{
            title: 'Time to First Token (TTFT) Comparison',
            xaxis: {{
                title: 'Benchmark Category',
                tickfont: {{ size: 14 }}
            }},
            yaxis: {{ title: 'Time (ms)' }},
            height: 450,
            showlegend: false,
            margin: {{ b: 100 }}
        }};

        Plotly.newPlot('ttft-chart', ttftData, ttftLayout);

        // Throughput Chart
        var throughputData = [{{
            x: {json.dumps(labels)},
            y: {json.dumps(all_throughput)},
            type: 'bar',
            marker: {{
                color: '#48bb78',
                line: {{ color: '#2f855a', width: 2 }}
            }},
            name: 'Throughput'
        }}];

        var throughputLayout = {{
            title: 'Output Throughput Comparison',
            xaxis: {{
                title: 'Benchmark Category',
                tickfont: {{ size: 14 }}
            }},
            yaxis: {{ title: 'Tokens/Second' }},
            height: 450,
            showlegend: false,
            margin: {{ b: 100 }}
        }};

        Plotly.newPlot('throughput-chart', throughputData, throughputLayout);

        // TPOT Chart
        var tpotData = [{{
            x: {json.dumps(labels)},
            y: {json.dumps(all_tpot)},
            type: 'bar',
            marker: {{
                color: '#ed8936',
                line: {{ color: '#c05621', width: 2 }}
            }},
            name: 'Mean TPOT'
        }}];

        var tpotLayout = {{
            title: 'Time Per Output Token (TPOT) Comparison',
            xaxis: {{
                title: 'Benchmark Category',
                tickfont: {{ size: 14 }}
            }},
            yaxis: {{ title: 'Time (ms)' }},
            height: 450,
            showlegend: false,
            margin: {{ b: 100 }}
        }};

        Plotly.newPlot('tpot-chart', tpotData, tpotLayout);
    </script>
</body>
</html>
'''

    with open(output_file, 'w') as f:
        f.write(html)

    print(f"✅ Report generated: {output_file}")


if __name__ == "__main__":
    results_dir = "/home/thiendo1/Desktop/test-modal/llm/benchmark_results"
    output_file = "/home/thiendo1/Desktop/test-modal/llm/benchmark_report.html"

    print("📊 Collecting benchmarks...")
    benchmarks = collect_all_benchmarks(results_dir)

    print(f"📝 Generating HTML report...")
    generate_html_report(benchmarks, output_file)

    print(f"\n✨ Done! Open {output_file} in your browser")