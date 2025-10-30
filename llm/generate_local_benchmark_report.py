"""
Generate an HTML report from local custom images benchmark results.

Usage:
    python generate_html_report.py
    python generate_html_report.py --timestamp 20250129_120000
    python generate_html_report.py --results-dir ./benchmark_results --open
"""

import json
import argparse
import webbrowser
from pathlib import Path
from datetime import datetime


def generate_html_report(data, output_path):
    """Generate HTML report with Plotly charts."""
    summary = data['summary']
    detailed = data['detailed_results']

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    successful = [r for r in detailed if r['success']]
    failed = [r for r in detailed if not r['success']]

    # Prepare data for charts
    image_names = [r['image_name'][:20] + '...' if len(r['image_name']) > 20 else r['image_name']
                   for r in successful]
    latencies = [r['latency'] * 1000 for r in successful]  # Convert to ms
    ttfts = [r['ttft'] * 1000 if r.get('ttft') else 0 for r in successful]
    output_tokens = [r['output_tokens'] for r in successful]

    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Custom Images Benchmark Report</title>
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

        .alert {{
            padding: 15px 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid;
        }}

        .alert-warning {{
            background: #fff3cd;
            border-color: #856404;
            color: #856404;
        }}

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
            <h1>🚀 Custom Images Benchmark Report</h1>
            <p>Generated on {timestamp}</p>
        </header>

        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Images</h3>
                <div class="value">{summary['num_images']}</div>
                <div class="label">Processed</div>
            </div>
            <div class="stat-card">
                <h3>Success Rate</h3>
                <div class="value">{summary['successful_requests'] / summary['num_images'] * 100:.1f}%</div>
                <div class="label">{summary['successful_requests']}/{summary['num_images']} requests</div>
            </div>
            <div class="stat-card">
                <h3>Avg Throughput</h3>
                <div class="value">{summary['output_token_throughput']:.1f}</div>
                <div class="label">tokens/sec</div>
            </div>
            <div class="stat-card">
                <h3>Avg TTFT</h3>
                <div class="value">{summary.get('mean_ttft_ms', 0):.0f}</div>
                <div class="label">ms</div>
            </div>
        </div>

        <div class="content">
            <div class="benchmark-section">
                <h2>📊 Performance Overview</h2>
                <table class="metrics-table">
                    <tbody>
                        <tr>
                            <td><strong>Server</strong></td>
                            <td>{summary['server_url']}</td>
                        </tr>
                        <tr>
                            <td><strong>Model</strong></td>
                            <td>{summary['model']}</td>
                        </tr>
                        <tr>
                            <td><strong>Timestamp</strong></td>
                            <td>{summary['timestamp']}</td>
                        </tr>
                        <tr>
                            <td><strong>Duration</strong></td>
                            <td>{summary['benchmark_duration']:.2f}s</td>
                        </tr>
                        <tr>
                            <td><strong>Max Concurrency</strong></td>
                            <td>{summary['max_concurrency']}</td>
                        </tr>
                        <tr>
                            <td><strong>Max Tokens</strong></td>
                            <td>{summary['max_tokens']}</td>
                        </tr>
                        <tr>
                            <td><strong>Temperature</strong></td>
                            <td>{summary['temperature']}</td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <div class="benchmark-section">
                <h2>📈 Performance Charts</h2>
                <div class="chart-container" id="latency-chart"></div>
                <div class="chart-container" id="ttft-chart"></div>
                <div class="chart-container" id="tokens-chart"></div>
            </div>

            <div class="benchmark-section">
                <h2>📊 Detailed Metrics</h2>
                <table class="metrics-table">
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Mean</th>
                            <th>Median</th>
                            <th>P99</th>
                        </tr>
                    </thead>
                    <tbody>
'''

    # Add metrics rows
    if summary.get('mean_ttft_ms'):
        html += f'''
                        <tr>
                            <td><strong>TTFT (Time to First Token)</strong></td>
                            <td>{summary['mean_ttft_ms']:.2f} ms</td>
                            <td>{summary['median_ttft_ms']:.2f} ms</td>
                            <td>{summary['p99_ttft_ms']:.2f} ms</td>
                        </tr>
'''

    if summary.get('mean_itl_ms'):
        html += f'''
                        <tr>
                            <td><strong>ITL (Inter-Token Latency)</strong></td>
                            <td>{summary['mean_itl_ms']:.2f} ms</td>
                            <td>{summary['median_itl_ms']:.2f} ms</td>
                            <td>{summary['p99_itl_ms']:.2f} ms</td>
                        </tr>
'''

    if summary.get('mean_tpot_ms'):
        html += f'''
                        <tr>
                            <td><strong>TPOT (Time per Output Token)</strong></td>
                            <td>{summary['mean_tpot_ms']:.2f} ms</td>
                            <td>{summary['median_tpot_ms']:.2f} ms</td>
                            <td>{summary['p99_tpot_ms']:.2f} ms</td>
                        </tr>
'''

    html += f'''
                        <tr>
                            <td><strong>E2EL (End-to-End Latency)</strong></td>
                            <td>{summary['mean_e2el_ms']:.2f} ms</td>
                            <td>{summary['median_e2el_ms']:.2f} ms</td>
                            <td>{summary['p99_e2el_ms']:.2f} ms</td>
                        </tr>
                        <tr>
                            <td><strong>Request Throughput</strong></td>
                            <td colspan="3">{summary['request_throughput']:.2f} req/s</td>
                        </tr>
                        <tr>
                            <td><strong>Token Throughput</strong></td>
                            <td colspan="3">{summary['output_token_throughput']:.2f} tok/s</td>
                        </tr>
                        <tr>
                            <td><strong>Total Output Tokens</strong></td>
                            <td colspan="3">{summary['total_output_tokens']}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
'''

    # Add failed requests section if any
    if failed:
        html += f'''
            <div class="benchmark-section">
                <h2>❌ Failed Requests</h2>
                <div class="alert alert-warning">
                    <strong>Warning:</strong> {len(failed)} request(s) failed during the benchmark.
                </div>
                <table class="metrics-table">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Image</th>
                            <th>Error</th>
                        </tr>
                    </thead>
                    <tbody>
'''
        for i, r in enumerate(failed, 1):
            html += f'''
                        <tr>
                            <td>{i}</td>
                            <td>{r['image_name']}</td>
                            <td style="color: #dc3545;">{r['error']}</td>
                        </tr>
'''
        html += '''
                    </tbody>
                </table>
            </div>
'''

    # Finish HTML
    html += f'''
        </div>

        <footer>
            <p>Custom Images Benchmark Report | Model: {summary['model']}</p>
        </footer>
    </div>

    <script>
        // Latency Chart
        var latencyData = [{{
            x: {json.dumps(image_names)},
            y: {json.dumps(latencies)},
            type: 'bar',
            marker: {{
                color: '#667eea',
                line: {{ color: '#764ba2', width: 2 }}
            }},
            name: 'E2E Latency'
        }}];

        var latencyLayout = {{
            title: 'End-to-End Latency per Image',
            xaxis: {{
                title: 'Image',
                tickangle: -45,
                tickfont: {{ size: 10 }}
            }},
            yaxis: {{ title: 'Latency (ms)' }},
            height: 500,
            showlegend: false,
            margin: {{ b: 150 }}
        }};

        Plotly.newPlot('latency-chart', latencyData, latencyLayout);

        // TTFT Chart
        var ttftData = [{{
            x: {json.dumps(image_names)},
            y: {json.dumps(ttfts)},
            type: 'bar',
            marker: {{
                color: '#48bb78',
                line: {{ color: '#2f855a', width: 2 }}
            }},
            name: 'TTFT'
        }}];

        var ttftLayout = {{
            title: 'Time to First Token (TTFT)',
            xaxis: {{
                title: 'Image',
                tickangle: -45,
                tickfont: {{ size: 10 }}
            }},
            yaxis: {{ title: 'Time (ms)' }},
            height: 500,
            showlegend: false,
            margin: {{ b: 150 }}
        }};

        Plotly.newPlot('ttft-chart', ttftData, ttftLayout);

        // Output Tokens Chart
        var tokensData = [{{
            x: {json.dumps(image_names)},
            y: {json.dumps(output_tokens)},
            type: 'bar',
            marker: {{
                color: '#ed8936',
                line: {{ color: '#c05621', width: 2 }}
            }},
            name: 'Output Tokens'
        }}];

        var tokensLayout = {{
            title: 'Output Tokens Generated',
            xaxis: {{
                title: 'Image',
                tickangle: -45,
                tickfont: {{ size: 10 }}
            }},
            yaxis: {{ title: 'Tokens' }},
            height: 500,
            showlegend: false,
            margin: {{ b: 150 }}
        }};

        Plotly.newPlot('tokens-chart', tokensData, tokensLayout);
    </script>
</body>
</html>
'''

    # Save HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    return output_path


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Benchmark Report - {timestamp}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        .header .subtitle {{
            font-size: 1.2em;
            opacity: 0.9;
        }}

        .content {{
            padding: 40px;
        }}

        .section {{
            margin-bottom: 40px;
        }}

        .section-title {{
            font-size: 1.8em;
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}

        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .card {{
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 20px;
            transition: transform 0.2s, box-shadow 0.2s;
        }}

        .card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}

        .card-title {{
            font-size: 0.9em;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 10px;
        }}

        .card-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}

        .card-unit {{
            font-size: 0.9em;
            color: #6c757d;
            margin-left: 5px;
        }}

        .metric-group {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }}

        .metric-group h3 {{
            color: #495057;
            margin-bottom: 15px;
            font-size: 1.3em;
        }}

        .metric-row {{
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #dee2e6;
        }}

        .metric-row:last-child {{
            border-bottom: none;
        }}

        .metric-label {{
            color: #6c757d;
            font-weight: 500;
        }}

        .metric-value {{
            font-weight: bold;
            color: #333;
        }}

        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
        }}

        .badge-success {{
            background: #d4edda;
            color: #155724;
        }}

        .badge-danger {{
            background: #f8d7da;
            color: #721c24;
        }}

        .badge-warning {{
            background: #fff3cd;
            color: #856404;
        }}

        .badge-info {{
            background: #d1ecf1;
            color: #0c5460;
        }}

        .progress-bar {{
            width: 100%;
            height: 30px;
            background: #e9ecef;
            border-radius: 15px;
            overflow: hidden;
            margin: 10px 0;
        }}

        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            transition: width 1s ease;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}

        th {{
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}

        td {{
            padding: 12px;
            border-bottom: 1px solid #dee2e6;
        }}

        tr:hover {{
            background: #f8f9fa;
        }}

        .status-success {{
            color: #28a745;
            font-weight: bold;
        }}

        .status-failed {{
            color: #dc3545;
            font-weight: bold;
        }}

        .alert {{
            padding: 15px 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}

        .alert-info {{
            background: #d1ecf1;
            border-left: 4px solid #0c5460;
            color: #0c5460;
        }}

        .alert-warning {{
            background: #fff3cd;
            border-left: 4px solid #856404;
            color: #856404;
        }}

        .footer {{
            background: #f8f9fa;
            padding: 20px 40px;
            text-align: center;
            color: #6c757d;
            border-top: 1px solid #dee2e6;
        }}

        .chart-placeholder {{
            background: #f8f9fa;
            border: 2px dashed #dee2e6;
            border-radius: 8px;
            padding: 40px;
            text-align: center;
            color: #6c757d;
            margin: 20px 0;
        }}

        @media print {{
            body {{
                background: white;
                padding: 0;
            }}

            .container {{
                box-shadow: none;
            }}

            .card:hover {{
                transform: none;
                box-shadow: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Benchmark Report</h1>
            <div class="subtitle">Vision Language Model Performance Analysis</div>
            <div style="margin-top: 20px; font-size: 0.9em; opacity: 0.8;">
                Generated: {generated_time}
            </div>
        </div>

        <div class="content">
            <!-- Overview Section -->
            <div class="section">
                <h2 class="section-title">📊 Overview</h2>

                <div class="grid">
                    <div class="card">
                        <div class="card-title">Timestamp</div>
                        <div class="card-value" style="font-size: 1.2em;">{timestamp}</div>
                    </div>
                    <div class="card">
                        <div class="card-title">Model</div>
                        <div class="card-value" style="font-size: 1.2em;">{model}</div>
                    </div>
                    <div class="card">
                        <div class="card-title">Duration</div>
                        <div class="card-value">{duration}<span class="card-unit">s</span></div>
                    </div>
                    <div class="card">
                        <div class="card-title">Total Images</div>
                        <div class="card-value">{num_images}</div>
                    </div>
                </div>

                <div class="grid">
                    <div class="card">
                        <div class="card-title">Successful Requests</div>
                        <div class="card-value" style="color: #28a745;">{successful}</div>
                    </div>
                    <div class="card">
                        <div class="card-title">Failed Requests</div>
                        <div class="card-value" style="color: #dc3545;">{failed}</div>
                    </div>
                    <div class="card">
                        <div class="card-title">Success Rate</div>
                        <div class="card-value">{success_rate}<span class="card-unit">%</span></div>
                    </div>
                    <div class="card">
                        <div class="card-title">Max Concurrency</div>
                        <div class="card-value">{max_concurrency}</div>
                    </div>
                </div>

                <div class="alert alert-info">
                    <strong>Server:</strong> {server_url}
                </div>
            </div>

            <!-- Performance Metrics Section -->
            <div class="section">
                <h2 class="section-title">⚡ Performance Metrics</h2>

                <div class="metric-group">
                    <h3>🚄 Throughput</h3>
                    <div class="metric-row">
                        <span class="metric-label">Request Throughput</span>
                        <span class="metric-value">{request_throughput} req/s</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Token Throughput</span>
                        <span class="metric-value">{token_throughput} tok/s</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Total Output Tokens</span>
                        <span class="metric-value">{total_tokens}</span>
                    </div>
                </div>

                {ttft_section}
                {itl_section}
                {tpot_section}
                {e2el_section}
            </div>

            <!-- Token Distribution Section -->
            <div class="section">
                <h2 class="section-title">📈 Token Distribution</h2>
                {token_stats}
            </div>

            <!-- Top Requests Section -->
            <div class="section">
                <h2 class="section-title">🏆 Top Slowest Requests</h2>
                {slowest_table}
            </div>

            <div class="section">
                <h2 class="section-title">⚡ Top Fastest Requests</h2>
                {fastest_table}
            </div>

            <!-- Failed Requests Section -->
            {failed_section}
        </div>

        <div class="footer">
            <p>Generated by Custom Images Benchmark Tool</p>
            <p style="font-size: 0.9em; margin-top: 5px;">Report generated at {generated_time}</p>
        </div>
    </div>
</body>
</html>
"""


def format_duration(seconds):
    """Format duration in seconds."""
    if seconds < 60:
        return f"{seconds:.2f}"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def generate_metric_section(title, icon, mean, median, p99):
    """Generate HTML for a metric section."""
    if mean is None:
        return ""

    return f"""
    <div class="metric-group">
        <h3>{icon} {title}</h3>
        <div class="metric-row">
            <span class="metric-label">Mean</span>
            <span class="metric-value">{mean:.2f} ms</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Median</span>
            <span class="metric-value">{median:.2f} ms</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">P99</span>
            <span class="metric-value">{p99:.2f} ms</span>
        </div>
    </div>
    """


def generate_token_stats(detailed_results):
    """Generate token distribution statistics."""
    successful = [r for r in detailed_results if r['success']]

    if not successful:
        return "<p>No successful requests to analyze.</p>"

    tokens = [r['output_tokens'] for r in successful]

    return f"""
    <div class="grid">
        <div class="card">
            <div class="card-title">Total Tokens</div>
            <div class="card-value">{sum(tokens)}</div>
        </div>
        <div class="card">
            <div class="card-title">Mean Tokens/Request</div>
            <div class="card-value">{sum(tokens) / len(tokens):.1f}</div>
        </div>
        <div class="card">
            <div class="card-title">Min Tokens</div>
            <div class="card-value">{min(tokens)}</div>
        </div>
        <div class="card">
            <div class="card-title">Max Tokens</div>
            <div class="card-value">{max(tokens)}</div>
        </div>
    </div>
    """


def generate_requests_table(results, title="Requests"):
    """Generate HTML table for requests."""
    if not results:
        return f"<p>No {title.lower()} to display.</p>"

    rows = []
    for i, r in enumerate(results, 1):
        ttft = f"{r['ttft'] * 1000:.2f}" if r.get('ttft') else "N/A"
        tpot = "N/A"
        if r.get('ttft') and r['output_tokens'] > 1:
            tpot_val = (r['latency'] - r['ttft']) / (r['output_tokens'] - 1)
            tpot = f"{tpot_val * 1000:.2f}"

        rows.append(f"""
        <tr>
            <td>{i}</td>
            <td>{r['image_name']}</td>
            <td>{r['latency'] * 1000:.2f} ms</td>
            <td>{ttft} ms</td>
            <td>{tpot} ms</td>
            <td>{r['output_tokens']}</td>
        </tr>
        """)

    return f"""
    <table>
        <thead>
            <tr>
                <th>#</th>
                <th>Image</th>
                <th>Latency</th>
                <th>TTFT</th>
                <th>TPOT</th>
                <th>Tokens</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """


def generate_failed_section(detailed_results):
    """Generate failed requests section."""
    failed = [r for r in detailed_results if not r['success']]

    if not failed:
        return ""

    rows = []
    for i, r in enumerate(failed, 1):
        rows.append(f"""
        <tr>
            <td>{i}</td>
            <td>{r['image_name']}</td>
            <td>{r['request_id']}</td>
            <td style="color: #dc3545;">{r['error']}</td>
        </tr>
        """)

    return f"""
    <div class="section">
        <h2 class="section-title">❌ Failed Requests</h2>
        <div class="alert alert-warning">
            <strong>Warning:</strong> {len(failed)} request(s) failed during the benchmark.
        </div>
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Image</th>
                    <th>Request ID</th>
                    <th>Error</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
    </div>
    """


def generate_html_report(data, output_path):
    """Generate HTML report from benchmark data."""
    summary = data['summary']
    detailed = data['detailed_results']

    successful = [r for r in detailed if r['success']]
    slowest = sorted(successful, key=lambda x: x['latency'], reverse=True)[:5]
    fastest = sorted(successful, key=lambda x: x['latency'])[:5]

    # Generate metric sections
    ttft_section = generate_metric_section(
        "Time to First Token (TTFT)",
        "⏱️",
        summary.get('mean_ttft_ms'),
        summary.get('median_ttft_ms'),
        summary.get('p99_ttft_ms')
    )

    itl_section = generate_metric_section(
        "Inter-Token Latency (ITL)",
        "🔄",
        summary.get('mean_itl_ms'),
        summary.get('median_itl_ms'),
        summary.get('p99_itl_ms')
    )

    tpot_section = generate_metric_section(
        "Time per Output Token (TPOT)",
        "📝",
        summary.get('mean_tpot_ms'),
        summary.get('median_tpot_ms'),
        summary.get('p99_tpot_ms')
    )

    e2el_section = generate_metric_section(
        "End-to-End Latency (E2EL)",
        "🎯",
        summary.get('mean_e2el_ms'),
        summary.get('median_e2el_ms'),
        summary.get('p99_e2el_ms')
    )

    # Generate HTML
    html = HTML_TEMPLATE.format(
        timestamp=summary['timestamp'],
        generated_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        model=summary['model'],
        duration=format_duration(summary['benchmark_duration']),
        num_images=summary['num_images'],
        successful=summary['successful_requests'],
        failed=summary['failed_requests'],
        success_rate=f"{summary['successful_requests'] / summary['num_images'] * 100:.2f}",
        max_concurrency=summary['max_concurrency'],
        server_url=summary['server_url'],
        request_throughput=f"{summary['request_throughput']:.2f}",
        token_throughput=f"{summary['output_token_throughput']:.2f}",
        total_tokens=summary['total_output_tokens'],
        ttft_section=ttft_section,
        itl_section=itl_section,
        tpot_section=tpot_section,
        e2el_section=e2el_section,
        token_stats=generate_token_stats(detailed),
        slowest_table=generate_requests_table(slowest, "slowest requests"),
        fastest_table=generate_requests_table(fastest, "fastest requests"),
        failed_section=generate_failed_section(detailed),
    )

    # Save HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"✅ HTML report generated: {output_path}")
    return output_path


def find_latest_results(results_dir):
    """Find the latest benchmark results."""
    results_dir = Path(results_dir)

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    timestamp_dirs = [d for d in results_dir.iterdir() if d.is_dir()]

    if not timestamp_dirs:
        raise FileNotFoundError(f"No benchmark results found in: {results_dir}")

    latest_dir = sorted(timestamp_dirs, key=lambda x: x.name)[-1]
    result_file = latest_dir / "custom_images" / "benchmark_results.json"

    if not result_file.exists():
        raise FileNotFoundError(f"No benchmark results found in: {latest_dir}")

    return result_file


def main():
    parser = argparse.ArgumentParser(
        description="Generate HTML report from local custom images benchmark results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./local_benchmark_results",
        help="Path to benchmark results directory"
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Specific timestamp to generate report for"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output HTML file path"
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open HTML report in browser after generation"
    )

    args = parser.parse_args()

    try:
        # Find results file
        if args.timestamp:
            result_file = Path(args.results_dir) / args.timestamp / "custom_images" / "benchmark_results.json"
            if not result_file.exists():
                raise FileNotFoundError(f"No results found for timestamp: {args.timestamp}")
        else:
            result_file = find_latest_results(args.results_dir)
            print(f"📁 Using latest results: {result_file}")

        # Load results
        with open(result_file, 'r') as f:
            data = json.load(f)

        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = result_file.parent / "report.html"

        # Generate HTML report
        html_path = generate_html_report(data, output_path)

        # Open in browser if requested
        if args.open:
            print(f"🌐 Opening report in browser...")
            webbrowser.open(f"file://{html_path.absolute()}")

        print(f"\n✨ Done! View your report at: {html_path}")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())