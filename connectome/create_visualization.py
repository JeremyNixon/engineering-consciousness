#!/usr/bin/env python3
"""
FlyWire Connectome 3D Visualization

Creates an interactive 3D visualization of fruit fly brain neurons
colored by neurotransmitter type with filtering by neural flow.

Data source: FlyWire Connectome (v783)
https://github.com/flyconnectome/flywire_annotations
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# Load data
print("Loading neuron annotations...")
df = pd.read_csv('data/neuron_annotations.tsv', sep='\t', low_memory=False)
print(f"Loaded {len(df)} neurons")

# Clean data - use soma positions, remove NaN
df = df.dropna(subset=['soma_x', 'soma_y', 'soma_z', 'top_nt'])
print(f"After filtering: {len(df)} neurons with complete data")

# Sample for visualization (full dataset would be too heavy for browser)
SAMPLE_SIZE = 25000
if len(df) > SAMPLE_SIZE:
    # Stratified sample to maintain neurotransmitter distribution
    df_sampled = df.groupby('top_nt', group_keys=False).apply(
        lambda x: x.sample(min(len(x), int(SAMPLE_SIZE * len(x) / len(df))), random_state=42)
    )
    print(f"Sampled {len(df_sampled)} neurons (stratified by neurotransmitter)")
else:
    df_sampled = df

# Neurotransmitter colors (neuroscience-inspired palette)
nt_colors = {
    'acetylcholine': '#2ecc71',  # Green - excitatory
    'glutamate': '#3498db',       # Blue - excitatory
    'gaba': '#e74c3c',            # Red - inhibitory
    'dopamine': '#9b59b6',        # Purple - modulatory
    'serotonin': '#f39c12',       # Orange - modulatory
    'octopamine': '#1abc9c',      # Teal - insect-specific
}

# Prepare data for each neurotransmitter type
traces_data = {}
for nt in nt_colors.keys():
    nt_df = df_sampled[df_sampled['top_nt'] == nt]
    if len(nt_df) > 0:
        traces_data[nt] = {
            'x': nt_df['soma_x'].tolist(),
            'y': nt_df['soma_y'].tolist(),
            'z': nt_df['soma_z'].tolist(),
            'flow': nt_df['flow'].tolist(),
            'super_class': nt_df['super_class'].tolist(),
            'cell_type': nt_df['cell_type'].fillna('unknown').tolist(),
            'confidence': nt_df['top_nt_conf'].tolist(),
            'count': len(nt_df),
            'color': nt_colors[nt]
        }

# Statistics for display
stats = {
    'total_neurons': len(df),
    'sampled_neurons': len(df_sampled),
    'by_nt': df['top_nt'].value_counts().to_dict(),
    'by_flow': df['flow'].value_counts().to_dict(),
    'by_class': df['super_class'].value_counts().to_dict()
}

# Generate HTML
html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FlyWire Connectome: 3D Neuron Visualization</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #e0e0e0;
        }
        .container { max-width: 1600px; margin: 0 auto; padding: 20px; }
        header {
            text-align: center;
            padding: 30px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            margin-bottom: 20px;
        }
        h1 {
            font-size: 2.5rem;
            background: linear-gradient(90deg, #00d2ff, #3a7bd5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        .subtitle { color: #888; font-size: 1.1rem; }
        .main-content { display: flex; gap: 20px; flex-wrap: wrap; }
        .plot-container {
            flex: 1;
            min-width: 800px;
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
            padding: 15px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        #plot { width: 100%; height: 700px; }
        .sidebar {
            width: 320px;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        .card {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .card h3 {
            font-size: 1rem;
            margin-bottom: 15px;
            color: #3a7bd5;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            cursor: pointer;
            transition: background 0.2s;
        }
        .legend-item:hover { background: rgba(255,255,255,0.05); }
        .legend-item:last-child { border-bottom: none; }
        .color-dot {
            width: 14px;
            height: 14px;
            border-radius: 50%;
            flex-shrink: 0;
        }
        .legend-label { flex: 1; text-transform: capitalize; }
        .legend-count { color: #888; font-size: 0.9rem; }
        .stat-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        .stat-row:last-child { border-bottom: none; }
        .stat-value { color: #3a7bd5; font-weight: 600; }
        .filter-section { margin-bottom: 15px; }
        .filter-section label { display: block; margin-bottom: 8px; font-size: 0.9rem; color: #888; }
        select {
            width: 100%;
            padding: 10px;
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 6px;
            color: #e0e0e0;
            font-size: 0.95rem;
        }
        .info-text { font-size: 0.85rem; color: #666; line-height: 1.5; margin-top: 10px; }
        footer {
            text-align: center;
            padding: 30px;
            color: #666;
            font-size: 0.85rem;
            border-top: 1px solid rgba(255,255,255,0.1);
            margin-top: 30px;
        }
        footer a { color: #3a7bd5; text-decoration: none; }
        footer a:hover { text-decoration: underline; }
        .controls { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 10px; }
        .btn {
            padding: 8px 16px;
            background: rgba(58, 123, 213, 0.3);
            border: 1px solid rgba(58, 123, 213, 0.5);
            border-radius: 6px;
            color: #e0e0e0;
            cursor: pointer;
            font-size: 0.85rem;
            transition: all 0.2s;
        }
        .btn:hover { background: rgba(58, 123, 213, 0.5); }
        .btn.active { background: rgba(58, 123, 213, 0.7); }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🪰 FlyWire Connectome Explorer</h1>
            <p class="subtitle">Interactive 3D visualization of <span id="neuron-count">NEURON_COUNT</span> neurons from the fruit fly brain</p>
        </header>

        <div class="main-content">
            <div class="plot-container">
                <div class="controls">
                    <button class="btn active" onclick="setView('default')">Default View</button>
                    <button class="btn" onclick="setView('top')">Top View (Dorsal)</button>
                    <button class="btn" onclick="setView('front')">Front View</button>
                    <button class="btn" onclick="setView('side')">Side View</button>
                </div>
                <div id="plot"></div>
            </div>

            <div class="sidebar">
                <div class="card">
                    <h3>Neurotransmitter Types</h3>
                    <div id="legend"></div>
                    <p class="info-text">Click legend items to toggle visibility. Neurotransmitters determine whether neurons excite or inhibit their targets.</p>
                </div>

                <div class="card">
                    <h3>Filter by Flow</h3>
                    <div class="filter-section">
                        <label>Neural Signal Direction</label>
                        <select id="flow-filter" onchange="filterByFlow(this.value)">
                            <option value="all">All Neurons</option>
                            <option value="intrinsic">Intrinsic (within brain)</option>
                            <option value="afferent">Afferent (sensory input)</option>
                            <option value="efferent">Efferent (motor output)</option>
                        </select>
                    </div>
                </div>

                <div class="card">
                    <h3>Dataset Statistics</h3>
                    <div class="stat-row">
                        <span>Total neurons (full dataset)</span>
                        <span class="stat-value" id="total-stat">TOTAL</span>
                    </div>
                    <div class="stat-row">
                        <span>Displayed (sampled)</span>
                        <span class="stat-value" id="sampled-stat">SAMPLED</span>
                    </div>
                    <div class="stat-row">
                        <span>Unique cell types</span>
                        <span class="stat-value">~7,000</span>
                    </div>
                    <div class="stat-row">
                        <span>Synaptic connections</span>
                        <span class="stat-value">~130M</span>
                    </div>
                </div>

                <div class="card">
                    <h3>Brain Regions</h3>
                    <div id="class-stats"></div>
                </div>
            </div>
        </div>

        <footer>
            Data: <a href="https://codex.flywire.ai/" target="_blank">FlyWire Connectome v783</a> |
            <a href="https://github.com/flyconnectome/flywire_annotations" target="_blank">GitHub</a> |
            <a href="https://zenodo.org/records/10676866" target="_blank">Zenodo</a><br>
            Citation: Dorkenwald et al. (2024), Schlegel et al. (2024) Nature
        </footer>
    </div>

    <script>
        // Data injected from Python
        const tracesData = TRACES_DATA;
        const stats = STATS_DATA;

        // Update stats display
        document.getElementById('neuron-count').textContent = stats.sampled_neurons.toLocaleString();
        document.getElementById('total-stat').textContent = stats.total_neurons.toLocaleString();
        document.getElementById('sampled-stat').textContent = stats.sampled_neurons.toLocaleString();

        // Build class stats
        const classStatsHtml = Object.entries(stats.by_class)
            .slice(0, 6)
            .map(([cls, count]) => `
                <div class="stat-row">
                    <span>${cls.replace('_', ' ')}</span>
                    <span class="stat-value">${count.toLocaleString()}</span>
                </div>
            `).join('');
        document.getElementById('class-stats').innerHTML = classStatsHtml;

        // Create Plotly traces
        let allTraces = [];
        const ntOrder = ['acetylcholine', 'glutamate', 'gaba', 'dopamine', 'serotonin', 'octopamine'];

        ntOrder.forEach(nt => {
            if (tracesData[nt]) {
                const d = tracesData[nt];
                allTraces.push({
                    type: 'scatter3d',
                    mode: 'markers',
                    name: nt,
                    x: d.x,
                    y: d.y,
                    z: d.z,
                    text: d.cell_type.map((ct, i) =>
                        `Type: ${ct}<br>Flow: ${d.flow[i]}<br>Class: ${d.super_class[i]}<br>Confidence: ${(d.confidence[i] * 100).toFixed(1)}%`
                    ),
                    hovertemplate: '<b>%{fullData.name}</b><br>%{text}<extra></extra>',
                    marker: {
                        size: 1.5,
                        color: d.color,
                        opacity: 0.7
                    },
                    customdata: d.flow
                });
            }
        });

        // Store original traces for filtering
        const originalTraces = JSON.parse(JSON.stringify(allTraces));

        // Layout configuration
        const layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            scene: {
                xaxis: { title: 'X (anterior-posterior)', color: '#666', gridcolor: 'rgba(255,255,255,0.1)' },
                yaxis: { title: 'Y (medial-lateral)', color: '#666', gridcolor: 'rgba(255,255,255,0.1)' },
                zaxis: { title: 'Z (dorsal-ventral)', color: '#666', gridcolor: 'rgba(255,255,255,0.1)' },
                bgcolor: 'rgba(0,0,0,0)',
                camera: { eye: { x: 1.5, y: 1.5, z: 1.2 } }
            },
            margin: { l: 0, r: 0, t: 0, b: 0 },
            showlegend: false,
            hovermode: 'closest'
        };

        // Create plot
        Plotly.newPlot('plot', allTraces, layout, { responsive: true });

        // Build legend
        const legendHtml = ntOrder.map(nt => {
            const d = tracesData[nt];
            if (!d) return '';
            return `
                <div class="legend-item" data-nt="${nt}" onclick="toggleTrace('${nt}')">
                    <div class="color-dot" style="background: ${d.color}"></div>
                    <span class="legend-label">${nt}</span>
                    <span class="legend-count">${d.count.toLocaleString()}</span>
                </div>
            `;
        }).join('');
        document.getElementById('legend').innerHTML = legendHtml;

        // Toggle trace visibility
        const visibility = {};
        ntOrder.forEach(nt => visibility[nt] = true);

        function toggleTrace(nt) {
            visibility[nt] = !visibility[nt];
            const traceIndex = ntOrder.indexOf(nt);
            Plotly.restyle('plot', { visible: visibility[nt] }, [traceIndex]);

            const legendItem = document.querySelector(`[data-nt="${nt}"]`);
            legendItem.style.opacity = visibility[nt] ? 1 : 0.4;
        }

        // Filter by flow type
        function filterByFlow(flowType) {
            const newTraces = originalTraces.map(trace => {
                if (flowType === 'all') {
                    return { ...trace };
                }
                const mask = trace.customdata.map(f => f === flowType);
                return {
                    ...trace,
                    x: trace.x.filter((_, i) => mask[i]),
                    y: trace.y.filter((_, i) => mask[i]),
                    z: trace.z.filter((_, i) => mask[i]),
                    text: trace.text.filter((_, i) => mask[i]),
                    customdata: trace.customdata.filter((_, i) => mask[i])
                };
            });
            Plotly.react('plot', newTraces, layout);
        }

        // Camera presets
        function setView(view) {
            const cameras = {
                default: { eye: { x: 1.5, y: 1.5, z: 1.2 } },
                top: { eye: { x: 0, y: 0, z: 2.5 } },
                front: { eye: { x: 0, y: 2.5, z: 0 } },
                side: { eye: { x: 2.5, y: 0, z: 0 } }
            };
            Plotly.relayout('plot', { 'scene.camera': cameras[view] });

            document.querySelectorAll('.controls .btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
        }
    </script>
</body>
</html>
'''

# Inject data
html = html_template.replace('TRACES_DATA', json.dumps(traces_data))
html = html_template.replace('TRACES_DATA', json.dumps(traces_data))
html = html.replace('STATS_DATA', json.dumps(stats))
html = html.replace('NEURON_COUNT', f"{len(df_sampled):,}")
html = html.replace('TOTAL', f"{stats['total_neurons']:,}")
html = html.replace('SAMPLED', f"{stats['sampled_neurons']:,}")

# Save
output_path = Path('flywire_connectome_3d.html')
with open(output_path, 'w') as f:
    f.write(html)

print(f"\nVisualization saved to: {output_path.absolute()}")
print(f"\nOpen in browser to explore the fruit fly connectome!")
