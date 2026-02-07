#!/usr/bin/env python3
"""
FlyWire Connectome Sunburst Visualization

Creates an interactive sunburst diagram showing the hierarchical organization
of fruit fly neurons: Super Class -> Cell Class -> Neurotransmitter Type

Data source: FlyWire Connectome (v783)
"""

import pandas as pd
import json
from collections import defaultdict

# Load data
print("Loading neuron annotations...")
df = pd.read_csv('data/neuron_annotations.tsv', sep='\t', low_memory=False)
print(f"Loaded {len(df)} neurons")

# Clean data
df = df.dropna(subset=['super_class', 'top_nt'])
df['cell_class'] = df['cell_class'].fillna('unclassified')
df['flow'] = df['flow'].fillna('unknown')

print(f"Processing {len(df)} neurons with classifications...")

# Build hierarchical data for sunburst
# Structure: super_class -> cell_class -> neurotransmitter -> count

hierarchy = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
flow_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

for _, row in df.iterrows():
    sc = row['super_class']
    cc = row['cell_class'] if row['cell_class'] else 'unclassified'
    nt = row['top_nt']
    fl = row['flow']
    hierarchy[sc][cc][nt] += 1
    flow_counts[sc][cc][fl] += 1

# Convert to sunburst format
ids = ['Total']
labels = ['All Neurons']
parents = ['']
values = [len(df)]
colors = ['#2c3e50']

# Color schemes
superclass_colors = {
    'optic': '#3498db',
    'central': '#2ecc71',
    'sensory': '#e74c3c',
    'visual_projection': '#9b59b6',
    'ascending': '#f39c12',
    'descending': '#1abc9c',
    'sensory_ascending': '#e67e22',
    'visual_centrifugal': '#8e44ad',
    'motor': '#c0392b',
    'endocrine': '#d35400'
}

nt_colors = {
    'acetylcholine': '#27ae60',
    'glutamate': '#2980b9',
    'gaba': '#c0392b',
    'dopamine': '#8e44ad',
    'serotonin': '#f39c12',
    'octopamine': '#16a085',
}

for sc, cc_dict in sorted(hierarchy.items(), key=lambda x: -sum(sum(nt.values()) for nt in x[1].values())):
    sc_total = sum(sum(nt_dict.values()) for nt_dict in cc_dict.values())
    sc_id = f"sc_{sc}"
    ids.append(sc_id)
    labels.append(sc.replace('_', ' ').title())
    parents.append('Total')
    values.append(sc_total)
    colors.append(superclass_colors.get(sc, '#7f8c8d'))

    for cc, nt_dict in sorted(cc_dict.items(), key=lambda x: -sum(x[1].values())):
        cc_total = sum(nt_dict.values())
        if cc_total < 50:  # Skip very small classes
            continue
        cc_id = f"cc_{sc}_{cc}"
        ids.append(cc_id)
        labels.append(cc.replace('_', ' ').title() if cc != 'unclassified' else 'Unclassified')
        parents.append(sc_id)
        values.append(cc_total)
        colors.append(superclass_colors.get(sc, '#7f8c8d'))

        for nt, count in sorted(nt_dict.items(), key=lambda x: -x[1]):
            if count < 10:  # Skip very small groups
                continue
            nt_id = f"nt_{sc}_{cc}_{nt}"
            ids.append(nt_id)
            labels.append(nt.title())
            parents.append(cc_id)
            values.append(count)
            colors.append(nt_colors.get(nt, '#95a5a6'))

sunburst_data = {
    'ids': ids,
    'labels': labels,
    'parents': parents,
    'values': values,
    'colors': colors
}

# Calculate statistics
stats = {
    'total': len(df),
    'superclasses': len(hierarchy),
    'nt_breakdown': df['top_nt'].value_counts().to_dict(),
    'flow_breakdown': df['flow'].value_counts().to_dict(),
    'top_cell_classes': df['cell_class'].value_counts().head(10).to_dict()
}

html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FlyWire Connectome: Neural Hierarchy</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 100%);
            min-height: 100vh;
            color: #e0e0e0;
        }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        header {
            text-align: center;
            padding: 40px 0;
        }
        h1 {
            font-size: 2.5rem;
            background: linear-gradient(90deg, #f093fb, #f5576c);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 15px;
        }
        .subtitle { color: #888; font-size: 1.1rem; max-width: 700px; margin: 0 auto; }
        .main-grid { display: grid; grid-template-columns: 1fr 380px; gap: 25px; margin-top: 30px; }
        .plot-container {
            background: rgba(255,255,255,0.03);
            border-radius: 16px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.08);
        }
        #sunburst { width: 100%; height: 750px; }
        .sidebar { display: flex; flex-direction: column; gap: 20px; }
        .card {
            background: rgba(255,255,255,0.04);
            border-radius: 14px;
            padding: 24px;
            border: 1px solid rgba(255,255,255,0.08);
        }
        .card h3 {
            font-size: 0.9rem;
            margin-bottom: 18px;
            color: #f5576c;
            text-transform: uppercase;
            letter-spacing: 1.5px;
        }
        .legend-row {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 10px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        .legend-row:last-child { border-bottom: none; }
        .legend-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            flex-shrink: 0;
        }
        .legend-label { flex: 1; font-size: 0.95rem; }
        .legend-value { color: #888; font-size: 0.9rem; }
        .stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        .stat-box {
            background: rgba(255,255,255,0.03);
            border-radius: 10px;
            padding: 18px;
            text-align: center;
        }
        .stat-number {
            font-size: 1.8rem;
            font-weight: 700;
            background: linear-gradient(90deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .stat-label { font-size: 0.8rem; color: #888; margin-top: 5px; text-transform: uppercase; }
        .instructions {
            background: rgba(102, 126, 234, 0.1);
            border-radius: 10px;
            padding: 16px;
            font-size: 0.9rem;
            line-height: 1.6;
            color: #aaa;
        }
        .instructions strong { color: #667eea; }
        footer {
            text-align: center;
            padding: 40px;
            color: #555;
            font-size: 0.85rem;
        }
        footer a { color: #667eea; text-decoration: none; }
        @media (max-width: 1100px) {
            .main-grid { grid-template-columns: 1fr; }
            .sidebar { order: -1; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🧠 Neural Hierarchy of Drosophila</h1>
            <p class="subtitle">
                Explore the organizational structure of ''' + f"{stats['total']:,}" + ''' neurons in the fruit fly brain.
                Click to zoom into each layer: Super Class → Cell Class → Neurotransmitter Type.
            </p>
        </header>

        <div class="main-grid">
            <div class="plot-container">
                <div id="sunburst"></div>
            </div>

            <div class="sidebar">
                <div class="card">
                    <h3>Overview</h3>
                    <div class="stat-grid">
                        <div class="stat-box">
                            <div class="stat-number">''' + f"{stats['total']:,}" + '''</div>
                            <div class="stat-label">Total Neurons</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-number">10</div>
                            <div class="stat-label">Super Classes</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-number">6</div>
                            <div class="stat-label">Neurotransmitters</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-number">~7K</div>
                            <div class="stat-label">Cell Types</div>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <h3>Neurotransmitter Legend</h3>
                    <div class="legend-row">
                        <div class="legend-dot" style="background: #27ae60"></div>
                        <span class="legend-label">Acetylcholine</span>
                        <span class="legend-value">Excitatory</span>
                    </div>
                    <div class="legend-row">
                        <div class="legend-dot" style="background: #2980b9"></div>
                        <span class="legend-label">Glutamate</span>
                        <span class="legend-value">Excitatory</span>
                    </div>
                    <div class="legend-row">
                        <div class="legend-dot" style="background: #c0392b"></div>
                        <span class="legend-label">GABA</span>
                        <span class="legend-value">Inhibitory</span>
                    </div>
                    <div class="legend-row">
                        <div class="legend-dot" style="background: #8e44ad"></div>
                        <span class="legend-label">Dopamine</span>
                        <span class="legend-value">Modulatory</span>
                    </div>
                    <div class="legend-row">
                        <div class="legend-dot" style="background: #f39c12"></div>
                        <span class="legend-label">Serotonin</span>
                        <span class="legend-value">Modulatory</span>
                    </div>
                    <div class="legend-row">
                        <div class="legend-dot" style="background: #16a085"></div>
                        <span class="legend-label">Octopamine</span>
                        <span class="legend-value">Modulatory (insect)</span>
                    </div>
                </div>

                <div class="card">
                    <h3>Signal Flow</h3>
                    <div class="legend-row">
                        <span class="legend-label">Intrinsic</span>
                        <span class="legend-value">''' + f"{stats['flow_breakdown'].get('intrinsic', 0):,}" + '''</span>
                    </div>
                    <div class="legend-row">
                        <span class="legend-label">Afferent (sensory)</span>
                        <span class="legend-value">''' + f"{stats['flow_breakdown'].get('afferent', 0):,}" + '''</span>
                    </div>
                    <div class="legend-row">
                        <span class="legend-label">Efferent (motor)</span>
                        <span class="legend-value">''' + f"{stats['flow_breakdown'].get('efferent', 0):,}" + '''</span>
                    </div>
                </div>

                <div class="card">
                    <div class="instructions">
                        <strong>How to explore:</strong><br>
                        • Click a segment to zoom in<br>
                        • Click the center to zoom out<br>
                        • Hover for details<br>
                        • Outer ring shows neurotransmitter types within each cell class
                    </div>
                </div>
            </div>
        </div>

        <footer>
            Data: <a href="https://codex.flywire.ai/" target="_blank">FlyWire Connectome v783</a> |
            <a href="https://github.com/flyconnectome/flywire_annotations" target="_blank">GitHub</a><br>
            Citation: Dorkenwald et al. (2024), Schlegel et al. (2024) Nature
        </footer>
    </div>

    <script>
        const data = ''' + json.dumps(sunburst_data) + ''';

        const trace = {
            type: 'sunburst',
            ids: data.ids,
            labels: data.labels,
            parents: data.parents,
            values: data.values,
            branchvalues: 'total',
            marker: {
                colors: data.colors,
                line: { width: 1, color: 'rgba(0,0,0,0.3)' }
            },
            hovertemplate: '<b>%{label}</b><br>Neurons: %{value:,}<br>%{percentParent:.1%} of parent<extra></extra>',
            textinfo: 'label+percent parent',
            insidetextorientation: 'radial',
            maxdepth: 3
        };

        const layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            margin: { t: 10, b: 10, l: 10, r: 10 },
            font: { color: '#e0e0e0', size: 11 }
        };

        Plotly.newPlot('sunburst', [trace], layout, { responsive: true });
    </script>
</body>
</html>
'''

# Save
output_path = 'flywire_hierarchy_sunburst.html'
with open(output_path, 'w') as f:
    f.write(html)

print(f"\nSunburst visualization saved to: {output_path}")
print("Open in browser to explore the neural hierarchy!")
