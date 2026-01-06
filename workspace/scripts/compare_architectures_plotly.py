"""
Compare Ridge Points Across Different CiM Architectures using Plotly

This script runs simulations on multiple CiM architectures and creates
interactive Plotly visualizations comparing ridge points and roofline models.
"""

import os
import sys
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

THIS_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(THIS_SCRIPT_DIR)

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from roofline_analysis import (
    get_model_layer_files,
    calculate_layer_profile,
    get_hardware_specs,
    HardwareSpec,
    LayerProfile,
    run_simulations_parallel,
    profile_layer_on_hardware,
    get_available_models,
    get_output_dir,
)


# All available CiM architectures
CIM_ARCHITECTURES = [
    "basic_analog",
    "isaac_isca_2016",
    "raella_isca_2023",
    "colonnade_jssc_2021",
    "wan_nature_2022",
    "jia_jssc_2020",
    "sinangil_jssc_2021",
    "wang_vlsi_2022",
    "lightning_sigc_2023",
    "albireo_isca_2021",
]


def get_all_hardware_specs() -> Dict[str, HardwareSpec]:
    """Get hardware specs for all architectures."""
    specs = {}
    for arch in CIM_ARCHITECTURES:
        specs[arch] = get_hardware_specs(arch)
    return specs


def create_ridge_point_comparison_table(specs: Dict[str, HardwareSpec]) -> str:
    """Create a text table comparing ridge points."""
    lines = [
        "\n" + "="*90,
        "  RIDGE POINT COMPARISON ACROSS CIM ARCHITECTURES",
        "="*90,
        "",
        f"{'Architecture':<30} {'Peak (TFLOPS)':>15} {'BW (GB/s)':>12} {'Ridge Point':>15}",
        f"{'':30} {'':>15} {'':>12} {'(MACs/Byte)':>15}",
        "-"*90,
    ]

    # Sort by ridge point
    sorted_specs = sorted(specs.items(), key=lambda x: x[1].ridge_point)

    for arch, hw in sorted_specs:
        lines.append(
            f"{hw.name:<30} {hw.peak_gflops/1000:>15.1f} "
            f"{hw.memory_bandwidth_gb_s:>12.0f} {hw.ridge_point:>15.1f}"
        )

    lines.extend([
        "-"*90,
        "",
        "Note: Ridge Point = Peak_GFLOPS / (2 * Memory_Bandwidth)",
        "      Lower ridge point → More likely to be compute-bound",
        "      Higher ridge point → More likely to be memory-bound",
    ])

    return "\n".join(lines)


def plot_hardware_specs_comparison(specs: Dict[str, HardwareSpec], save_path: str = None) -> go.Figure:
    """Create a multi-panel comparison of hardware specs using Plotly."""

    sorted_items = sorted(specs.items(), key=lambda x: x[1].ridge_point)
    names = [hw.name for _, hw in sorted_items]
    ridge_points = [hw.ridge_point for _, hw in sorted_items]
    peak_tflops = [hw.peak_gflops / 1000 for _, hw in sorted_items]
    bandwidths = [hw.memory_bandwidth_gb_s for _, hw in sorted_items]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Ridge Point (MACs/Byte)",
            "Peak Performance (TFLOPS)",
            "Memory Bandwidth (GB/s)",
            "Peak vs Bandwidth Relationship"
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    colors = px.colors.qualitative.Set2[:len(names)]

    # Ridge Point bar chart
    fig.add_trace(
        go.Bar(x=names, y=ridge_points, marker_color=colors, name="Ridge Point",
               text=[f"{v:.0f}" for v in ridge_points], textposition='outside'),
        row=1, col=1
    )

    # Peak Performance bar chart
    fig.add_trace(
        go.Bar(x=names, y=peak_tflops, marker_color=colors, name="Peak TFLOPS",
               text=[f"{v:.0f}" for v in peak_tflops], textposition='outside'),
        row=1, col=2
    )

    # Bandwidth bar chart
    fig.add_trace(
        go.Bar(x=names, y=bandwidths, marker_color=colors, name="Bandwidth",
               text=[f"{v:.0f}" for v in bandwidths], textposition='outside'),
        row=2, col=1
    )

    # Scatter plot: Peak vs Bandwidth with ridge point as color
    fig.add_trace(
        go.Scatter(
            x=bandwidths, y=peak_tflops,
            mode='markers+text',
            marker=dict(size=20, color=ridge_points, colorscale='Viridis',
                       showscale=True, colorbar=dict(title="Ridge Point")),
            text=[n.split()[0] for n in names],  # Short names
            textposition='top center',
            name="Architectures"
        ),
        row=2, col=2
    )

    fig.update_xaxes(tickangle=-45, row=1, col=1)
    fig.update_xaxes(tickangle=-45, row=1, col=2)
    fig.update_xaxes(tickangle=-45, row=2, col=1)
    fig.update_xaxes(title_text="Bandwidth (GB/s)", row=2, col=2)
    fig.update_yaxes(title_text="Peak (TFLOPS)", row=2, col=2)

    fig.update_layout(
        title={
            'text': "CiM Architecture Specifications Comparison",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        template="plotly_white",
        height=900,
        width=1200,
        showlegend=False,
    )

    if save_path:
        fig.write_html(save_path)
        svg_path = save_path.replace('.html', '.svg')
        fig.write_image(svg_path)
        print(f"Saved hardware specs comparison to {save_path} and {svg_path}")

    return fig


def plot_rooflines_comparison(
    specs: Dict[str, HardwareSpec],
    layer_profiles: List[LayerProfile] = None,
    model_name: str = None,
    save_path: str = None
) -> go.Figure:
    """Create interactive roofline comparison plot using Plotly."""

    fig = go.Figure()

    # OI range (MACs/Byte)
    oi_range = np.logspace(-1, 3, 500)

    # Color palette
    colors = px.colors.qualitative.Set1

    # Plot roofline for each architecture
    for i, (arch, hw) in enumerate(sorted(specs.items(), key=lambda x: x[1].ridge_point)):
        color = colors[i % len(colors)]

        # Memory roof: GFLOPS = 2 * OI * BW
        memory_roof = 2 * oi_range * hw.memory_bandwidth_gb_s
        compute_roof = np.full_like(oi_range, hw.peak_gflops)
        roofline = np.minimum(memory_roof, compute_roof)

        # Add roofline trace
        fig.add_trace(go.Scatter(
            x=oi_range, y=roofline,
            mode='lines',
            name=f"{hw.name}",
            line=dict(color=color, width=2),
            hovertemplate=(
                f"<b>{hw.name}</b><br>"
                "OI: %{x:.1f} MACs/Byte<br>"
                "Performance: %{y:.0f} GFLOPS<br>"
                f"Ridge Point: {hw.ridge_point:.1f} MACs/B<br>"
                "<extra></extra>"
            )
        ))

        # Add ridge point marker
        fig.add_trace(go.Scatter(
            x=[hw.ridge_point],
            y=[hw.peak_gflops],
            mode='markers',
            marker=dict(symbol='diamond', size=12, color=color, line=dict(width=2, color='black')),
            name=f"Ridge: {hw.ridge_point:.0f}",
            showlegend=False,
            hovertemplate=(
                f"<b>{hw.name} Ridge Point</b><br>"
                f"OI: {hw.ridge_point:.1f} MACs/Byte<br>"
                f"Peak: {hw.peak_gflops/1000:.0f} TFLOPS<br>"
                "<extra></extra>"
            )
        ))

    # Add layer profiles if provided
    if layer_profiles:
        # Group by operator category
        op_categories = defaultdict(list)
        for p in layer_profiles:
            op_categories[p.operator_category].append(p)

        op_colors = px.colors.qualitative.Dark24
        for i, (cat, profiles) in enumerate(op_categories.items()):
            ois = [p.operational_intensity for p in profiles]
            # Use max roofline performance at each OI
            max_hw = max(specs.values(), key=lambda h: h.peak_gflops)
            perfs = [min(max_hw.peak_gflops, 2 * oi * max_hw.memory_bandwidth_gb_s) for oi in ois]

            fig.add_trace(go.Scatter(
                x=ois, y=perfs,
                mode='markers',
                marker=dict(size=8, color=op_colors[i % len(op_colors)], opacity=0.6),
                name=f"{cat} ({len(profiles)} layers)",
                hovertemplate=(
                    f"<b>{cat}</b><br>"
                    "OI: %{x:.1f} MACs/Byte<br>"
                    "Roofline Perf: %{y:.0f} GFLOPS<br>"
                    "<extra></extra>"
                )
            ))

    fig.update_layout(
        title={
            'text': f"Roofline Comparison: {len(specs)} CiM Architectures" +
                   (f" with {model_name} Layers" if model_name else ""),
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title="Operational Intensity (MACs/Byte)",
        yaxis_title="Performance (GFLOPS)",
        xaxis_type="log",
        yaxis_type="log",
        template="plotly_white",
        height=700,
        width=1100,
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        hovermode='closest',
    )

    # Add ridge point explanation
    fig.add_annotation(
        text="◆ = Ridge Point (where memory and compute roofs meet)",
        xref="paper", yref="paper",
        x=0.02, y=0.02,
        showarrow=False,
        font=dict(size=11),
        bgcolor="white",
        borderpad=4
    )

    if save_path:
        fig.write_html(save_path)
        svg_path = save_path.replace('.html', '.svg')
        fig.write_image(svg_path)
        print(f"Saved roofline comparison to {save_path} and {svg_path}")

    return fig


def plot_ridge_point_scatter(specs: Dict[str, HardwareSpec], save_path: str = None) -> go.Figure:
    """Create a scatter plot showing Peak vs Ridge Point relationship."""

    names = [hw.name for hw in specs.values()]
    ridge_points = [hw.ridge_point for hw in specs.values()]
    peak_tflops = [hw.peak_gflops / 1000 for hw in specs.values()]
    bandwidths = [hw.memory_bandwidth_gb_s for hw in specs.values()]

    fig = go.Figure()

    # Scatter plot with bandwidth as size
    fig.add_trace(go.Scatter(
        x=ridge_points,
        y=peak_tflops,
        mode='markers+text',
        marker=dict(
            size=[b/3 for b in bandwidths],  # Scale bandwidth for bubble size
            color=ridge_points,
            colorscale='RdYlGn_r',
            showscale=True,
            colorbar=dict(title="Ridge Point<br>(MACs/Byte)"),
            line=dict(width=2, color='black')
        ),
        text=names,
        textposition='top center',
        textfont=dict(size=10),
        customdata=list(zip(names, bandwidths)),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Ridge Point: %{x:.1f} MACs/Byte<br>"
            "Peak: %{y:.0f} TFLOPS<br>"
            "Bandwidth: %{customdata[1]:.0f} GB/s<br>"
            "<extra></extra>"
        )
    ))

    fig.update_layout(
        title={
            'text': "Peak Performance vs Ridge Point<br><sup>Bubble size = Memory Bandwidth</sup>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title="Ridge Point (MACs/Byte)",
        yaxis_title="Peak Performance (TFLOPS)",
        template="plotly_white",
        height=600,
        width=900,
    )

    if save_path:
        fig.write_html(save_path)
        svg_path = save_path.replace('.html', '.svg')
        fig.write_image(svg_path)
        print(f"Saved ridge point scatter to {save_path} and {svg_path}")

    return fig


def run_architecture_comparison(
    model_name: str = "gpt2_medium",
    num_layers: int = 20,
    output_dir: str = None,
    architectures: List[str] = None
):
    """Run comprehensive architecture comparison with Plotly visualizations."""

    if output_dir is None:
        output_dir = os.path.join(THIS_SCRIPT_DIR, "..", "outputs", "architecture_comparison")
    os.makedirs(output_dir, exist_ok=True)

    # Get hardware specs
    if architectures is None:
        architectures = CIM_ARCHITECTURES

    specs = {arch: get_hardware_specs(arch) for arch in architectures}

    # Print comparison table
    print(create_ridge_point_comparison_table(specs))

    # Get layer profiles for the model
    print(f"\nProfiling {model_name} layers...")
    try:
        layer_files = get_model_layer_files(model_name)[:num_layers]
        layer_profiles = [calculate_layer_profile(f, i) for i, f in enumerate(layer_files)]
        print(f"  Profiled {len(layer_profiles)} layers")
    except Exception as e:
        print(f"  Warning: Could not load {model_name} layers: {e}")
        layer_profiles = None

    # Generate Plotly visualizations
    print("\nGenerating Plotly visualizations...")

    # 1. Hardware Specs Comparison (multi-panel)
    fig1 = plot_hardware_specs_comparison(
        specs,
        save_path=os.path.join(output_dir, "hardware_specs_comparison.html")
    )

    # 2. Roofline Comparison
    fig2 = plot_rooflines_comparison(
        specs,
        layer_profiles=layer_profiles,
        model_name=model_name if layer_profiles else None,
        save_path=os.path.join(output_dir, "roofline_comparison.html")
    )

    # 3. Ridge Point Scatter (Peak vs Ridge)
    fig3 = plot_ridge_point_scatter(
        specs,
        save_path=os.path.join(output_dir, "ridge_point_scatter.html")
    )

    print(f"\n{'='*70}")
    print(f"  All visualizations saved to: {output_dir}")
    print(f"{'='*70}")
    print("\nGenerated files:")
    print("  - hardware_specs_comparison.html/svg : Multi-panel specs comparison")
    print("  - roofline_comparison.html/svg       : Interactive roofline overlay")
    print("  - ridge_point_scatter.html/svg       : Peak vs Ridge bubble chart")

    return specs, layer_profiles


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare Ridge Points Across CiM Architectures with Plotly"
    )
    parser.add_argument("--model", type=str, default="gpt2_medium",
                        help="Model to use for layer profiling")
    parser.add_argument("--num-layers", type=int, default=20,
                        help="Number of layers to profile")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--architectures", type=str, nargs="+", default=None,
                        help="Specific architectures to compare")

    args = parser.parse_args()

    run_architecture_comparison(
        model_name=args.model,
        num_layers=args.num_layers,
        output_dir=args.output_dir,
        architectures=args.architectures
    )
