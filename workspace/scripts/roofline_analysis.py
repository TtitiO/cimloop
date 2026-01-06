"""
Roofline Model Analysis for DNN Layers on CiM (Compute-in-Memory) Hardware

This script provides comprehensive profiling and visualization of DNN model layers
(GPT-2, ResNet, VGG, MobileNet, Vision Transformer, etc.) on various CiM hardware
architectures, calculating operational intensity and plotting roofline models.

=== KEY CONCEPTS ===
- Operational Intensity (OI): MACs per Byte of data movement (algorithm property)
- Hardware Performance: Measured in OP/s or MAC/s (configurable: 1 MAC = 2 OPs default)
- Roofline Model: Visualizes compute vs memory bottlenecks for each operator
- Weight Stationary: CiM-specific assumption where weights are stored in memory
  cells and not counted in data movement during inference
- Multi-level OI: Support for OI at different memory hierarchy levels (DRAM, GBUF, CIM port)

=== FEATURES ===
1. Unit Normalization & Configuration:
   - Explicit configuration for OP/s vs MAC/s (1 MAC = 2 OPs or 1 MAC = 1 unit)
   - Configurable OI byte boundary (DRAM, global buffer, CIM port, PE SRAM)
   - Multi-level OI support for different memory hierarchy levels
   - Weight stationary mode for CiM architectures

2. Self-Consistency Validation:
   - Recompute Sim.GOPS from GMACs, cycles, frequency
   - Back-compute required bandwidth from OI and achieved performance
   - Detect mismatches and inconsistencies in reported metrics
   - Identify memory-bound vs compute-bound layers

3. Layer Profiling:
   - Parse layer dimensions (C, M, P) from YAML workload files
   - Calculate MACs, data movement, and operational intensity per layer
   - Categorize operators (Linear/FC, Convolution, Attention, etc.)

4. Hardware Support:
   - Multiple CiM architectures: RRAM-based (ISAAC, RAELLA, Wan et al.)
     and SRAM-based (Colonnade, Jia, Sinangil, Wang)
   - Configurable peak GFLOPS and memory bandwidth specifications
   - Hardware specs derived from published papers and simulations

5. Visualization (Interactive Plotly):
   - Roofline plots with memory-bound/compute-bound regions
   - Multi-level roofline plots showing different memory hierarchy levels
   - Layer analysis charts (OI distribution, compute, data movement)
   - Hardware comparison across multiple architectures
   - Export to HTML (interactive) and PNG (static)

6. Simulation Integration:
   - Optional Timeloop simulations for actual hardware performance
   - Parallel execution support for faster analysis
   - Comparison of theoretical vs simulated performance

7. CLI Interface:
   - `analyze`: Run roofline analysis for a model on specific hardware
   - `compare`: Compare rooflines across multiple hardware architectures
   - `validate`: Generate validation report for consistency checks
   - `list`: Show available models, macros, or output files
   - `clean`: Remove generated output files

=== USAGE EXAMPLES ===
    # Basic analysis
    python roofline_analysis.py analyze --model resnet18 --macro basic_analog

    # With simulation
    python roofline_analysis.py analyze --model gpt2_medium --simulate

    # With validation report
    python roofline_analysis.py analyze --model gpt2_medium --simulate --validate

    # Multi-level OI analysis
    python roofline_analysis.py analyze --model gpt2_medium --multilevel-oi

    # Compare architectures
    python roofline_analysis.py compare --model gpt2_medium

    # List and clean
    python roofline_analysis.py list models
    python roofline_analysis.py clean --model gpt2_medium --dry-run

    # colonnade architecture analysis (system auto-detected from macro name)
    docker exec cimloop-tutorial-1 python3 scripts/roofline_analysis.py analyze --model gpt2_medium --macro colonnade_jssc_2021 --simulate

=== OUTPUT ===
- Summary tables (.txt) with layer-by-layer metrics
- Validation reports (.txt) with consistency checks
- Interactive roofline plots (.html)
- Static roofline plots (.png)
- Layer analysis charts with OI, MACs, and data movement breakdowns

=== VALIDATION FRAMEWORK ===
Step 1: Normalize units and definitions (one-time setup)
  - Lock definition of "ops" (OP/s vs MAC/s) via RooflineConfig
  - Define byte boundary for OI (DRAM, GBUF, CIM port, etc.)
  - Enable multi-level OI for fusion decisions

Step 2: Self-consistency checks
  2.1: Recompute Sim.GOPS from GMACs, cycles, frequency
  2.2: Back-compute required bandwidth from OI and achieved performance
  - Verify reported metrics match recomputed values
  - Identify memory-bound vs compute-bound layers
  - Check bandwidth utilization and compatibility
"""

import os
import sys
import re
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from typing import Any
from collections import defaultdict
from tqdm import tqdm
import joblib

THIS_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(THIS_SCRIPT_DIR)

from utils import get_spec, run_mapper, path_from_model_dir, parallel_test, delayed
from tl_output_parsing import MacroOutputStats, MacroOutputStatsList


# ============================================================================
# CONFIGURATION: Units and Definitions
# ============================================================================
@dataclass
class RooflineConfig:
    """
    Configuration for roofline analysis to ensure consistent unit definitions.

    This configuration enforces clear definitions for performance metrics and
    operational intensity calculations across the entire analysis pipeline.
    """

    # === Performance Unit Configuration ===
    # Decide whether to use OP/s (where 1 MAC = 2 OPs) or MAC/s (where 1 MAC = 1 unit)
    use_ops_not_macs: bool = True  # True: Use OP/s with MAC=2 OPs, False: Use MAC/s

    # === Operational Intensity Byte Boundary ===
    # Define what "Byte" means in OI calculations for each memory hierarchy level
    oi_byte_boundary: str = "cim_port"  # Options: "dram", "global_buffer", "cim_port", "pe_sram"

    # === Multi-level OI Support ===
    # Enable calculation of OI at multiple memory hierarchy levels
    enable_multilevel_oi: bool = False

    # === Weight Stationary Assumption ===
    # For CiM: weights stored in memory cells, not counted in data movement
    weight_stationary: bool = True

    @property
    def ops_per_mac(self) -> int:
        """Number of operations per MAC (1 MAC = 1 multiply + 1 add)"""
        return 2 if self.use_ops_not_macs else 1

    @property
    def performance_unit(self) -> str:
        """String representation of performance unit"""
        return "OP/s" if self.use_ops_not_macs else "MAC/s"

    @property
    def oi_unit(self) -> str:
        """String representation of OI unit"""
        mac_unit = "OP" if self.use_ops_not_macs else "MAC"
        return f"{mac_unit}/Byte"

    def to_performance_unit(self, macs: float) -> float:
        """Convert MACs to configured performance unit (OPs or MACs)"""
        return macs * self.ops_per_mac

    def get_oi_description(self) -> str:
        """Get detailed description of OI calculation"""
        boundary_desc = {
            "dram": "DRAM bytes (system bottleneck level)",
            "global_buffer": "Global buffer bytes (chip-level)",
            "cim_port": "CIM macro port bytes (input/output only)",
            "pe_sram": "PE local SRAM bytes"
        }

        ws_desc = " (weights excluded - weight stationary)" if self.weight_stationary else " (weights included)"

        return (
            f"OI = {self.oi_unit} at {boundary_desc.get(self.oi_byte_boundary, 'custom')}"
            f"{ws_desc}"
        )


# Default configuration - can be overridden per analysis
DEFAULT_ROOFLINE_CONFIG = RooflineConfig(
    use_ops_not_macs=True,           # Use OP/s where 1 MAC = 2 OPs
    oi_byte_boundary="cim_port",     # Measure data movement at CIM macro ports
    enable_multilevel_oi=False,      # Single-level OI for now
    weight_stationary=True           # CiM assumption: weights in memory cells
)


# Operator type categorization for transformers and CNNs
OPERATOR_CATEGORIES = {
    # Transformer operators
    "Conv1D": "Linear/FC",
    "GPT2Attention": "Attention",
    "MatMul": "MatMul",
    "Linear": "Linear/FC",
    "Embedding": "Embedding",
    "LayerNorm": "Normalization",
    "Softmax": "Softmax",
    # CNN operators
    "Conv2D": "Convolution",
    "Conv": "Convolution",
    "DepthwiseConv": "Depthwise Conv",
    "Pooling": "Pooling",
    "BatchNorm": "Normalization",
    "FC": "Linear/FC",
    "Dense": "Linear/FC",
}


@dataclass
class LayerProfile:
    """Profile data for a single layer."""
    layer_idx: int
    layer_id: str              # e.g., "000", "001"
    operator_type: str         # e.g., "Conv1D", "GPT2Attention"
    operator_category: str     # e.g., "Linear/FC", "Attention"
    model_name: str            # e.g., "gpt2_medium", "resnet18"

    # Problem dimensions
    input_channels: int        # C
    output_channels: int       # M
    sequence_length: int       # P (or spatial dimension)

    # Compute metrics (Algorithm)
    total_macs: int            # Total MAC operations (algorithm metric)

    # Data movement metrics (in bytes)
    input_bytes: int
    weight_bytes: int
    output_bytes: int
    total_bytes: int           # Primary data movement (at configured boundary)

    # Operational Intensity: MACs per Byte (algorithm property)
    operational_intensity: float  # MACs / Byte (at configured boundary)

    # Multi-level OI support (for different memory hierarchy levels)
    oi_dram: Optional[float] = None           # OI at DRAM level (system bottleneck)
    oi_global_buffer: Optional[float] = None  # OI at global buffer/cache level
    oi_cim_port: Optional[float] = None       # OI at CIM macro input/output ports
    oi_pe_sram: Optional[float] = None        # OI at PE local SRAM level

    # Bytes at each level (for multi-level analysis)
    bytes_dram: Optional[int] = None
    bytes_global_buffer: Optional[int] = None
    bytes_cim_port: Optional[int] = None
    bytes_pe_sram: Optional[int] = None

    # Hardware execution metrics (if simulated)
    cycles: Optional[int] = None
    energy_pj: Optional[float] = None
    achieved_gflops: Optional[float] = None  # Hardware performance in GFLOPS

    @property
    def total_flops(self) -> int:
        """Hardware FLOPS = 2 * MACs (multiply + add)"""
        return 2 * self.total_macs

    @property
    def display_name(self) -> str:
        """Human-readable layer name"""
        return f"{self.operator_type}_{self.layer_id}"

    def get_oi_at_level(self, level: str) -> Optional[float]:
        """Get OI at specific memory hierarchy level."""
        level_map = {
            "dram": self.oi_dram,
            "global_buffer": self.oi_global_buffer,
            "cim_port": self.oi_cim_port,
            "pe_sram": self.oi_pe_sram
        }
        return level_map.get(level, self.operational_intensity)

    def get_bytes_at_level(self, level: str) -> Optional[int]:
        """Get bytes moved at specific memory hierarchy level."""
        level_map = {
            "dram": self.bytes_dram,
            "global_buffer": self.bytes_global_buffer,
            "cim_port": self.bytes_cim_port,
            "pe_sram": self.bytes_pe_sram
        }
        return level_map.get(level, self.total_bytes)


@dataclass
class HardwareSpec:
    """Hardware specification for roofline model."""
    name: str
    peak_gops: float              # Peak GOPS (integer operations for CIM)
    memory_bandwidth_gb_s: float  # GB/s

    # Detailed macro specifications (optional)
    array_rows: Optional[int] = None          # Number of rows in CIM array
    array_cols: Optional[int] = None          # Number of columns in CIM array
    technology_nm: Optional[int] = None       # Technology node in nm
    voltage: Optional[float] = None           # Operating voltage
    weight_bits: Optional[int] = None         # Weight precision
    input_bits: Optional[int] = None          # Input precision
    output_bits: Optional[int] = None         # Output precision
    adc_resolution: Optional[int] = None      # ADC resolution in bits
    cell_type: Optional[str] = None           # Memory cell type (SRAM, RRAM, etc.)
    frequency_mhz: Optional[float] = None     # Operating frequency in MHz

    @property
    def peak_ops_per_sec(self) -> float:
        return self.peak_gops * 1e9

    @property
    def memory_bandwidth_bytes_per_sec(self) -> float:
        return self.memory_bandwidth_gb_s * 1e9

    @property
    def array_size_str(self) -> str:
        """Human-readable array size string."""
        if self.array_rows and self.array_cols:
            return f"{self.array_rows}×{self.array_cols}"
        return "N/A"

    @property
    def precision_str(self) -> str:
        """Human-readable precision string."""
        if self.weight_bits and self.input_bits and self.output_bits:
            return f"W{self.weight_bits}/I{self.input_bits}/O{self.output_bits}"
        return "N/A"


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

@dataclass
class LayerValidation:
    """Validation metrics for a single layer to check self-consistency."""
    layer_idx: int
    layer_id: str

    # Original metrics
    reported_gmacs: float           # GMACs reported
    reported_sim_gops: Optional[float]    # Simulated GOPS reported
    cycles: Optional[int]
    frequency_mhz: Optional[float]
    operational_intensity: float

    # Recomputed metrics (for validation)
    recomputed_sim_gops: Optional[float] = None    # From GMACs, cycles, freq
    recomputed_sim_gmacs_per_sec: Optional[float] = None

    # Bandwidth analysis
    required_bandwidth_gb_s: Optional[float] = None  # BW needed to achieve perf
    is_memory_bound: Optional[bool] = None

    # Consistency flags
    gops_mismatch: Optional[bool] = None
    gops_error_percent: Optional[float] = None

    @property
    def is_valid_for_analysis(self) -> bool:
        """Check if this layer has enough data for validation"""
        return (self.cycles is not None and
                self.cycles > 0 and
                self.frequency_mhz is not None and
                self.frequency_mhz > 0)


def validate_layer_consistency(
    profile: LayerProfile,
    hardware: HardwareSpec,
    config: RooflineConfig = DEFAULT_ROOFLINE_CONFIG,
    tolerance_percent: float = 5.0
) -> LayerValidation:
    """
    Perform self-consistency checks on a single layer.

    Step 2.1: Recompute Sim.GOPS from GMACs, cycles, frequency
    Step 2.2: Back-compute implied bandwidth from OI and achieved perf

    Args:
        profile: Layer profile with simulation data
        hardware: Hardware specification
        config: Roofline configuration
        tolerance_percent: Acceptable error percentage for consistency checks
    """
    validation = LayerValidation(
        layer_idx=profile.layer_idx,
        layer_id=profile.layer_id,
        reported_gmacs=profile.total_macs / 1e9,
        reported_sim_gops=profile.achieved_gflops,
        cycles=profile.cycles,
        frequency_mhz=hardware.frequency_mhz,
        operational_intensity=profile.operational_intensity
    )

    # Skip validation if no simulation data
    if not validation.is_valid_for_analysis:
        return validation

    # === Step 2.1: Recompute Sim.GOPS from GMACs, cycles, frequency ===
    if profile.cycles and hardware.frequency_mhz:
        # Time = cycles / frequency
        time_sec = profile.cycles / (hardware.frequency_mhz * 1e6)

        # MAC/s = GMACs / time
        gmacs_per_sec = validation.reported_gmacs / time_sec
        validation.recomputed_sim_gmacs_per_sec = gmacs_per_sec

        # GOPS = MAC/s * ops_per_mac (typically 2 for 1 MAC = 2 OPs)
        validation.recomputed_sim_gops = gmacs_per_sec * config.ops_per_mac

        # Check consistency with reported Sim.GOPS
        if profile.achieved_gflops is not None:
            error = abs(validation.recomputed_sim_gops - profile.achieved_gflops)
            validation.gops_error_percent = (error / profile.achieved_gflops) * 100
            validation.gops_mismatch = validation.gops_error_percent > tolerance_percent

    # === Step 2.2: Back-compute required bandwidth ===
    if profile.achieved_gflops is not None:
        # Convert GOPS to GMAC/s if needed
        if config.use_ops_not_macs:
            achieved_gmacs_per_sec = profile.achieved_gflops / config.ops_per_mac
        else:
            achieved_gmacs_per_sec = profile.achieved_gflops

        # Required BW (GB/s) = MAC/s / OI
        # Because: OI = MAC/Byte, so Byte/s = MAC/s / OI, and GB/s = Byte/s / 1e9
        if profile.operational_intensity > 0:
            validation.required_bandwidth_gb_s = achieved_gmacs_per_sec / profile.operational_intensity

            # Determine if memory bound (required BW > available BW)
            validation.is_memory_bound = (
                validation.required_bandwidth_gb_s >= hardware.memory_bandwidth_gb_s * 0.9
            )

    return validation


def create_validation_report(
    layer_profiles: List[LayerProfile],
    hardware: HardwareSpec,
    config: RooflineConfig = DEFAULT_ROOFLINE_CONFIG,
    model_name: str = None
) -> str:
    """
    Generate a comprehensive validation report checking self-consistency.

    This implements:
    - Step 2.1: Recompute Sim.GOPS from GMACs, cycles, frequency
    - Step 2.2: Back-compute implied bandwidth from OI and achieved perf
    """
    if model_name is None and layer_profiles:
        model_name = layer_profiles[0].model_name

    lines = [
        f"\n{'='*120}",
        f"  VALIDATION REPORT: {model_name.upper()} on {hardware.name}",
        f"{'='*120}",
        f"",
        f"=== CONFIGURATION ===",
        f"Performance Unit:    {config.performance_unit} (1 MAC = {config.ops_per_mac} {config.performance_unit.split('/')[0]})",
        f"OI Definition:       {config.oi_unit}",
        f"OI Byte Boundary:    {config.oi_byte_boundary}",
        f"Weight Stationary:   {config.weight_stationary}",
        f"{config.get_oi_description()}",
        f"",
        f"=== HARDWARE SPECS ===",
        f"Peak Performance:    {hardware.peak_gops:.0f} G{config.performance_unit}",
        f"Memory Bandwidth:    {hardware.memory_bandwidth_gb_s:.1f} GB/s",
        f"Frequency:           {hardware.frequency_mhz:.0f} MHz" if hardware.frequency_mhz else "Frequency:           N/A",
        f"Ridge Point OI:      {hardware.peak_gops / (config.ops_per_mac * hardware.memory_bandwidth_gb_s):.2f} {config.oi_unit}",
        f"",
        f"{'='*120}",
        f"  STEP 2.1: Recompute Sim.GOPS from GMACs, cycles, frequency",
        f"{'='*120}",
        f"",
    ]

    # Validate each layer
    validations = [validate_layer_consistency(p, hardware, config) for p in layer_profiles]

    # Filter valid layers (those with simulation data)
    valid_layers = [v for v in validations if v.is_valid_for_analysis]

    if not valid_layers:
        lines.append("No simulation data available for validation.")
        lines.append("Run with --simulate flag to generate validation data.")
        return "\n".join(lines)

    # Table header
    header = (
        f"{'Layer':<12} {'GMACs':>8} {'Cycles':>12} {'Reported':>12} {'Recomputed':>12} "
        f"{'Error':>8} {'Status':<8}"
    )
    lines.append(header)
    lines.append(f"{' '*12} {' '*8} {' '*12} {'G' + config.performance_unit:>12} {'G' + config.performance_unit:>12} {'(%)':>8}")
    lines.append("-" * 120)

    mismatch_count = 0
    for v in valid_layers:
        if v.recomputed_sim_gops is not None:
            status = "❌ MISMATCH" if v.gops_mismatch else "✓ OK"
            if v.gops_mismatch:
                mismatch_count += 1

            lines.append(
                f"{v.layer_id:<12} {v.reported_gmacs:>8.2f} {v.cycles:>12,} "
                f"{v.reported_sim_gops:>12.2f} {v.recomputed_sim_gops:>12.2f} "
                f"{v.gops_error_percent:>8.2f} {status:<8}"
            )

    lines.extend([
        "-" * 120,
        f"",
        f"Layers validated: {len(valid_layers)}/{len(layer_profiles)}",
        f"Mismatches found: {mismatch_count}",
        f"",
    ])

    # === Step 2.2: Bandwidth Analysis ===
    lines.extend([
        f"{'='*120}",
        f"  STEP 2.2: Back-compute Required Bandwidth from OI and Achieved Performance",
        f"{'='*120}",
        f"",
        f"Formula: Required_BW (GB/s) = Achieved_GMAC/s / OI",
        f"         where OI is in MACs/Byte",
        f"",
    ])

    header2 = (
        f"{'Layer':<12} {'OI':>8} {'Achieved':>12} {'Required':>12} {'Available':>12} "
        f"{'Utilization':>12} {'Bottleneck':<15}"
    )
    lines.append(header2)
    lines.append(
        f"{' '*12} {'(MAC/B)':>8} {'GMAC/s':>12} {'BW (GB/s)':>12} {'BW (GB/s)':>12} "
        f"{'(%)':>12}"
    )
    lines.append("-" * 120)

    memory_bound_count = 0
    compute_bound_count = 0

    for v in valid_layers:
        if v.required_bandwidth_gb_s is not None:
            utilization = (v.required_bandwidth_gb_s / hardware.memory_bandwidth_gb_s) * 100

            # Determine bottleneck
            if v.is_memory_bound:
                bottleneck = "Memory-Bound"
                memory_bound_count += 1
            else:
                bottleneck = "Compute-Bound"
                compute_bound_count += 1

            # Get achieved GMAC/s
            achieved_gmacs = v.recomputed_sim_gmacs_per_sec or (v.reported_sim_gops / config.ops_per_mac if v.reported_sim_gops else 0)

            lines.append(
                f"{v.layer_id:<12} {v.operational_intensity:>8.2f} {achieved_gmacs:>12.2f} "
                f"{v.required_bandwidth_gb_s:>12.2f} {hardware.memory_bandwidth_gb_s:>12.2f} "
                f"{utilization:>12.1f} {bottleneck:<15}"
            )

    lines.extend([
        "-" * 120,
        f"",
        f"Memory-Bound Layers: {memory_bound_count} (performance limited by bandwidth)",
        f"Compute-Bound Layers: {compute_bound_count} (performance limited by peak compute)",
        f"",
    ])

    # Analysis summary
    if valid_layers:
        avg_bw_utilization = np.mean([
            v.required_bandwidth_gb_s / hardware.memory_bandwidth_gb_s * 100
            for v in valid_layers if v.required_bandwidth_gb_s is not None
        ])

        max_bw_required = max([
            v.required_bandwidth_gb_s
            for v in valid_layers if v.required_bandwidth_gb_s is not None
        ])

        lines.extend([
            f"{'='*120}",
            f"  BANDWIDTH ANALYSIS SUMMARY",
            f"{'='*120}",
            f"Average BW Utilization:  {avg_bw_utilization:.1f}%",
            f"Peak BW Required:        {max_bw_required:.2f} GB/s",
            f"Hardware BW Available:   {hardware.memory_bandwidth_gb_s:.2f} GB/s",
            f"",
        ])

        if avg_bw_utilization > 100:
            lines.append(
                "⚠️  WARNING: Average bandwidth utilization exceeds 100%!"
            )
            lines.append(
                "    This suggests either:"
            )
            lines.append(
                "    1) The hardware bandwidth spec is underestimated"
            )
            lines.append(
                "    2) The OI calculation boundary doesn't match the bandwidth measurement level"
            )
            lines.append(
                "    3) Simulation includes optimizations (data reuse, caching) not captured in OI"
            )
            lines.append("")

    lines.extend([
        f"{'='*120}",
        f"  RECOMMENDATIONS",
        f"{'='*120}",
        f"",
        f"1. If mismatches found in Step 2.1:",
        f"   - Verify that Sim.GOPS is correctly calculated as (GMACs/time) * {config.ops_per_mac}",
        f"   - Check for rounding errors or hidden unit conversions",
        f"   - Ensure frequency is consistent across all calculations",
        f"",
        f"2. If bandwidth utilization > 100% in Step 2.2:",
        f"   - Verify OI byte boundary matches bandwidth measurement level",
        f"   - Current OI boundary: {config.oi_byte_boundary}",
        f"   - Consider multi-level OI analysis (DRAM vs. on-chip)",
        f"   - Check if data reuse/caching affects actual data movement",
        f"",
        f"3. For fusion decisions:",
        f"   - Analyze OI at both system level (DRAM) and local level (CIM port)",
        f"   - Use multi-level roofline analysis (enable_multilevel_oi=True)",
        f"",
    ])

    return "\n".join(lines)


def get_model_layer_files(model_name: str = "gpt2_medium", workload_dir: str = None) -> List[str]:
    """Get all layer YAML files for a model sorted by layer index."""
    if workload_dir is None:
        workload_dir = path_from_model_dir("workloads", model_name)

    if not os.path.exists(workload_dir):
        raise FileNotFoundError(f"Model workload directory not found: {workload_dir}")

    layer_files = []
    for f in sorted(os.listdir(workload_dir)):
        if f.endswith('.yaml'):
            layer_files.append(os.path.join(workload_dir, f))
    return layer_files


def get_available_models() -> List[str]:
    """Get list of available models in the workloads directory."""
    workloads_dir = path_from_model_dir("workloads")
    models = []
    for item in os.listdir(workloads_dir):
        item_path = os.path.join(workloads_dir, item)
        if os.path.isdir(item_path) and not item.startswith('_'):
            # Check if it has YAML files
            yaml_files = [f for f in os.listdir(item_path) if f.endswith('.yaml')]
            if yaml_files:
                models.append(item)
    return sorted(models)


def get_output_dir(output_dir: str = None, model_name: str = None, macro: str = None) -> str:
    """Get the default output directory for roofline analysis."""
    if output_dir is None:
        base_dir = os.path.join(THIS_SCRIPT_DIR, "..", "outputs", "roofline_outputs")
        if model_name and macro:
            output_dir = os.path.join(base_dir, model_name, macro)
        elif model_name:
            output_dir = os.path.join(base_dir, model_name)
        else:
            output_dir = base_dir
    return output_dir


def clean_model_outputs(model_name: str = None, output_dir: str = None, dry_run: bool = False) -> List[str]:
    """
    Clean output files for a specific model or all models.

    Args:
        model_name: Model name to clean (e.g., 'gpt2_medium'). If None, cleans all files.
        output_dir: Output directory path. Uses default if None.
        dry_run: If True, only list files that would be deleted without actually deleting.

    Returns:
        List of deleted (or would-be-deleted) file paths.
    """
    base_dir = get_output_dir(output_dir)

    if not os.path.exists(base_dir):
        print(f"Output directory does not exist: {base_dir}")
        return []

    deleted_files = []

    if model_name is None:
        # Delete all files and directories recursively
        for root, dirs, files in os.walk(base_dir):
            for filename in files:
                filepath = os.path.join(root, filename)
                deleted_files.append(filepath)
                if not dry_run:
                    os.remove(filepath)
                    print(f"  Deleted: {filepath}")
                else:
                    print(f"  Would delete: {filepath}")
    else:
        # Delete all files in the model's directory
        model_dir = os.path.join(base_dir, model_name)
        if os.path.exists(model_dir):
            for root, dirs, files in os.walk(model_dir):
                for filename in files:
                    filepath = os.path.join(root, filename)
                    deleted_files.append(filepath)
                    if not dry_run:
                        os.remove(filepath)
                        print(f"  Deleted: {filepath}")
                    else:
                        print(f"  Would delete: {filepath}")

    return deleted_files


def clean_old_simulations(base_output_dir: str = None, dry_run: bool = False) -> List[str]:
    """
    Clean old Timeloop simulation directories (timestamp-based folders).
    These are folders like "12888.281473838294048" from previous runs.

    Args:
        base_output_dir: Base output directory. If None, uses default workspace outputs.
        dry_run: If True, only list directories that would be deleted.

    Returns:
        List of deleted (or would-be-deleted) directory paths.
    """
    if base_output_dir is None:
        base_output_dir = os.path.join(THIS_SCRIPT_DIR, "..", "outputs")

    base_output_dir = os.path.abspath(base_output_dir)

    if not os.path.exists(base_output_dir):
        print(f"Output directory does not exist: {base_output_dir}")
        return []

    deleted_dirs = []

    # Find timestamp-based directories
    for item in os.listdir(base_output_dir):
        item_path = os.path.join(base_output_dir, item)

        # Check if it's a directory and looks like a timestamp (digits and dots)
        if os.path.isdir(item_path) and item.replace('.', '').replace('_', '').isdigit():
            deleted_dirs.append(item_path)

            if not dry_run:
                import shutil
                shutil.rmtree(item_path)
                print(f"  Deleted directory: {item_path}")
            else:
                print(f"  Would delete directory: {item_path}")

    return deleted_dirs


def list_model_outputs(model_name: str = None, output_dir: str = None) -> Dict[str, Dict[str, List[str]]]:
    """
    List output files grouped by model and hardware architecture.

    Args:
        model_name: Filter by specific model. If None, lists all.
        output_dir: Output directory path.

    Returns:
        Dictionary mapping model names to dictionaries of hardware architectures and their files.
    """
    base_dir = get_output_dir(output_dir)

    if not os.path.exists(base_dir):
        return {}

    # Group files by model and hardware
    model_files = defaultdict(lambda: defaultdict(list))

    # Walk through model directories
    for model in os.listdir(base_dir):
        model_path = os.path.join(base_dir, model)
        if not os.path.isdir(model_path):
            continue

        # Skip timestamp-based simulation directories (e.g., "12888.281473838294048")
        if model.replace('.', '').replace('_', '').isdigit():
            continue

        # If filtering by model, skip others
        if model_name and model != model_name:
            continue

        # Walk through hardware architecture directories
        for hardware in os.listdir(model_path):
            hardware_path = os.path.join(model_path, hardware)
            if not os.path.isdir(hardware_path):
                continue

            # Count layer directories and regular files separately
            layer_count = 0
            files = []
            for item in sorted(os.listdir(hardware_path)):
                item_path = os.path.join(hardware_path, item)
                if os.path.isfile(item_path):
                    files.append(item)
                elif os.path.isdir(item_path) and item.isdigit():
                    layer_count += 1

            # Add summary of layer directories
            if layer_count > 0:
                files.insert(0, f"[{layer_count} layer directories: 0/ through {layer_count-1}/]")

            model_files[model][hardware] = files

    # Filter by model if specified
    if model_name:
        return {model_name: dict(model_files.get(model_name, {}))}

    return {k: dict(v) for k, v in model_files.items()}


def parse_layer_info(yaml_path: str) -> Dict:
    """Parse layer dimensions and metadata from YAML file."""
    with open(yaml_path, 'r') as f:
        content = f.read()

    result = {}

    # Extract instance dimensions: {C: xxx, M: xxx, P: xxx}
    match = re.search(r'instance:\s*\{([^}]+)\}', content)
    if match:
        instance_str = match.group(1)
        for pair in instance_str.split(','):
            if ':' in pair:
                key, val = pair.split(':')
                key = key.strip()
                val = val.strip()
                try:
                    result[key] = int(val)
                except ValueError:
                    pass

    # Extract operator name
    name_match = re.search(r'^\s*name:\s*(\w+)', content, re.MULTILINE)
    if name_match:
        result['operator_name'] = name_match.group(1)
    else:
        result['operator_name'] = 'Unknown'

    # Extract model/dnn name
    dnn_match = re.search(r'dnn_name:\s*(\w+)', content)
    if dnn_match:
        result['model_name'] = dnn_match.group(1)
    else:
        result['model_name'] = 'unknown'

    return result


def calculate_layer_profile(yaml_path: str, layer_idx: int,
                           input_bits: int = 8, weight_bits: int = 8,
                           output_bits: int = 8,
                           weight_stationary: bool = True,
                           config: RooflineConfig = None) -> LayerProfile:
    """
    Calculate operational intensity and other metrics for a layer.

    Args:
        yaml_path: Path to layer YAML file
        layer_idx: Layer index
        input_bits: Input precision in bits
        weight_bits: Weight precision in bits
        output_bits: Output precision in bits
        weight_stationary: If True, weights are stored in CiM array and not
                          counted in data movement (typical for CiM).
                          If False, weights are also moved (typical for GPUs).
        config: Roofline configuration (uses DEFAULT_ROOFLINE_CONFIG if None)
    """
    if config is None:
        config = DEFAULT_ROOFLINE_CONFIG

    info = parse_layer_info(yaml_path)

    # Extract dimensions (defaults for typical transformer layer)
    C = info.get('C', 1024)  # Input channels
    M = info.get('M', 1024)  # Output channels
    P = info.get('P', 256)   # Sequence length / spatial dimension

    # Get operator info
    operator_name = info.get('operator_name', 'Unknown')
    model_name = info.get('model_name', 'unknown')
    operator_category = OPERATOR_CATEGORIES.get(operator_name, operator_name)

    # Extract layer ID from filename
    layer_id = os.path.basename(yaml_path).replace('.yaml', '')

    # For inference, assume batch_size = 1
    batch_size = 1

    # Calculate MACs for matrix multiplication
    # MACs = batch * seq_len * in_features * out_features
    total_macs = batch_size * P * C * M

    # Calculate data sizes in bytes
    bytes_per_input = input_bits / 8
    bytes_per_weight = weight_bits / 8
    bytes_per_output = output_bits / 8

    # Input: (batch, seq_len, in_features)
    input_bytes = int(batch_size * P * C * bytes_per_input)

    # Weights: (in_features, out_features)
    weight_bytes = int(C * M * bytes_per_weight)

    # Output: (batch, seq_len, out_features)
    output_bytes = int(batch_size * P * M * bytes_per_output)

    # === Multi-level OI Support ===
    # Calculate data movement at different memory hierarchy levels

    # Level 1: CIM Port (input/output only, weights in memory cells)
    bytes_cim_port = input_bytes + output_bytes
    oi_cim_port = total_macs / bytes_cim_port if bytes_cim_port > 0 else 0

    # Level 2: PE SRAM (may include partial weights if not fully weight-stationary)
    bytes_pe_sram = bytes_cim_port  # Simplified: same as CIM port for now
    oi_pe_sram = total_macs / bytes_pe_sram if bytes_pe_sram > 0 else 0

    # Level 3: Global Buffer (includes data tiling/reuse)
    # For now, assume similar to CIM port. In real systems, this would be lower
    # due to data reuse at higher levels reducing off-chip traffic
    bytes_global_buffer = bytes_cim_port
    oi_global_buffer = total_macs / bytes_global_buffer if bytes_global_buffer > 0 else 0

    # Level 4: DRAM (system bottleneck - includes all data movement to/from chip)
    # For traditional architectures, this includes weights
    if config.weight_stationary:
        bytes_dram = input_bytes + output_bytes
    else:
        bytes_dram = input_bytes + weight_bytes + output_bytes
    oi_dram = total_macs / bytes_dram if bytes_dram > 0 else 0

    # Select primary OI based on configured boundary
    boundary_to_bytes = {
        "cim_port": bytes_cim_port,
        "pe_sram": bytes_pe_sram,
        "global_buffer": bytes_global_buffer,
        "dram": bytes_dram
    }

    total_bytes = boundary_to_bytes.get(config.oi_byte_boundary, bytes_cim_port)
    operational_intensity = total_macs / total_bytes if total_bytes > 0 else 0

    profile = LayerProfile(
        layer_idx=layer_idx,
        layer_id=layer_id,
        operator_type=operator_name,
        operator_category=operator_category,
        model_name=model_name,
        input_channels=C,
        output_channels=M,
        sequence_length=P,
        total_macs=total_macs,
        input_bytes=input_bytes,
        weight_bytes=weight_bytes,
        output_bytes=output_bytes,
        total_bytes=total_bytes,
        operational_intensity=operational_intensity,
    )

    # Add multi-level OI if enabled
    if config.enable_multilevel_oi:
        profile.oi_cim_port = oi_cim_port
        profile.oi_pe_sram = oi_pe_sram
        profile.oi_global_buffer = oi_global_buffer
        profile.oi_dram = oi_dram
        profile.bytes_cim_port = bytes_cim_port
        profile.bytes_pe_sram = bytes_pe_sram
        profile.bytes_global_buffer = bytes_global_buffer
        profile.bytes_dram = bytes_dram

    return profile


def profile_layer_on_hardware(
    macro: str,
    layer_path: str,
    variables: dict = None,
    system: str = None,
    output_dir: str = None
) -> Optional[MacroOutputStats]:
    """
    Run Timeloop simulation for a layer on specific hardware.

    Args:
        macro: CiM macro architecture name
        layer_path: Path to the layer YAML file
        variables: Optional dictionary of variables to override
        system: System architecture name
        output_dir: Directory to save timeloop mapper output files.
                   If provided, all mapper output files will be saved here.
    """
    try:
        spec = get_spec(
            macro=macro,
            layer=layer_path,
            system=system,
            max_utilization=False,
        )

        if variables:
            spec.variables.update(variables)

        spec.architecture.name2leaf("macro").attributes["has_power_gating"] = True

        return run_mapper(spec=spec, output_dir=output_dir)
    except Exception as e:
        print(f"Error running layer {layer_path}: {e}")
        return None


def run_simulation_for_layer(
    macro: str,
    layer_file: str,
    layer_idx: int,
    system: str = None,
    output_dir: str = None
) -> Tuple[Optional[MacroOutputStats], LayerProfile]:
    """
    Run Timeloop simulation for a single layer and return both simulation stats
    and updated layer profile with actual achieved performance.

    Args:
        macro: CiM macro architecture name
        layer_file: Path to the layer YAML file
        layer_idx: Layer index
        system: System architecture name
        output_dir: Directory to save timeloop mapper output files for this layer.
                   If provided, all mapper files (timeloop-mapper.*) will be saved here.
    """
    # First calculate theoretical profile
    profile = calculate_layer_profile(layer_file, layer_idx)

    # Run simulation
    sim_stats = profile_layer_on_hardware(
        macro=macro,
        layer_path=layer_file,
        system=system,
        output_dir=output_dir
    )

    if sim_stats is not None:
        # Update profile with simulation results
        profile.cycles = sim_stats.cycles
        profile.energy_pj = sim_stats.energy * 1e12  # Convert to pJ

        # Calculate achieved GFLOPS from simulation
        # TOPS from simulation already accounts for 2x (MAC = 2 ops)
        profile.achieved_gflops = sim_stats.tops * 1000  # TOPS to GFLOPS

    return sim_stats, profile


def run_simulations_parallel(
    macro: str,
    layer_files: List[str],
    system: str = None,
    n_jobs: int = 8,
    base_output_dir: str = None
) -> Tuple[List[Optional[MacroOutputStats]], List[LayerProfile]]:
    """
    Run Timeloop simulations for multiple layers in parallel.
    Returns both simulation statistics and updated layer profiles.

    Args:
        macro: CiM macro architecture name
        layer_files: List of paths to layer YAML files
        system: System architecture name
        n_jobs: Number of parallel jobs
        base_output_dir: Base directory for output. If provided, each layer's
                        mapper output will be saved to base_output_dir/<layer_idx>/
    """
    from joblib import Parallel, delayed as jl_delayed

    def run_single(layer_file, idx):
        # Determine output directory for this layer
        layer_output_dir = None
        if base_output_dir:
            layer_output_dir = os.path.join(base_output_dir, str(idx))
            os.makedirs(layer_output_dir, exist_ok=True)

        return run_simulation_for_layer(macro, layer_file, idx, system, layer_output_dir)

    results = list(tqdm(
        Parallel(return_as="generator", n_jobs=n_jobs)(
            jl_delayed(run_single)(f, i) for i, f in enumerate(layer_files)
        ),
        total=len(layer_files),
        desc="Running simulations"
    ))

    sim_stats = [r[0] for r in results]
    profiles = [r[1] for r in results]

    return sim_stats, profiles


def create_simulation_comparison_table(
    layer_profiles: List[LayerProfile],
    hardware: HardwareSpec,
    model_name: str = None
) -> str:
    """Create a comparison table showing theoretical vs simulated performance."""
    if model_name is None and layer_profiles:
        model_name = layer_profiles[0].model_name

    lines = [
        f"\n{'='*110}",
        f"  {model_name.upper()} - Simulation vs Theoretical Comparison",
        f"{'='*110}",
        f"Hardware: {hardware.name}",
        f"",
    ]

    # Check if any layer has simulation data
    has_sim_data = any(p.achieved_gflops is not None for p in layer_profiles)

    if has_sim_data:
        header = (
            f"{'Layer':<18} {'Op':<12} {'OI':>8} {'Theo.GFLOPS':>12} "
            f"{'Sim.GFLOPS':>11} {'Efficiency':>10} {'Cycles':>12} {'Energy(mJ)':>10}"
        )
        lines.append(header)
        lines.append("-" * len(header))

        for p in layer_profiles:
            # Theoretical achievable GFLOPS (use roofline model)
            theo_gflops = min(hardware.peak_gops, 2 * p.operational_intensity * hardware.memory_bandwidth_gb_s)

            if p.achieved_gflops is not None:
                efficiency = (p.achieved_gflops / theo_gflops * 100) if theo_gflops > 0 else 0
                energy_mj = p.energy_pj / 1e9 if p.energy_pj else 0
                line = (
                    f"{p.display_name:<18} {p.operator_type:<12} {p.operational_intensity:>8.1f} "
                    f"{theo_gflops:>12.1f} {p.achieved_gflops:>11.1f} {efficiency:>9.1f}% "
                    f"{p.cycles:>12,} {energy_mj:>10.3f}"
                )
            else:
                line = (
                    f"{p.display_name:<18} {p.operator_type:<12} {p.operational_intensity:>8.1f} "
                    f"{theo_gflops:>12.1f} {'N/A':>11} {'N/A':>10} "
                    f"{'N/A':>12} {'N/A':>10}"
                )
            lines.append(line)

        lines.append("-" * len(header))

        # Summary statistics
        simulated = [p for p in layer_profiles if p.achieved_gflops is not None]
        if simulated:
            avg_efficiency = np.mean([
                p.achieved_gflops / min(hardware.peak_gops, 2 * p.operational_intensity * hardware.memory_bandwidth_gb_s)
                for p in simulated
            ]) * 100

            total_cycles = sum(p.cycles for p in simulated if p.cycles)
            total_energy = sum(p.energy_pj for p in simulated if p.energy_pj) / 1e9  # mJ

            lines.extend([
                f"",
                f"{'='*110}",
                f"  SIMULATION SUMMARY",
                f"{'='*110}",
                f"Layers Simulated: {len(simulated)}/{len(layer_profiles)}",
                f"Average Efficiency: {avg_efficiency:.1f}% of roofline",
                f"Total Cycles: {total_cycles:,}",
                f"Total Energy: {total_energy:.3f} mJ",
            ])
    else:
        lines.extend([
            "No simulation data available.",
            "Run with --simulate flag to get actual hardware performance.",
        ])

    return "\n".join(lines)


def save_plotly_figure(fig: go.Figure, save_path: str):
    """Save a Plotly figure to both HTML and PNG formats."""
    if save_path:
        # Remove extension if present (only common image/html extensions)
        for ext in ['.html', '.png', '.jpg', '.jpeg', '.svg', '.pdf']:
            if save_path.lower().endswith(ext):
                save_path = save_path[:-len(ext)]
                break

        # Save HTML
        html_path = f"{save_path}.html"
        fig.write_html(html_path)
        print(f"Saved interactive plot to {html_path}")

        # Save PNG
        png_path = f"{save_path}.png"
        try:
            fig.write_image(png_path, scale=2)
            print(f"Saved static plot to {png_path}")
        except Exception as e:
            print(f"Warning: Could not save PNG (install kaleido: pip install kaleido): {e}")


def plot_roofline_with_simulation(
    hardware: HardwareSpec,
    layer_profiles: List[LayerProfile],
    model_name: str = None,
    title: str = None,
    save_path: str = None,
    figsize: Tuple[int, int] = (14, 10)
):
    """
    Plot empirical roofline model derived from simulation results.

    The roofline is now based on:
    - Empirical peak compute: Maximum achieved GOPS across all layers
    - Empirical sustained bandwidth: Calculated from memory-bound layers

    Layer markers show actual simulated performance positioned on the empirical roofline.
    """
    if model_name is None and layer_profiles:
        model_name = layer_profiles[0].model_name

    fig = go.Figure()

    # Define the range of operational intensities to plot (MACs/Byte)
    oi_min, oi_max = 0.1, 1000
    oi_range = np.logspace(np.log10(oi_min), np.log10(oi_max), 500)

    # Memory and compute roofs
    memory_roof = 2 * oi_range * hardware.memory_bandwidth_gb_s
    compute_roof = np.full_like(oi_range, hardware.peak_gops)
    roofline = np.minimum(memory_roof, compute_roof)

    # Plot the roofline
    fig.add_trace(go.Scatter(
        x=oi_range, y=roofline,
        mode='lines',
        name='Roofline',
        line=dict(color='blue', width=3)
    ))

    # Ridge point (intersection)
    ridge_oi = hardware.peak_gops / (2 * hardware.memory_bandwidth_gb_s)
    ridge_perf = hardware.peak_gops

    # Add ridge point marker
    fig.add_trace(go.Scatter(
        x=[ridge_oi], y=[ridge_perf],
        mode='markers+text',
        name='Ridge Point',
        marker=dict(
            color='red',
            size=14,
            symbol='diamond',
            line=dict(color='darkred', width=2)
        ),
        text=[f'Ridge Point<br>OI={ridge_oi:.1f}'],
        textposition='top right',
        textfont=dict(size=11, color='darkred'),
        hovertemplate=(
            '<b>Ridge Point</b><br>'
            f'OI: {ridge_oi:.1f} MACs/Byte<br>'
            f'Performance: {ridge_perf:.0f} GFLOPS<extra></extra>'
        )
    ))

    # Add reference points along the roofline at key OI values
    reference_ois = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
    reference_perfs = [min(hardware.peak_gops, 2 * oi * hardware.memory_bandwidth_gb_s) for oi in reference_ois]

    fig.add_trace(go.Scatter(
        x=reference_ois, y=reference_perfs,
        mode='markers+text',
        name='Reference Points',
        marker=dict(
            color='navy',
            size=8,
            symbol='circle',
            line=dict(color='white', width=1)
        ),
        text=[f'{oi}' for oi in reference_ois],
        textposition='top center',
        textfont=dict(size=9, color='navy'),
        hovertemplate=(
            '<b>Reference Point</b><br>'
            'OI: %{x:.1f} MACs/Byte<br>'
            'Performance: %{y:.0f} GFLOPS<extra></extra>'
        )
    ))

    # Group layers by operator category
    categories = list(set(p.operator_category for p in layer_profiles))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    color_map = {cat: colors[i % len(colors)] for i, cat in enumerate(categories)}

    # Plot each layer by category - show actual simulated performance
    has_sim_data = any(p.achieved_gflops is not None for p in layer_profiles)

    for cat in categories:
        cat_profiles = [p for p in layer_profiles if p.operator_category == cat]

        # Plot actual simulated performance (or all layers if no sim data)
        sim_profiles = [p for p in cat_profiles if p.achieved_gflops is not None] if has_sim_data else cat_profiles

        if sim_profiles:
            sim_ois = [p.operational_intensity for p in sim_profiles]
            sim_gflops = [p.achieved_gflops if p.achieved_gflops is not None
                         else min(hardware.peak_gops, 2 * p.operational_intensity * hardware.memory_bandwidth_gb_s)
                         for p in sim_profiles]
            sim_labels = [str(p.layer_idx) for p in sim_profiles]

            fig.add_trace(go.Scatter(
                x=sim_ois, y=sim_gflops,
                mode='markers',
                name=cat,
                marker=dict(
                    color=color_map[cat],
                    size=12,
                    line=dict(color='white', width=1.5)
                ),
                text=sim_labels,
                hovertemplate=(
                    f'<b>{cat}</b><br>'
                    'Layer: %{text}<br>'
                    'OI: %{x:.1f} MACs/Byte<br>'
                    'Performance: %{y:.0f} GFLOPS<extra></extra>'
                )
            ))

    # Update layout
    fig.update_layout(
        title=dict(
            text=title or f'Empirical Roofline: {model_name} on {hardware.name}',
            x=0.5
        ),
        xaxis=dict(
            title='Operational Intensity (MACs/Byte)',
            type='log',
            range=[np.log10(oi_min), np.log10(oi_max)],
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title='Performance (GFLOPS)',
            type='log',
            range=[0, np.log10(hardware.peak_gops * 2)],
            gridcolor='lightgray'
        ),
        legend=dict(x=0.7, y=0.1),
        width=figsize[0] * 70,
        height=figsize[1] * 70,
        plot_bgcolor='white',
        annotations=[
            dict(
                text=(f'<b>Hardware:</b> {hardware.name}<br>'
                      f'<b>Empirical Peak:</b> {hardware.peak_gops:.2f} GOPS<br>'
                      f'<b>Empirical BW:</b> {hardware.memory_bandwidth_gb_s:.1f} GB/s<br>'
                      f'<b>Ridge Point OI:</b> {ridge_oi:.1f} MACs/Byte'),
                xref='paper', yref='paper',
                x=0.02, y=0.98,
                showarrow=False,
                bgcolor='white',
                bordercolor='lightgray',
                borderwidth=1,
                align='left'
            )
        ]
    )

    save_plotly_figure(fig, save_path)
    return fig


def get_hardware_specs(macro: str) -> HardwareSpec:
    """
    Get hardware specifications for roofline model.
    Performance is in GFLOPS (1 MAC = 2 FLOPS for hardware measurement).

    Note: These are theoretical peak values. Actual achievable performance
    depends on the specific workload and mapping efficiency.
    """
    # CiM hardware specs - values based on published papers and simulations
    # Peak GFLOPS can be very high for CiM due to massive parallelism
    hardware_configs = {
        # === Analog/RRAM-based CiM ===
        "basic_analog": HardwareSpec(
            name="Basic Analog CiM",
            peak_gops=100000.0,   # 100 TFLOPS
            memory_bandwidth_gb_s=100.0,
            array_rows=128,
            array_cols=128,
            technology_nm=65,
            voltage=0.8,
            weight_bits=8,
            input_bits=8,
            output_bits=8,
            adc_resolution=8,
            cell_type="RRAM",
            frequency_mhz=500.0,
        ),
        "isaac_isca_2016": HardwareSpec(
            name="ISAAC (ISCA 2016)",
            peak_gops=160000.0,   # 160 TFLOPS - High-performance ReRAM design
            memory_bandwidth_gb_s=128.0,
            array_rows=128,
            array_cols=128,
            technology_nm=32,
            voltage=1.0,
            weight_bits=2,
            input_bits=16,
            output_bits=16,
            adc_resolution=8,
            cell_type="RRAM",
            frequency_mhz=1200.0,
        ),
        "raella_isca_2023": HardwareSpec(
            name="RAELLA (ISCA 2023)",
            peak_gops=120000.0,   # 120 TFLOPS - Row-efficient architecture
            memory_bandwidth_gb_s=200.0,
            array_rows=256,
            array_cols=256,
            technology_nm=28,
            voltage=0.9,
            weight_bits=4,
            input_bits=8,
            output_bits=8,
            adc_resolution=6,
            cell_type="RRAM",
            frequency_mhz=1000.0,
        ),
        "wan_nature_2022": HardwareSpec(
            name="Wan et al. (Nature 2022)",
            peak_gops=200000.0,   # 200 TFLOPS - Full-system integration
            memory_bandwidth_gb_s=160.0,
            array_rows=256,
            array_cols=256,
            technology_nm=40,
            voltage=1.0,
            weight_bits=4,
            input_bits=8,
            output_bits=16,
            adc_resolution=8,
            cell_type="RRAM",
            frequency_mhz=100.0,
        ),

        # === SRAM-based CiM ===
        "colonnade_jssc_2021": HardwareSpec(
            name="Colonnade (JSSC 2021)",
            # 128×128 digital bit-serial CIM macro, 65nm
            # Peak at 1-bit: 567 GOPS @ 138 MHz (Table II & III)
            # Input BW: 128 rows × 138 MHz × 1 bit = 17.7 Gbps ≈ 2.2 GB/s
            # Ridge Point: 567 GOPS / 2.2 GB/s ≈ 258 OP/Byte
            peak_gops=567.0,        # 567 GOPS at 1-bit precision (Table II)
            memory_bandwidth_gb_s=2.2,  # Input bandwidth from bit-serial architecture
            array_rows=128,
            array_cols=128,
            technology_nm=65,
            voltage=0.8,
            weight_bits=1,  # Bit-serial
            input_bits=1,   # Bit-serial
            output_bits=1,
            adc_resolution=1,  # Digital, no ADC
            cell_type="SRAM",
            frequency_mhz=138.0,  # Maximum frequency at 1-bit (Table II)
        ),
        "jia_jssc_2020": HardwareSpec(
            name="Jia et al. (JSSC 2020)",
            peak_gops=64000.0,    # 64 TFLOPS - Programmable SRAM
            memory_bandwidth_gb_s=200.0,
            array_rows=256,
            array_cols=256,
            technology_nm=22,
            voltage=0.8,
            weight_bits=8,
            input_bits=8,
            output_bits=20,
            adc_resolution=5,
            cell_type="SRAM",
            frequency_mhz=500.0,
        ),
        "sinangil_jssc_2021": HardwareSpec(
            name="Sinangil (JSSC 2021)",
            peak_gops=48000.0,    # 48 TFLOPS - Energy-efficient SRAM
            memory_bandwidth_gb_s=180.0,
            array_rows=256,
            array_cols=64,
            technology_nm=7,
            voltage=0.75,
            weight_bits=8,
            input_bits=8,
            output_bits=20,
            adc_resolution=6,
            cell_type="SRAM",
            frequency_mhz=1000.0,
        ),
        "wang_vlsi_2022": HardwareSpec(
            name="Wang et al. (VLSI 2022)",
            peak_gops=96000.0,    # 96 TFLOPS - High-density SRAM
            memory_bandwidth_gb_s=240.0,
            array_rows=512,
            array_cols=512,
            technology_nm=28,
            voltage=0.9,
            weight_bits=8,
            input_bits=8,
            output_bits=20,
            adc_resolution=8,
            cell_type="SRAM",
            frequency_mhz=800.0,
        ),

        # === Digital/Mixed-Signal CiM ===
        "lightning_sigc_2023": HardwareSpec(
            name="Lightning (SIGCOMM 2023)",
            peak_gops=256000.0,   # 256 TFLOPS - Optical/photonic accelerator
            memory_bandwidth_gb_s=400.0,
            array_rows=64,
            array_cols=64,
            technology_nm=45,
            voltage=1.0,
            weight_bits=8,
            input_bits=8,
            output_bits=8,
            adc_resolution=8,
            cell_type="Photonic",
            frequency_mhz=10000.0,  # Optical speed
        ),
        "albireo_isca_2021": HardwareSpec(
            name="Albireo (ISCA 2021)",
            peak_gops=140000.0,   # 140 TFLOPS - Digital CiM
            memory_bandwidth_gb_s=175.0,
            array_rows=128,
            array_cols=128,
            technology_nm=14,
            voltage=0.8,
            weight_bits=4,
            input_bits=8,
            output_bits=16,
            adc_resolution=4,
            cell_type="SRAM",
            frequency_mhz=1500.0,
        ),

        # === Custom SRAM Design ===
        "sram_1mb_custom": HardwareSpec(
            name="Custom SRAM 1MB (1024x1024x8)",
            peak_gops=120000.0,   # 120 TFLOPS - Large SRAM array with high parallelism
            memory_bandwidth_gb_s=300.0,  # High bandwidth due to large array
            array_rows=1024,
            array_cols=1024,
            technology_nm=28,
            voltage=0.9,
            weight_bits=8,
            input_bits=8,
            output_bits=24,
            adc_resolution=8,
            cell_type="SRAM",
            frequency_mhz=1000.0,
        ),
    }

    return hardware_configs.get(macro, HardwareSpec(
        name=f"CiM ({macro})",
        peak_gops=100000.0,
        memory_bandwidth_gb_s=100.0,
    ))


def get_hardware_specs_from_simulation(
    macro: str,
    sim_stats: List[MacroOutputStats],
    layer_profiles: List[LayerProfile],
    config: RooflineConfig = DEFAULT_ROOFLINE_CONFIG
) -> HardwareSpec:
    """
    Derive empirical hardware specifications from simulation results.

    This extracts:
    1. Empirical peak compute: Maximum achieved GOPS across all layers
    2. Empirical sustained bandwidth: Calculated from memory-bound layers using
       the formula: BW = Achieved_GOPS / (ops_per_mac * OI)
       where OI is in MACs/Byte

    Args:
        macro: Hardware macro name
        sim_stats: Simulation statistics for each layer
        layer_profiles: Layer profiles with OI and achieved performance
        config: Roofline configuration for unit conversions

    Returns:
        HardwareSpec with empirical peak and bandwidth values
    """
    # Collect achieved performance data
    achieved_data = []
    for stats, profile in zip(sim_stats, layer_profiles):
        if stats is not None and profile.achieved_gflops is not None:
            gops = profile.achieved_gflops  # Already in correct units from profile
            oi = profile.operational_intensity  # MACs/Byte
            achieved_data.append((oi, gops))

    if not achieved_data:
        # Fallback to theoretical specs
        theoretical = get_hardware_specs(macro)
        theoretical.name = f"{theoretical.name} (empirical - no sim data)"
        return theoretical

    # === 1. EMPIRICAL PEAK COMPUTE ===
    # Maximum achieved GOPS across all layers (typically from compute-bound layers)
    empirical_peak_gops = max(g for _, g in achieved_data)

    print(f"\n{'='*70}")
    print(f"  EMPIRICAL ROOFLINE EXTRACTION")
    print(f"{'='*70}")
    print(f"Empirical Peak Compute: {empirical_peak_gops:.2f} G{config.performance_unit}")

    # === 2. EMPIRICAL SUSTAINED BANDWIDTH ===
    # Calculate from memory-bound layers (low OI) where:
    # Achieved_GOPS = ops_per_mac * OI * BW
    # Therefore: BW = Achieved_GOPS / (ops_per_mac * OI)

    # Find ridge point from theoretical specs to help identify memory-bound layers
    theoretical = get_hardware_specs(macro)
    theoretical_ridge_oi = theoretical.peak_gops / (config.ops_per_mac * theoretical.memory_bandwidth_gb_s)

    # Memory-bound layers typically have OI < ridge_point
    # Use layers with OI < 0.5 * ridge_point for more conservative estimate
    threshold_oi = 0.5 * theoretical_ridge_oi
    memory_bound_layers = [(oi, g) for oi, g in achieved_data if oi < threshold_oi]

    if memory_bound_layers:
        # Calculate bandwidth from each memory-bound layer
        bandwidths = []
        for oi, gops in memory_bound_layers:
            # BW (GB/s) = Achieved_GOPS / (ops_per_mac * OI_MACs/Byte)
            bw = gops / (config.ops_per_mac * oi)
            bandwidths.append(bw)

        # Use the maximum sustained bandwidth from memory-bound layers
        empirical_bandwidth = max(bandwidths)
        print(f"Empirical Sustained Bandwidth: {empirical_bandwidth:.2f} GB/s")
        print(f"  (calculated from {len(memory_bound_layers)} memory-bound layers with OI < {threshold_oi:.1f})")
    else:
        # No clearly memory-bound layers, use average from all layers
        bandwidths = [g / (config.ops_per_mac * oi) for oi, g in achieved_data]
        empirical_bandwidth = np.median(bandwidths)  # Use median for robustness
        print(f"Empirical Sustained Bandwidth: {empirical_bandwidth:.2f} GB/s")
        print(f"  (calculated from median of all layers - no clear memory-bound region)")

    empirical_ridge_oi = empirical_peak_gops / (config.ops_per_mac * empirical_bandwidth)
    print(f"Empirical Ridge Point OI: {empirical_ridge_oi:.2f} {config.oi_unit}")
    print(f"{'='*70}\n")

    # Get other specs from theoretical hardware
    return HardwareSpec(
        name=f"{theoretical.name} (Empirical)",
        peak_gops=empirical_peak_gops,
        memory_bandwidth_gb_s=empirical_bandwidth,
        array_rows=theoretical.array_rows,
        array_cols=theoretical.array_cols,
        technology_nm=theoretical.technology_nm,
        voltage=theoretical.voltage,
        weight_bits=theoretical.weight_bits,
        input_bits=theoretical.input_bits,
        output_bits=theoretical.output_bits,
        adc_resolution=theoretical.adc_resolution,
        cell_type=theoretical.cell_type,
        frequency_mhz=theoretical.frequency_mhz,
    )


def plot_roofline_model(
    hardware: HardwareSpec,
    layer_profiles: List[LayerProfile],
    model_name: str = None,
    title: str = None,
    save_path: str = None,
    figsize: Tuple[int, int] = (14, 10)
):
    """
    Plot the roofline model with layer positions.

    X-axis: Operational Intensity (MACs/Byte) - Algorithm property
    Y-axis: Performance (GFLOPS) - Hardware measurement

    Note: The roofline converts OI to performance using:
    Performance = min(Peak_GFLOPS, 2 * OI * Bandwidth)
    The factor of 2 accounts for 1 MAC = 2 FLOPS
    """
    if model_name is None and layer_profiles:
        model_name = layer_profiles[0].model_name

    fig = go.Figure()

    # Define the range of operational intensities to plot (MACs/Byte)
    oi_min, oi_max = 0.1, 1000
    oi_range = np.logspace(np.log10(oi_min), np.log10(oi_max), 500)

    # Memory bandwidth ceiling: Performance (GFLOPS) = 2 * OI * Bandwidth
    memory_roof = 2 * oi_range * hardware.memory_bandwidth_gb_s

    # Compute ceiling: flat line at peak performance
    compute_roof = np.full_like(oi_range, hardware.peak_gops)

    # The actual roofline is the minimum of the two
    roofline = np.minimum(memory_roof, compute_roof)

    # Ridge point (intersection)
    ridge_oi = hardware.peak_gops / (2 * hardware.memory_bandwidth_gb_s)
    ridge_perf = hardware.peak_gops  # Performance at ridge point

    # Add ridge point marker
    fig.add_trace(go.Scatter(
        x=[ridge_oi], y=[ridge_perf],
        mode='markers+text',
        name='Ridge Point',
        marker=dict(
            color='red',
            size=14,
            symbol='diamond',
            line=dict(color='darkred', width=2)
        ),
        text=[f'Ridge Point<br>OI={ridge_oi:.1f}'],
        textposition='top right',
        textfont=dict(size=11, color='darkred'),
        hovertemplate=(
            '<b>Ridge Point</b><br>'
            f'OI: {ridge_oi:.1f} MACs/Byte<br>'
            f'Performance: {ridge_perf:.0f} GFLOPS<extra></extra>'
        )
    ))

    # Memory-bound region (shaded)
    memory_bound_oi = oi_range[oi_range <= ridge_oi]
    memory_bound_roof = 2 * memory_bound_oi * hardware.memory_bandwidth_gb_s
    fig.add_trace(go.Scatter(
        x=np.concatenate([memory_bound_oi, memory_bound_oi[::-1]]),
        y=np.concatenate([memory_bound_roof, np.ones_like(memory_bound_oi)]),
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.1)',
        line=dict(width=0),
        name='Memory-bound region',
        showlegend=True
    ))

    # Compute-bound region (shaded)
    compute_bound_oi = oi_range[oi_range >= ridge_oi]
    fig.add_trace(go.Scatter(
        x=np.concatenate([compute_bound_oi, compute_bound_oi[::-1]]),
        y=np.concatenate([np.full_like(compute_bound_oi, hardware.peak_gops),
                         np.ones_like(compute_bound_oi)]),
        fill='toself',
        fillcolor='rgba(0, 255, 0, 0.1)',
        line=dict(width=0),
        name='Compute-bound region',
        showlegend=True
    ))

    # Plot the roofline
    fig.add_trace(go.Scatter(
        x=oi_range, y=roofline,
        mode='lines',
        name='Roofline',
        line=dict(color='blue', width=3)
    ))

    # Add reference points along the roofline at key OI values
    reference_ois = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
    reference_perfs = [min(hardware.peak_gops, 2 * oi * hardware.memory_bandwidth_gb_s) for oi in reference_ois]

    fig.add_trace(go.Scatter(
        x=reference_ois, y=reference_perfs,
        mode='markers+text',
        name='Reference Points',
        marker=dict(
            color='navy',
            size=8,
            symbol='circle',
            line=dict(color='white', width=1)
        ),
        text=[f'{oi}' for oi in reference_ois],
        textposition='top center',
        textfont=dict(size=9, color='navy'),
        hovertemplate=(
            '<b>Reference Point</b><br>'
            'OI: %{x:.1f} MACs/Byte<br>'
            'Performance: %{y:.0f} GFLOPS<extra></extra>'
        )
    ))

    # Group layers by operator category for coloring
    categories = list(set(p.operator_category for p in layer_profiles))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    color_map = {cat: colors[i % len(colors)] for i, cat in enumerate(categories)}

    # Plot each layer by category
    for cat in categories:
        cat_profiles = [p for p in layer_profiles if p.operator_category == cat]
        ois = [p.operational_intensity for p in cat_profiles]
        perfs = []
        hover_texts = []

        for p in cat_profiles:
            achievable = min(hardware.peak_gops, 2 * p.operational_intensity * hardware.memory_bandwidth_gb_s)
            perf = p.achieved_gflops if p.achieved_gflops is not None else achievable
            perfs.append(perf)
            hover_texts.append(
                f'<b>{p.display_name}</b><br>'
                f'Category: {cat}<br>'
                f'OI: {p.operational_intensity:.1f} MACs/Byte<br>'
                f'Performance: {perf:.0f} GFLOPS<br>'
                f'MACs: {p.total_macs/1e9:.2f} G'
            )

        fig.add_trace(go.Scatter(
            x=ois, y=perfs,
            mode='markers',
            name=cat,
            marker=dict(
                color=color_map[cat],
                size=12,
                line=dict(color='black', width=1)
            ),
            hovertemplate='%{text}<extra></extra>',
            text=hover_texts
        ))

    # Update layout
    fig.update_layout(
        title=dict(
            text=title or f'Roofline Model: {model_name} on {hardware.name}',
            x=0.5
        ),
        xaxis=dict(
            title='Operational Intensity (MACs/Byte)',
            type='log',
            range=[np.log10(oi_min), np.log10(oi_max)],
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title='Performance (GFLOPS)',
            type='log',
            range=[0, np.log10(hardware.peak_gops * 2)],
            gridcolor='lightgray'
        ),
        legend=dict(x=0.7, y=0.1),
        width=figsize[0] * 70,
        height=figsize[1] * 70,
        plot_bgcolor='white',
        annotations=[
            dict(
                text=(f'<b>Hardware:</b> {hardware.name}<br>'
                      f'<b>Peak:</b> {hardware.peak_gops/1000:.1f} TFLOPS<br>'
                      f'<b>Memory BW:</b> {hardware.memory_bandwidth_gb_s:.0f} GB/s'),
                xref='paper', yref='paper',
                x=0.02, y=0.98,
                showarrow=False,
                bgcolor='white',
                bordercolor='lightgray',
                borderwidth=1,
                align='left'
            )
        ]
    )

    save_plotly_figure(fig, save_path)
    return fig


def plot_layer_analysis(
    layer_profiles: List[LayerProfile],
    hardware: HardwareSpec,
    model_name: str = None,
    save_path: str = None,
    figsize: Tuple[int, int] = (16, 12)
):
    """Create detailed analysis plots for layer profiling using Plotly."""
    if model_name is None and layer_profiles:
        model_name = layer_profiles[0].model_name

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Operational Intensity by Layer',
            'Compute (GMACs) by Layer',
            'Data Movement by Layer',
            'Layer Statistics'
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "table"}]]
    )

    layer_indices = list(range(len(layer_profiles)))
    layer_names = [p.display_name for p in layer_profiles]

    # Plot 1: Operational Intensity by Layer
    ois = [p.operational_intensity for p in layer_profiles]
    fig.add_trace(
        go.Bar(
            x=layer_indices, y=ois,
            marker_color='steelblue',
            name='OI',
            hovertemplate='Layer %{x}<br>OI: %{y:.1f} MACs/Byte<extra></extra>'
        ),
        row=1, col=1
    )
    fig.update_yaxes(type='log', title_text='OI (MACs/Byte)', row=1, col=1)
    fig.update_xaxes(title_text='Layer Index', row=1, col=1)

    # Plot 2: MACs by Layer
    macs = [p.total_macs / 1e9 for p in layer_profiles]
    fig.add_trace(
        go.Bar(
            x=layer_indices, y=macs,
            marker_color='steelblue',
            name='GMACs',
            hovertemplate='Layer %{x}<br>GMACs: %{y:.2f}<extra></extra>'
        ),
        row=1, col=2
    )
    fig.update_yaxes(title_text='GMACs', row=1, col=2)
    fig.update_xaxes(title_text='Layer Index', row=1, col=2)

    # Plot 3: Data Movement by Layer
    total_mb = [p.total_bytes / 1e6 for p in layer_profiles]
    fig.add_trace(
        go.Bar(
            x=layer_indices, y=total_mb,
            marker_color='darkorange',
            name='Data (MB)',
            hovertemplate='Layer %{x}<br>Data: %{y:.2f} MB<extra></extra>'
        ),
        row=2, col=1
    )
    fig.update_yaxes(title_text='Data Movement (MB)', row=2, col=1)
    fig.update_xaxes(title_text='Layer Index', row=2, col=1)

    # Plot 4: Layer Statistics Table
    total_macs_sum = sum(p.total_macs for p in layer_profiles)
    total_bytes_sum = sum(p.total_bytes for p in layer_profiles)
    avg_oi = np.mean(ois)

    fig.add_trace(
        go.Table(
            header=dict(values=['Metric', 'Value'], fill_color='lightgray', align='left'),
            cells=dict(
                values=[
                    ['Model', 'Hardware', 'Total Layers', 'Total GMACs', 'Total Data Movement', 'Avg OI'],
                    [model_name, hardware.name, len(layer_profiles),
                     f'{total_macs_sum/1e9:.2f}', f'{total_bytes_sum/1e6:.2f} MB',
                     f'{avg_oi:.1f} MACs/Byte']
                ],
                fill_color='white',
                align='left'
            )
        ),
        row=2, col=2
    )

    fig.update_layout(
        title=dict(text=f'{model_name} - Layer Analysis', x=0.5),
        showlegend=False,
        width=figsize[0] * 70,
        height=figsize[1] * 70
    )

    save_plotly_figure(fig, save_path)
    return fig


def create_summary_table(layer_profiles: List[LayerProfile], hardware: HardwareSpec,
                        model_name: str = None) -> str:
    """Create a summary table of layer profiling results."""
    if model_name is None and layer_profiles:
        model_name = layer_profiles[0].model_name

    header = (
        f"{'Layer':<20} {'Operator':<15} {'C':>6} {'M':>6} {'P':>5} "
        f"{'GMACs':>8} {'Data(MB)':>9} {'OI':>8}"
    )
    separator = "-" * len(header)

    lines = [
        f"\n{'='*90}",
        f"  {model_name.upper()} - Layer Profiling Summary",
        f"{'='*90}",
        f"Hardware: {hardware.name}",
        f"Peak Performance: {hardware.peak_gops/1000:.1f} TFLOPS ({hardware.peak_gops:.0f} GFLOPS)",
        f"Memory Bandwidth: {hardware.memory_bandwidth_gb_s:.0f} GB/s",
        f"",
        f"Note: OI = Operational Intensity (MACs/Byte) - Algorithm property",
        f"      Performance measured in FLOPS (1 MAC = 2 FLOPS)",
        f"{'='*90}\n",
        header,
        separator
    ]

    for p in layer_profiles:
        line = (
            f"{p.display_name:<20} {p.operator_type:<15} {p.input_channels:>6} {p.output_channels:>6} "
            f"{p.sequence_length:>5} {p.total_macs/1e9:>8.2f} "
            f"{p.total_bytes/1e6:>9.2f} {p.operational_intensity:>8.1f}"
        )
        lines.append(line)

    lines.append(separator)

    # Summary statistics
    total_macs = sum(p.total_macs for p in layer_profiles)
    total_data = sum(p.total_bytes for p in layer_profiles)
    avg_oi = np.mean([p.operational_intensity for p in layer_profiles])

    # Group by operator type
    op_stats = defaultdict(lambda: {'count': 0, 'macs': 0, 'ois': []})
    for p in layer_profiles:
        op_stats[p.operator_type]['count'] += 1
        op_stats[p.operator_type]['macs'] += p.total_macs
        op_stats[p.operator_type]['ois'].append(p.operational_intensity)

    lines.extend([
        f"\n{'='*90}",
        f"  SUMMARY STATISTICS",
        f"{'='*90}",
        f"Total Layers: {len(layer_profiles)}",
        f"Total GMACs: {total_macs/1e9:.2f}",
        f"Total Data Movement: {total_data/1e6:.2f} MB",
        f"Average Operational Intensity: {avg_oi:.1f} MACs/Byte",
        f"",
        f"{'='*90}",
        f"  OPERATOR TYPE ANALYSIS",
        f"{'='*90}",
    ])

    lines.append(f"{'Operator':<20} {'Count':>8} {'GMACs':>10} {'Avg OI':>10}")
    lines.append("-" * 50)

    for op_type, stats in sorted(op_stats.items(), key=lambda x: -x[1]['macs']):
        avg_op_oi = np.mean(stats['ois'])
        lines.append(f"{op_type:<20} {stats['count']:>8} {stats['macs']/1e9:>10.2f} {avg_op_oi:>10.1f}")

    return "\n".join(lines)


def create_combined_summary(
    layer_profiles: List[LayerProfile],
    hardware: HardwareSpec,
    macro: str,
    model_name: str = None
) -> str:
    """
    Create a combined summary that includes both layer profiling and simulation results.
    This is the single summary file output for each analysis run.
    """
    if model_name is None and layer_profiles:
        model_name = layer_profiles[0].model_name

    # Check if simulation data is available
    has_sim_data = any(p.achieved_gflops is not None for p in layer_profiles)

    # Calculate summary statistics
    total_macs = sum(p.total_macs for p in layer_profiles)
    total_data = sum(p.total_bytes for p in layer_profiles)
    avg_oi = np.mean([p.operational_intensity for p in layer_profiles])

    # Use OPS instead of FLOPS for CiM (1 MAC = 2 OPs)
    peak_ops = hardware.peak_gops  # Already in GOPS

    # Detect if using empirical roofline
    is_empirical = "(Empirical)" in hardware.name

    lines = [
        f"\n{'='*100}",
        f"  {model_name.upper()} - Analysis Summary",
        f"{'='*100}",
        f"Hardware: {hardware.name}",
        f"Peak Performance: {peak_ops:.2f} GOPS{' (Empirical from simulations)' if is_empirical else ' (Theoretical)'}",
        f"Memory Bandwidth: {hardware.memory_bandwidth_gb_s:.2f} GB/s{' (Empirical sustained)' if is_empirical else ' (Theoretical)'}",
        f"",
    ]

    # Add CIM macro specs
    lines.append(f"--- CIM Macro Specifications ---")
    if hardware.array_rows and hardware.array_cols:
        lines.append(f"  Array Size:        {hardware.array_size_str} ({hardware.array_rows * hardware.array_cols:,} cells)")
    if hardware.cell_type:
        lines.append(f"  Cell Type:         {hardware.cell_type}")
    if hardware.technology_nm:
        lines.append(f"  Technology Node:   {hardware.technology_nm} nm")
    if hardware.voltage:
        lines.append(f"  Operating Voltage: {hardware.voltage} V")
    if hardware.frequency_mhz:
        lines.append(f"  Frequency:         {hardware.frequency_mhz:.0f} MHz")
    if hardware.weight_bits and hardware.input_bits and hardware.output_bits:
        lines.append(f"  Precision:         {hardware.precision_str} (Weight/Input/Output bits)")
    if hardware.adc_resolution:
        lines.append(f"  ADC Resolution:    {hardware.adc_resolution} bits")

    # Calculate ridge point (OI at which memory and compute ceilings meet)
    ridge_oi = hardware.peak_gops / (2 * hardware.memory_bandwidth_gb_s)
    lines.append(f"  Ridge Point OI:    {ridge_oi:.1f} MACs/Byte")
    lines.append(f"")

    # Add simulation results summary if available
    if has_sim_data:
        simulated = [p for p in layer_profiles if p.achieved_gflops is not None]
        avg_efficiency = np.mean([
            p.achieved_gflops / min(hardware.peak_gops, 2 * p.operational_intensity * hardware.memory_bandwidth_gb_s)
            for p in simulated
        ]) * 100

        total_cycles = sum(p.cycles for p in simulated if p.cycles)
        total_energy = sum(p.energy_pj for p in simulated if p.energy_pj) / 1e9  # mJ

        lines.extend([
            f"--- Simulation Results ---",
            f"Layers Simulated: {len(simulated)}/{len(layer_profiles)}",
            f"Average Efficiency: {avg_efficiency:.1f}% of roofline",
            f"Total Cycles: {total_cycles:,}",
            f"Total Energy: {total_energy:.3f} mJ",
            f"",
            f"Note: Efficiency >100% indicates the roofline memory bandwidth ({hardware.memory_bandwidth_gb_s:.1f} GB/s)",
            f"      underestimates actual I/O capacity. Timeloop simulates macro in isolation",
            f"      with ideal input supply (no external memory bottleneck modeled).",
            f"",
        ])

    lines.extend([
        f"Note: OI = Operational Intensity (MACs/Byte) - Algorithm property",
        f"      Performance measured in OPS (1 MAC = 2 OPs)",
        f"{'='*100}",
        f"",
    ])

    # Layer-by-layer table
    if has_sim_data:
        header = (
            f"{'Layer':<18} {'Op':<14} {'GMACs':>8} {'OI':>7} "
            f"{'Theo.GOPS':>10} {'Sim.GOPS':>9} {'Eff%':>7} "
            f"{'Cycles':>12} {'Energy(mJ)':>10}"
        )
    else:
        header = (
            f"{'Layer':<20} {'Operator':<15} {'C':>6} {'M':>6} {'P':>5} "
            f"{'GMACs':>8} {'Data(MB)':>9} {'OI':>8}"
        )

    lines.append(header)
    lines.append("-" * len(header))

    for p in layer_profiles:
        theo_gops = min(hardware.peak_gops, 2 * p.operational_intensity * hardware.memory_bandwidth_gb_s)

        if has_sim_data:
            if p.achieved_gflops is not None:
                efficiency = (p.achieved_gflops / theo_gops * 100) if theo_gops > 0 else 0
                energy_mj = p.energy_pj / 1e9 if p.energy_pj else 0
                line = (
                    f"{p.display_name:<18} {p.operator_type:<14} {p.total_macs/1e9:>8.2f} "
                    f"{p.operational_intensity:>7.1f} {theo_gops:>10.1f} {p.achieved_gflops:>9.1f} "
                    f"{efficiency:>6.1f}% {p.cycles:>12,} {energy_mj:>10.3f}"
                )
            else:
                line = (
                    f"{p.display_name:<18} {p.operator_type:<14} {p.total_macs/1e9:>8.2f} "
                    f"{p.operational_intensity:>7.1f} {theo_gops:>10.1f} {'N/A':>9} "
                    f"{'N/A':>7} {'N/A':>12} {'N/A':>10}"
                )
        else:
            line = (
                f"{p.display_name:<20} {p.operator_type:<15} {p.input_channels:>6} {p.output_channels:>6} "
                f"{p.sequence_length:>5} {p.total_macs/1e9:>8.2f} "
                f"{p.total_bytes/1e6:>9.2f} {p.operational_intensity:>8.1f}"
            )
        lines.append(line)

    lines.append("-" * len(header))

    # Summary statistics section
    lines.extend([
        f"",
        f"{'='*100}",
        f"  SUMMARY STATISTICS",
        f"{'='*100}",
        f"Total Layers: {len(layer_profiles)}",
        f"Total GMACs: {total_macs/1e9:.2f}",
        f"Total Data Movement: {total_data/1e6:.2f} MB",
        f"Average Operational Intensity: {avg_oi:.1f} MACs/Byte",
        f"",
    ])

    # Operator type analysis
    op_stats = defaultdict(lambda: {'count': 0, 'macs': 0, 'ois': []})
    for p in layer_profiles:
        op_stats[p.operator_type]['count'] += 1
        op_stats[p.operator_type]['macs'] += p.total_macs
        op_stats[p.operator_type]['ois'].append(p.operational_intensity)

    lines.extend([
        f"{'='*100}",
        f"  OPERATOR TYPE ANALYSIS",
        f"{'='*100}",
        f"{'Operator':<20} {'Count':>8} {'GMACs':>10} {'Avg OI':>10}",
        "-" * 50,
    ])

    for op_type, stats in sorted(op_stats.items(), key=lambda x: -x[1]['macs']):
        avg_op_oi = np.mean(stats['ois'])
        lines.append(f"{op_type:<20} {stats['count']:>8} {stats['macs']/1e9:>10.2f} {avg_op_oi:>10.1f}")

    return "\n".join(lines)


def save_layer_profile(layer_profile: LayerProfile, hardware: HardwareSpec, layer_dir: str):
    """Save detailed information for a single layer."""
    os.makedirs(layer_dir, exist_ok=True)

    summary_path = os.path.join(layer_dir, "layer_info.txt")
    with open(summary_path, 'w') as f:
        f.write(f"{'='*70}\n")
        f.write(f"  Layer {layer_profile.layer_idx}: {layer_profile.display_name}\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Operator Type: {layer_profile.operator_type}\n")
        f.write(f"Operator Category: {layer_profile.operator_category}\n")
        f.write(f"Model: {layer_profile.model_name}\n\n")

        f.write(f"--- Problem Dimensions ---\n")
        f.write(f"Input Channels (C):  {layer_profile.input_channels:>10,}\n")
        f.write(f"Output Channels (M): {layer_profile.output_channels:>10,}\n")
        f.write(f"Sequence Length (P): {layer_profile.sequence_length:>10,}\n\n")

        f.write(f"--- Compute ---\n")
        f.write(f"Total MACs:  {layer_profile.total_macs:>15,}\n")
        f.write(f"Total FLOPs: {layer_profile.total_flops:>15,} (MACs * 2)\n")
        f.write(f"GMACs:       {layer_profile.total_macs/1e9:>15.2f}\n\n")

        f.write(f"--- Data Movement ---\n")
        f.write(f"Input Data:  {layer_profile.input_bytes:>15,} bytes ({layer_profile.input_bytes/1e6:>8.2f} MB)\n")
        f.write(f"Weight Data: {layer_profile.weight_bytes:>15,} bytes ({layer_profile.weight_bytes/1e6:>8.2f} MB)\n")
        f.write(f"Output Data: {layer_profile.output_bytes:>15,} bytes ({layer_profile.output_bytes/1e6:>8.2f} MB)\n")
        f.write(f"Total Data:  {layer_profile.total_bytes:>15,} bytes ({layer_profile.total_bytes/1e6:>8.2f} MB)\n\n")

        f.write(f"--- Operational Intensity ---\n")
        f.write(f"OI: {layer_profile.operational_intensity:.2f} MACs/Byte\n\n")

        f.write(f"--- Hardware: {hardware.name} ---\n")
        theo_gflops = min(hardware.peak_gops, 2 * layer_profile.operational_intensity * hardware.memory_bandwidth_gb_s)
        f.write(f"Theoretical Performance: {theo_gflops:>10.2f} GFLOPS\n")

        if layer_profile.achieved_gflops is not None:
            efficiency = (layer_profile.achieved_gflops / theo_gflops * 100) if theo_gflops > 0 else 0
            f.write(f"\n--- Simulation Results ---\n")
            f.write(f"Achieved Performance: {layer_profile.achieved_gflops:>10.2f} GFLOPS\n")
            f.write(f"Efficiency:           {efficiency:>10.1f}% of roofline\n")
            f.write(f"Cycles:               {layer_profile.cycles:>10,}\n")
            f.write(f"Energy:               {layer_profile.energy_pj/1e9:>10.3f} mJ\n")


def run_roofline_analysis(
    model_name: str = "gpt2_medium",
    macro: str = "basic_analog",
    run_simulation: bool = False,
    output_dir: str = None,
    num_layers: int = None,
    generate_validation: bool = False,
    roofline_config: RooflineConfig = None,
    system: str = None
):
    """
    Main function to run the roofline analysis.

    Args:
        model_name: Name of the DNN model to analyze (e.g., "gpt2_medium", "resnet18")
        macro: Name of the CiM macro architecture to analyze
        run_simulation: Whether to run actual Timeloop simulations
        output_dir: Directory to save output files
        num_layers: Number of layers to analyze (None for all)
        generate_validation: Whether to generate validation report (requires simulation)
        roofline_config: Roofline configuration (uses DEFAULT_ROOFLINE_CONFIG if None)
        system: System architecture name (default: uses macro name if available, else ws_dummy_buffer_one_macro_bw_limited)
    """
    if roofline_config is None:
        roofline_config = DEFAULT_ROOFLINE_CONFIG

    # Auto-detect system based on macro name if not specified
    if system is None:
        system_yaml_path = path_from_model_dir(f"arch/4_system/{macro}.yaml")
        if os.path.exists(system_yaml_path):
            system = macro
            print(f"Using system architecture: {system} (matched to macro)")
        else:
            system = "ws_dummy_buffer_one_macro_bw_limited"
            print(f"Using default system architecture: {system}")

    output_dir = get_output_dir(output_dir, model_name, macro)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Roofline Analysis: {model_name} on {macro}")
    print(f"{'='*70}\n")

    # Print configuration
    print(f"Configuration:")
    print(f"  Performance Unit: {roofline_config.performance_unit}")
    print(f"  OI Definition:    {roofline_config.oi_unit}")
    print(f"  OI Byte Boundary: {roofline_config.oi_byte_boundary}")
    print(f"  Multi-level OI:   {roofline_config.enable_multilevel_oi}")
    print()

    # Get hardware specifications
    hardware = get_hardware_specs(macro)
    print(f"Hardware: {hardware.name}")
    print(f"  Peak Performance: {hardware.peak_gops:.2f} G{roofline_config.performance_unit} ({hardware.peak_gops/1000:.3f} T{roofline_config.performance_unit})")
    print(f"  Memory Bandwidth: {hardware.memory_bandwidth_gb_s:.2f} GB/s\n")

    # Get layer files
    try:
        layer_files = get_model_layer_files(model_name)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Available models: {get_available_models()}")
        return None, None

    if num_layers:
        layer_files = layer_files[:num_layers]

    print(f"Found {len(layer_files)} {model_name} layers to analyze\n")

    # Profile each layer
    print("Profiling layers...")
    layer_profiles = []
    for i, layer_file in enumerate(tqdm(layer_files, desc="Calculating layer profiles")):
        profile = calculate_layer_profile(layer_file, i, config=roofline_config)
        layer_profiles.append(profile)

        # Save per-layer profile
        layer_dir = os.path.join(output_dir, str(i))
        save_layer_profile(profile, hardware, layer_dir)

    # Create detailed layer analysis plots
    analysis_path = os.path.join(output_dir, f"layer_analysis_{model_name}_{macro}")
    plot_layer_analysis(
        layer_profiles,
        hardware,
        model_name=model_name,
        save_path=analysis_path
    )

    # Optionally run actual simulations
    if run_simulation:
        print("\n" + "="*70)
        print("  Running Timeloop Simulations")
        print("="*70 + "\n")
        print(f"Mapper output files will be saved to: {output_dir}/<layer_idx>/\n")

        # Run simulations for all layers, saving mapper outputs to each layer folder
        sim_stats, layer_profiles = run_simulations_parallel(
            macro=macro,
            layer_files=layer_files,
            system=system,
            n_jobs=min(8, len(layer_files)),
            base_output_dir=output_dir
        )

        # Count successful simulations
        successful = sum(1 for s in sim_stats if s is not None)
        print(f"\nSimulation complete: {successful}/{len(layer_files)} layers successful")

        # Extract empirical hardware specs from simulation results
        print("\nExtracting empirical roofline from simulation data...")
        hardware_empirical = get_hardware_specs_from_simulation(
            macro=macro,
            sim_stats=sim_stats,
            layer_profiles=layer_profiles,
            config=roofline_config
        )

        # Replace theoretical hardware with empirical for all subsequent operations
        hardware = hardware_empirical

        # Update per-layer files with simulation results
        for i, profile in enumerate(layer_profiles):
            layer_dir = os.path.join(output_dir, str(i))
            save_layer_profile(profile, hardware, layer_dir)

    # Create and save combined summary (includes simulation results if available)
    summary = create_combined_summary(layer_profiles, hardware, macro, model_name)
    print(summary)

    # Save to single combined summary file
    summary_path = os.path.join(output_dir, f"{model_name}_{macro}_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"\nSaved summary to {summary_path}")

    # Generate validation report if requested and simulation data is available
    if generate_validation and run_simulation:
        print("\n" + "="*70)
        print("  Generating Validation Report")
        print("="*70 + "\n")

        validation_report = create_validation_report(
            layer_profiles,
            hardware,
            config=roofline_config,
            model_name=model_name
        )

        print(validation_report)

        # Save validation report
        validation_path = os.path.join(output_dir, f"{model_name}_{macro}_validation.txt")
        with open(validation_path, 'w') as f:
            f.write(validation_report)
        print(f"\nSaved validation report to {validation_path}")
    elif generate_validation and not run_simulation:
        print("\n⚠️  Validation report requires simulation data. Run with --simulate flag.")

    # Create roofline plot (single file, updated with simulation data if available)
    roofline_path = os.path.join(output_dir, f"roofline_{model_name}_{macro}")
    if run_simulation:
        # Use simulation plot with empirical roofline
        plot_roofline_with_simulation(
            hardware,
            layer_profiles,
            model_name=model_name,
            title=f"{model_name} - Empirical Roofline on {hardware.name}",
            save_path=roofline_path
        )
    else:
        # Use theoretical roofline plot
        plot_roofline_model(
            hardware,
            layer_profiles,
            model_name=model_name,
            title=f"{model_name} - Theoretical Roofline on {hardware.name}",
            save_path=roofline_path
        )

    return layer_profiles, hardware


def compare_hardware_rooflines(
    model_name: str = "gpt2_medium",
    macros: List[str] = None,
    output_dir: str = None,
    num_layers: int = 50
):
    """
    Compare roofline models across different hardware architectures using Plotly.
    """
    if macros is None:
        macros = ["basic_analog", "isaac_isca_2016", "raella_isca_2023"]

    # Use comparison directory under the model
    if output_dir is None:
        output_dir = get_output_dir(None, model_name, "comparison")
    os.makedirs(output_dir, exist_ok=True)

    # Get layer profiles (same for all hardware)
    layer_files = get_model_layer_files(model_name)[:num_layers]
    layer_profiles = [calculate_layer_profile(f, i) for i, f in enumerate(layer_files)]

    # Create subplots
    fig = make_subplots(
        rows=1, cols=len(macros),
        subplot_titles=[get_hardware_specs(m).name for m in macros],
        shared_yaxes=True
    )

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for col_idx, macro in enumerate(macros, 1):
        hardware = get_hardware_specs(macro)

        # Plot roofline (using MACs/Byte for x-axis, GFLOPS for y-axis)
        oi_range = np.logspace(-1, 3, 500)
        memory_roof = 2 * oi_range * hardware.memory_bandwidth_gb_s  # GFLOPS
        compute_roof = np.full_like(oi_range, hardware.peak_gops)
        roofline = np.minimum(memory_roof, compute_roof)

        # Add roofline
        fig.add_trace(
            go.Scatter(
                x=oi_range, y=roofline,
                mode='lines',
                name=f'Roofline ({hardware.name})',
                line=dict(color='blue', width=2),
                showlegend=(col_idx == 1)
            ),
            row=1, col=col_idx
        )

        # Add reference points along the roofline
        reference_ois = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
        reference_perfs = [min(hardware.peak_gops, 2 * oi * hardware.memory_bandwidth_gb_s) for oi in reference_ois]

        fig.add_trace(
            go.Scatter(
                x=reference_ois, y=reference_perfs,
                mode='markers',
                name='Reference Points',
                marker=dict(color='navy', size=6, symbol='circle'),
                hovertemplate='OI: %{x:.1f}<br>Perf: %{y:.0f} GFLOPS<extra></extra>',
                showlegend=(col_idx == 1)
            ),
            row=1, col=col_idx
        )

        # Plot layers
        layer_ois = [p.operational_intensity for p in layer_profiles]
        layer_perfs = [
            min(hardware.peak_gops, 2 * p.operational_intensity * hardware.memory_bandwidth_gb_s)
            for p in layer_profiles
        ]

        fig.add_trace(
            go.Scatter(
                x=layer_ois, y=layer_perfs,
                mode='markers',
                name='Layers',
                marker=dict(color='steelblue', size=8, opacity=0.7),
                hovertemplate='OI: %{x:.1f} MACs/Byte<br>Perf: %{y:.0f} GFLOPS<extra></extra>',
                showlegend=(col_idx == 1)
            ),
            row=1, col=col_idx
        )

        # Update axes for this subplot
        fig.update_xaxes(
            type='log', title_text='OI (MACs/Byte)',
            range=[-1, 3], row=1, col=col_idx
        )
        fig.update_yaxes(
            type='log',
            range=[1, np.log10(hardware.peak_gops * 2)],
            row=1, col=col_idx
        )

    # Update first y-axis label
    fig.update_yaxes(title_text='Performance (GFLOPS)', row=1, col=1)

    # Update layout
    fig.update_layout(
        title=dict(text=f'{model_name} - Hardware Comparison', x=0.5),
        width=400 * len(macros),
        height=500,
        showlegend=True
    )

    comparison_path = os.path.join(output_dir, f"hardware_comparison_{model_name}")
    save_plotly_figure(fig, comparison_path)

    return fig


def get_available_macros() -> List[str]:
    """Get list of available macro architectures."""
    macro_dir = path_from_model_dir("arch", "1_macro")
    macros = []
    for item in os.listdir(macro_dir):
        item_path = os.path.join(macro_dir, item)
        if os.path.isdir(item_path) and not item.startswith('_'):
            macros.append(item)
    return sorted(macros)


def _add_common_args(parser):
    """Add common arguments to a parser."""
    parser.add_argument("--model", "-m", type=str, default="gpt2_medium",
                        help="DNN model to analyze (default: gpt2_medium)")
    parser.add_argument("--output-dir", "-o", type=str, default=None,
                        help="Output directory for results")


def _cmd_list(args):
    """Handle 'list' subcommand."""
    if args.what == "models":
        print("Available models:")
        for model in get_available_models():
            print(f"  - {model}")
    elif args.what == "macros":
        print("Available macro architectures:")
        print(f"{'Macro':<30} {'Peak TFLOPS':>12} {'BW (GB/s)':>10}")
        print("-" * 54)
        for macro in get_available_macros():
            hw = get_hardware_specs(macro)
            print(f"{macro:<30} {hw.peak_gops/1000:>12.0f} {hw.memory_bandwidth_gb_s:>10.0f}")
    elif args.what == "outputs":
        print("Output files by model and hardware:")
        model_files = list_model_outputs(output_dir=args.output_dir)
        for model, hardware_dict in sorted(model_files.items()):
            print(f"\n  {model}:")
            for hardware, files in sorted(hardware_dict.items()):
                print(f"    {hardware}:")
                for f in files:
                    print(f"      - {f}")
        if not model_files:
            print("  (no output files found)")


def _cmd_clean(args):
    """Handle 'clean' subcommand."""
    if args.old_sims:
        print("Cleaning old simulation directories...")
        deleted = clean_old_simulations(dry_run=args.dry_run)
        action = "Would delete" if args.dry_run else "Deleted"
        print(f"\n{action} {len(deleted)} directories.")
    elif args.all:
        print("Cleaning all roofline output files...")
        model = None
        deleted = clean_model_outputs(model_name=model, output_dir=args.output_dir, dry_run=args.dry_run)
        action = "Would delete" if args.dry_run else "Deleted"
        print(f"\n{action} {len(deleted)} files.")

        # Also clean old simulations if cleaning all
        if not args.dry_run:
            print("\nAlso cleaning old simulation directories...")
            deleted_dirs = clean_old_simulations(dry_run=False)
            print(f"Deleted {len(deleted_dirs)} directories.")
    else:
        print(f"Cleaning roofline output files for model: {args.model}...")
        model = args.model
        deleted = clean_model_outputs(model_name=model, output_dir=args.output_dir, dry_run=args.dry_run)
        action = "Would delete" if args.dry_run else "Deleted"
        print(f"\n{action} {len(deleted)} files.")


def _cmd_analyze(args):
    """Handle 'analyze' subcommand (default roofline analysis)."""

    # Create custom config if multilevel OI is requested
    config = None
    if hasattr(args, 'multilevel_oi') and args.multilevel_oi:
        config = RooflineConfig(
            use_ops_not_macs=True,
            oi_byte_boundary="cim_port",
            enable_multilevel_oi=True,
            weight_stationary=True
        )

    run_roofline_analysis(
        model_name=args.model,
        macro=args.macro,
        run_simulation=args.simulate,
        output_dir=args.output_dir,
        num_layers=args.num_layers,
        generate_validation=getattr(args, 'validate', False),
        roofline_config=config,
        system=args.system
    )


def _cmd_compare(args):
    """Handle 'compare' subcommand."""
    compare_hardware_rooflines(model_name=args.model, output_dir=args.output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Roofline Analysis for DNN Models on CiM Hardware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  %(prog)s analyze --model resnet18 --macro basic_analog

  # With simulation
  %(prog)s analyze --model gpt2_medium --simulate

  # With simulation and validation report
  %(prog)s analyze --model gpt2_medium --simulate --validate

  # Multi-level OI analysis
  %(prog)s analyze --model gpt2_medium --multilevel-oi

  # Compare architectures
  %(prog)s compare --model gpt2_medium

  # List and clean
  %(prog)s list models
  %(prog)s list macros
  %(prog)s clean --model gpt2_medium --dry-run
  %(prog)s clean --old-sims  # Clean old simulation directories
"""
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # --- 'analyze' subcommand (default roofline analysis) ---
    analyze_parser = subparsers.add_parser("analyze", help="Run roofline analysis for a model")
    _add_common_args(analyze_parser)
    analyze_parser.add_argument("--macro", type=str, default="basic_analog",
                                help="CiM macro architecture (default: basic_analog)")
    analyze_parser.add_argument("--system", type=str, default=None,
                                help="System architecture (default: auto-detects from macro name if available)")
    analyze_parser.add_argument("--simulate", "-s", action="store_true",
                                help="Run actual Timeloop simulations")
    analyze_parser.add_argument("--validate", "-v", action="store_true",
                                help="Generate validation report (requires --simulate)")
    analyze_parser.add_argument("--multilevel-oi", action="store_true",
                                help="Enable multi-level OI analysis (DRAM, GBUF, CIM port)")
    analyze_parser.add_argument("--num-layers", "-n", type=int, default=None,
                                help="Number of layers to analyze (default: all)")
    analyze_parser.set_defaults(func=_cmd_analyze)

    # --- 'compare' subcommand ---
    compare_parser = subparsers.add_parser("compare", help="Compare theoretical rooflines across hardware")
    _add_common_args(compare_parser)
    compare_parser.set_defaults(func=_cmd_compare)

    # --- 'list' subcommand ---
    list_parser = subparsers.add_parser("list", help="List available resources")
    list_parser.add_argument("what", choices=["models", "macros", "outputs"],
                             help="What to list: models, macros, or outputs")
    list_parser.add_argument("--output-dir", "-o", type=str, default=None,
                             help="Output directory (for listing outputs)")
    list_parser.set_defaults(func=_cmd_list)

    # --- 'clean' subcommand ---
    clean_parser = subparsers.add_parser("clean", help="Clean output files")
    clean_parser.add_argument("--model", "-m", type=str, default="gpt2_medium",
                              help="Model to clean outputs for")
    clean_parser.add_argument("--all", "-a", action="store_true",
                              help="Clean all roofline output files")
    clean_parser.add_argument("--old-sims", action="store_true",
                              help="Clean old timestamp-based simulation directories")
    clean_parser.add_argument("--output-dir", "-o", type=str, default=None,
                              help="Output directory")
    clean_parser.add_argument("--dry-run", action="store_true",
                              help="Show what would be deleted without deleting")
    clean_parser.set_defaults(func=_cmd_clean)

    args = parser.parse_args()

    # Run the appropriate command
    if args.command is None:
        parser.print_help()
    else:
        args.func(args)
