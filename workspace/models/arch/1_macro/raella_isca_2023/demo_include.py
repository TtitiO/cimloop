# =====================================================================================================================
import sys
import os

from _tests import scripts
from scripts.notebook_utils import *

MACRO_NAME = "raella_isca_2023"
from scripts import utils as utl
import scripts

os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
from IPython.display import clear_output


def get_energy_breakdown(variables):
    result = utl.parallel_test(
        [
            utl.delayed(utl.quick_run)(
                macro=MACRO_NAME,
                variables=variables,
                tile="raella",
            )
        ]
    )

    analog_compute = ["row_drivers", "output_center_offset_correct", "cim_unit"]
    # if  in result[0].per_component_energy:
    #     analog_compute.append("cim_unit")

    for f in ["combine_per_component_energy", "combine_per_component_area"]:
        callfunc = getattr(result, f)
        callfunc(["adc"], "ADC")
        callfunc(analog_compute, "Analog Compute")
        callfunc(
            [
                "timely_psubbuf",
                "shift_add",
            ],
            "Digital Compute",
        )
        callfunc(
            [
                "input_buffer",
                "inter_macro_network",
                "eDRAM_buf",
                "output_register",
            ],
            "Input + Output Movement",
        )
    result.clear_zero_areas()
    result.clear_zero_energies()
    for r in result:
        r.per_component_area /= (
            32 * 512 * variables["ARRAY_ROWS"] * variables["BITS_PER_CELL"] * 0.032**2
        )
    return result


def diplay_distributions(dp, ax: plt.Axes):
    dp.return_column_sums = True
    x = (
        (torch.randn((10000, dp.chunker.block_size)) * 0.3)
        .clamp(0, 1)
        .to(torch.float64)
    )
    y = (
        (torch.randn((10000, dp.chunker.block_size)) * 0.3)
        .clamp(-1, 1)
        .to(torch.float64)
    )
    z = dp.forward(x, y)
    # Display a histogram of the values
    hist = np.histogram(z.flatten().detach().numpy(), bins=200)
    # Smooth the histogram
    ax.plot(hist[1][1:], np.convolve(hist[0], np.ones(20) / 20, mode="same"))
    ax.set_xlabel("Value at ADC")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Values Seen by ADC")
    minval, maxval = dp.quant_out.get_range()
    if z.min() >= 0:
        maxval -= minval
        minval = 0
    ax.axvline(minval, color="r", linestyle="dashed", linewidth=1)
    ax.axvline(maxval, color="r", linestyle="dashed", linewidth=1)


def run_raella(
    adc_bits: int,
    n_rows: int,
    n_cols: int,
    center_offset: int,
    adaptive_weight_slicing: bool,
    speculation: int,
    device_type: str,
):

    variables = dict(
        SPECULATION_ENABLED=speculation,
        ADC_RESOLUTION=adc_bits,
        ARRAY_ROWS=n_rows,
        ARRAY_COLS=n_cols,
        N_ADC_PER_BANK=n_cols // 64,
    )
    if not adaptive_weight_slicing:
        variables["BITS_PER_CELL"] = 4
    else:
        variables["BITS_PER_CELL"] = 3
    if not center_offset:
        variables["WEIGHT_ENCODING_FUNC"] = "offset_encode_hist"
        variables["CIM_UNIT_WIDTH_CELLS"] = 1

    basepath = "/home/workspace/models/memory_cells/"
    if device_type == "Protonic":
        variables["CELL_CONFIG"] = basepath + "/ecram_demo.cell.yaml"
    else:
        variables["CELL_CONFIG"] = basepath + "/sram_jia_jssc_2020.cell.yaml"
        variables["BITS_PER_CELL"] = 1
    variables["CELL_CONFIG"] = f'"{variables["CELL_CONFIG"]}"'

    result = get_energy_breakdown(variables)
    fig, axs = plt.subplots(1, 3, figsize=(15, 3))
    plt.subplots_adjust(wspace=0.4)  # Increase the space between the plots

    bar_side_by_side(
        {"": r.per_compute("per_component_energy") * 1e15 for r in result},
        ylabel="Energy (fJ/MAC)",
        title=f"Energy Breakdown",
        ax=axs[0],
    )
    bar_side_by_side(
        {"": r.per_component_area * 1e12 for r in result},
        ylabel="Area (F^2/bit)",
        title=f"Area Breakdown",
        ax=axs[1],
    )
    # Remove axs[1] legend
    axs[1].get_legend().remove()
    # Move the axs[0] legend to a global x=-0.1 y=0.5 position
    axs[0].legend(loc="center left", bbox_to_anchor=(-1.1, 0.5))

    input_slicing = tuple([1] * 8)
    weight_slicing = (4, 2, 2) if adaptive_weight_slicing else (4, 4)
    if device_type == "SRAM":
        weight_slicing = (1, 1, 1, 1, 1, 1, 1, 1)
    dp = GenericDatapath(
        n_rows,
        input_slicing,
        weight_slicing,
        input_signed_slices=False,
        weight_signed_slices=center_offset,
        adc_bins=2**adc_bits,
        adc_step_size=1,
    )
    dp.quant_out.signed = True
    diplay_distributions(dp, axs[2])
    s = f"{n_rows}x{n_cols} Array\n{adc_bits}b ADC"
    for text, val in [
        ("Center Offset", center_offset),
        ("Adaptive Weight Slicing", adaptive_weight_slicing),
        ("Speculation", speculation),
    ]:
        if val:
            s += f"\n{text}"
    s += "\nTotal Energy: {:.2f} fJ/MAC".format(result[0].per_compute("energy") * 1e15)
    fig.suptitle(s, x=-0.2, y=0.5)
    # fig.show()


# =====================================================================================================================
# =====================================================================================================================
# =====================================================================================================================

import math
from typing import Any, Optional, Union, Tuple
from torch import nn, Tensor
import torch
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

INT = torch.int32
FLOAT = torch.float32


class BidirectionalSequential(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, x):# -> Union[Tensor, Tuple[Tensor, *Tuple[Any, ...]]]:
        metadata = []
        for module in self:
            x, m = module.forward(x)
            metadata.append(m)
        return x, metadata

    def reverse(self, x: Tensor, *metadata: Any):# -> Tensor:
        for i, module in enumerate(reversed(self)):
            x = module.reverse(x, metadata[-i - 1])
        return x


class BidirectionalModule(nn.Module):
    def forward(self, x: Tensor):# -> Union[Tensor, Tuple[Tensor, *Tuple[Any, ...]]]:
        return x, None

    def reverse(self, x: Tensor, *metadata: Any):# -> Tensor:
        return x


class Slicer(nn.Module):
    def __init__(
        self,
        slices: Tuple[int, ...],
        signed: bool = False,
        two_sided_signed: bool = False,
    ):
        super().__init__()
        assert slices
        self.slices: Tuple[int, ...] = slices
        self.signed: bool = signed
        self.two_sided_signed: bool = two_sided_signed
        if self.signed and self.two_sided_signed:
            raise ValueError("Cannot have two-sided signed and signed")

        masks = []
        shifts = [0]
        for s in slices[::-1]:
            masks.append((1 << s) - 1)
            shifts.append(s + shifts[-1])

        self.masks = torch.tensor(masks[::-1], dtype=INT)
        self.shifts = torch.tensor(shifts[-2::-1], dtype=INT)
        self.shifts_float = (1 << self.shifts).to(dtype=FLOAT)

    def forward(self, x: Tensor):# -> Union[Tensor, Tuple[Tensor, *Tuple[Any, ...]]]:
        self.masks = self.masks.to(x.device)
        self.shifts = self.shifts.to(x.device)
        m = self.masks.reshape(-1, *([1] * len(x.shape)))
        s = self.shifts.reshape(-1, *([1] * len(x.shape)))
        if self.signed:
            return ((x.abs() >> s) & m) * x.sign().unsqueeze(0)
        if self.two_sided_signed:
            x_pos = (torch.clamp(x, 0, None) >> s) & m
            x_neg = (torch.clamp(-x, 0, None) >> s) & m
            return torch.cat([x_pos, x_neg], dim=0)
        return (x >> s) & m

    def pad(self, x: Tensor):# -> Tensor:
        return x.unsqueeze(0)

    def reverse(self, x: Tensor, *metadata: Any):# -> Tensor:
        if self.two_sided_signed:
            x_pos, x_neg = torch.chunk(x, 2, dim=0)
            x = x_pos - x_neg

        if torch.is_floating_point(x):
            self.shifts_float = self.shifts_float.to(x.device)
            s = self.shifts_float.reshape(-1, *([1] * (len(x.shape) - 1)))
            return (x * s).sum(dim=0)

        self.masks = self.masks.to(x.device)
        s = self.shifts.reshape(-1, *([1] * (len(x.shape) - 1)))
        return (x << s).sum(dim=0)


class Quantizer(BidirectionalModule):
    def __init__(
        self,
        n_steps: int,
        signed: bool,
        step_size: Optional[float] = None,
        dim: Union[int, Tuple[int]] = (-1),
    ):
        super().__init__()
        self.n_steps = n_steps
        self.signed = signed
        self.dim = dim
        self.step_size = step_size

        if self.n_steps % 2 == 0 and self.signed:
            raise ValueError("n_steps must be odd to include 0")

    def offset(self, x: Tensor):# -> Union[Tensor, Tuple[Tensor, *Tuple[Any, ...]]]:
        if self.signed or torch.min(x) >= 0:
            return x, None
        offs = x.amin(dim=self.dim, keepdim=True)
        return x - offs, offs.squeeze(self.dim)

    def get_sum(self, x: Tensor):# -> Tensor:
        return x.sum(dim=self.dim)

    def forward(self, x: Tensor):# -> Union[Tensor, Tuple[Tensor, *Tuple[Any, ...]]]:
        if self.step_size is None:
            maxval = x.amax(dim=self.dim, keepdim=True)
            minval = (
                x.amin(dim=self.dim, keepdim=True)
                if self.signed
                else torch.zeros_like(maxval)
            )
            max_abs = torch.max(minval.abs(), maxval.abs())
            step_size = max_abs / (self.n_steps - 1) * (2 if self.signed else 1)
        else:
            step_size = torch.tensor(self.step_size, dtype=FLOAT, device=x.device)

        min_allowed, max_allowed = self.get_range(step_size)

        quantized = torch.round(x.clamp(min_allowed, max_allowed) / step_size).to(
            dtype=INT
        )
        step_size = step_size.squeeze(self.dim) if self.step_size is None else step_size
        return quantized, step_size

    def reverse(self, x: Tensor, *metadata: Any):# -> Tensor:
        step_size = metadata[0]
        return x * step_size

    def get_range(self, step_size: Optional[torch.Tensor] = None):
        if step_size is None:
            step_size = self.step_size

        if step_size is None:
            raise ValueError("Step size not set")

        if not isinstance(step_size, torch.Tensor):
            step_size = torch.tensor(step_size)

        min_allowed = (
            -step_size * (self.n_steps - 1) / 2
            if self.signed
            else torch.zeros_like(step_size)
        )
        max_allowed = min_allowed + step_size * self.n_steps
        return min_allowed, max_allowed


class Chunker(BidirectionalModule):
    def __init__(self, block_size: int):
        super().__init__()
        self.block_size = block_size

    def forward(self, x: Tensor):# -> Union[Tensor, Tuple[Tensor, *Tuple[Any, ...]]]:
        # Split into blocks of size block_size. Pad to block_size
        n_blocks = math.ceil(x.shape[-1] / self.block_size)
        n_per_block = math.ceil(x.shape[-1] / n_blocks)
        residual = n_blocks * n_per_block - x.shape[-1]

        padded = torch.nn.functional.pad(x, (0, residual))
        x = padded.reshape(*padded.shape[:-1], n_blocks, n_per_block)
        x = torch.moveaxis(x, -2, 0)
        return x

    def reverse(self, x: Tensor, *metadata: Any):# -> Tensor:
        return x.sum(dim=0)


class GaussianNoise(BidirectionalModule):
    def __init__(self, std: float, cache_shape: bool = False):
        super().__init__()
        self.std = std
        self.cache = {}
        self.cache_shape = cache_shape

    def forward(self, x: Tensor):# -> Union[Tensor, Tuple[Tensor, *Tuple[Any, ...]]]:
        x = x.to(dtype=FLOAT)
        cachekey = (x.shape, x.device, x.dtype)
        r = self.cache.get(cachekey, torch.randn_like(x))
        if self.cache_shape:
            self.cache[cachekey] = r
        return x * (1 + r * self.std)


class PassThrough(BidirectionalModule):
    def forward(self, x: Tensor):# -> Union[Tensor, Tuple[Tensor, *Tuple[Any, ...]]]:
        return x, None

    def reverse(self, x: Tensor, *metadata: Any):# -> Tensor:
        return x

    def offset(self, x: Tensor):# -> Union[Tensor, Tuple[Tensor, *Tuple[Any, ...]]]:
        return x, torch.zeros_like(x)

    def reverse_offset(self, x: Tensor, *metadata: Any):# -> Tensor:
        return x

    def pad(self, x: Tensor):# -> Tensor:
        return x


class GenericDatapath:
    def __init__(
        self,
        array_rows: int,
        input_slices: Tuple[int, ...],
        weight_slices: Tuple[int, ...],
        input_signed_slices: bool,
        weight_signed_slices: bool,
        adc_bins: int,
        adc_step_size: Optional[float] = None,
        two_sided_signed_inputs: bool = False,
        two_sided_signed_weights: bool = False,
        digitize: bool = False,
        cache_weights: bool = False,
    ):
        input_bits = sum(input_slices)
        weight_bits = sum(weight_slices)

        input_signed = input_signed_slices or two_sided_signed_inputs
        weight_signed = weight_signed_slices or two_sided_signed_weights
        self.quant_in = Quantizer(2**input_bits - input_signed, input_signed)
        self.quant_weight = Quantizer(2**weight_bits - weight_signed, weight_signed)

        self.slicer_in = Slicer(
            input_slices, input_signed_slices, two_sided_signed_inputs
        )
        self.slicer_weight = Slicer(
            weight_slices, weight_signed_slices, two_sided_signed_weights
        )

        self.chunker = Chunker(array_rows)

        outputs_signed = input_signed_slices or weight_signed_slices
        # When outputs get to the ADC, dim -3 is sample, -2 is chunk, -1 is always size 1
        self.quant_out = None
        if adc_bins is not None:
            self.quant_out = Quantizer(
                adc_bins - outputs_signed,
                outputs_signed,
                adc_step_size,
                dim=(-1, -2, -3),
            )

        self.input_noiser = None
        self.weight_noiser = None
        self.output_noiser = None
        self.digitize = digitize
        self.cache_weights = cache_weights
        self._cached_weights = None
        self.return_column_sums = False

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # ============ Forward pass ============
        # Quantize inputs and weights to 8 bits
        x_resign, x_offs = self.quant_in.offset(x)
        x_quant, x_scale = self.quant_in(x_resign)
        x_sliced = self.slicer_weight.pad(self.slicer_in(x_quant))
        x_chunked = self.chunker(x_sliced)
        x_chunked = (
            self.input_noiser(x_chunked) if self.input_noiser is not None else x_chunked
        )
        x_chunked = x_chunked.round() if self.digitize else x_chunked

        if self._cached_weights is None:
            y_resign, y_offs = self.quant_weight.offset(y)
            y_quant, y_scale = self.quant_weight(y_resign)
            y_sliced = self.slicer_weight(self.slicer_in.pad(y_quant))
            y_chunked = self.chunker(y_sliced)
            y_chunked = (
                self.weight_noiser(y_chunked)
                if self.weight_noiser is not None
                else y_chunked
            )
            y_chunked = y_chunked.round() if self.digitize else y_chunked
        else:
            y_chunked, y_offs, y_scale = self._cached_weights

        if self.cache_weights:
            self._cached_weights = y_chunked, y_offs, y_scale

        # Perform dot products within each CiM array
        z = (x_chunked.unsqueeze(-2) @ y_chunked.unsqueeze(-1)).squeeze(-1)
        z = self.output_noiser(z) if self.output_noiser is not None else z
        if self.return_column_sums:
            return z.squeeze(-1)

        # ============ Reverse pass ============
        if self.quant_out is not None:
            z = self.quant_out.reverse(*self.quant_out(z))  # ADC
        z_sliced = self.chunker.reverse(z)  # Sum across arrays
        z_quant = self.slicer_in.reverse(
            self.slicer_weight.reverse(z_sliced)
        )  # Merge slices

        # Integer outputs -> floating point outputs
        z = self.quant_in.reverse(self.quant_weight.reverse(z_quant, y_scale), x_scale)

        z += (
            x_offs.flatten() * y.sum(dim=-1) + y_offs.flatten() * x.sum(dim=-1)
        ).unsqueeze(-1) - x_offs * y_offs * x.shape[-1]
        return z

    def process_inputs(self, x: torch.Tensor, include_nonidealities: bool = True):
        x_resign, x_offs = self.quant_in.offset(x)
        x_quant, x_scale = self.quant_in(x_resign)
        x_sliced = self.slicer_weight.pad(self.slicer_in(x_quant))
        if self.input_noiser is not None and include_nonidealities:
            x_sliced = self.input_noiser(x_sliced)
        return x_sliced, (self.quant_in.get_sum(x), x_offs, x_scale)

    def process_weights(self, y: torch.Tensor, include_nonidealities: bool = True):
        y_resign, y_offs = self.quant_weight.offset(y)
        y_quant, y_scale = self.quant_weight(y_resign)
        y_sliced = self.slicer_weight(self.slicer_in.pad(y_quant))
        if self.weight_noiser is not None and include_nonidealities:
            y_sliced = self.weight_noiser(y_sliced)
        return y_sliced, (self.quant_weight.get_sum(y), y_offs, y_scale)

    def process_outputs(
        self,
        z: torch.Tensor,
        x_meta: Tuple[Tensor, Tensor],
        y_meta: Tuple[Tensor, Tensor],
        x_n: int,
        include_nonidealities: bool = True,
    ):
        x_sum, x_offs, x_scale = x_meta
        y_sum, y_offs, y_scale = y_meta

        if include_nonidealities:
            z = self.output_noiser(z) if self.output_noiser is not None else z
            z = (
                self.quant_out.reverse(*self.quant_out(z))
                if self.quant_out is not None
                else z
            )
        z = self.chunker.reverse(z)  # Sum across arrays
        z = self.slicer_in.reverse(self.slicer_weight.reverse(z))

        zsx = (-1,) + (1,) * (z.dim() - 1)
        zsy = (1,) + (-1,) + (1,) * (z.dim() - 2)
        x_scale, x_sum = x_scale.reshape(zsx), x_sum.reshape(zsx)
        y_scale, y_sum = y_scale.reshape(zsy), y_sum.reshape(zsy)

        z = self.quant_in.reverse(z, x_scale)
        z = self.quant_weight.reverse(z, y_scale)

        # z += ((x_offs * y_sum + y_offs * x_sum) - x_offs * y_offs * x_n)
        if x_offs is not None:
            x_offs = x_offs.reshape(zsx)
            z += x_offs * y_sum
        if y_offs is not None:
            y_offs = y_offs.reshape(zsy)
            z += y_offs * x_sum
        if x_offs is not None and y_offs is not None:
            z -= x_offs * y_offs * x_n
        return z

    def chunk(self, x: torch.Tensor):
        return self.chunker(x)

    def dechunk(self, x: torch.Tensor):
        return self.chunker.reverse(x)


class ConvDatapath(torch.nn.Module):
    def __init__(self, conv_layer: nn.Conv2d, datapath: GenericDatapath):
        super().__init__()
        self.enabled = True
        self.include_nonidealities = True
        self.conv_layer = conv_layer
        self.datapath = datapath
        self.datapath.quant_in.dim = (-3, -2, -1)
        self.datapath.quant_weight.dim = (-3, -2, -1)

    def forward(self, x: torch.Tensor):# -> torch.Tensor:
        if not self.enabled:
            return self.conv_layer(x)

        # Flatten the convolution operation
        batch_size, in_channels, height, width = x.shape
        out_channels, _, kernel_height, kernel_width = self.conv_layer.weight.shape
        stride = self.conv_layer.stride
        pad = self.conv_layer.padding
        h = (height + 2 * pad[0] - kernel_height) // stride[0] + 1
        w = (width + 2 * pad[1] - kernel_width) // stride[1] + 1

        x_padded = torch.nn.functional.pad(x, (pad[1], pad[1], pad[0], pad[0]))

        x_processed, x_meta = self.datapath.process_inputs(
            x_padded, include_nonidealities=self.include_nonidealities
        )
        y_processed, y_meta = self.datapath.process_weights(
            self.conv_layer.weight, include_nonidealities=self.include_nonidealities
        )

        x_processed, y_processed = x_processed.to(dtype=FLOAT), y_processed.to(
            dtype=FLOAT
        )

        # Flatten the last 3 dimensions of y_processed
        y_flat = y_processed.reshape(
            *y_processed.shape[:-4], 1, y_processed.shape[-4], -1
        )
        y_chunked = self.datapath.chunk(y_flat)

        new_shape = [
            max(a, b)
            for a, b in list(zip((1,) + x_processed.shape, y_chunked.shape))[:-3]
        ] + [batch_size, out_channels, h, w]

        # out = torch.zeros(batch_size, out_channels, h, w, dtype=FLOAT, device=x.device)
        out = torch.zeros(tuple(new_shape), dtype=FLOAT, device=x.device)
        for i in range(h):
            for j in range(w):
                x_slice = x_processed[
                    ...,
                    i * stride[0] : i * stride[0] + kernel_height,
                    j * stride[1] : j * stride[1] + kernel_width,
                ]
                x_flat = x_slice.reshape(*x_slice.shape[:-3], 1, -1)
                x_chunked = self.datapath.chunk(x_flat)
                z = (
                    (x_chunked.unsqueeze(-2) @ y_chunked.unsqueeze(-1))
                    .squeeze(-1)
                    .squeeze(-1)
                )
                out[..., i, j] = z

        out = self.datapath.process_outputs(out, x_meta, y_meta, x_flat.shape[-1])

        # import matplotlib.pyplot as plt
        # import numpy as np
        # conv_out = self.conv_layer(x)
        # conv_out = conv_out.detach().cpu().numpy().flatten()
        # out2 = out.detach().cpu().numpy().flatten()

        # plt.scatter(conv_out, out2, alpha=0.5)
        # plt.xlabel('Conv Output')
        # plt.ylabel('Out')
        # plt.title('Scatter plot of Conv Output vs Out')
        # plt.show()

        # r_squared = np.corrcoef(conv_out, out2)[0, 1] ** 2
        # if r_squared < 0.99:
        #     assert False

        # out[..., i, j] = self.datapath.process_outputs(z.unsqueeze(-1), x_meta, y_meta).reshape(batch_size, out_channels)

        print(f"Completed layer {self.conv_layer}")

        return out
