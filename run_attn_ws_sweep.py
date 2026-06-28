#!/usr/bin/env python3
import argparse
import csv
import itertools
import math
import shlex
import subprocess
import sys
from pathlib import Path


NCU_METRICS = (
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,"
    "gpu__time_duration.sum"
)
CSV_FIELDS = ["B", "H", "S", "D", "time", "tflops", "tc_utilization"]


def parse_int_list(value):
    return [int(item) for item in value.replace(" ", ",").split(",") if item.strip()]


def parse_float(value):
    value = value.strip().strip('"').replace(",", "")
    if not value or value.lower() in {"n/a", "nan"}:
        return math.nan
    return float(value)


def time_to_ms(value, unit):
    unit = unit.strip().lower()
    if unit in {"ns", "nsec", "nsecond", "nanosecond", "nanoseconds"}:
        return value / 1e6
    if unit in {"us", "usec", "usecond", "microsecond", "microseconds"}:
        return value / 1e3
    if unit in {"ms", "msec", "msecond", "millisecond", "milliseconds"}:
        return value
    if unit in {"s", "sec", "second", "seconds"}:
        return value * 1e3
    return value


def metric_value_from_row(row, metric_name):
    for idx, cell in enumerate(row):
        if cell.strip().strip('"') != metric_name:
            continue
        for value_idx in range(len(row) - 1, idx, -1):
            try:
                value = parse_float(row[value_idx])
            except ValueError:
                continue
            unit = row[value_idx - 1] if value_idx - 1 > idx else ""
            return value, unit
    return None


def metric_value_from_text_line(line, metric_name):
    if metric_name not in line:
        return None
    fields = line.split(metric_name, 1)[1].strip().split()
    if not fields:
        return None
    try:
        value = parse_float(fields[-1])
    except ValueError:
        return None
    unit = fields[-2] if len(fields) >= 2 else ""
    return value, unit


def parse_ncu_output(output):
    time_ms = math.nan
    tc_utilization = math.nan

    for line in output.splitlines():
        time_result = metric_value_from_text_line(line, "gpu__time_duration.sum")
        if time_result is not None:
            value, unit = time_result
            time_ms = time_to_ms(value, unit)
            continue

        tc_result = metric_value_from_text_line(
            line, "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"
        )
        if tc_result is not None:
            value, _unit = tc_result
            tc_utilization = value

    for row in csv.reader(output.splitlines()):
        if not row:
            continue

        time_result = metric_value_from_row(row, "gpu__time_duration.sum")
        if time_result is not None:
            value, unit = time_result
            time_ms = time_to_ms(value, unit)
            continue

        tc_result = metric_value_from_row(
            row, "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"
        )
        if tc_result is not None:
            value, _unit = tc_result
            tc_utilization = value

    return time_ms, tc_utilization


def attention_flops(B, H, S, D):
    return 4.0 * B * H * S * S * D


def format_number(value):
    if math.isnan(value):
        return "nan"
    return f"{value:.6f}"


def compile_shape(args, src, bin_path, B, H, S, D):
    cmd = [
        args.nvcc,
        "-std=c++17",
        f"-arch={args.arch}",
        "-O3",
        str(src),
        "-o",
        str(bin_path),
        "-lcuda",
        f"-DATTN_B={B}",
        f"-DATTN_H={H}",
        f"-DATTN_S={S}",
        f"-DATTN_D={D}",
        f"-DATTN_WARMUP={args.warmup}",
    ]
    print(shlex.join(cmd), flush=True)
    if args.dry_run:
        return True

    completed = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        print(completed.stdout, file=sys.stderr)
        print(
            f"compile failed for B={B} H={H} S={S} D={D} "
            f"with return code {completed.returncode}",
            file=sys.stderr,
        )
        return False
    return True


def profile_shape(args, bin_path, B, H, S, D):
    cmd = [
        args.ncu,
        "--metrics",
        NCU_METRICS,
        "--profile-from-start",
        "on",
        "--launch-skip",
        str(args.launch_skip),
        "--launch-count",
        str(args.launch_count),
        str(bin_path),
    ]
    if args.kernel_name:
        cmd[1:1] = [
            "--kernel-name-base",
            "function",
            "--kernel-name",
            args.kernel_name,
        ]
    if args.ncu_csv:
        cmd[1:1] = ["--csv", "--page", "raw"]

    print(shlex.join(cmd), flush=True)
    if args.dry_run:
        return math.nan, math.nan, math.nan

    completed = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    time_ms, tc_utilization = parse_ncu_output(completed.stdout)

    if completed.returncode != 0 or math.isnan(time_ms) or math.isnan(tc_utilization):
        print(completed.stdout, file=sys.stderr)
    if completed.returncode != 0:
        print(
            f"ncu failed for B={B} H={H} S={S} D={D} "
            f"with return code {completed.returncode}",
            file=sys.stderr,
        )
    elif math.isnan(time_ms) or math.isnan(tc_utilization):
        print(f"failed to parse ncu metrics for B={B} H={H} S={S} D={D}", file=sys.stderr)

    if math.isnan(time_ms):
        tflops = math.nan
    else:
        tflops = attention_flops(B, H, S, D) / (time_ms * 1e-3) / 1e12
    return time_ms, tflops, tc_utilization


def iter_shapes(args):
    return itertools.product(args.batch_sizes, args.num_heads, args.seq_lens, args.head_dims)


def append_row(path, row):
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow(row)


def build_parser():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Sweep attn_ws shapes, collect ncu time/tensor-core utilization, and save CSV."
    )
    parser.add_argument("--batch-sizes", type=parse_int_list, default=[1, 2, 4, 8, 16])
    parser.add_argument("--num-heads", type=parse_int_list, default=[8, 16, 32])
    parser.add_argument("--seq-lens", type=parse_int_list, default=[8192, 16384])
    parser.add_argument("--head-dims", type=parse_int_list, default=[128])
    parser.add_argument("--csv", type=Path, default=script_dir / "attn_ws_ncu_sweep.csv")
    parser.add_argument("--src", type=Path, default=script_dir / "attn_ws.cu")
    parser.add_argument("--bin", type=Path, default=script_dir / "attn_ws_shape")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--launch-skip", type=int, default=10)
    parser.add_argument("--launch-count", type=int, default=1)
    parser.add_argument("--ncu", default="ncu")
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--arch", default="sm_90a")
    parser.add_argument("--kernel-name", default="")
    parser.add_argument("--ncu-csv", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    shapes = list(iter_shapes(args))

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    print(f"num shapes = {len(shapes)}")
    print(f"csv        = {args.csv}")
    print(f"warmup     = {args.warmup}")
    print(f"ncu skip   = {args.launch_skip}")
    print(f"ncu count  = {args.launch_count}")

    for idx, (B, H, S, D) in enumerate(shapes, 1):
        print(f"[{idx}/{len(shapes)}] B={B} H={H} S={S} D={D}", flush=True)
        if compile_shape(args, args.src, args.bin, B, H, S, D):
            time_ms, tflops, tc_utilization = profile_shape(args, args.bin, B, H, S, D)
        else:
            time_ms, tflops, tc_utilization = math.nan, math.nan, math.nan

        row = {
            "B": B,
            "H": H,
            "S": S,
            "D": D,
            "time": format_number(time_ms),
            "tflops": format_number(tflops),
            "tc_utilization": format_number(tc_utilization),
        }
        append_row(args.csv, row)
        print(row, flush=True)


if __name__ == "__main__":
    main()
