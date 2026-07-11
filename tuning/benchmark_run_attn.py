#!/usr/bin/env python3
import argparse
import itertools
import json
import os
import re
import subprocess
from argparse import Namespace
from pathlib import Path
from collections import defaultdict

from solver import solve

def solution_cast(solutions):
    bn_to_solutions = defaultdict(list)
    for solution in solutions:
        bn_to_solutions[solution["BN"]].append(solution)

    keep_ids = set()
    for bn_solutions in bn_to_solutions.values():
        max_num_smem_p = max(solution["NumSmemP"] for solution in bn_solutions)
        if max_num_smem_p % 4 == 0:
            threshold = max_num_smem_p
        else:
            multiples_of_4 = [
                solution["NumSmemP"]
                for solution in bn_solutions
                if solution["NumSmemP"] % 4 == 0
            ]
            if not multiples_of_4:
                continue
            threshold = max(multiples_of_4)

        for solution in bn_solutions:
            if solution["NumSmemP"] <= threshold:
                keep_ids.add(id(solution))

    return [
        solution
        for solution in solutions
        if id(solution) in keep_ids
    ]

def get_configs(B, H, S, D, kernel_idx=2, bn_min=112):
    num_thread_producer = 1 * 128
    num_thread_consumer = 2 * 128
    # count = 0
    configs = []
    for num_smem in range(1, 3):
        solutions = solution_cast(solve(D, num_smem, KernelIdx=kernel_idx))

        for solution in solutions:
            # if S % solution["BM"] != 0 or S % solution["BN"] != 0 or solution["BN"] < bn_min:
            if solution["BN"] < bn_min or solution["BN"] > 256:
                continue

            for producer_reg in [24, 32, 48]:
                for consumer_reg in [184, 192, 200, 208, 216, 224, 232, 240]:
                    n = solution['RepeatN']
                    num_smp = solution['NumSmemP']
                    BM, BN = solution["BM"], solution["BN"]

                    max_reg = num_thread_producer * producer_reg + num_thread_consumer * consumer_reg
                    consumer_reg_max = solution["EstimatedRegsPerThread"]
                    # print(max_reg, consumer_reg_max)

                    cfg = {
                            "B": B, "H": H, "S": S, "D": D,
                            "BM": BM, "BN": BN,
                            "NUM_SMEM": num_smem,
                            "PRODUCER_REG_DEALLOC": producer_reg, "CONSUMER_REG_ALLOC": consumer_reg,
                            "P_SMEM_K_TILES": num_smp, "KERNEL_IDX": kernel_idx, "NUM_THREADS": 384,
                            "SmemSize": solution['SmemSize'], "RegSize": solution['RegSize']
                        }
                    if max_reg < 65536 and consumer_reg_max <= consumer_reg:
                        if kernel_idx == 1 and n == 1 and num_smp == 0:
                            configs.append(cfg)
                        elif kernel_idx == 2 and n == 1 and num_smp != 0:
                            configs.append(cfg)
                            # print(solution)
                            # count += 1
                        elif kernel_idx == 3 and n != 1 and num_smp != 0:
                            configs.append(cfg)
                            # print(solution)
                            # count += 1
    # print(count)
    return configs



DEFAULTS = [
    {
        "B": 1,
        "H": 1,
        "S": 1024,
        "D": 64,
        "BM": 256,
        "BN": 128,
        "NUM_SMEM": 1,
        "PRODUCER_REG_DEALLOC": 24,
        "CONSUMER_REG_ALLOC": 240,
        "P_SMEM_K_TILES": 4,
        "KERNEL_IDX": 3,
        "NUM_THREADS": 384,
    },
]


KERNEL_NAMES = {
    1: "runAttnWS2StageBaselineKernel",
    2: "runAttnWS2StageSmemPKernel",
    3: "runAttnWSForNSmemPKernel",
}


PTXAS_PERF_LOSS_RE = re.compile(
    r"ptxas\s+info\s*:\s*\(C75\d{2}\).*Potential Performance Loss",
    re.IGNORECASE,
)


PARAM_ORDER = tuple(DEFAULTS[0].keys())


def replace_constexpr(source: str, name: str, value: int, ctype: str = "int") -> str:
    pattern = rf"(constexpr\s+{re.escape(ctype)}\s+{re.escape(name)}\s*=\s*)[^;]+;"
    repl = rf"\g<1>{value};"
    source, count = re.subn(pattern, repl, source, count=1)
    if count != 1:
        raise RuntimeError(f"Could not find constexpr {ctype} {name} in source")
    return source


def config_tag(params: dict[str, int]) -> str:
    return (
        f"k{params['KERNEL_IDX']}_b{params['B']}_h{params['H']}_s{params['S']}_d{params['D']}"
        f"_bm{params['BM']}_bn{params['BN']}_smem{params['NUM_SMEM']}"
        f"_p{params['P_SMEM_K_TILES']}_prd{params['PRODUCER_REG_DEALLOC']}"
        f"_cra{params['CONSUMER_REG_ALLOC']}_nt{params['NUM_THREADS']}"
    )


def config_paths(build_dir: Path, params: dict[str, int]) -> tuple[Path, Path]:
    generated = build_dir / f"attn_test_{config_tag(params)}.cu"
    return generated, generated.with_suffix("")


def make_source(source_path: Path, build_dir: Path, params: dict[str, int], force: bool) -> Path:
    generated, _ = config_paths(build_dir, params)
    if generated.exists() and not force:
        print(f"[cache] reuse source: {generated}")
        return generated

    text = source_path.read_text()
    for name in (
        "BM",
        "BN",
        "B",
        "H",
        "S",
        "D",
        "NUM_SMEM",
        "P_SMEM_K_TILES",
        "KERNEL_IDX",
        "NUM_THREADS",
    ):
        text = replace_constexpr(text, name, params[name])

    text = replace_constexpr(text, "PRODUCER_REG_DEALLOC", params["PRODUCER_REG_DEALLOC"], "uint32_t")
    text = replace_constexpr(text, "CONSUMER_REG_ALLOC", params["CONSUMER_REG_ALLOC"], "uint32_t")

    build_dir.mkdir(parents=True, exist_ok=True)
    generated.write_text(text)
    print(f"[gen] wrote source: {generated}")
    return generated


def run_cmd(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str] | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        capture_output=capture,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate runAttn template parameters, compile attn_test.cu, and run benchmark_attn."
    )
    tuning_dir = Path(__file__).resolve().parent
    parser.add_argument("--source", type=Path, default=tuning_dir / "attn_test.cu")
    parser.add_argument("--build-dir", type=Path, default=tuning_dir / "build_run_attn")
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--arch", default="sm_90a")
    parser.add_argument("--device", default=None, help="CUDA_VISIBLE_DEVICES value, for example 0 or 1")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--force-generate", action="store_true", help="Regenerate cached .cu sources")
    parser.add_argument("--force-compile", action="store_true", help="Recompile even when cached binary exists")
    parser.add_argument("--keep-going", action="store_true", help="Continue a sweep after a failed config")

    for name in PARAM_ORDER:
        parser.add_argument(f"--{name.lower()}", type=int, nargs="+", default=None)

    return parser.parse_args()


def param_grid(args: argparse.Namespace) -> list[dict[str, int]]:
    override_names = [name for name in PARAM_ORDER if getattr(args, name.lower()) is not None]
    if not override_names:
        return [dict(config) for config in DEFAULTS]

    override_values = [getattr(args, name.lower()) for name in override_names]
    configs = []
    for base_config in DEFAULTS:
        for combo in itertools.product(*override_values):
            config = dict(base_config)
            config.update(zip(override_names, combo))
            configs.append(config)
    return configs


def validate_params(params: dict[str, int]) -> None:
    kernel_idx = params["KERNEL_IDX"]
    if kernel_idx not in KERNEL_NAMES:
        raise ValueError(f"--kernel_idx must be one of {sorted(KERNEL_NAMES)}")
    if params["S"] % params["BM"] != 0:
        raise ValueError("--s must be divisible by --bm because grid.x uses S / BM")
    if params["S"] % params["BN"] != 0:
        raise ValueError("--s must be divisible by --bn because the kernels iterate K/V tiles by BN")
    if params["D"] % 64 != 0:
        raise ValueError("--d must be divisible by 64 because create_tensor_map requires global_width % 64 == 0")
    if kernel_idx == 3 and params["BM"] % 128 != 0:
        raise ValueError("--bm must be divisible by 128 for attnWSKForNSmemPKernel")


def print_params(params: dict[str, int]) -> None:
    print("runAttn template parameters:")
    print("  " + ", ".join(f"{name}={value}" for name, value in params.items()))
    print(f"selected kernel: {params['KERNEL_IDX']} ({KERNEL_NAMES[params['KERNEL_IDX']]})")


def compile_config(args: argparse.Namespace, source_path: Path, generated: Path, exe: Path) -> None:
    if exe.exists() and not args.force_compile:
        print(f"[cache] reuse binary: {exe}")
        return

    if not generated.exists():
        raise RuntimeError(f"Generated source is missing: {generated}")

    compile_cmd = [
        args.nvcc,
        "-std=c++17",
        f"-arch={args.arch}",
        "-O3",
        str(generated),
        "-o",
        str(exe),
        "-lcuda",
        "-Xptxas=-v",
    ]
    print("+ " + " ".join(compile_cmd), flush=True)
    result = subprocess.run(
        compile_cmd,
        cwd=source_path.parent,
        text=True,
        capture_output=True,
    )
    compile_output = "\n".join(
        part
        for part in (result.stdout.strip(), result.stderr.strip())
        if part
    )
    if compile_output:
        print(compile_output)

    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            compile_cmd,
            output=result.stdout,
            stderr=result.stderr,
        )

    match = PTXAS_PERF_LOSS_RE.search(compile_output)
    if match:
        if exe.exists():
            exe.unlink()
        raise RuntimeError(f"Discard compile due to ptxas warning: {match.group(0)}")


def run_config(args: argparse.Namespace, source_path: Path, exe: Path) -> str:
    env = os.environ.copy()
    if args.device is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.device

    result = run_cmd([str(exe)], cwd=source_path.parent, env=env, capture=True)
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    print("benchmark_attn output:")
    print(stdout if stdout else "<empty>")
    if stderr:
        print("stderr:")
        print(stderr)

    return stdout


def parse_benchmark_output(output: str) -> dict[str, float]:
    for line in reversed(output.splitlines()):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 8:
            continue
        try:
            return {
                "B": int(parts[0]),
                "H": int(parts[1]),
                "S": int(parts[2]),
                "D": int(parts[3]),
                "BM": int(parts[4]),
                "BN": int(parts[5]),
                "avg_ms": float(parts[6]),
                "tflops": float(parts[7]),
            }
        except ValueError:
            continue
    raise ValueError(f"Could not parse benchmark_attn output: {output!r}")


def benchmark_get_configs(
    B: int,
    H: int,
    S: int,
    D: int,
    kernel_idx: int = 2,
    bn_min: int = 112,
    rank: int = 10,
    source: Path | None = None,
    build_dir: Path | None = None,
    nvcc: str = "nvcc",
    arch: str = "sm_90a",
    device: str | None = None,
    force_generate: bool = False,
    force_compile: bool = False,
    keep_going: bool = True,
    output_path: Path | None = None,
) -> list[dict]:
    if rank <= 0:
        raise ValueError("rank must be positive")

    tuning_dir = Path(__file__).resolve().parent
    source_path = (source or tuning_dir / "attn_test.cu").resolve()
    build_dir = (build_dir or tuning_dir / "build_run_attn").resolve()
    build_dir.mkdir(parents=True, exist_ok=True)

    args = Namespace(
        nvcc=nvcc,
        arch=arch,
        device=device,
        force_generate=force_generate,
        force_compile=force_compile,
    )

    configs = get_configs(B, H, S, D, kernel_idx=kernel_idx, bn_min=bn_min)
    top_results: list[dict] = []
    failed_results: list[dict] = []
    print(f"generated configs: {len(configs)}")

    for idx, params in enumerate(configs, 1):
        print(f"\n=== autotune config {idx}/{len(configs)} ===")
        try:
            validate_params(params)
            print_params(params)

            generated, exe = config_paths(build_dir, params)
            if exe.exists() and not force_compile:
                print(f"[cache] found binary before source generation: {exe}")
            else:
                generated = make_source(source_path, build_dir, params, force_generate)
            print(f"source: {generated}")
            print(f"binary: {exe}")

            compile_config(args, source_path, generated, exe)
            output = run_config(args, source_path, exe)
            metrics = parse_benchmark_output(output)
            result = {
                "config": dict(params),
                "metrics": metrics,
                "output": output,
            }

            top_results.append(result)
            top_results.sort(key=lambda item: item["metrics"]["tflops"], reverse=True)
            del top_results[rank:]

            best = top_results[0]
            print(
                f"[best] tflops={best['metrics']['tflops']:.3f}, "
                f"avg_ms={best['metrics']['avg_ms']:.6f}, "
                f"config={config_tag(best['config'])}"
            )
        except Exception as exc:
            failed = {"config": dict(params), "error": str(exc)}
            failed_results.append(failed)
            print(f"[failed] {exc}")
            if not keep_going:
                raise

    if output_path is None:
        output_path = build_dir / ".." / "results" / (
            f"top{rank}_k{kernel_idx}_b{B}_h{H}_s{S}_d{D}_bnmin{bn_min}.json"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rank": rank,
        "total_configs": len(configs),
        "failed_count": len(failed_results),
        "top_results": top_results,
        "failed_results": failed_results,
    }
    output_path.write_text(json.dumps(payload, indent=2))

    print(f"\n=== top {rank} ===")
    for idx, result in enumerate(top_results, 1):
        metrics = result["metrics"]
        print(
            f"{idx}. tflops={metrics['tflops']:.3f}, "
            f"avg_ms={metrics['avg_ms']:.6f}, "
            f"config={config_tag(result['config'])}"
        )
    print(f"saved: {output_path}")
    return top_results


def main() -> None:
    args = parse_args()
    source_path = args.source.resolve()
    build_dir = args.build_dir.resolve()
    build_dir.mkdir(parents=True, exist_ok=True)

    configs = param_grid(args)
    summaries: list[tuple[dict[str, int], str, str]] = []
    print(f"configs: {len(configs)}")

    for idx, params in enumerate(configs, 1):
        print(f"\n=== config {idx}/{len(configs)} ===")
        try:
            validate_params(params)
            print_params(params)

            generated, exe = config_paths(build_dir, params)
            if exe.exists() and not args.force_compile:
                print(f"[cache] found binary before source generation: {exe}")
            else:
                generated = make_source(source_path, build_dir, params, args.force_generate)
            print(f"source: {generated}")
            print(f"binary: {exe}")

            compile_config(args, source_path, generated, exe)

            if args.compile_only:
                summaries.append((params, "compiled", ""))
                continue

            output = run_config(args, source_path, exe)
            summaries.append((params, "ok", output))
        except Exception as exc:
            summaries.append((params, "failed", str(exc)))
            if not args.keep_going:
                raise
            print(f"[failed] {exc}")

    print("\n=== summary ===")
    for params, status, output in summaries:
        result_line = output.splitlines()[-1] if output else ""
        print(f"{status}: {config_tag(params)}")
        if result_line:
            print(f"  result: {result_line}")


if __name__ == "__main__":
    B = 1
    H = 16
    S = 32768
    bn_min = 160

    for D in [64, 128]:
        if D == 128:
            bn_min = 96
        # for cfg in get_configs(B, H, S, D, kernel_idx=3, bn_min=bn_min):
        #     print(cfg)
        for kernel_idx in range(1, 4):
            benchmark_get_configs(
                B,
                H,
                S,
                D,
                kernel_idx=kernel_idx,
                rank=15,
                bn_min=bn_min,
            )

    # main()
