#!/usr/bin/env python3
"""Configure, compile, and run attn_test.cu.

Every kernel parameter accepts one or more values.  Supplying multiple values
builds the Cartesian product, which makes small parameter sweeps convenient.
"""

import json
import os
import re
import subprocess
from pathlib import Path

from solver import keep_solve, radical_slove

CONFIG_NAMES = (
    "BM",
    "BN",
    "B",
    "H",
    "S",
    "D",
    "NUM_SMEM",
    "PRODUCER_REG_DEALLOC",
    "CONSUMER_REG_ALLOC",
    "P_SMEM_K_TILES",
    "Q_REG_K_TILES",
    "NUM_THREADS",
    "KERNEL_IDX",
)

PTXAS_PERF_LOSS_RE = re.compile(
    r"ptxas\s+info\s*:\s*\(C75\d{2}\).*Potential Performance Loss",
    re.IGNORECASE,
)

KERNEL_NAMES = {
    1: "runAttnWSBaselineKernel",
    2: "runAttnWS2StageKernel",
    3: "runAttnWSForNKernel",
}

SOURCE_DIR = Path(__file__).resolve().parent


def config_tag(params: dict[str, int]) -> str:
    return (
        f"k{params['KERNEL_IDX']}_b{params['B']}_h{params['H']}"
        f"_s{params['S']}_d{params['D']}_bm{params['BM']}_bn{params['BN']}"
        f"_smem{params['NUM_SMEM']}_p{params['P_SMEM_K_TILES']}"
        f"_q{params['Q_REG_K_TILES']}_prd{params['PRODUCER_REG_DEALLOC']}"
        f"_cra{params['CONSUMER_REG_ALLOC']}_nt{params['NUM_THREADS']}"
    )


def config_paths(build_dir: Path, params: dict[str, int]) -> tuple[Path, Path]:
    generated = build_dir / f"attn_qk_reg_{config_tag(params)}.cu"
    return generated, generated.with_suffix("")


def replace_constexpr(source: str, name: str, value: int, ctype: str = "int") -> str:
    pattern = rf"(constexpr\s+{re.escape(ctype)}\s+{re.escape(name)}\s*=\s*)[^;]+;"
    source, count = re.subn(pattern, rf"\g<1>{value};", source, count=1)
    if count != 1:
        raise RuntimeError(f"Could not find constexpr {ctype} {name} in source")
    return source


def make_source(source: str, generated: Path, params: dict[str, int]) -> None:
    if generated.exists():
        print(f"[cache] reuse source: {generated}")
        return

    for name in CONFIG_NAMES:
        ctype = "uint32_t" if name in {
            "PRODUCER_REG_DEALLOC", "CONSUMER_REG_ALLOC"
        } else "int"
        source = replace_constexpr(source, name, params[name], ctype)

    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_text(source)


def compile_config(args: dict[str, str], generated: Path, exe: Path) -> None:
    if exe.exists():
        print(f"[cache] reuse binary: {exe}")
        return
    cmd = [
        args["nvcc"], "-std=c++17", "-arch={}".format(args["arch"]), "-O3",
        "-I", str(SOURCE_DIR), str(generated), "-o", str(exe),
        "-lcuda", "-Xptxas=-v",
    ]
    print("+ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=SOURCE_DIR, text=True, capture_output=True)
    output = "\n".join(x for x in (result.stdout.strip(), result.stderr.strip()) if x)
    if output:
        print(output)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, cmd, output=result.stdout, stderr=result.stderr)
    warning = PTXAS_PERF_LOSS_RE.search(output)
    if warning:
        exe.unlink(missing_ok=True)
        raise RuntimeError(f"discarded binary due to ptxas warning: {warning.group(0)}")


def run_config(args: dict[str, str], exe: Path) -> str:
    env = os.environ.copy()
    if args["device"] is not None:
        env["CUDA_VISIBLE_DEVICES"] = args["device"]
    print(f"+ {exe}", flush=True)
    result = subprocess.run(
        [str(exe)], cwd=SOURCE_DIR, env=env, check=True,
        text=True, capture_output=True,
    )
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip())
    return result.stdout.strip()


def print_params(params: dict[str, int]) -> None:
    print("runAttn template parameters:")
    print("  " + ", ".join(f"{name}={value}" for name, value in params.items()))
    print(f"selected kernel: {params['KERNEL_IDX']} ({KERNEL_NAMES[params['KERNEL_IDX']]})")


def validate_params(params: dict[str, int]) -> None:
    kernel_idx = params["KERNEL_IDX"]
    if kernel_idx not in KERNEL_NAMES:
        raise ValueError(f"--kernel_idx must be one of {sorted(KERNEL_NAMES)}")
    if params["D"] % 64 != 0:
        raise ValueError("--d must be divisible by 64 because create_tensor_map requires global_width % 64 == 0")
    if kernel_idx == 3 and params["BM"] % 128 != 0:
        raise ValueError("--bm must be divisible by 128 for attnWSKForNSmemPKernel")


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


def get_configs(B, H, S, D, kernel_idx=1, bn_min=112, solve_func=keep_solve):
    num_thread_producer = 1 * 128
    num_thread_consumer = 2 * 128
    # count = 0
    configs = []
    solutions = solve_func(D, KernelIdx=kernel_idx)

    for solution in solutions:
        if solution["BN"] < bn_min or solution["BN"] > 256:
            continue

        for producer_reg in [24, 32, 48]:
            for consumer_reg in [184, 192, 200, 208, 216, 224, 232, 240]:
                n = solution['RepeatN']
                num_smem_p, num_reg_q = solution['NumSmemP'], solution['NumRegQ']
                BM, BN = solution["BM"], solution["BN"]

                max_reg = num_thread_producer * producer_reg + num_thread_consumer * consumer_reg
                consumer_reg_max = solution["EstimatedRegsPerThread"]
                # print(max_reg, consumer_reg_max)

                cfg = {
                        "B": B, "H": H, "S": S, "D": D, "BM": BM, "BN": BN, "NUM_SMEM": num_smem,
                        "PRODUCER_REG_DEALLOC": producer_reg, "CONSUMER_REG_ALLOC": consumer_reg,
                        "P_SMEM_K_TILES": num_smem_p, "Q_REG_K_TILES": num_reg_q, 
                        "NUM_THREADS": 384, "KERNEL_IDX": kernel_idx, 
                        "SmemSize": solution['SmemSize'], "RegSize": solution['RegSize']
                    }
                if max_reg < 65536 and consumer_reg_max <= consumer_reg:
                    if kernel_idx == 1 and n == 1 and num_smem_p == 0 and num_reg_q == 0:
                        configs.append(cfg)
                    elif kernel_idx == 2 and n == 1 and (num_smem_p != 0 or num_reg_q != 0):
                        configs.append(cfg)
                    elif kernel_idx == 3 and n != 1:
                        configs.append(cfg)
    # print(count)
    return configs


def benchmark(
    B: int,
    H: int,
    S: int,
    D: int,
    kernel_idx: int = 1,
    bn_min: int = 112,
    solve_func: function = keep_solve,
    rank: int = 10,
    source_dir: Path | None = None,
    build_dir: Path | None = None,
    result_dir: Path | None = None,
    nvcc: str = "nvcc",
    arch: str = "sm_90a",
    device: str = "0",
    keep_going: bool = True,
    output_path: Path | None = None,
) -> list[dict]:
    if rank <= 0:
        raise ValueError("rank must be positive")

    source_path = (source_dir or SOURCE_DIR / "attn_test.cu").resolve()
    build_dir = (build_dir or SOURCE_DIR / "build_run_attn").resolve()
    result_dir = (result_dir or SOURCE_DIR / "results").resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    print(result_dir)

    args = {"nvcc": nvcc, "arch": arch, "device": device}

    configs = get_configs(B, H, S, D, kernel_idx=kernel_idx, bn_min=bn_min, solve_func=solve_func)
    source = source_path.read_text()
    top_results: list[dict] = []
    failed_results: list[dict] = []
    print(f"generated configs: {len(configs)}")

    for idx, params in enumerate(configs, 1):
        print(f"\n=== autotune config {idx}/{len(configs)} ===")
        try:
            validate_params(params)
            print_params(params)

            generated, exe = config_paths(build_dir, params)
            if exe.exists():
                print(f"[cache] found binary before source generation: {exe}")
            else:
                make_source(source, generated, params)
            print(f"source: {generated}")
            print(f"binary: {exe}")

            compile_config(args, generated, exe)
            output = run_config(args, exe)
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
        output_path = result_dir / (
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

if __name__ == "__main__":
    B = 1
    H = 16
    S = 32768
    bn_min = 160

    for D in [64, 128]:
        if D == 128:
            bn_min = 96
        for cfg in get_configs(B, H, S, D, kernel_idx=3, bn_min=bn_min):
            print(cfg)
        # for kernel_idx in range(1, 4):
        #     benchmark(
        #         B,
        #         H,
        #         S,
        #         D,
        #         kernel_idx=kernel_idx,
        #         rank=15,
        #         bn_min=bn_min,
        #     )