import torch
from flash_attn_interface import flash_attn_func

# ============ 参数 ============
BATCH   = 1
HEADS   = 16
SEQ_LEN = 114*160*2
HEAD_DIM = 64
CAUSAL  = False
DTYPE   = torch.float16   # 或 torch.bfloat16
WARMUP  = 100
REPEAT  = 1000
# ==============================

device = "cuda"
torch.manual_seed(42)

# FlashAttention 期望 layout: (batch, seqlen, nheads, headdim)
q = torch.randn(BATCH, SEQ_LEN, HEADS, HEAD_DIM, dtype=DTYPE, device=device)
k = torch.randn(BATCH, SEQ_LEN, HEADS, HEAD_DIM, dtype=DTYPE, device=device)
v = torch.randn(BATCH, SEQ_LEN, HEADS, HEAD_DIM, dtype=DTYPE, device=device)

softmax_scale = 1.0 / (HEAD_DIM ** 0.5)

# ---------- Warmup ----------
for _ in range(WARMUP):
    out = flash_attn_func(q, k, v, causal=CAUSAL, softmax_scale=softmax_scale)
torch.cuda.synchronize()

# ---------- Benchmark ----------
start_ev = torch.cuda.Event(enable_timing=True)
end_ev   = torch.cuda.Event(enable_timing=True)

start_ev.record()
for _ in range(REPEAT):
    out = flash_attn_func(q, k, v, causal=CAUSAL, softmax_scale=softmax_scale)
end_ev.record()
torch.cuda.synchronize()

total_ms = start_ev.elapsed_time(end_ev)   # 单位 ms
avg_ms   = total_ms / REPEAT

# ---------- FLOPs (前向) ----------
# FA fwd ≈ 4 * B * H * S_q * S_k * D  (causal 时除 2，论文惯例)
flops = 4 * BATCH * HEADS * SEQ_LEN * SEQ_LEN * HEAD_DIM
if CAUSAL:
    flops //= 2

tflops = (flops / (avg_ms * 1e-3)) / 1e12

# ---------- 输出 ----------
print(f"BATCH:{BATCH:>5} HEADS:{HEADS:>5} SEQ:{SEQ_LEN:>5} D:{HEAD_DIM:>4}  causal={CAUSAL}")
print(f"Warmup  : {WARMUP}  iters")
print(f"Repeat  : {REPEAT}  iters")
print(f"Avg lat.: {avg_ms:.3f} ms")
print(f"TFLOPS  : {tflops:.2f} TFLOPS")