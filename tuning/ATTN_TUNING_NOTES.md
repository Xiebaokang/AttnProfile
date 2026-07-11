# Attention Kernel 调优与资源建模笔记

本文总结 `AttnProfile/tuning` 中 attention kernel 的序列 padding、尾部 mask、WGMMA shape、shared-memory 大小、寄存器模型和参数扫描策略。

## 1. 非整除序列长度

原始 kernel 使用：

```cpp
grid.x = S / BM;
for (int iw = 0; iw < S; iw += BN) { ... }
```

当 `S % BM != 0` 时，Q/O 的最后一个 tile 会被遗漏；当 `S % BN != 0` 时，最后一个 KV tile 不完整。

现在使用两个物理长度：

```cpp
SQO = ceil_div(S, BM) * BM;
SKV = ceil_div(S, BN) * BN;
```

- Q/O 按 `SQO` 分配并创建 TMA tensor map。
- K/V 按 `SKV` 分配并创建 TMA tensor map。
- `grid.x = SQO / BM`。
- kernel 参数仍传入真实长度 `S`。
- benchmark 的有效 FLOPs 仍按真实 `S` 计算。

## 2. 为什么 KV 不能只补零

如果 padding K 为 0，则对应 score 为：

```text
Q · 0 = 0
```

它进入 softmax 后会贡献 `exp(0)`，改变 softmax 分母。因此只补零不能保证结果与原始长度 attention 等价。

正确方式与 FA3 一致：

1. tile 数量向上取整；
2. TMA 对物理越界区域安全填充；
3. QK GEMM 后、softmax 前，将 `global_k >= S` 的 score 设置为 `-FLT_MAX`；
4. padding token 对 softmax 的贡献变为 0。

即：

```text
物理 padding + 逻辑 sequence-length mask
```

三个 kernel 都已经加入尾部 KV mask。Kernel 3 原来的：

```cpp
N * S / BN
```

也改成了：

```cpp
N * ceil_div(S, BN)
```

## 3. Q 方向 padding

Q 的最后一个不完整 BM tile 可以完整计算，输出写到按 `SQO` 分配的 O buffer。使用方只读取真实 `S` 行。

这避免了尾部 block 的复杂分支，同时保证真实 query 全部被处理。

## 4. FLOPs 的两种定义

有效 attention FLOPs：

```cpp
useful_flops = 4 * B * H * S * S * D;
```

其中 QK 和 PV 各贡献：

```text
2 * B * H * S * S * D
```

padding 后的实际执行 FLOPs 近似为：

```cpp
executed_flops = 4 * B * H * SQO * SKV * D;
```

- useful TFLOPS 表示算法有效吞吐，适合比较不同 tile 配置。
- executed TFLOPS 表示包含 padding 的硬件计算吞吐。
- 同一个 `B/H/S/D` sweep 中，按 useful TFLOPS 排序等价于按 latency 从小到大排序。

## 5. WGMMA RS shape 支持

原来的 `wgmma_ss` 支持：

```text
16, 32, 64, 96, 112, 128,
144, 160, 176, 192, 208, 224, 240, 256
```

而 `wgmma_rs` 原来只支持：

```text
16, 32, 64, 128, 256
```

现在 `wgmma_rs` 已补齐与 `wgmma_ss` 相同的原生 `m64nNk16` PTX shape，并使用统一接口：

```cpp
float (&d)[(WGMMA_N + 15) / 16][8]
```

新增 shape 使用单条原生 WGMMA 指令，没有拆成多条指令，避免 shared-memory B descriptor 偏移和 swizzle 布局问题。

## 6. Shared-memory 精确大小

solver 原来的 shared-memory 公式与 C++ `sizeof` 不一致，原因包括：

1. 漏算 `Qmbar/Kempty/Vempty/Kfull/Vfull`；
2. 漏算字段的 `alignas(128)`；
3. 漏算 union 最终向 128 bytes 对齐；
4. P 的实际列数不是简单的 `P * MMA_K`，而是：

```cpp
P_SMEM_COLS = ceil_div(P_SMEM_K_TILES * MMA_K, 64) * 64;
```

solver 现在通过 `smem_size_bytes()` 逐字段模拟真实 C++ 布局：

```text
NumSmemP == 0 -> SMemWSBaseline
NumSmemP > 0  -> SMemWS
```

多组参数已确认 Python 结果与 C++ `sizeof` 完全一致。

另外，baseline launcher 原来错误地使用：

```cpp
sizeof(SMemWS<...>)
```

现已改成：

```cpp
sizeof(SMemWSBaseline<...>)
```

## 7. Hopper shared-memory 上限

`229376 bytes` 只是 224 KiB，并非 Hopper opt-in 单 block 的真实上限。

H100/H800 常见的 opt-in 上限是：

```text
227 KiB = 232448 bytes
```

因此 `229504 bytes` 的动态 shared memory 仍可能正常运行：

```text
229504 < 232448
```

实际值应从设备查询：

```cpp
cudaDeviceGetAttribute(
    &max_smem_optin,
    cudaDevAttrMaxSharedMemoryPerBlockOptin,
    device
);
```

动态 shared-memory 安全上限应为：

```text
sharedMemPerBlockOptin - kernel static shared memory
```

静态 shared memory 可通过 `cudaFuncGetAttributes()` 查询。

## 8. 三个 kernel 的主要寄存器组成

令：

```python
RepeatN = BM // (NumConsumer * MMA_M)
```

每个 consumer thread 的主要寄存器估算：

```python
AccSRegs = BN // 2
AccORegs = RepeatN * D // 2
RegPRegs = (BN // MMA_K - NumSmemP) * 4
SoftmaxRegs = RepeatN * 6
```

含义：

- `AccSRegs`：FP32 QK/softmax accumulator。
- `AccORegs`：需要跨 KV 循环保存的 FP32 output accumulator。
- `RegPRegs`：没有放入 shared memory 的 FP16 P，每个 32-bit register 保存两个 FP16。
- `SoftmaxRegs`：`scores_max`、`scores_max_prev` 和 `logsum`。

Kernel 3 的 `acc_s` 和 register P 一次只对应一个 query 子 tile，因此其每线程大小不随 `RepeatN` 增长；但 `acc_o` 和 softmax persistent state 需要保存全部 query 子 tile，因此随 `RepeatN` 增长。

## 9. 两种寄存器模型

### 9.1 生命周期重叠模型

较乐观模型假设编译器能够复用不同阶段的寄存器：

```python
QKStageRegs = AccSRegs + AccORegs + SoftmaxRegs
PVStageRegs = RegPRegs + AccORegs + SoftmaxRegs

OptimisticRegs = max(QKStageRegs, PVStageRegs) + Overhead
```

这个模型接近理想的活跃寄存器下界，但可能低估 inline PTX、WGMMA asynchronous operand 和转换阶段造成的生命周期重叠。

### 9.2 保守累加模型

当前 solver 采用：

```python
CombinedRegs = (
    AccSRegs
    + RegPRegs
    + AccORegs
    + SoftmaxRegs
)

EstimatedRegsPerThread = CombinedRegs + Overhead
```

它假设 `acc_s`、`acc_s_cast/register P` 和 `acc_o` 同时占用寄存器，是较保守的上界。

当前 overhead：

```text
Kernel 1: 8 registers/thread
Kernel 2: 12 registers/thread
Kernel 3: 16 registers/thread
```

选择保守累加模型的主要原因是保留 Kernel 2/3 策略的建模意义：

```text
NumSmemP 增大
-> RegPRegs 减少
-> EstimatedRegsPerThread 减少
-> shared memory 分担 register P 压力
```

实际 PTXAS 寄存器数通常介于乐观模型和保守模型之间。

## 10. Kernel 3 中 P 墠大但 RegSize 不变的旧现象

在旧的阶段峰值模型中，例如：

```text
BM=256, BN=256, D=64, RepeatN=2
```

有：

```text
AccSRegs=128
AccORegs=64
SoftmaxRegs=12
QKStageRegs=204
```

当 P 从 14 增加到 15：

```text
RegPRegs: 8 -> 4
PVStageRegs: 84 -> 80
```

但 QK 阶段仍为 204，所以 `max(QK, PV)` 不变。

切回保守累加模型后：

```text
P=14: EstimatedRegsPerThread=228, RegSize=233472
P=15: EstimatedRegsPerThread=224, RegSize=229376
P=16: EstimatedRegsPerThread=220, RegSize=225280
```

这样 P shared-memory 策略会直接反映在寄存器估算中。

## 11. RegSize 的当前含义

为兼容现有日志，solver 仍输出 `RegSize`：

```python
RegSize = (
    EstimatedRegsPerThread
    * NumConsumerThreads
    * 4
)
```

它表示全部 consumer threads 的估算寄存器存储字节数，不包含 producer allocation。

真正的 CTA allocation 约束单独检查：

```python
TotalAllocatedRegs = (
    NumProducerThreads * PRODUCER_REG_DEALLOC
    + NumConsumerThreads * CONSUMER_REG_ALLOC
)

TotalAllocatedRegs <= 65536
```

## 12. solve 与 producer/consumer 寄存器扫描

`get_configs()` 会扫描：

```python
PRODUCER_REG_DEALLOC = [24, 32, 48]
CONSUMER_REG_ALLOC = [184, 192, ..., 240]
```

当前方式先以扫描范围最大值 240 求候选全集，然后逐组合过滤：

```text
solve(ConsumerRegLimit=240)
        ↓
EstimatedRegsPerThread <= consumer_reg
        ↓
producer*128 + consumer*256 <= 65536
```

如果寄存器模型准确，这与针对每个 consumer 上限重复调用 solve 等价，但更高效。

需要注意：保守模型可能把 PTXAS 实际可运行的配置提前过滤。solver 是候选预筛选，最终事实仍应以以下结果为准：

- PTXAS `Used N registers`；
- spill stores/loads；
- `cudaFuncSetAttribute` 是否成功；
- kernel 是否正常运行；
- 实际 benchmark latency。

## 13. Solver 的职责边界

推荐将完整调优流程理解为：

```text
solver
  生成资源上理论可行、偏保守的候选
        ↓
NVCC/PTXAS
  检查实际寄存器数、spill 和编译告警
        ↓
CUDA runtime
  检查 opt-in shared memory 和 launch 合法性
        ↓
benchmark
  以真实 latency / useful TFLOPS 选择最终配置
```

solver 不应被视为编译器寄存器分配的精确替代。

## 14. 当前关键实现决策

1. 非整除 S 使用 Q/O 与 K/V 独立物理 padding。
2. KV padding 必须配合 softmax 前 `-inf` mask。
3. useful FLOPs 继续按真实 S 计算。
4. shared-memory 大小严格模拟 C++ `sizeof`。
5. Hopper SM90 默认可考虑 `232448-byte` opt-in 上限，但最好运行时查询。
6. WGMMA RS 支持与 SS 相同的 N shape。
7. solver 当前使用保守寄存器累加模型，以体现 P shared-memory 对寄存器压力的分担。
8. PTXAS 和实际 benchmark 是最终判断依据。

## 15. 猜想：QK 混合 RS/SS，用寄存器余量换取 shared-memory 空间

> 本节是后续设计猜想，尚未在当前 kernel 中实现或完成数值、性能验证。其目标是在 D 增大、shared memory 先于 registers 成为瓶颈时，进一步扩大可行的 BN。

### 15.1 问题背景

当前 Kernel 2/3 的主要资源迁移方向是：

```text
增大 P_SMEM_K_TILES
    ↓
将更多 softmax P 从 registers 搬到 shared memory
    ↓
降低 acc_s_cast/register P 的寄存器压力
```

当寄存器先成为瓶颈时，这种方式有意义。但随着 D 增大，shared memory 的增长很快：

```text
Q bytes = BM * D * sizeof(fp16)
K bytes = BN * D * NUM_SMEM * sizeof(fp16)
V bytes = BN * D * NUM_SMEM * sizeof(fp16)
```

此时可能出现：

```text
shared memory 已接近上限
registers 仍有余量
```

继续增大 `P_SMEM_K_TILES` 不仅无法放入更多配置，反而会进一步挤压 shared memory。因此可以考虑反向迁移：

```text
将部分 Q 从 shared memory 搬到 registers
```

并在 QK GEMM 中混合使用 WGMMA RS 和 SS。

### 15.2 QK RS 能减少哪部分 shared memory

当前 QK 为：

```text
Q: shared memory
K: shared memory
QK: WGMMA SS
```

改成 RS 后：

```text
Q: registers，作为 WGMMA A operand
K: shared memory，作为 WGMMA B operand
QK: WGMMA RS
```

因此 RS 可以减少或消除：

```text
BM * D * sizeof(fp16)
```

的 Q shared-memory 空间。

它不能直接减少 K shared memory，因为当前 Hopper FP16 WGMMA RS 的自然形式是：

```text
A = registers
B = shared memory
```

在 QK 中对应：

```text
A = Q
B = K
```

所以该策略本质上是 `Q: shared -> registers`，而不是 `K: shared -> registers`。

### 15.3 方案 A：全部 Q 使用 RS

最激进的方案是不在 shared memory 中长期保存完整 Q，而是将所需 Q fragment 保存在 consumer registers 中。

一个 `m64k16` RS A operand 每线程需要 4 个 32-bit registers。一个 query tile 沿 D 方向包含 `D/16` 个 fragment，因此：

```python
QRegsPerQueryTile = 4 * (D // 16)
                  = D // 4
```

如果 Kernel 3 同时保留 `RepeatN` 个 query tile：

```python
QRegsPerThread = RepeatN * D // 4
```

例如：

```text
BM=256
D=128
RepeatN=2

QRegsPerThread = 2 * 128 / 4 = 64
```

收益：

- 可移除完整 Q shared-memory buffer；
- Q 只加载一次，之后可跨全部 KV blocks 重用；
- 在 registers 有余量、shared memory 不足时可能允许更大的 BN。

代价：

- Q registers 跨整个 KV 循环长期存活；
- D 或 RepeatN 较大时寄存器压力明显；
- 可能挤压 `acc_s`、`acc_o`、softmax state 和地址/pipeline 临时寄存器；
- 需要处理 global/shared layout 到 WGMMA RS A fragment layout 的转换。

### 15.4 方案 B：按 query/M tile 混合 RS 与 SS

更平衡的方案是只把部分 query tile 放入 registers，其余 Q 仍放在 shared memory。

引入模板参数：

```cpp
Q_REG_TILES
```

定义：

```cpp
constexpr int Q_TOTAL_TILES = BM / (2 * MMA_M);
constexpr int Q_SMEM_TILES = Q_TOTAL_TILES - Q_REG_TILES;
```

例如 `BM=256`、`RepeatN=2`：

```text
n=0: Q in registers -> QK RS
n=1: Q in shared    -> QK SS
```

此时 Q shared memory 从：

```python
BM * D * sizeof(fp16)
```

降低到：

```python
QSmemBytes = (
    Q_SMEM_TILES
    * 2 * MMA_M
    * D
    * sizeof(fp16)
)
```

新增 Q register 压力约为：

```python
QRegPerThread = Q_REG_TILES * D // 4
```

编译期分派可以写成：

```cpp
if constexpr (n < Q_REG_TILES) {
    wgmma_rs<QK_MMA_N>(
        acc_s,
        q_reg[n][k / MMA_K],
        KAddr
    );
} else {
    wgmma_ss<QK_MMA_N>(
        acc_s,
        QSmemAddr,
        KAddr
    );
}
```

该方式可以连续调节 shared/register trade-off，比“全 RS”更适合 solver 扫描。

### 15.5 方案 C：沿 D/K 维度混合 RS 与 SS

另一种理论方案是在同一个 query tile 内，部分 D fragment 使用 RS，部分使用 SS：

```text
Q[:, 0:64]   -> RS
Q[:, 64:128] -> SS
```

两种 WGMMA 可以写入同一个 QK accumulator：

```cpp
wgmma_rs<..., ScaleD=0>(...);  // 初始化 accumulator
wgmma_ss<..., ScaleD=1>(...);  // 继续累加
```

但这种方式需要把 Q shared-memory buffer 压缩为只包含 SS 使用的列，导致以下实现复杂度显著增加：

- TMA tensor map 的 box/stride；
- shared-memory swizzle；
- descriptor stride；
- D fragment 到压缩 Q layout 的映射；
- RS/SS accumulator 一致性和初始化顺序。

因此优先考虑按 query/M tile 混合，而不是按 D/K fragment 混合。

### 15.6 Q 如何进入 RS registers

PV 的 register P 由 softmax 直接在 registers 中产生：

```text
acc_s -> softmax -> fp16 acc_s_cast
```

因此 PV 的 register P 不需要额外 global/shared-to-register 搬运。

Q 不同，它最初位于 global memory。TMA 支持：

```text
global -> shared
```

但不能直接完成：

```text
global -> registers
```

所以 Q RS 至少有两条加载路径。

#### 路径 1：普通 global load 直接进入 registers

consumer threads 可以使用向量化 global load：

```cpp
uint4 q_vec = *reinterpret_cast<const uint4*>(gQ + offset);
```

然后将数据重排为 WGMMA RS A operand 所需的 4-register fragment。

优点：

- 真正消除 Q shared-memory buffer；
- shared-memory 节省最大。

难点：

- 必须推导 128 consumer threads 对 Q fragment 的精确映射；
- 保证 global memory 合并访问；
- global layout 到 WGMMA A registers 的重排；
- 普通 global load 没有 TMA mbarrier 式流水；
- 需要避免多个 consumer warpgroup 重复加载相同 Q 数据。

#### 路径 2：小型 Q staging shared memory

保留只容纳一个或两个 `m64k16` fragment 的 Q staging buffer：

```text
single buffer: 64 * 16 * sizeof(fp16)
double buffer: 2 * 64 * 16 * sizeof(fp16)
```

流程：

```text
global/TMA -> Q staging smem
ldmatrix   -> Q registers
复用 staging buffer
```

与完整 Q buffer 相比，这仍能显著减少 shared-memory 占用。

但不应在每个 KV block 都重新加载 Q。更合理的方式是：

1. prologue 将 Q fragment 搬入 registers；
2. Q registers 跨全部 KV blocks 保持存活；
3. 每个 KV block 只更新 K/V pipeline。

对于长序列，Q 的一次性加载成本会被大量 KV blocks 摊薄。

### 15.7 smem-to-register load 能否与 WGMMA overlap

`ldmatrix` 是同步 shared-to-register 指令，没有与 TMA 完全相同的异步完成/mbarrier 机制。但 WGMMA 自身是异步的，理论上可利用 tensor-core pipeline 与 LSU/shared-memory pipeline 的并行性隐藏下一次 Q fragment load。

一种双缓冲形式：

```cpp
load_q_fragment(q_reg[0], q_smem[0]);

for (int k = 0; k < D; k += MMA_K) {
    wgmma_rs(..., q_reg[cur], ...);
    warpgroup_commit_batch();

    // 当前 WGMMA 在 tensor core pipeline 中执行时，
    // 尝试通过 LSU/ldmatrix 准备下一片 Q。
    load_q_fragment(q_reg[next], q_smem[next]);

    cur ^= 1;
}
```

要获得 overlap，需要满足：

- 当前 WGMMA 不再依赖即将被覆盖的 Q registers；
- 使用两套 Q register fragment；
- 不要过早调用 `warpgroup_wait`；
- producer 已完成下一片 Q staging；
- 双缓冲增加的 registers 不反过来成为新瓶颈。

不过对于 attention，优先猜想仍是“Q 一次加载、跨 KV blocks 长期复用”。如果采用这种方式，Q load 只发生在 prologue，复杂的循环内 overlap 可能没有必要。

### 15.8 与 P shared-memory 策略组成双向资源迁移

如果引入 `Q_REG_TILES`，solver 可以同时扫描：

```text
P_SMEM_K_TILES
Q_REG_TILES
```

两个参数的方向相反：

```text
P_SMEM_K_TILES 增大：
    register P -> shared memory

Q_REG_TILES 增大：
    shared Q -> registers
```

对应资源模型可以扩展为：

```python
QSmemBytes = (
    (RepeatN - Q_REG_TILES)
    * 2 * MMA_M
    * D
    * ElementWidth
)

QRegPerThread = Q_REG_TILES * D // 4
```

保守寄存器模型加入：

```python
CombinedRegs = (
    AccSRegs
    + RegPRegs
    + AccORegs
    + SoftmaxRegs
    + QRegPerThread
)
```

shared-memory 模型则将原来的完整 Q 数组替换为 `Q_SMEM_TILES` 对应大小，并加入可能存在的 Q staging buffer。

这使 solver 能探索以下区域：

```text
shared 紧张、register 宽松 -> 增大 Q_REG_TILES
register 紧张、shared 宽松 -> 增大 P_SMEM_K_TILES
两者都紧张              -> 减小 BM/BN、NUM_SMEM 或 D tile
```

### 15.9 该策略无法解决的限制

即使完整移除 Q shared memory，K/V 仍需要：

```python
KVSmemBytes = (
    2
    * BN
    * D
    * NUM_SMEM
    * sizeof(fp16)
)
```

例如：

```text
D=128
BN=256
NUM_SMEM=2

K/V bytes = 2 * 256 * 128 * 2 * 2
          = 262144 bytes
```

仅 K/V 就已经超过常见 Hopper 单 block opt-in shared-memory 上限。因此 QK RS 只能扩大一部分可行域，不能消除 K/V pipeline 本身的容量限制。

必要时还需要结合：

- `NUM_SMEM=1`；
- K 和 V 使用不同 stage 数；
- 延迟 V load；
- K/V shared-memory 区域生命周期复用；
- 更细粒度 producer/consumer pipeline；
- 降低 BN；
- 对 D 方向进一步分块。

### 15.10 推荐的第一版实验

优先在 Kernel 3 中实现按 query tile 划分的混合模式：

1. 增加模板参数 `Q_REG_TILES`；
2. 第一版只支持 `Q_REG_TILES=0` 和 `1`；
3. 对 register Q 先采用小型 shared staging + `ldmatrix`，降低直接 global-to-register 映射风险；
4. Q 在 prologue 加载一次，并跨全部 KV blocks 保存；
5. `n < Q_REG_TILES` 使用 QK RS，其余 `n` 使用 QK SS；
6. solver 同时加入 `QSmemBytes` 和 `QRegPerThread`；
7. 分别记录 PTXAS registers、spill、shared-memory 大小和 latency。

需要重点验证：

- RS 与 SS QK accumulator 的数值布局是否完全一致；
- Q register fragment 的 lane mapping；
- Q registers 的实际生命周期；
- Q register 常驻是否造成 spill；
- Q staging/mbarrier 是否影响现有 K/V pipeline；
- `Q_REG_TILES` 增大后能否实际提升 BN；
- 增大的 BN 是否足以抵消额外 Q load/rearrange 成本；
- 长序列与短序列下的收益是否不同。

### 15.11 当前猜想结论

QK 的部分 RS + 部分 SS 在架构上可行，最适合按 query/M tile 划分。它能够形成与 P shared-memory 策略互补的双向资源迁移：

```text
P: registers -> shared memory
Q: shared memory -> registers
```

Q 的 shared-to-register 操作理论上可与异步 WGMMA 部分 overlap，但对长序列更值得优先尝试的是在 prologue 中一次性加载 Q，并跨 KV blocks 重用。该策略能减少 Q shared memory、扩大一部分 BN 可行域，但最终仍受 K/V shared-memory 容量和 pipeline stage 数限制。
