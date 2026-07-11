
def ceil_div(a: int, b: int) -> int:
    """计算 ceil(a / b)，要求 b > 0。"""
    if b <= 0:
        raise ValueError("b 必须大于 0")
    return -(-a // b)


def align_up(value: int, alignment: int) -> int:
    """与 C++ alignas 对象布局一致地向上对齐。"""
    return ceil_div(value, alignment) * alignment


def smem_size_bytes(
    BM: int,
    BN: int,
    HD: int,
    NumSmem: int,
    NumSmemP: int,
    MMA_K: int = 16,
    ElementWidth: int = 2,
) -> int:
    """精确模拟 SMemWSBaseline / SMemWS 的 sizeof 结果。

    NumSmemP == 0 对应 SMemWSBaseline；大于 0 对应 SMemWS。
    """
    offset = 0

    def add_array(size: int, alignment: int) -> None:
        nonlocal offset
        offset = align_up(offset, alignment)
        offset += size

    add_array(BM * HD * ElementWidth, 128)                 # Q
    add_array(BN * HD * NumSmem * ElementWidth, 128)      # K
    add_array(BN * HD * NumSmem * ElementWidth, 128)      # V

    if NumSmemP > 0:
        # 与 SMemWS::P_SMEM_COLS 完全一致，P 的列数按 64 对齐。
        p_smem_cols = align_up(NumSmemP * MMA_K, 64)
        add_array(BM * p_smem_cols * ElementWidth, 128)   # P

    add_array(8, 8)                                        # Qmbar
    add_array(NumSmem * 8, 8)                              # Kempty
    add_array(NumSmem * 8, 8)                              # Vempty
    add_array(NumSmem * 8, 8)                              # Kfull
    add_array(NumSmem * 8, 8)                              # Vfull

    struct_size = align_up(offset, 128)
    output_size = align_up(BM * HD * ElementWidth, 128)
    return align_up(max(struct_size, output_size), 128)


def estimate_registers_per_thread(
    KernelIdx: int,
    BM: int,
    BN: int,
    HD: int,
    NumSmemP: int,
    MMA_M: int = 64,
    MMA_K: int = 16,
    NumConsumer: int = 2,
) -> dict[str, int]:
    """保守估算单个 consumer thread 的寄存器数。

    为了保留 P shared-memory 策略的建模意义，这里假设 acc_s、register P
    和 acc_o 同时占用寄存器，不采用跨阶段生命周期复用。
    """
    if KernelIdx not in (1, 2, 3):
        raise ValueError("KernelIdx 必须是 1、2 或 3")

    repeat_n = BM // (NumConsumer * MMA_M)
    acc_s_regs = BN // 2
    acc_o_regs = repeat_n * HD // 2
    reg_p_regs = (BN // MMA_K - NumSmemP) * 4
    softmax_regs = repeat_n * 6

    qk_stage_regs = acc_s_regs + acc_o_regs + softmax_regs
    pv_stage_regs = reg_p_regs + acc_o_regs + softmax_regs
    combined_regs = acc_s_regs + reg_p_regs + acc_o_regs + softmax_regs
    overhead = {1: 8, 2: 12, 3: 16}[KernelIdx]
    estimated = combined_regs + overhead

    return {
        "AccSRegs": acc_s_regs,
        "AccORegs": acc_o_regs,
        "RegPRegs": reg_p_regs,
        "SoftmaxRegs": softmax_regs,
        "QKStageRegs": qk_stage_regs,
        "PVStageRegs": pv_stage_regs,
        "CombinedRegs": combined_regs,
        "RegisterOverhead": overhead,
        "EstimatedRegsPerThread": estimated,
    }


def solve(
    HD: int = 64,
    NumSmem: int = 2,
    MMA_M: int = 64,
    MMA_K: int = 16,
    NumConsumer: int = 2,
    ElementWidth: int = 2,
    SmemLimit: int = 232_448,
    KernelIdx: int = 2,
    ConsumerRegLimit: int = 240,
):
    """
    求解所有满足约束的整数解：

        BM = NumConsumer * MMA_M 的正整数倍

        RepeatN = BM / MMA_M / NumConsumer

        SmemSize = sizeof(SMemWSBaseline)  if NumSmemP == 0
                   sizeof(SMemWS)          otherwise
            <= SmemLimit

        EstimatedRegsPerThread 使用保守累加模型，将 acc_s、register P、
        acc_o 和 softmax persistent state 全部计入。

        BN 是 MMA_K 的正整数倍

        0 <= NumSmemP <= BN / MMA_K
    """

    parameters = {
        "MMA_M": MMA_M,
        "MMA_K": MMA_K,
        "HD": HD,
        "NumConsumer": NumConsumer,
        "NumSmem": NumSmem,
        "ElementWidth": ElementWidth,
        "SmemLimit": SmemLimit,
        "KernelIdx": KernelIdx,
        "ConsumerRegLimit": ConsumerRegLimit,
    }

    for name, value in parameters.items():
        if value <= 0:
            raise ValueError(f"{name} 必须是正整数")

    # ---------------------------------------------------------
    # BM 必须为 NumConsumer * MMA_M 的正整数倍
    # ---------------------------------------------------------
    BM_step = NumConsumer * MMA_M

    # 即使 BN 和 NumSmemP 取最有利的值，也必须满足固定开销：
    #
    # SmemSize >= BM * HD * ElementWidth
    # RegSize  >= BM * HD * 4
    #
    # 因此可以得到 BM 的绝对最大值。
    BM_max_by_smem = SmemLimit // (HD * ElementWidth)
    BM_max = BM_max_by_smem

    # 向下对齐到 BM_step。
    BM_max = (BM_max // BM_step) * BM_step

    solutions = []

    for BM in range(BM_step, BM_max + 1, BM_step):
        RepeatN = BM // MMA_M // NumConsumer

        # RepeatN 实际上等于 BM / BM_step，因此必然为正整数。
        assert RepeatN >= 1
        assert BM == RepeatN * MMA_M * NumConsumer

        # -----------------------------------------------------
        # 去掉 SmemSize 中只与 BM 有关的固定部分：
        #
        # SmemSize =
        #     BM * HD * ElementWidth
        #     + 2 * BN * HD * ElementWidth * NumSmem
        #     + NumSmemP * MMA_K * BM * ElementWidth
        #
        # 因为 BN、HD、ElementWidth、NumSmem 都为正数，
        # max() 中第一项一定不小于第二项。
        # -----------------------------------------------------
        S_remaining = (
            SmemLimit
            - BM * HD * ElementWidth
        )

        if S_remaining < 0:
            continue

        # -----------------------------------------------------
        # 使用 NumSmemP = 0 推导共享内存下的 BN 最大值：
        #
        # 2 * BN * HD * ElementWidth * NumSmem
        #     <= S_remaining
        # -----------------------------------------------------
        smem_bn_coefficient = (
            2
            * HD
            * ElementWidth
            * NumSmem
        )

        BN_max_by_smem = (
            S_remaining // smem_bn_coefficient
        )

        # 先由 shared memory 给出安全的枚举上界，寄存器峰值在最终
        # NumSmemP 循环中按 kernel 的计算阶段检查。
        BN_max = BN_max_by_smem

        # BN 必须是 MMA_K 的整数倍。
        BN_max = (BN_max // MMA_K) * MMA_K

        for BN in range(MMA_K, BN_max + 1, MMA_K):
            num_p_tiles = BN // MMA_K

            # -------------------------------------------------
            # 共享内存给出 NumSmemP 上界：
            #
            # BM*HD*ElementWidth
            # + 2*BN*HD*ElementWidth*NumSmem
            # + P*MMA_K*BM*ElementWidth
            # <= SmemLimit
            #
            # P <=
            # (
            #   S_remaining
            #   - 2*BN*HD*ElementWidth*NumSmem
            # )
            # / (MMA_K*BM*ElementWidth)
            # -------------------------------------------------
            smem_after_bn = (
                S_remaining
                - smem_bn_coefficient * BN
            )

            if smem_after_bn < 0:
                continue

            p_max_smem = (
                smem_after_bn
                // (
                    MMA_K
                    * BM
                    * ElementWidth
                )
            )

            # 原始 pipeline/tile 数量约束。
            p_max_tile = num_p_tiles

            # Baseline 没有 P shared-memory buffer；kernel 2/3 至少需要
            # 一个 P tile，且不能超过 BN/MMA_K。
            P_min = 0 if KernelIdx == 1 else 1
            P_max = min(
                p_max_smem,
                p_max_tile,
            )

            if KernelIdx == 1:
                P_max = min(P_max, 0)

            if P_min > P_max:
                continue

            for NumSmemP in range(P_min, P_max + 1):
                # ---------------------------------------------
                # 使用原始公式计算共享内存。
                # ---------------------------------------------
                SmemSize = smem_size_bytes(
                    BM=BM,
                    BN=BN,
                    HD=HD,
                    NumSmem=NumSmem,
                    NumSmemP=NumSmemP,
                    MMA_K=MMA_K,
                    ElementWidth=ElementWidth,
                )

                # ---------------------------------------------
                # 使用原始公式计算寄存器资源。
                #
                # 分开检查两个除法项是否为整数，
                # 避免无意中使用浮点数或向下取整。
                # ---------------------------------------------
                reg_estimate = estimate_registers_per_thread(
                    KernelIdx=KernelIdx,
                    BM=BM,
                    BN=BN,
                    HD=HD,
                    NumSmemP=NumSmemP,
                    MMA_M=MMA_M,
                    MMA_K=MMA_K,
                    NumConsumer=NumConsumer,
                )
                estimated_regs = reg_estimate["EstimatedRegsPerThread"]
                # 保留 RegSize 字段兼容现有日志；其含义改为全部 consumer
                # threads 的估算峰值寄存器存储字节数。
                RegSize = estimated_regs * NumConsumer * 128 * 4

                # ---------------------------------------------
                # 最终防御性验证。
                # ---------------------------------------------
                if BM % BM_step != 0:
                    continue

                if BN % MMA_K != 0:
                    continue

                if not (
                    0
                    <= NumSmemP
                    <= BN // MMA_K
                ):
                    continue

                if SmemSize > SmemLimit:
                    continue

                if estimated_regs > ConsumerRegLimit:
                    continue

                solutions.append(
                    {
                        "BM": BM,
                        "BN": BN,
                        "NumSmem": NumSmem,
                        "NumSmemP": NumSmemP,
                        "RepeatN": RepeatN,
                        "SmemSize": SmemSize,
                        "RegSize": RegSize,
                        **reg_estimate,
                    }
                )

    return solutions


def print_solutions(solutions):
    print(f"总解数：{len(solutions)}")

    current_bm = None

    for solution in solutions:
        if solution["BM"] != current_bm:
            current_bm = solution["BM"]
            print(
                f"\nBM={current_bm}, "
                f"RepeatN={solution['RepeatN']}"
            )

        print(
            "  [RESULT] "
            f"BN={solution['BN']}, "
            f"NumSmemP={solution['NumSmemP']}, "
            f"SmemSize={solution['SmemSize']}, "
            f"EstimatedRegsPerThread={solution['EstimatedRegsPerThread']}, "
            f"CombinedRegs={solution['CombinedRegs']}, "
            f"QKStageRegs={solution['QKStageRegs']}, "
            f"PVStageRegs={solution['PVStageRegs']}"
        )


if __name__ == "__main__":
    solutions = solve(
        HD = 64,
        NumSmem = 2,
        MMA_M = 64,
        MMA_K = 16,
        NumConsumer = 2,
        ElementWidth = 2,
        SmemLimit = 232_448,
        KernelIdx = 2,
        ConsumerRegLimit = 240,
    )

    print_solutions(solutions)
