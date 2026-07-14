
def keep_solve(
    HD: int = 64,
    MMA_M: int = 64,
    MMA_K: int = 16,
    NumConsumer: int = 2,
    ElementWidth: int = 2,
    SmemLimit: int = 232_448,
    KernelIdx: int = 1,
    ConsumerRegLimit: int = 240,
):
    if KernelIdx == 1:
        # 计算 smemsize union SMemWSBaseline 的大小
        # 保守估计寄存器的用量
        # 解出所有BM、BN、num_smem等参数
        # 约束：BM为MMA_M*NumConsumer的整数倍、相同其他配置下num_smem_p参数的最大值必须是4的倍数、num_reg_q参数
        pass
    elif KernelIdx == 2:
        # 计算 smemsize union SMemWSBaseline 的大小
        # 保守估计寄存器的用量
        pass
    elif KernelIdx == 3:
        pass
    else:
        # 抛出异常 raise
        pass

def radical_slove(
    HD: int = 64,
    MMA_M: int = 64,
    MMA_K: int = 16,
    NumConsumer: int = 2,
    ElementWidth: int = 2,
    SmemLimit: int = 232_448,
    KernelIdx: int = 1,
    ConsumerRegLimit: int = 240,
):
    pass
