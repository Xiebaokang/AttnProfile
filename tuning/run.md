# 执行脚本

## 1. 设置主函数

```py
if __name__ == "__main__":
    B = 1
    H = 16
    S = 32768
    bn_min = 112
    for D in [64, 128]:
        if D == 128:
            bn_min = 96
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
```

## 2. 执行脚本

```bash
# create tmux session
tmux new -s attn

# shift 2-dev GPU
export CUDA_VISIBLE_DEVICES=1
python benchmark_run_attn.py > ./logs/log 2>&1

# list tmux session
tmux ls

# back tmux session
tmux attach -t attn

# back origin Terminal
Ctrl + B  -> D

# exit session
Ctrl + D
```



# 检查config

## 1. 设置主函数

```py
if __name__ == "__main__":
    B = 1
    H = 16
    S = 32768
    bn_min = 112
    for D in [64, 128]:
        if D == 128:
            bn_min = 96
        for cfg in get_configs(B, H, S, D, kernel_idx=3, bn_min=bn_min):
            print(cfg)
```

## 2. 执行脚本

```bash
python benchmark_run_attn.py > ./logs/logg 2>&1
```