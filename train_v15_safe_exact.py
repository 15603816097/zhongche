import numpy as np

import train_v15_official as base


# Exact V8 integration 中 speed_rpm 是主要退化来源。
# 保留 V16 已验证的其他变量 robust alpha，只把 speed_rpm 从 0.15 降到 0.0。
SAFE_ALPHAS = np.asarray([1.0, 1.0, 1.0, 0.0, 1.0, 0.5], dtype=np.float64)


def main():
    base.ALPHAS = SAFE_ALPHAS.copy()
    base.main()


if __name__ == "__main__":
    main()
