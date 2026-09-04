from __future__ import annotations

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
EXTERNAL = ROOT / "external_data"


def find_kaist_root() -> Path:
    base = EXTERNAL / "kaist"
    candidates = [p for p in base.rglob("*") if p.is_dir() and (p / "vibration").is_dir()]
    if not candidates:
        raise FileNotFoundError(f"Cannot find KAIST root under {base}")
    return sorted(candidates, key=lambda p: len(str(p)))[0]


def _short_stats(arr: np.ndarray) -> str:
    arr = np.asarray(arr)
    if arr.size == 0:
        return "empty"
    if not np.issubdtype(arr.dtype, np.number):
        return "non-numeric"
    flat = arr.astype(np.float64, copy=False).reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return "no finite values"
    q = np.quantile(finite, [0.0, 0.01, 0.5, 0.99, 1.0])
    return (
        f"min={q[0]:.6g} p01={q[1]:.6g} median={q[2]:.6g} "
        f"p99={q[3]:.6g} max={q[4]:.6g} mean={finite.mean():.6g} std={finite.std():.6g}"
    )


def _walk_mat(obj, name: str, depth: int = 0, max_depth: int = 6) -> None:
    indent = "  " * depth
    if depth > max_depth:
        print(f"{indent}{name}: <max depth>")
        return

    if hasattr(obj, "_fieldnames"):
        fields = list(getattr(obj, "_fieldnames") or [])
        print(f"{indent}{name}: MATLAB struct fields={fields}")
        for field in fields:
            _walk_mat(getattr(obj, field), f"{name}.{field}", depth + 1, max_depth)
        return

    if isinstance(obj, np.ndarray):
        print(f"{indent}{name}: ndarray shape={obj.shape} dtype={obj.dtype} {_short_stats(obj)}")
        if obj.dtype.names:
            for field in obj.dtype.names:
                _walk_mat(obj[field], f"{name}.{field}", depth + 1, max_depth)
        elif obj.dtype == object and obj.size <= 20:
            for i, value in enumerate(obj.reshape(-1)):
                _walk_mat(value, f"{name}[{i}]", depth + 1, max_depth)
        return

    if np.isscalar(obj):
        print(f"{indent}{name}: scalar type={type(obj).__name__} value={obj!r}")
        return

    print(f"{indent}{name}: type={type(obj).__name__}")


def inspect_mat(path: Path) -> None:
    print("\n" + "=" * 120)
    print(f"MAT DEEP INSPECT: {path}")
    print("=" * 120)
    from scipy.io import loadmat

    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    for key, value in data.items():
        if key.startswith("__"):
            continue
        _walk_mat(value, key)


def inspect_tdms(path: Path) -> None:
    print("\n" + "=" * 120)
    print(f"TDMS DEEP INSPECT: {path}")
    print("=" * 120)
    from nptdms import TdmsFile

    tdms = TdmsFile.read(path)
    print(f"file properties: {dict(tdms.properties)}")
    for group in tdms.groups():
        print(f"group={group.name!r} properties={dict(group.properties)}")
        for channel in group.channels():
            arr = np.asarray(channel[:])
            props = dict(channel.properties)
            print(
                f"  channel={channel.name!r} len={len(arr)} dtype={arr.dtype} "
                f"{_short_stats(arr)}"
            )
            interesting = {
                k: v
                for k, v in props.items()
                if any(
                    token in str(k).lower()
                    for token in ["unit", "wf_", "sample", "time", "rate", "increment", "start"]
                )
            }
            print(f"    properties={interesting}")


def main() -> None:
    root = find_kaist_root()
    print(f"KAIST root: {root}")

    vib_dir = root / "vibration"
    ac_dir = root / "acoustic"
    tdms_dir = root / "current_temp"

    mat_candidates = [
        vib_dir / "0Nm_Normal.mat",
        vib_dir / "2Nm_Normal.mat",
        vib_dir / "4Nm_Normal.mat",
        vib_dir / "0Nm_BPFI_03.mat",
        ac_dir / "0Nm_Normal.mat",
        ac_dir / "0Nm_BPFI_03.mat",
    ]
    for path in mat_candidates:
        if path.exists():
            inspect_mat(path)

    tdms_candidates = [
        tdms_dir / "0Nm_Normal.tdms",
        tdms_dir / "2Nm_Normal.tdms",
        tdms_dir / "4Nm_Normal.tdms",
        tdms_dir / "0Nm_BPFI_03.tdms",
    ]
    for path in tdms_candidates:
        if path.exists():
            inspect_tdms(path)

    print("\n" + "=" * 120)
    print("DEEP INSPECTION COMPLETE")
    print("=" * 120)


if __name__ == "__main__":
    main()
