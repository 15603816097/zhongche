from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
EXTERNAL = ROOT / "external_data"


def find_kaist_root() -> Path:
    base = EXTERNAL / "kaist"
    if not base.exists():
        raise FileNotFoundError(f"KAIST directory not found: {base}")
    candidates = [p for p in base.rglob("*") if p.is_dir() and (p / "vibration").is_dir()]
    if not candidates:
        raise FileNotFoundError(f"Cannot find KAIST root containing vibration/: {base}")
    return sorted(candidates, key=lambda p: len(str(p)))[0]


def inspect_mat(path: Path) -> None:
    print(f"\nMAT: {path}")
    try:
        from scipy.io import whosmat

        items = whosmat(path)
        for name, shape, dtype in items:
            if name.startswith("__"):
                continue
            print(f"  {name:28s} shape={shape!s:18s} dtype={dtype}")
        if items:
            return
    except Exception as exc:
        print(f"  scipy.io.whosmat failed: {type(exc).__name__}: {exc}")

    try:
        import h5py

        with h5py.File(path, "r") as f:
            def visitor(name, obj):
                if hasattr(obj, "shape"):
                    print(f"  {name:28s} shape={obj.shape} dtype={obj.dtype}")
            f.visititems(visitor)
    except Exception as exc:
        print(f"  h5py fallback failed: {type(exc).__name__}: {exc}")


def inspect_tdms(path: Path) -> None:
    print(f"\nTDMS: {path}")
    try:
        from nptdms import TdmsFile
    except ImportError:
        print("  nptdms is not installed. Install with: pip install nptdms")
        return

    try:
        tdms = TdmsFile.read(path)
        for group in tdms.groups():
            print(f"  group={group.name!r}")
            for channel in group.channels():
                data = channel[:]
                props = dict(channel.properties)
                unit = props.get("unit_string", props.get("NI_UnitDescription", ""))
                print(
                    f"    channel={channel.name!r:32s} len={len(data):8d} "
                    f"dtype={getattr(data, 'dtype', None)} unit={unit!r}"
                )
    except Exception as exc:
        print(f"  TDMS read failed: {type(exc).__name__}: {exc}")


def inspect_zip(path: Path) -> None:
    if not path.exists():
        return
    try:
        with zipfile.ZipFile(path) as zf:
            files = [n for n in zf.namelist() if not n.endswith("/")]
        print(f"  ZIP {path.name}: {len(files)} files")
        for name in files[:10]:
            print(f"    {name}")
        if len(files) > 10:
            print(f"    ... ({len(files)-10} more)")
    except Exception as exc:
        print(f"  ZIP inspect failed {path.name}: {type(exc).__name__}: {exc}")


def inspect_kaist() -> None:
    print("=" * 110)
    print("KAIST DATASET")
    print("=" * 110)
    root = find_kaist_root()
    print(f"root: {root}")

    vibration_dir = root / "vibration"
    acoustic_dir = root / "acoustic"
    tdms_dir = root / "current_temp"

    vib = sorted(vibration_dir.glob("*.mat"))
    ac = sorted(acoustic_dir.glob("*.mat"))
    td = sorted(tdms_dir.glob("*.tdms"))

    print(f"vibration .mat : {len(vib)}")
    print(f"acoustic .mat  : {len(ac)}")
    print(f"current/temp   : {len(td)}")

    inspect_zip(root / "vibration.zip")
    inspect_zip(root / "acoustic.zip")
    inspect_zip(root / "current,temp.zip")

    vib_stems = {p.stem for p in vib}
    ac_stems = {p.stem for p in ac}
    td_stems = {p.stem for p in td}
    print("\nFilename overlap:")
    print(f"  vibration ∩ current/temp : {len(vib_stems & td_stems)}")
    print(f"  vibration ∩ acoustic     : {len(vib_stems & ac_stems)}")
    print(f"  all three modalities     : {len(vib_stems & ac_stems & td_stems)}")
    if ac_stems:
        missing_ac = sorted((vib_stems & td_stems) - ac_stems)
        print(f"  matched vib+tdms but missing acoustic: {len(missing_ac)}")
        for x in missing_ac[:12]:
            print(f"    {x}")
        if len(missing_ac) > 12:
            print(f"    ... ({len(missing_ac)-12} more)")

    preferred = ["0Nm_Normal", "2Nm_Normal", "4Nm_Normal", "0Nm_BPFI_03"]
    for stem in preferred:
        p = vibration_dir / f"{stem}.mat"
        if p.exists():
            inspect_mat(p)
            break
    for stem in preferred:
        p = acoustic_dir / f"{stem}.mat"
        if p.exists():
            inspect_mat(p)
            break
    for stem in preferred:
        p = tdms_dir / f"{stem}.tdms"
        if p.exists():
            inspect_tdms(p)
            break


def inspect_metropt() -> None:
    print("\n" + "=" * 110)
    print("METROPT-3 DATASET")
    print("=" * 110)
    csv_path = EXTERNAL / "metropt" / "MetroPT3(AirCompressor).csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    print(f"csv: {csv_path}")
    print(f"size: {csv_path.stat().st_size / 1024**2:.1f} MiB")

    df = pd.read_csv(csv_path, nrows=10000)
    print(f"sample shape: {df.shape}")
    print("columns:")
    for i, c in enumerate(df.columns):
        print(f"  [{i:02d}] {c!r} dtype={df[c].dtype}")

    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        stats = numeric.describe(percentiles=[0.01, 0.5, 0.99]).T[
            ["min", "1%", "50%", "99%", "max", "mean", "std"]
        ]
        print("\nNumeric statistics from first 10,000 rows:")
        with pd.option_context("display.max_rows", 100, "display.width", 180):
            print(stats.to_string())

    print("\nfirst 3 rows:")
    with pd.option_context("display.max_columns", 100, "display.width", 220):
        print(df.head(3).to_string(index=False))


def main() -> int:
    print(f"project root : {ROOT}")
    print(f"external dir : {EXTERNAL}")
    if not EXTERNAL.exists():
        print("ERROR: external_data/ does not exist")
        return 2

    try:
        inspect_kaist()
    except Exception as exc:
        print(f"\nKAIST inspection ERROR: {type(exc).__name__}: {exc}")

    try:
        inspect_metropt()
    except Exception as exc:
        print(f"\nMetroPT inspection ERROR: {type(exc).__name__}: {exc}")

    print("\n" + "=" * 110)
    print("INSPECTION COMPLETE")
    print("=" * 110)
    return 0


if __name__ == "__main__":
    sys.exit(main())
