"""Deterministic ingestion of the three tidy tables read by the replay environments.

    criteo_uplift_real/tidy.pkl   from criteo-research-uplift-v2.1.csv.gz      (Criteo Uplift v2.1, 13,979,592 rows)
    lenta/tidy.pkl                from lenta_dataset.csv.gz                     (Lenta, 687,029 rows)
    hillstrom/tidy.pkl            from the MineThatData e-mail challenge csv    (Hillstrom, 64,000 rows)

Lenta and Hillstrom: the two builders below are the code that wrote our pickles (a dataset-cache module of an
earlier, unrelated project of ours, run once on 2026-05-20; only its download helper, cache bookkeeping and unrelated
datasets are omitted, the table construction is verbatim).  Criteo: the code that wrote our pickle on 2026-10-02 was
not kept; ``build_criteo`` is a reconstruction (csv read with pandas defaults, features cast to float32, the four
indicators to int8) and is validated by content against our pickle with ``check_tidy_content.py``.

Pickle bytes.  The loaders pin the pickle BYTES (``dsswm/envs/{data_v6,lenta_v6,hillstrom_v9}.py``).  The bytes
depend on the pandas version and on the storage of the column-name index: our Lenta and Hillstrom pickles were written
without pyarrow (string storage "python"), our Criteo pickle with pyarrow (string storage "pyarrow").  ``ingest`` sets
that storage per table (``STRING_STORAGE``).  With pandas 3.0.6, numpy 2.5.3 and pyarrow 25.0.1 this script reproduces
all three pickles byte for byte (checked 2026-10-06).  With other versions, or without pyarrow, the bytes may differ;
validate a re-ingest with ``check_tidy_content.py`` (row / column counts, column names, dtypes, index and every value),
which is version independent.  See README "Data".

Usage (cwd = exp/code; DATA_DIR as in the README; raw files where the table below says):
    python ingest_tidy.py --table lenta|hillstrom|criteo|all [--data-dir DIR] [--out-dir DIR] [--force]

Raw files read (relative to the data dir):
    criteo_uplift_real/criteo-research-uplift-v2.1.csv.gz
    lenta/raw/lenta_dataset.csv.gz                       (also accepted: lenta/lenta_dataset.csv.gz)
    hillstrom/raw/hillstrom.csv                          (also accepted: hillstrom/<original csv file name>)
Outputs are written to <out-dir>/<dir>/tidy.pkl (out-dir defaults to the data dir; an existing tidy.pkl is never
overwritten without --force).
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import pandas as pd

RAW = {
    "criteo": ("criteo_uplift_real", ["criteo-research-uplift-v2.1.csv.gz"]),
    "lenta": ("lenta", ["raw/lenta_dataset.csv.gz", "lenta_dataset.csv.gz"]),
    "hillstrom": ("hillstrom", ["raw/hillstrom.csv",
                                "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"]),
}
RAW_SHA256 = {
    "criteo": "2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc",
    "lenta": "b531544f6c072d22f232d91e20ffcaca265acf224e02687502a40e6d56135682",
    "hillstrom": "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece",
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ builders
def build_hillstrom(raw: Path) -> pd.DataFrame:
    """Verbatim table construction of the original builder."""
    df = pd.read_csv(raw)
    # Tidy: encode treatment (multi-arm: 0=No E-Mail, 1=Mens E-Mail, 2=Womens E-Mail)
    arm_map = {"No E-Mail": 0, "Mens E-Mail": 1, "Womens E-Mail": 2}
    tidy = pd.DataFrame(
        {
            "treatment": df["segment"].map(arm_map).astype("int8"),
            "outcome": df["spend"].astype("float64"),
            "binary": df["visit"].astype("int8"),
            "weight": 1.0,
            "x_recency": df["recency"].astype("float32"),
            "x_history": df["history"].astype("float32"),
            "x_mens": df["mens"].astype("int8"),
            "x_womens": df["womens"].astype("int8"),
            "x_newbie": df["newbie"].astype("int8"),
            "x_channel": df["channel"].astype("category").cat.codes.astype("int8"),
            "x_zip": df["zip_code"].astype("category").cat.codes.astype("int8"),
            "x_history_segment": df["history_segment"]
            .astype("category")
            .cat.codes.astype("int8"),
            "conversion": df["conversion"].astype("int8"),
        }
    )
    return tidy


def build_lenta(raw: Path) -> pd.DataFrame:
    """Verbatim table construction of the original builder."""
    df = pd.read_csv(raw, compression="gzip")
    # The Lenta uplift dataset uses 'group' in {'test', 'control'} and 'response_att' (binary).  The published
    # competition file has no continuous-revenue column; 'response_amt' is the row-wise sum of the 18 'sale_sum_*'
    # category aggregates.  (Not read by any replay environment of this package, which uses response_att.)
    treat = (df["group"].astype(str).str.lower() == "test").astype("int8")
    sale_cols = [c for c in df.columns if c.startswith("sale_sum_")]
    response_amt = df[sale_cols].fillna(0.0).sum(axis=1).astype("float64")
    binary = df["response_att"].astype("int8")
    # Keep a slim numeric covariate slice (8 cols).
    feature_cols = [
        "age",
        "children",
        "main_format",
        "months_from_register",
        "promo_share_15d",
        "food_share_1m",
        "k_var_cheque_3m",
        "mean_discount_depth_15d",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    tidy = pd.DataFrame(
        {
            "treatment": treat,
            "outcome": response_amt,
            "binary": binary,
            "weight": 1.0,
            "response_att": binary,
            "response_amt": response_amt,
        }
    )
    covariates = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    covariates.columns = [f"x_{c}" for c in covariates.columns]
    tidy = pd.concat([tidy.reset_index(drop=True), covariates.reset_index(drop=True)], axis=1)
    return tidy


CRITEO_FLOAT = [f"f{i}" for i in range(12)]
CRITEO_INT = ["treatment", "conversion", "visit", "exposure"]


def build_criteo(raw: Path) -> pd.DataFrame:
    """Reconstruction (original code not kept): the csv.gz read with pandas defaults (float64 / int64), then the 12
    features cast to float32 and the four 0/1 indicators to int8, column order and RangeIndex as in the file."""
    df = pd.read_csv(raw, compression="gzip")
    if list(df.columns) != CRITEO_FLOAT + CRITEO_INT:
        raise RuntimeError(f"unexpected Criteo columns {list(df.columns)}")
    for c in CRITEO_FLOAT:
        df[c] = df[c].astype("float32")
    for c in CRITEO_INT:
        df[c] = df[c].astype("int8")
    return df


BUILDERS = {"criteo": build_criteo, "lenta": build_lenta, "hillstrom": build_hillstrom}
STRING_STORAGE = {"criteo": "pyarrow", "lenta": "python", "hillstrom": "python"}


def _storage(table: str) -> str:
    want = STRING_STORAGE[table]
    if want == "pyarrow":
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            print(f"[ingest] {table}: pyarrow not installed; string storage 'python' instead (pickle bytes will differ "
                  f"from ours, content is unaffected)", flush=True)
            return "python"
    return want


def find_raw(data_dir: Path, table: str) -> Path:
    d, cands = RAW[table]
    for c in cands:
        p = data_dir / d / c
        if p.exists():
            return p
    raise FileNotFoundError(f"{table}: none of {[str(Path(d) / c) for c in cands]} under {data_dir}")


def ingest(table: str, data_dir: Path, out_dir: Path, force: bool = False, check_raw: bool = True) -> Path:
    raw = find_raw(data_dir, table)
    if check_raw:
        got = sha256_file(raw)
        if got != RAW_SHA256[table]:
            raise RuntimeError(f"{table}: raw file {raw.name} sha256 {got} differs from the expected "
                               f"{RAW_SHA256[table]}")
    out = out_dir / RAW[table][0] / "tidy.pkl"
    if out.exists() and not force:
        raise SystemExit(f"{out} exists; pass --force to overwrite (or --out-dir elsewhere)")
    with pd.option_context("mode.string_storage", _storage(table)):
        df = BUILDERS[table](raw)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(out)
    print(f"[ingest] {table}: {len(df)} rows x {df.shape[1]} cols -> {out} (sha256 {sha256_file(out)})", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", choices=("criteo", "lenta", "hillstrom", "all"), required=True)
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-raw-check", action="store_true", help="skip the raw-file sha256 check")
    a = ap.parse_args()
    data_dir = Path(a.data_dir)
    out_dir = Path(a.out_dir) if a.out_dir else data_dir
    tables = ("criteo", "lenta", "hillstrom") if a.table == "all" else (a.table,)
    for t in tables:
        ingest(t, data_dir, out_dir, a.force, not a.no_raw_check)


if __name__ == "__main__":
    main()
