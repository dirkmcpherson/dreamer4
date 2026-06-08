#!/usr/bin/env python
"""Download LIBERO demonstration datasets (HDF5) from HuggingFace.

Source repo: https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets (public).
Sizes verified 2026-06-08 via the HF API:

    suite            size      tasks   role
    ---------------------------------------------------------------
    libero_spatial   ~7.5 GB   10      single-suite (spatial reasoning)
    libero_object    ~7.8 GB   10      single-suite (object) -- good POC
    libero_goal      ~7.5 GB   10      single-suite (goal)
    libero_10        ~13.4 GB  10      long-horizon eval (aka LIBERO-Long)
    libero_90        ~67.4 GB  90      MAIN multi-task training suite
    ---------------------------------------------------------------
    ALL              ~103.6 GB 130

Recommended pulls:
    multi-task train + eval : --suites libero_90 libero_10   (~81 GB)
    quick POC               : --suites libero_object         (~7.8 GB)
    everything              : --suites all                   (~104 GB)

Requires: huggingface_hub (installed by infra/update_cluster_env.sh).
Download is resumable -- re-run with the same --out to continue.

Usage:
    python infra/download_libero.py --out /scratch/$USER/data/libero \
        --suites libero_90 libero_10
"""
import argparse

REPO_ID = "yifengzhu-hf/LIBERO-datasets"
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]
APPROX_GB = {
    "libero_spatial": 7.5, "libero_object": 7.8, "libero_goal": 7.5,
    "libero_10": 13.4, "libero_90": 67.4,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="./data/libero",
                    help="destination directory (use cluster scratch, e.g. /scratch/$USER/data/libero)")
    ap.add_argument("--suites", nargs="+", default=["libero_90", "libero_10"],
                    metavar="SUITE", help="suites to fetch, or 'all' (default: libero_90 libero_10)")
    ap.add_argument("--workers", type=int, default=8, help="parallel download workers")
    ap.add_argument("--dry-run", action="store_true", help="print plan + size estimate, don't download")
    args = ap.parse_args()

    suites = SUITES if "all" in args.suites else args.suites
    bad = [s for s in suites if s not in SUITES]
    if bad:
        ap.error(f"unknown suite(s): {bad}. choose from {SUITES} or 'all'")

    est = sum(APPROX_GB[s] for s in suites)
    print(f"Repo   : {REPO_ID}")
    print(f"Out    : {args.out}")
    print(f"Suites : {suites}")
    print(f"Approx : ~{est:.1f} GB  (HDF5; resumable)")

    if args.dry_run:
        print("[dry-run] not downloading.")
        return

    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=args.out,
        allow_patterns=[f"{s}/*" for s in suites],
        max_workers=args.workers,
    )
    print(f"\nDone. Demos under {args.out}/<suite>/*.hdf5")


if __name__ == "__main__":
    main()
