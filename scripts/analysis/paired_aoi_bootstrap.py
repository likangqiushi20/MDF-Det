"""Paired AOI-level uncertainty analysis for the first-round revision table.

The six F1 pairs below are the values reported in Table 1 of the first-round
revision.  AOIs, rather than adjacent video frames, are resampled because the
frames within an AOI are temporally correlated.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np


AOIS = np.asarray(["01", "02", "03", "34", "40", "41"])
MDF_DET = np.asarray([0.848, 0.914, 0.888, 0.941, 0.835, 0.843])
HM_NET = np.asarray([0.759, 0.893, 0.840, 0.927, 0.750, 0.825])
SEED = 20260907
RESAMPLES = 10_000
T_CRITICAL_DF5_975 = 2.570581835636305


def main() -> None:
    difference = MDF_DET - HM_NET
    mean_difference = float(np.mean(difference))
    sample_std = float(np.std(difference, ddof=1))
    standard_error = sample_std / np.sqrt(len(difference))
    t_interval = [
        mean_difference - T_CRITICAL_DF5_975 * standard_error,
        mean_difference + T_CRITICAL_DF5_975 * standard_error,
    ]

    rng = np.random.default_rng(SEED)
    sampled_indexes = rng.integers(
        0, len(AOIS), size=(RESAMPLES, len(AOIS)), endpoint=False
    )
    bootstrap_means = difference[sampled_indexes].mean(axis=1)
    bootstrap_interval = np.percentile(bootstrap_means, [2.5, 97.5]).tolist()

    leave_one_out = []
    for index, aoi in enumerate(AOIS):
        retained = np.arange(len(AOIS)) != index
        leave_one_out.append(
            {
                "excluded_aoi": str(aoi),
                "mean_paired_f1_improvement": float(np.mean(difference[retained])),
            }
        )

    # Exact paired sign-flip randomization test over all 2^6 sign assignments.
    sign_flip_means = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(difference)):
        sign_flip_means.append(float(np.mean(difference * np.asarray(signs))))
    observed_abs = abs(mean_difference)
    exact_two_sided_p = float(
        np.mean(np.abs(sign_flip_means) >= observed_abs - 1e-15)
    )

    result = {
        "input_source": "First-round revision, Table 1 (reported three-decimal F1 values)",
        "comparison": "MDF-Det minus HM-Net",
        "resampling_unit": "AOI",
        "reason_for_aoi_unit": "Adjacent WAMI frames are temporally correlated",
        "seed": SEED,
        "bootstrap_resamples": RESAMPLES,
        "per_aoi": [
            {
                "aoi": str(aoi),
                "mdf_det_f1": float(mdf),
                "hm_net_f1": float(hm),
                "paired_difference": float(diff),
            }
            for aoi, mdf, hm, diff in zip(AOIS, MDF_DET, HM_NET, difference)
        ],
        "summary": {
            "mean_mdf_det_f1": float(np.mean(MDF_DET)),
            "mean_hm_net_f1": float(np.mean(HM_NET)),
            "mean_paired_f1_improvement": mean_difference,
            "sample_std_of_paired_improvement": sample_std,
            "bootstrap_95_percentile_ci": bootstrap_interval,
            "paired_t_95_ci": t_interval,
            "exact_sign_flip_two_sided_p": exact_two_sided_p,
            "positive_aoi_improvements": int(np.sum(difference > 0)),
            "total_aois": int(len(AOIS)),
            "minimum_improvement": float(np.min(difference)),
            "maximum_improvement": float(np.max(difference)),
        },
        "leave_one_aoi_out": leave_one_out,
        "limitations": [
            "Only six geographic units are available.",
            "The calculation uses the three-decimal values reported in the revision table.",
            "The interval describes cross-AOI uncertainty, not repeated-training randomness.",
        ],
    }

    output = Path("result/revision/paired_aoi_bootstrap_results.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
