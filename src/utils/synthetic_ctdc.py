from __future__ import annotations

import numpy as np
import pandas as pd


AGE_BANDS = ("0--8", "9--17", "18--20", "21--23", "24--26",
             "27--29", "30--38", "39--47", "48+")


def generate_ctdc_style_synthetic_records(
    n_records: int = 200, random_state: int = 42
) -> pd.DataFrame:
    """Generate CTDC-shaped records for public-safe mapping and workflow tests."""

    rng = np.random.default_rng(random_state)
    labels = rng.choice(["Sex", "Labor", "Both"], size=n_records, p=[0.45, 0.4, 0.15])
    rows = []
    for idx, label in enumerate(labels, start=1):
        sex_like = label in {"Sex", "Both"}
        labor_like = label in {"Labor", "Both"}
        both_like = label == "Both"

        young_probability = 0.35 if sex_like else 0.10
        age = (
            rng.choice(["0--8", "9--17"])
            if rng.random() < young_probability
            else rng.choice(AGE_BANDS[2:])
        )
        close_relation = rng.random() < (0.45 if sex_like else 0.20)

        rows.append(
            {
                "case_id": f"CTDC-SYN-{idx:04d}",
                "yearOfRegistration": int(rng.integers(2016, 2024)),
                "gender": rng.choice(
                    ["Woman", "Man", "Trans/Transgender/NonConforming"],
                    p=[0.58, 0.39, 0.03],
                ),
                "ageBroad": age,
                "citizenship": rng.choice(["USA", "MEX", "GTM", "PHL"]),
                "CountryOfExploitation": "USA",
                "traffickMonths": rng.choice(
                    ["0--12 (0-1 yr)", "13--24 (1-2 yrs)", "25+ (2+ yrs)"]
                ),
                "meansDebtBondageEarnings": _ctdc_flag(
                    rng, 0.55 if labor_like else 0.10
                ),
                "meansThreats": _ctdc_flag(rng, 0.45 if both_like else 0.30),
                "meansAbusePsyPhySex": _ctdc_flag(
                    rng, 0.55 if sex_like else 0.25
                ),
                "meansFalsePromises": _ctdc_flag(
                    rng, 0.45 if labor_like else 0.18
                ),
                "meansDrugsAlcohol": _ctdc_flag(rng, 0.35 if sex_like else 0.12),
                "meansDenyBasicNeeds": _ctdc_flag(
                    rng, 0.55 if labor_like else 0.25
                ),
                "meansExcessiveWorkHours": _ctdc_flag(
                    rng, 0.70 if labor_like else 0.08
                ),
                "meansWithholdDocs": _ctdc_flag(
                    rng, 0.50 if labor_like else 0.08
                ),
                "isForcedLabour": 1.0 if labor_like else np.nan,
                "isSexualExploit": 1.0 if sex_like else np.nan,
                "isOtherExploit": np.nan,
                "sectorOfLabourAgriculture": _ctdc_flag(
                    rng, 0.20 if labor_like else 0.02
                ),
                "sectorOfLabourConstruction": _ctdc_flag(
                    rng, 0.25 if labor_like else 0.02
                ),
                "sectorOfLabourDomesticWork": _ctdc_flag(
                    rng, 0.25 if labor_like else 0.02
                ),
                "sectorOfLabourHospitality": _ctdc_flag(
                    rng, 0.20 if labor_like else 0.02
                ),
                "sectorOfSexProstitution": _ctdc_flag(
                    rng, 0.75 if sex_like else 0.02
                ),
                "sectorOfSexPornography": _ctdc_flag(
                    rng, 0.20 if sex_like else 0.01
                ),
                "recruiterRelationIntimatePartner": 1.0
                if close_relation and rng.random() < 0.45
                else np.nan,
                "recruiterRelationFriend": 1.0
                if close_relation and rng.random() < 0.35
                else np.nan,
                "recruiterRelationFamily": 1.0
                if close_relation and rng.random() < 0.25
                else np.nan,
                "recruiterRelationOther": _ctdc_flag(
                    rng, 0.30 if labor_like else 0.12
                ),
            }
        )
    return pd.DataFrame(rows)


def _ctdc_flag(rng: np.random.Generator, probability: float) -> float:
    return 1.0 if rng.random() < probability else np.nan
