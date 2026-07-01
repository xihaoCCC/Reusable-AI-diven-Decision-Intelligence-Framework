from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import pandas as pd

from src.data_mapping.htcds_plus_schema import HTCDSPlusSchema
from src.data_mapping.model_features import ModelFeatureConfig


class MappingStatus(str, Enum):
    COVERED = "covered"
    PARTIAL = "partial"
    EXTENSION = "extension"


@dataclass(frozen=True)
class CTDCFieldMapping:
    source_field: str
    status: MappingStatus
    htcds_field: str | None = None
    extension_field: str | None = None
    modeling_role: str = "available_not_selected"
    notes: str = ""


CTDC_FIELD_MAPPINGS: tuple[CTDCFieldMapping, ...] = (
    CTDCFieldMapping("yearOfRegistration", MappingStatus.EXTENSION,
                     extension_field="case.registration_year", modeling_role="filter",
                     notes="Operational registration year has no field in the cleaned standard."),
    CTDCFieldMapping("gender", MappingStatus.COVERED, "Gender", modeling_role="feature",
                     notes="CTDC categories are translated to cleaned HTCDS values."),
    CTDCFieldMapping("ageBroad", MappingStatus.EXTENSION,
                     extension_field="person.age_at_identification_broad", modeling_role="feature",
                     notes="Age is measured at identification/assistance, not when first trafficked."),
    CTDCFieldMapping("citizenship", MappingStatus.COVERED, "Citizenship"),
    CTDCFieldMapping("CountryOfExploitation", MappingStatus.EXTENSION,
                     extension_field="exploitation.country", modeling_role="filter",
                     notes="Useful location field absent from the cleaned standard."),
    CTDCFieldMapping("traffickMonths", MappingStatus.EXTENSION,
                     extension_field="trafficking.duration_broad",
                     notes="Duration band has no field in the cleaned standard."),
    CTDCFieldMapping("meansDebtBondageEarnings", MappingStatus.PARTIAL, "MethodsOfControl",
                     "control.debt_bondage_or_takes_earnings", "feature",
                     "Composite flag cannot distinguish debt bondage from taken earnings."),
    CTDCFieldMapping("meansThreats", MappingStatus.PARTIAL, "MethodsOfControl",
                     "control.threats", "feature",
                     "Composite flag merges threats, use of children, and law-enforcement threats."),
    CTDCFieldMapping("meansAbusePsyPhySex", MappingStatus.PARTIAL, "MethodsOfControl",
                     "control.abuse", "feature",
                     "Composite flag merges psychological, physical, and sexual abuse."),
    CTDCFieldMapping("meansFalsePromises", MappingStatus.COVERED, "MethodsOfControl",
                     "control.false_promises", "feature"),
    CTDCFieldMapping("meansDrugsAlcohol", MappingStatus.COVERED, "MethodsOfControl",
                     "control.psychoactive_substances", "feature"),
    CTDCFieldMapping("meansDenyBasicNeeds", MappingStatus.PARTIAL, "MethodsOfControl",
                     "control.restricted_needs", "feature",
                     "Composite flag merges finance, movement, medical care, and necessities."),
    CTDCFieldMapping("meansExcessiveWorkHours", MappingStatus.COVERED, "MethodsOfControl",
                     "control.excessive_working_hours", "feature"),
    CTDCFieldMapping("meansWithholdDocs", MappingStatus.COVERED, "MethodsOfControl",
                     "control.withholds_documents", "feature"),
    CTDCFieldMapping("isForcedLabour", MappingStatus.COVERED, "TypeOfExploitation",
                     modeling_role="target"),
    CTDCFieldMapping("isSexualExploit", MappingStatus.COVERED, "TypeOfExploitation",
                     modeling_role="target"),
    CTDCFieldMapping("isOtherExploit", MappingStatus.COVERED, "TypeOfExploitation",
                     modeling_role="target_filter"),
    CTDCFieldMapping("sectorOfLabourAgriculture", MappingStatus.PARTIAL,
                     "ForcedLabourIndustry",
                     notes="CTDC agriculture is narrower than the combined HTCDS industry value."),
    CTDCFieldMapping("sectorOfLabourConstruction", MappingStatus.COVERED,
                     "ForcedLabourIndustry"),
    CTDCFieldMapping("sectorOfLabourDomesticWork", MappingStatus.COVERED,
                     "ForcedLabourIndustry"),
    CTDCFieldMapping("sectorOfLabourHospitality", MappingStatus.COVERED,
                     "ForcedLabourIndustry"),
    CTDCFieldMapping("sectorOfSexProstitution", MappingStatus.COVERED,
                     "TypeOfSexExploitation"),
    CTDCFieldMapping("sectorOfSexPornography", MappingStatus.COVERED,
                     "TypeOfSexExploitation"),
    CTDCFieldMapping("recruiterRelationIntimatePartner", MappingStatus.PARTIAL,
                     "RelationshipToTrafficker", "recruiter.intimate_partner", "feature",
                     "Specific CTDC relationship collapses to HTCDS Acquainted."),
    CTDCFieldMapping("recruiterRelationFriend", MappingStatus.PARTIAL,
                     "RelationshipToTrafficker", "recruiter.friend", "feature",
                     "Specific CTDC relationship collapses to HTCDS Acquainted."),
    CTDCFieldMapping("recruiterRelationFamily", MappingStatus.COVERED,
                     "RelationshipToTrafficker", "recruiter.family", "feature"),
    CTDCFieldMapping("recruiterRelationOther", MappingStatus.PARTIAL,
                     "RelationshipToTrafficker", "recruiter.other", "feature",
                     "Other recruiter relationships cannot be represented precisely."),
)


DIRECT_CONTROL_VALUES = {
    "meansFalsePromises": "FalsePromises",
    "meansDrugsAlcohol": "PsychoactiveSubstance",
    "meansExcessiveWorkHours": "ExcessiveWorkingHours",
    "meansWithholdDocs": "WithholdsDocuments",
}

EXTENSION_SOURCE_FIELDS = {
    mapping.source_field: mapping.extension_field
    for mapping in CTDC_FIELD_MAPPINGS
    if mapping.extension_field
}

AGE_ORDER = ("0--8", "9--17", "18--20", "21--23", "24--26",
             "27--29", "30--38", "39--47", "48+")
AGE_ORDINAL = {value: index for index, value in enumerate(AGE_ORDER)}


class CTDCHTCDSPlusMapper:
    """Map CTDC records to official HTCDS fields and curated HTCDS+ fields."""

    def __init__(self, schema: HTCDSPlusSchema) -> None:
        self.schema = schema
        self._validate_mapping_specification()

    def map_records(self, records: pd.DataFrame, id_col: str = "case_id") -> pd.DataFrame:
        mapped = pd.DataFrame(index=records.index)
        mapped["case_id"] = (
            records[id_col].astype(str)
            if id_col in records.columns
            else pd.Series(records.index, index=records.index).map(lambda i: f"ctdc-{i}")
        )

        mapped["Gender"] = _map_gender(
            records.get("gender", pd.Series(pd.NA, index=records.index, dtype="object"))
        )
        mapped["Citizenship"] = records.get("citizenship", pd.Series(pd.NA, index=records.index))
        mapped["MethodsOfControl"] = records.apply(_map_direct_controls, axis=1)
        mapped["TypeOfExploitation"] = records.apply(_map_exploitation_type, axis=1)
        mapped["ForcedLabourIndustry"] = records.apply(_map_labour_industries, axis=1)
        mapped["TypeOfSexExploitation"] = records.apply(_map_sex_exploitation, axis=1)
        mapped["RelationshipToTrafficker"] = records.apply(_map_relationship, axis=1)

        for source_field, extension_field in EXTENSION_SOURCE_FIELDS.items():
            source = records.get(source_field, pd.Series(pd.NA, index=records.index))
            if source_field.startswith(("means", "recruiterRelation")):
                mapped[extension_field] = _binary(source)
            elif source_field == "ageBroad":
                mapped[extension_field] = source.replace({"09--17": "9--17"})
            else:
                mapped[extension_field] = source

        self._validate_standardized_values(mapped)
        self._validate_extension_values(mapped)
        return mapped

    def coverage_report(
        self, available_columns: Iterable[str] | None = None
    ) -> pd.DataFrame:
        available = set(available_columns) if available_columns is not None else None
        rows = []
        for mapping in CTDC_FIELD_MAPPINGS:
            rows.append(
                {
                    "ctdc_field": mapping.source_field,
                    "available": available is None or mapping.source_field in available,
                    "mapping_status": mapping.status.value,
                    "htcds_field": mapping.htcds_field or "",
                    "extension_field": mapping.extension_field or "",
                    "modeling_role": mapping.modeling_role,
                    "notes": mapping.notes,
                }
            )
        return pd.DataFrame(rows)

    def _validate_mapping_specification(self) -> None:
        invalid = sorted(
            {
                mapping.htcds_field
                for mapping in CTDC_FIELD_MAPPINGS
                if mapping.htcds_field and not self.schema.has_field(mapping.htcds_field)
            }
        )
        if invalid:
            raise ValueError(f"CTDC mapping references unknown HTCDS fields: {invalid}")
        invalid_extensions = sorted(
            {
                mapping.extension_field
                for mapping in CTDC_FIELD_MAPPINGS
                if mapping.extension_field and not self.schema.has_field(mapping.extension_field)
            }
        )
        if invalid_extensions:
            raise ValueError(
                f"CTDC mapping references unknown HTCDS+ fields: {invalid_extensions}"
            )

    def _validate_standardized_values(self, mapped: pd.DataFrame) -> None:
        standard_fields = (
            "Gender",
            "Citizenship",
            "MethodsOfControl",
            "TypeOfExploitation",
            "ForcedLabourIndustry",
            "TypeOfSexExploitation",
            "RelationshipToTrafficker",
        )
        for field_name in standard_fields:
            invalid = mapped[field_name].map(
                lambda value: not self.schema.validate_value(field_name, value)
            )
            if invalid.any():
                row_indexes = mapped.index[invalid].tolist()[:5]
                raise ValueError(
                    f"Mapped values for {field_name} violate the HTCDS schema "
                    f"at rows {row_indexes}"
                )

    def _validate_extension_values(self, mapped: pd.DataFrame) -> None:
        for field_name in self.schema.extension_field_names:
            if field_name not in mapped.columns:
                continue
            invalid = mapped[field_name].map(
                lambda value: not self.schema.validate_value(field_name, value)
            )
            if invalid.any():
                row_indexes = mapped.index[invalid].tolist()[:5]
                raise ValueError(
                    f"Mapped values for {field_name} violate the HTCDS+ schema "
                    f"at rows {row_indexes}"
                )


def build_ctdc_exploitation_model_frame(
    mapped_records: pd.DataFrame,
    feature_config: ModelFeatureConfig,
) -> pd.DataFrame:
    """Derive only the configured model inputs from mapped standard/extension fields."""

    output = pd.DataFrame(index=mapped_records.index)
    output["case_id"] = mapped_records["case_id"]
    output["exploitation_type"] = mapped_records["TypeOfExploitation"].map(
        {"SexualExploitation": "Sex", "ForcedLabour": "Labor", "Both": "Both"}
    )

    gender = mapped_records["Gender"]
    output["person.gender_feminine"] = (gender == "Feminine").astype(float)
    output["person.gender_masculine"] = (gender == "Masculine").astype(float)
    output["person.gender_trans_or_nonconforming"] = gender.isin(
        ["TransgenderFeminine", "TransgenderMasculine", "NonConforming"]
    ).astype(float)
    output["person.gender_unknown"] = gender.isna().astype(float)

    age = mapped_records["person.age_at_identification_broad"]
    output["person.age_at_identification_ordinal"] = age.map(AGE_ORDINAL).astype(float)
    output["person.age_unknown"] = age.isna().astype(float)
    output["person.minor_or_young"] = age.isin(["0--8", "9--17"]).astype(float)

    for field_name in (
        "control.debt_bondage_or_takes_earnings",
        "control.threats",
        "control.abuse",
        "control.false_promises",
        "control.psychoactive_substances",
        "control.restricted_needs",
        "control.excessive_working_hours",
        "control.withholds_documents",
        "recruiter.intimate_partner",
        "recruiter.friend",
        "recruiter.family",
        "recruiter.other",
    ):
        output[field_name] = mapped_records[field_name]

    feature_config.validate_columns(set(output.columns))
    selected = ["case_id", feature_config.label_field, *feature_config.model_features]
    return output[selected].copy()


def add_decision_support_features(model_frame: pd.DataFrame) -> pd.DataFrame:
    """Add transparent triage inputs without changing selected model features."""

    output = model_frame.copy()
    output["threats"] = output["control.threats"]
    output["abuse_indicators"] = output["control.abuse"]
    output["denial_basic_needs"] = output["control.restricted_needs"]
    output["drugs_alcohol_control"] = output["control.psychoactive_substances"]
    output["false_promises"] = output["control.false_promises"]
    output["minor_young_indicator"] = output["person.minor_or_young"]
    output["close_recruiter_relationship"] = output[
        ["recruiter.intimate_partner", "recruiter.friend", "recruiter.family"]
    ].max(axis=1)
    output["debt_bondage"] = output["control.debt_bondage_or_takes_earnings"]
    output["excessive_work_hours"] = output["control.excessive_working_hours"]
    output["withheld_documents"] = output["control.withholds_documents"]
    output["control_score"] = output[
        ["control.threats", "control.psychoactive_substances",
         "control.debt_bondage_or_takes_earnings", "control.withholds_documents"]
    ].mean(axis=1)
    output["vulnerability_score"] = output[
        ["control.abuse", "control.restricted_needs", "person.minor_or_young"]
    ].mean(axis=1)
    output["relationship_score"] = output["close_recruiter_relationship"]
    return output


def _map_gender(series: pd.Series) -> pd.Series:
    values = {
        "Woman": "Feminine",
        "Man": "Masculine",
        "Trans/Transgender/NonConforming": "NonConforming",
    }
    return series.map(values)


def _map_direct_controls(row: pd.Series) -> tuple[str, ...] | object:
    source_values = [row.get(source) for source in DIRECT_CONTROL_VALUES]
    if all(pd.isna(value) for value in source_values):
        return pd.NA
    return tuple(
        value for source, value in DIRECT_CONTROL_VALUES.items() if _is_active(row.get(source))
    )


def _map_exploitation_type(row: pd.Series) -> object:
    forced = _is_active(row.get("isForcedLabour"))
    sexual = _is_active(row.get("isSexualExploit"))
    other = _is_active(row.get("isOtherExploit"))
    if forced and sexual:
        return "Both"
    if forced:
        return "ForcedLabour"
    if sexual:
        return "SexualExploitation"
    if other:
        return "Other"
    return pd.NA


def _map_labour_industries(row: pd.Series) -> tuple[str, ...]:
    values = (
        ("sectorOfLabourAgriculture", "A_AgricultureForestryAndFishing"),
        ("sectorOfLabourConstruction", "F_Construction"),
        ("sectorOfLabourDomesticWork", "T_ActivitiesOfHouseholdsAsEmployers"),
        ("sectorOfLabourHospitality", "I_AccommodationAndFoodServiceActivities"),
    )
    return tuple(value for source, value in values if _is_active(row.get(source)))


def _map_sex_exploitation(row: pd.Series) -> tuple[str, ...]:
    values = (
        ("sectorOfSexProstitution", "Prostitution"),
        ("sectorOfSexPornography", "Pornography"),
    )
    return tuple(value for source, value in values if _is_active(row.get(source)))


def _map_relationship(row: pd.Series) -> object:
    if _is_active(row.get("recruiterRelationFamily")):
        return "Familial"
    if any(
        _is_active(row.get(field))
        for field in (
            "recruiterRelationIntimatePartner",
            "recruiterRelationFriend",
            "recruiterRelationOther",
        )
    ):
        return "Acquainted"
    return pd.NA


def _binary(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    output = pd.Series(float("nan"), index=series.index, dtype="float64")
    output.loc[numeric.eq(0)] = 0.0
    output.loc[numeric.eq(1)] = 1.0
    return output


def _is_active(value: object) -> bool:
    try:
        return float(value) == 1.0
    except (TypeError, ValueError):
        return False
