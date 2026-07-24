from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from digital_twin.data.schemas import (
    assert_valid,
    validate_events,
    validate_observations,
    validate_participants,
)
from digital_twin.data_adapters.base import AdapterResult, BaseAdapter


SOURCE_DATASET = "salzburg-salivary-hormones-osf-qevzh"
TRANSFORMATION_VERSION = "salzburg-qevzh-adapter-v1"
RELATIVE_EPOCH = pd.Timestamp("2001-01-01")
RAW_FILE = "DD_Females_Hormones_anonymized.csv"
REQUIRED_COLUMNS = {
    "VP",
    "Period",
    "LHOvulatoryKit",
    "Estradiol",
    "Progesterone",
    "cycle",
    "cycle_day_forward",
    "cycle_day_backward",
    "expected_backward_count",
    "expected_cycle_length",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _participant_key(value: object) -> str:
    digest = hashlib.sha256(f"{SOURCE_DATASET}:{value}".encode("utf-8")).hexdigest()[:16]
    return f"SAL-{digest}"


class SalzburgHormoneAdapter(BaseAdapter):
    """Adapter for the anonymous OSF qevzh daily salivary-hormone release.

    Calendar dates are not released. Curated timestamps therefore use a
    participant-relative epoch and preserve source row order. Retrospective
    backward-count columns are audited but are never emitted as online
    observations or model features.
    """

    source_name = SOURCE_DATASET
    access_classification = "public_deidentified_participant_data"
    shareable = False
    source_license = "public_OSF_download_no_explicit_dataset_license_observed_2026-07-23"

    @staticmethod
    def _release_root(source_dir: Path) -> Path:
        source_dir = Path(source_dir).resolve()
        if (source_dir / RAW_FILE).is_file():
            return source_dir
        candidates = list(source_dir.rglob(RAW_FILE))
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Expected one {RAW_FILE} below {source_dir}; found {len(candidates)}"
            )
        return candidates[0].parent

    @staticmethod
    def _read(root: Path) -> pd.DataFrame:
        frame = pd.read_csv(root / RAW_FILE)
        missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(f"{RAW_FILE} is missing required columns: {missing}")
        return frame

    @staticmethod
    def _validate(frame: pd.DataFrame) -> None:
        if frame["VP"].isna().any() or frame["cycle"].isna().any():
            raise ValueError("Participant and cycle identifiers must be nonmissing")
        keys = ["VP", "cycle", "cycle_day_forward"]
        duplicates = int(frame.duplicated(keys).sum())
        if duplicates:
            raise ValueError(f"Duplicate participant-cycle-day keys: {duplicates}")
        if not frame.groupby("VP", sort=False)["cycle"].apply(
            lambda values: values.is_monotonic_increasing
        ).all():
            raise ValueError("Cycles are not ordered within participant")
        if not frame.groupby(["VP", "cycle"], sort=False)["cycle_day_forward"].apply(
            lambda values: values.is_monotonic_increasing
        ).all():
            raise ValueError("Cycle days are not ordered")
        if not frame["Period"].dropna().isin([0, 1]).all():
            raise ValueError("Period must be binary")
        if not frame["LHOvulatoryKit"].dropna().isin([0, 1]).all():
            raise ValueError("LHOvulatoryKit must be binary")

    def inspect(self, source_dir: Path) -> dict[str, object]:
        root = self._release_root(source_dir)
        frame = self._read(root)
        self._validate(frame)
        grouped = frame.groupby(["VP", "cycle"], sort=False)
        cycle_summary = grouped.agg(
            rows=("cycle_day_forward", "size"),
            backward_missing=("cycle_day_backward", lambda values: int(values.isna().sum())),
            lh_positive=("LHOvulatoryKit", "sum"),
        )
        complete = cycle_summary["backward_missing"].eq(0)
        boundary = cycle_summary["backward_missing"].eq(cycle_summary["rows"])
        return {
            "source": SOURCE_DATASET,
            "access_classification": self.access_classification,
            "raw_file": str(root / RAW_FILE),
            "raw_sha256": _sha256(root / RAW_FILE),
            "rows": int(len(frame)),
            "participants": int(frame["VP"].nunique()),
            "cycle_records": int(len(cycle_summary)),
            "complete_cycle_intervals_from_raw": int(complete.sum()),
            "terminal_boundary_rows": int(boundary.sum()),
            "lh_positive_cycles": int(cycle_summary["lh_positive"].gt(0).sum()),
            "estradiol_coverage": float(frame["Estradiol"].notna().mean()),
            "progesterone_coverage": float(frame["Progesterone"].notna().mean()),
            "absolute_dates_available": False,
            "future_derived_columns_prohibited_online": [
                "cycle_day_backward",
                "expected_backward_count",
            ],
            "source_discrepancy_note": (
                "The raw file yields 137 complete intervals; the article reports 136. "
                "The adapter preserves the raw release and reports this discrepancy."
            ),
        }

    @staticmethod
    def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        frame["participant_id"] = frame["VP"].map(_participant_key)
        frame["source_cycle_number"] = pd.to_numeric(frame["cycle"], errors="raise").astype(int)
        frame["cycle_day"] = pd.to_numeric(
            frame["cycle_day_forward"], errors="raise"
        ).astype(int)
        frame["study_day"] = frame.groupby("participant_id", sort=False).cumcount() + 1
        frame["relative_date"] = (
            RELATIVE_EPOCH + pd.to_timedelta(frame["study_day"] - 1, unit="D")
        )
        frame["cycle_id"] = (
            frame["participant_id"]
            + "-C"
            + frame["source_cycle_number"].astype(str).str.zfill(3)
        )
        return frame

    @staticmethod
    def _participants(frame: pd.DataFrame) -> pd.DataFrame:
        ids = frame[["participant_id"]].drop_duplicates().reset_index(drop=True)
        return ids.assign(
            source_dataset=SOURCE_DATASET,
            age=np.nan,
            age_unit="not_released",
            reproductive_stage="regularly_cycling_age_18_35_by_study_eligibility",
            contraceptive_status="no_hormonal_contraceptive_use_past_6_months_by_eligibility",
            medication_status="not_released",
            device_type="none",
            timezone="not_released_relative_time_only",
            enrollment_date=pd.NaT,
            withdrawal_date=pd.NaT,
            available_covariates="three_pre_study_cycle_lengths_summarized_as_expected_cycle_length",
            source_id_rekeyed=True,
        )

    @staticmethod
    def _cycles(frame: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for (participant_id, cycle_number), group in frame.groupby(
            ["participant_id", "source_cycle_number"], sort=False
        ):
            group = group.sort_values("cycle_day")
            backward_missing = int(group["cycle_day_backward"].isna().sum())
            complete = backward_missing == 0
            length = int(group["cycle_day"].max()) if complete else np.nan
            plausible = bool(complete and 15 <= float(length) <= 60)
            if not complete:
                reason = "right_censored_terminal_menses_boundary"
            elif not plausible:
                reason = "cycle_interval_outlier_review_required"
            else:
                reason = "eligible"
            start_day = int(group["study_day"].min())
            rows.append(
                {
                    "participant_id": participant_id,
                    "study_interval": 1,
                    "cycle_id": str(group["cycle_id"].iloc[0]),
                    "source_cycle_number": int(cycle_number),
                    "cycle_start_day": start_day,
                    "cycle_end_day": int(group["study_day"].max()),
                    "next_cycle_start_day": start_day + int(length) if complete else np.nan,
                    "cycle_length_days": length,
                    "expected_cycle_length_history": float(
                        pd.to_numeric(group["expected_cycle_length"], errors="coerce").iloc[0]
                    ),
                    "left_censored": False,
                    "right_censored": not complete,
                    "complete_event_interval": complete,
                    "eligible_for_primary_evaluation": plausible,
                    "eligibility_reason": reason,
                    "time_basis": "participant_relative_day_source_row_order",
                    "label_source": "self_reported_menstruation_onset",
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _observation_frame(
        frame: pd.DataFrame,
        *,
        raw_column: str,
        signal_name: str,
        unit: str,
        values: pd.Series,
        measurement_hour: int,
        device: str,
        assay: str,
    ) -> pd.DataFrame:
        event_time = frame["relative_date"] + pd.Timedelta(hours=12)
        measurement_time = frame["relative_date"] + pd.Timedelta(hours=measurement_hour)
        missing = values.isna()
        out = pd.DataFrame(
            {
                "participant_id": frame["participant_id"],
                "source_dataset": SOURCE_DATASET,
                "source_record_id": (
                    RAW_FILE
                    + ":"
                    + frame["participant_id"]
                    + ":"
                    + frame["source_cycle_number"].astype(str)
                    + ":"
                    + frame["cycle_day"].astype(str)
                    + ":"
                    + signal_name
                ),
                "cycle_id": frame["cycle_id"],
                "signal_name": signal_name,
                "value": pd.to_numeric(values, errors="coerce"),
                "unit": unit,
                "event_time": event_time,
                "measurement_time": measurement_time,
                "report_time": measurement_time,
                "availability_time": measurement_time,
                "device": device,
                "assay": assay,
                "quality_flag": np.where(missing, "missing", "pass"),
                "missingness_reason": np.where(missing, "not_recorded_or_failed_assay", ""),
                "raw_column": raw_column,
                "transformation_version": TRANSFORMATION_VERSION,
                "study_interval": 1,
                "day_in_study": frame["study_day"].astype(int),
                "cycle_day": frame["cycle_day"].astype(int),
                "availability_assumption": "same_evening_exact_time_not_released",
            }
        )
        return out

    @staticmethod
    def _events(frame: pd.DataFrame) -> pd.DataFrame:
        onset = frame.loc[pd.to_numeric(frame["Period"], errors="coerce").eq(1)].copy()
        time = onset["relative_date"] + pd.Timedelta(hours=12)
        return pd.DataFrame(
            {
                "participant_id": onset["participant_id"],
                "cycle_id": onset["cycle_id"],
                "event_type": "menstruation_onset",
                "event_time_lower": time,
                "event_time_upper": time,
                "event_source": "daily_period_indicator",
                "certainty": "participant_reported",
                "availability_time": onset["relative_date"] + pd.Timedelta(hours=21),
                "study_interval": 1,
                "day_in_study": onset["study_day"].astype(int),
                "cycle_day": onset["cycle_day"].astype(int),
                "time_basis": "participant_relative_day_source_row_order",
            }
        ).reset_index(drop=True)

    @staticmethod
    def _references(frame: pd.DataFrame) -> pd.DataFrame:
        positive = frame.loc[
            pd.to_numeric(frame["LHOvulatoryKit"], errors="coerce").eq(1)
        ].copy()
        time = positive["relative_date"] + pd.Timedelta(hours=12)
        return pd.DataFrame(
            {
                "participant_id": positive["participant_id"],
                "cycle_id": positive["cycle_id"],
                "event_type": "urinary_lh_positive",
                "event_time_lower": time,
                "event_time_upper": time,
                "event_source": "participant_urinary_lh_kit",
                "certainty": "lh_surge_surrogate_not_confirmed_ovulation",
                "availability_time": positive["relative_date"] + pd.Timedelta(hours=21),
                "study_interval": 1,
                "day_in_study": positive["study_day"].astype(int),
                "cycle_day": positive["cycle_day"].astype(int),
                "reference_method": "first released positive urinary LH indicator",
            }
        ).reset_index(drop=True)

    @staticmethod
    def _quality(
        frame: pd.DataFrame, cycles: pd.DataFrame, references: pd.DataFrame
    ) -> pd.DataFrame:
        ledger = frame.groupby("participant_id", as_index=False).agg(
            source_rows=("study_day", "size"),
            source_cycle_records=("source_cycle_number", "nunique"),
            estradiol_coverage=("Estradiol", lambda values: float(values.notna().mean())),
            progesterone_coverage=("Progesterone", lambda values: float(values.notna().mean())),
        )
        cycle_summary = cycles.groupby("participant_id", as_index=False).agg(
            complete_cycle_intervals=("complete_event_interval", "sum"),
            primary_eligible_cycles=("eligible_for_primary_evaluation", "sum"),
            right_censored_records=("right_censored", "sum"),
        )
        lh = (
            references.groupby("participant_id")
            .size()
            .rename("urinary_lh_positive_events")
            .reset_index()
        )
        ledger = ledger.merge(cycle_summary, on="participant_id", how="left").merge(
            lh, on="participant_id", how="left"
        )
        ledger["urinary_lh_positive_events"] = (
            ledger["urinary_lh_positive_events"].fillna(0).astype(int)
        )
        ledger["access_classification"] = SalzburgHormoneAdapter.access_classification
        ledger["absolute_dates_available"] = False
        ledger["future_columns_excluded_from_online_features"] = (
            "cycle_day_backward,expected_backward_count"
        )
        return ledger

    def convert(self, source_dir: Path) -> AdapterResult:
        root = self._release_root(source_dir)
        inspection = self.inspect(root)
        raw = self._read(root)
        self._validate(raw)
        frame = self._prepare(raw)
        participants = self._participants(frame)
        cycles = self._cycles(frame)

        observations = pd.concat(
            [
                self._observation_frame(
                    frame,
                    raw_column="Estradiol",
                    signal_name="salivary_estradiol",
                    unit="pg/mL",
                    values=frame["Estradiol"],
                    measurement_hour=20,
                    device="none",
                    assay="Salimetrics_ELISA_saliva",
                ),
                self._observation_frame(
                    frame,
                    raw_column="Progesterone",
                    signal_name="salivary_progesterone",
                    unit="pg/mL",
                    values=frame["Progesterone"],
                    measurement_hour=20,
                    device="none",
                    assay="Salimetrics_ELISA_saliva",
                ),
                self._observation_frame(
                    frame,
                    raw_column="Period",
                    signal_name="bleeding_reported",
                    unit="binary",
                    values=frame["Period"],
                    measurement_hour=21,
                    device="daily_diary",
                    assay="self_report",
                ),
            ],
            ignore_index=True,
        )
        events = self._events(frame)
        references = self._references(frame)
        hormone_measurements = observations.loc[
            observations["signal_name"].isin(
                {"salivary_estradiol", "salivary_progesterone"}
            )
        ].reset_index(drop=True)
        quality = self._quality(frame, cycles, references)

        assert_valid(
            validate_participants(participants),
            validate_observations(observations),
            validate_events(events),
            validate_events(references),
        )
        provenance = [
            {
                "path": str((root / RAW_FILE).resolve()),
                "sha256": str(inspection["raw_sha256"]),
                "role": "participant_level_daily_source",
            }
        ]
        readme = root.parent.parent / "README.md"
        if readme.is_file():
            provenance.append(
                {
                    "path": str(readme.resolve()),
                    "sha256": _sha256(readme),
                    "role": "source_readme",
                }
            )
        return AdapterResult(
            participants=participants,
            daily_observations=observations,
            hormone_measurements=hormone_measurements,
            events=events,
            reference_intervals=references,
            cycles=cycles,
            data_quality=quality,
            provenance=provenance,
        )
