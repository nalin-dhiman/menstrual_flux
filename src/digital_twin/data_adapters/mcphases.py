from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from digital_twin.data.schemas import (
    assert_valid,
    validate_events,
    validate_observations,
    validate_participants,
)
from digital_twin.data_adapters.base import AdapterResult, BaseAdapter


SOURCE_DATASET = "mcPHASES-1.0.0"
TRANSFORMATION_VERSION = "mcphases-1.0.0-adapter-v1"
RELATIVE_EPOCH = pd.Timestamp("2000-01-01")

REQUIRED_COLUMNS = {
    "subject-info.csv": {
        "id",
        "birth_year",
        "age_of_first_menarche",
    },
    "height_and_weight.csv": {
        "id",
        "height_2022",
        "weight_2022",
        "height_2024",
        "weight_2024",
    },
    "hormones_and_selfreport.csv": {
        "id",
        "study_interval",
        "day_in_study",
        "phase",
        "lh",
        "estrogen",
        "pdg",
        "flow_volume",
        "appetite",
        "exerciselevel",
        "headaches",
        "cramps",
        "sorebreasts",
        "fatigue",
        "sleepissue",
        "moodswing",
        "stress",
        "foodcravings",
        "indigestion",
        "bloating",
    },
    "computed_temperature.csv": {
        "id",
        "study_interval",
        "sleep_end_day_in_study",
        "nightly_temperature",
    },
    "resting_heart_rate.csv": {"id", "study_interval", "day_in_study", "value"},
    "heart_rate_variability_details.csv": {"id", "study_interval", "day_in_study", "rmssd"},
    "sleep_score.csv": {"id", "study_interval", "day_in_study", "overall_score"},
    "respiratory_rate_summary.csv": {
        "id",
        "study_interval",
        "day_in_study",
        "full_sleep_breathing_rate",
    },
    "active_minutes.csv": {
        "id",
        "study_interval",
        "day_in_study",
        "sedentary",
        "lightly",
        "moderately",
        "very",
    },
}

LIKERT_MAP = {
    "Not at all": 0.0,
    "Very Low/Little": 1.0,
    "Very Low": 1.0,
    "Low": 2.0,
    "Moderate": 3.0,
    "High": 4.0,
    "Very High": 5.0,
    "1": 1.0,
    "2": 2.0,
    "3": 3.0,
    "4": 4.0,
    "5": 5.0,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _participant_key(value: object) -> str:
    digest = hashlib.sha256(f"{SOURCE_DATASET}:{value}".encode("utf-8")).hexdigest()[:16]
    return f"MCP-{digest}"


def _relative_time(day: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(day, errors="coerce")
    return RELATIVE_EPOCH + pd.to_timedelta(numeric - 1, unit="D") + pd.Timedelta(hours=12)


def _as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _as_likert(series: pd.Series) -> pd.Series:
    mapped = series.astype("string").map(LIKERT_MAP)
    unknown = series.notna() & mapped.isna()
    if unknown.any():
        values = sorted(series.loc[unknown].astype(str).unique())
        raise ValueError(f"Unmapped Likert categories: {values}")
    return mapped.astype(float)


class McPhasesAdapter(BaseAdapter):
    """Release-specific adapter for the restricted mcPHASES v1.0.0 files.

    Absolute calendar dates and actual availability timestamps are absent from
    the release. All curated timestamps therefore use a documented relative
    epoch and same-day availability assumption. Released Mira phase labels are
    retained as device-defined references, never as ultrasound-confirmed truth.
    """

    source_name = "mcPHASES"

    @staticmethod
    def _release_root(source_dir: Path) -> Path:
        source_dir = Path(source_dir).resolve()
        if (source_dir / "hormones_and_selfreport.csv").exists():
            return source_dir
        candidates = list(source_dir.rglob("hormones_and_selfreport.csv"))
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Expected one mcPHASES release below {source_dir}; found {len(candidates)} hormone files"
            )
        return candidates[0].parent

    @staticmethod
    def _read_csv(root: Path, name: str) -> pd.DataFrame:
        return pd.read_csv(root / name, dtype={"id": "string"}, low_memory=False)

    @staticmethod
    def _checksum_manifest(root: Path) -> dict[str, str]:
        manifest = root / "SHA256SUMS.txt"
        if not manifest.exists():
            raise FileNotFoundError(manifest)
        expected: dict[str, str] = {}
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, name = line.split(maxsplit=1)
            expected[name.lstrip("*")] = digest
        required_manifest_entries = set(REQUIRED_COLUMNS) | {"README.txt", "LICENSE.txt"}
        unlisted = sorted(required_manifest_entries - set(expected))
        if unlisted:
            raise ValueError(f"Required release files are not covered by SHA256SUMS.txt: {unlisted}")
        missing = sorted(name for name in expected if not (root / name).is_file())
        if missing:
            raise FileNotFoundError(f"Files listed in SHA256SUMS.txt are missing: {missing}")
        mismatches = sorted(name for name, digest in expected.items() if _sha256(root / name) != digest)
        if mismatches:
            raise ValueError(f"Checksum verification failed: {mismatches}")
        return expected

    @staticmethod
    def _validate_release_columns(root: Path) -> None:
        missing_files = sorted(name for name in REQUIRED_COLUMNS if not (root / name).is_file())
        if missing_files:
            raise FileNotFoundError(f"Required mcPHASES files are missing: {missing_files}")
        for name, required in REQUIRED_COLUMNS.items():
            actual = set(pd.read_csv(root / name, nrows=0).columns)
            missing = sorted(required - actual)
            if missing:
                raise ValueError(f"{name} is missing required columns: {missing}")

    def inspect(self, source_dir: Path) -> dict[str, object]:
        root = self._release_root(source_dir)
        self._validate_release_columns(root)
        checksums = self._checksum_manifest(root)
        subjects = self._read_csv(root, "subject-info.csv")
        daily = self._read_csv(root, "hormones_and_selfreport.csv")
        if subjects["id"].isna().any() or subjects["id"].duplicated().any():
            raise ValueError("subject-info.csv must contain unique, nonmissing participant IDs")
        unknown = set(daily["id"].dropna()) - set(subjects["id"])
        if unknown:
            raise ValueError("hormones_and_selfreport.csv contains IDs absent from subject-info.csv")
        duplicate_keys = int(daily.duplicated(["id", "study_interval", "day_in_study"]).sum())
        if duplicate_keys:
            raise ValueError(f"Duplicate hormone/self-report daily keys: {duplicate_keys}")
        phase_counts = daily["phase"].fillna("<missing>").value_counts().to_dict()
        return {
            "source": self.source_name,
            "release_version": "1.0.0",
            "access_classification": "restricted_health_data",
            "checksums_verified": len(checksums),
            "required_tables_verified": len(REQUIRED_COLUMNS),
            "participants": int(subjects["id"].nunique()),
            "daily_rows": len(daily),
            "study_intervals": sorted(map(int, daily["study_interval"].dropna().unique())),
            "phase_counts": {str(key): int(value) for key, value in phase_counts.items()},
            "coverage": {
                column: float(daily[column].notna().mean())
                for column in ("lh", "estrogen", "pdg", "flow_volume", "appetite", "stress")
            },
            "absolute_dates_available": False,
            "availability_timestamps_available": False,
            "phase_reference": "Mira proprietary interpretation; not clinical ovulation confirmation",
        }

    @staticmethod
    def _annotate_cycles(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        pieces: list[pd.DataFrame] = []
        cycle_rows: list[dict[str, object]] = []
        for (participant_id, interval), group in daily.groupby(
            ["participant_id", "study_interval"], sort=True, dropna=False
        ):
            group = group.sort_values("day_in_study").copy()
            previous_phase = group["phase"].shift()
            previous_day = group["day_in_study"].shift()
            adjacent = group["day_in_study"].sub(previous_day).eq(1)
            onset = (
                group["phase"].eq("Menstrual")
                & previous_phase.notna()
                & ~previous_phase.eq("Menstrual")
                & adjacent
            )
            group["is_menstruation_onset"] = onset
            group["cycle_number"] = onset.cumsum().astype(int)
            prefix = f"{participant_id}-I{int(interval)}"
            group["cycle_id"] = group["cycle_number"].map(
                lambda number: f"{prefix}-C{number:03d}" if number > 0 else f"{prefix}-PRE"
            )

            first_day = int(group["day_in_study"].min())
            last_day = int(group["day_in_study"].max())
            onset_days = group.loc[onset, "day_in_study"].astype(int).tolist()
            first_onset = onset_days[0] if onset_days else None
            cycle_rows.append(
                {
                    "participant_id": participant_id,
                    "study_interval": int(interval),
                    "cycle_id": f"{prefix}-PRE",
                    "cycle_start_day": np.nan,
                    "cycle_end_day": (first_onset - 1) if first_onset is not None else last_day,
                    "next_cycle_start_day": first_onset,
                    "cycle_length_days": np.nan,
                    "left_censored": True,
                    "right_censored": not bool(onset_days),
                    "complete_event_interval": False,
                    "eligible_for_primary_evaluation": False,
                    "eligibility_reason": "left_censored_cycle_start",
                }
            )
            for index, start in enumerate(onset_days, start=1):
                next_start = onset_days[index] if index < len(onset_days) else None
                length = (next_start - start) if next_start is not None else np.nan
                plausible = bool(next_start is not None and 15 <= float(length) <= 60)
                if next_start is None:
                    reason = "right_censored_next_menses"
                elif not plausible:
                    reason = "cycle_interval_outlier_review_required"
                else:
                    reason = "eligible"
                cycle_rows.append(
                    {
                        "participant_id": participant_id,
                        "study_interval": int(interval),
                        "cycle_id": f"{prefix}-C{index:03d}",
                        "cycle_start_day": start,
                        "cycle_end_day": (next_start - 1) if next_start is not None else last_day,
                        "next_cycle_start_day": next_start,
                        "cycle_length_days": length,
                        "left_censored": False,
                        "right_censored": next_start is None,
                        "complete_event_interval": next_start is not None,
                        "eligible_for_primary_evaluation": plausible,
                        "eligibility_reason": reason,
                    }
                )
            pieces.append(group)
        cycles = pd.DataFrame(cycle_rows)
        cycles["time_basis"] = "relative_day_in_study"
        cycles["label_source"] = "released_mira_phase_transition"
        return pd.concat(pieces, ignore_index=True), cycles

    @staticmethod
    def _missing_reason(
        raw_column: str,
        interval: pd.Series,
        values: pd.Series,
        available_participants: pd.Series | None = None,
        participant_id: pd.Series | None = None,
    ) -> pd.Series:
        reason = pd.Series("", index=values.index, dtype="string")
        missing = values.isna()
        reason.loc[missing] = "not_recorded"
        if raw_column == "pdg":
            reason.loc[missing & interval.eq(2022)] = "structurally_unavailable_interval_1"
        if raw_column in {
            "flow_volume",
            "appetite",
            "exerciselevel",
            "headaches",
            "cramps",
            "sorebreasts",
            "fatigue",
            "sleepissue",
            "moodswing",
            "stress",
            "foodcravings",
            "indigestion",
            "bloating",
        }:
            reason.loc[missing & interval.eq(2024)] = "optional_diary_not_reported_interval_2"
        if available_participants is not None and participant_id is not None:
            no_stream = ~participant_id.isin(set(available_participants.dropna().astype(str)))
            reason.loc[missing & no_stream] = "device_stream_unavailable_for_participant"
            reason.loc[missing & ~no_stream] = "device_measurement_not_available"
        return reason

    @staticmethod
    def _observation_rows(
        base: pd.DataFrame,
        *,
        raw_column: str,
        signal_name: str,
        values: pd.Series,
        unit: str,
        source_file: str,
        device: str,
        assay: str,
        missing_reason: pd.Series,
        transformation_note: str,
        source_count: pd.Series | None = None,
    ) -> pd.DataFrame:
        out = base[
            [
                "participant_id",
                "study_interval",
                "day_in_study",
                "cycle_id",
                "phase",
                "relative_time",
            ]
        ].copy()
        out["source_dataset"] = SOURCE_DATASET
        out["source_record_id"] = (
            source_file
            + ":"
            + out["participant_id"]
            + ":"
            + out["study_interval"].astype(str)
            + ":"
            + out["day_in_study"].astype(int).astype(str)
            + ":"
            + signal_name
        )
        out["signal_name"] = signal_name
        out["value"] = values.to_numpy()
        out["unit"] = unit
        out["event_time"] = out["relative_time"]
        out["measurement_time"] = out["relative_time"]
        out["report_time"] = out["relative_time"]
        out["availability_time"] = out["relative_time"]
        out["device"] = device
        out["assay"] = assay
        out["quality_flag"] = np.where(
            out["value"].isna(),
            "missing",
            "pass_relative_day_availability_assumed_same_day",
        )
        out["missingness_reason"] = missing_reason.to_numpy()
        out["raw_column"] = raw_column
        out["transformation_version"] = TRANSFORMATION_VERSION
        out["transformation_note"] = transformation_note
        out["released_phase_label"] = out.pop("phase")
        out["time_basis"] = "relative_day_in_study_no_absolute_date"
        out["source_record_count"] = (
            source_count.to_numpy() if source_count is not None else np.ones(len(out), dtype=int)
        )
        return out

    @staticmethod
    def _daily_median(
        root: Path,
        name: str,
        day_column: str,
        value_column: str,
        participant_map: dict[str, str],
        filter_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        frame = McPhasesAdapter._read_csv(root, name)
        frame["participant_id"] = frame["id"].map(participant_map)
        frame["study_interval"] = pd.to_numeric(frame["study_interval"], errors="raise").astype(int)
        frame["day_in_study"] = pd.to_numeric(frame[day_column], errors="raise").astype(int)
        if filter_fn is not None:
            frame = filter_fn(frame)
        frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
        keys = ["participant_id", "study_interval", "day_in_study"]
        aggregate = (
            frame.groupby(keys, as_index=False)
            .agg(value=(value_column, "median"), source_record_count=(value_column, "size"))
        )
        return aggregate, frame["participant_id"]

    @staticmethod
    def _participants(subjects: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
        participant_map = {str(value): _participant_key(value) for value in subjects["id"]}
        birth_year = pd.to_numeric(subjects["birth_year"], errors="coerce")
        participants = pd.DataFrame(
            {
                "participant_id": subjects["id"].map(participant_map),
                "source_dataset": SOURCE_DATASET,
                "age": 2022 - birth_year,
                "age_unit": "years_approximate_from_birth_year",
                "reproductive_stage": "naturally_cycling_adult_eligibility",
                "contraceptive_status": "no_recent_hormonal_contraceptive_use_at_enrollment",
                "medication_status": "not_released",
                "device_type": "Mira Plus; Fitbit Sense; Dexcom G6 interval_1",
                "timezone": "America/Toronto_study_region_not_participant_verified",
                "enrollment_date": pd.NaT,
                "withdrawal_date": pd.NaT,
                "available_covariates": (
                    "approximate_age,age_at_menarche,height,weight,education,"
                    "ethnicity,sexual_activity,menstrual_health_literacy"
                ),
                "source_id_rekeyed": True,
                "data_minimization_note": "sensitive released covariate values omitted from curated participant table",
            }
        )
        return participants, participant_map

    @staticmethod
    def _events(daily: pd.DataFrame) -> pd.DataFrame:
        onset = daily.loc[daily["is_menstruation_onset"]].copy()
        events = pd.DataFrame(
            {
                "participant_id": onset["participant_id"],
                "cycle_id": onset["cycle_id"],
                "event_type": "menstruation_onset",
                "event_time_lower": onset["relative_time"],
                "event_time_upper": onset["relative_time"],
                "event_source": "mcphases_released_mira_phase_transition",
                "certainty": "device_defined_not_clinically_adjudicated",
                "availability_time": onset["relative_time"],
                "study_interval": onset["study_interval"].astype(int),
                "day_in_study": onset["day_in_study"].astype(int),
                "time_basis": "relative_day_in_study_availability_assumed_same_day",
            }
        )
        return events.reset_index(drop=True)

    @staticmethod
    def _reference_intervals(daily: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for (participant_id, interval), group in daily.groupby(
            ["participant_id", "study_interval"], sort=True
        ):
            group = group.sort_values("day_in_study").reset_index(drop=True)
            segment = (
                group["phase"].ne(group["phase"].shift())
                | group["day_in_study"].sub(group["day_in_study"].shift()).ne(1)
            ).cumsum()
            for _, window in group.loc[group["phase"].eq("Fertility")].groupby(segment):
                start = window.iloc[0]
                end = window.iloc[-1]
                rows.append(
                    {
                        "participant_id": participant_id,
                        "cycle_id": start["cycle_id"],
                        "event_type": "mira_fertility_window",
                        "event_time_lower": start["relative_time"],
                        "event_time_upper": end["relative_time"],
                        "event_source": "mcphases_released_mira_phase_label",
                        "certainty": "proprietary_device_defined_not_clinical_ovulation",
                        "availability_time": end["relative_time"],
                        "study_interval": int(interval),
                        "reference_method": "contiguous released Fertility phase days",
                    }
                )
            previous = group.shift()
            transition = (
                group["phase"].eq("Luteal")
                & previous["phase"].eq("Fertility")
                & group["day_in_study"].sub(previous["day_in_study"]).eq(1)
            )
            for index in group.index[transition]:
                current = group.loc[index]
                prior = group.loc[index - 1]
                rows.append(
                    {
                        "participant_id": participant_id,
                        "cycle_id": current["cycle_id"],
                        "event_type": "mira_fertility_to_luteal_transition",
                        "event_time_lower": prior["relative_time"],
                        "event_time_upper": current["relative_time"],
                        "event_source": "mcphases_released_mira_phase_transition",
                        "certainty": "proprietary_device_defined_not_clinical_ovulation",
                        "availability_time": current["relative_time"],
                        "study_interval": int(interval),
                        "reference_method": "interval between adjacent Fertility and Luteal labels",
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def _quality_ledger(
        daily: pd.DataFrame, observations: pd.DataFrame, cycles: pd.DataFrame
    ) -> pd.DataFrame:
        keys = ["participant_id", "study_interval"]
        ledger = daily.groupby(keys, as_index=False).agg(
            first_day=("day_in_study", "min"),
            last_day=("day_in_study", "max"),
            daily_grid_rows=("day_in_study", "size"),
            menstruation_onsets=("is_menstruation_onset", "sum"),
        )
        coverage = (
            observations.assign(observed=observations["value"].notna())
            .groupby(keys + ["signal_name"])["observed"]
            .mean()
            .unstack("signal_name")
            .add_prefix("coverage_")
            .reset_index()
        )
        cycle_summary = cycles.groupby(keys, as_index=False).agg(
            cycle_records=("cycle_id", "size"),
            complete_event_intervals=("complete_event_interval", "sum"),
            primary_eligible_cycles=("eligible_for_primary_evaluation", "sum"),
            left_censored_records=("left_censored", "sum"),
            right_censored_records=("right_censored", "sum"),
        )
        ledger = ledger.merge(coverage, on=keys, how="left").merge(cycle_summary, on=keys, how="left")
        ledger["access_classification"] = "restricted_health_data"
        ledger["absolute_dates_available"] = False
        ledger["availability_timestamps_available"] = False
        ledger["phase_reference"] = "Mira proprietary interpretation"
        return ledger

    def convert(self, source_dir: Path) -> AdapterResult:
        root = self._release_root(source_dir)
        inspection = self.inspect(root)
        subjects = self._read_csv(root, "subject-info.csv")
        participants, participant_map = self._participants(subjects)

        daily = self._read_csv(root, "hormones_and_selfreport.csv")
        daily["participant_id"] = daily["id"].map(participant_map)
        daily["study_interval"] = pd.to_numeric(daily["study_interval"], errors="raise").astype(int)
        daily["day_in_study"] = pd.to_numeric(daily["day_in_study"], errors="raise").astype(int)
        daily = daily.sort_values(["participant_id", "study_interval", "day_in_study"])
        daily, cycles = self._annotate_cycles(daily)
        daily["relative_time"] = _relative_time(daily["day_in_study"])

        observation_frames: list[pd.DataFrame] = []
        hormone_specs = [
            ("lh", "lh", "mIU/mL"),
            ("estrogen", "e3g", "ng/mL"),
            ("pdg", "pdg", "mcg/mL"),
        ]
        for raw_column, signal_name, unit in hormone_specs:
            values = _as_numeric(daily[raw_column])
            reason = self._missing_reason(
                raw_column, daily["study_interval"], values
            )
            observation_frames.append(
                self._observation_rows(
                    daily,
                    raw_column=raw_column,
                    signal_name=signal_name,
                    values=values,
                    unit=unit,
                    source_file="hormones_and_selfreport.csv",
                    device="Mira Plus",
                    assay="home urinary hormone analyzer",
                    missing_reason=reason,
                    transformation_note="numeric value preserved; E3G released under raw column estrogen",
                )
            )

        bleeding = daily["flow_volume"].map(
            lambda value: np.nan if pd.isna(value) else float(value != "Not at all")
        )
        observation_frames.append(
            self._observation_rows(
                daily,
                raw_column="flow_volume",
                signal_name="bleeding_reported",
                values=bleeding,
                unit="binary",
                source_file="hormones_and_selfreport.csv",
                device="daily diary",
                assay="self_report",
                missing_reason=self._missing_reason(
                    "flow_volume", daily["study_interval"], bleeding
                ),
                transformation_note="1 for any released flow category other than Not at all; 0 otherwise",
            )
        )

        symptoms = [
            "appetite",
            "exerciselevel",
            "headaches",
            "cramps",
            "sorebreasts",
            "fatigue",
            "sleepissue",
            "moodswing",
            "stress",
            "foodcravings",
            "indigestion",
            "bloating",
        ]
        for raw_column in symptoms:
            values = _as_likert(daily[raw_column])
            observation_frames.append(
                self._observation_rows(
                    daily,
                    raw_column=raw_column,
                    signal_name=f"self_report_{raw_column}",
                    values=values,
                    unit="ordinal_0_5",
                    source_file="hormones_and_selfreport.csv",
                    device="daily diary",
                    assay="self_report",
                    missing_reason=self._missing_reason(
                        raw_column, daily["study_interval"], values
                    ),
                    transformation_note="released Likert category mapped explicitly to documented 0-5 scale",
                )
            )

        wearable_specs = [
            (
                "computed_temperature.csv",
                "sleep_end_day_in_study",
                "nightly_temperature",
                "nightly_skin_temperature",
                "degC",
                "Fitbit Sense",
                None,
            ),
            (
                "resting_heart_rate.csv",
                "day_in_study",
                "value",
                "resting_heart_rate",
                "bpm",
                "Fitbit Sense",
                lambda frame: frame.loc[pd.to_numeric(frame["value"], errors="coerce") > 0],
            ),
            (
                "heart_rate_variability_details.csv",
                "day_in_study",
                "rmssd",
                "sleep_hrv_rmssd",
                "ms",
                "Fitbit Sense",
                None,
            ),
            (
                "sleep_score.csv",
                "day_in_study",
                "overall_score",
                "sleep_overall_score",
                "score",
                "Fitbit Sense",
                None,
            ),
            (
                "respiratory_rate_summary.csv",
                "day_in_study",
                "full_sleep_breathing_rate",
                "sleep_respiratory_rate",
                "breaths/min",
                "Fitbit Sense",
                None,
            ),
        ]
        merge_keys = ["participant_id", "study_interval", "day_in_study"]
        for (
            source_file,
            day_column,
            raw_column,
            signal_name,
            unit,
            device,
            filter_fn,
        ) in wearable_specs:
            aggregate, available_participants = self._daily_median(
                root,
                source_file,
                day_column,
                raw_column,
                participant_map,
                filter_fn,
            )
            merged = daily[merge_keys].merge(aggregate, on=merge_keys, how="left")
            values = merged["value"]
            reason = self._missing_reason(
                raw_column,
                daily["study_interval"],
                values,
                available_participants,
                daily["participant_id"],
            )
            observation_frames.append(
                self._observation_rows(
                    daily,
                    raw_column=raw_column,
                    signal_name=signal_name,
                    values=values,
                    unit=unit,
                    source_file=source_file,
                    device=device,
                    assay="device_derived",
                    missing_reason=reason,
                    transformation_note="median of released records by participant, interval, and assigned day",
                    source_count=merged["source_record_count"].fillna(0).astype(int),
                )
            )

        active = self._read_csv(root, "active_minutes.csv")
        active["participant_id"] = active["id"].map(participant_map)
        active["study_interval"] = pd.to_numeric(active["study_interval"], errors="raise").astype(int)
        active["day_in_study"] = pd.to_numeric(active["day_in_study"], errors="raise").astype(int)
        available_active = active["participant_id"]
        active_columns = [
            ("sedentary", "sedentary_minutes"),
            ("lightly", "light_activity_minutes"),
            ("moderately", "moderate_activity_minutes"),
            ("very", "vigorous_activity_minutes"),
        ]
        for raw_column, signal_name in active_columns:
            aggregate = (
                active.assign(raw_value=pd.to_numeric(active[raw_column], errors="coerce"))
                .groupby(merge_keys, as_index=False)
                .agg(value=("raw_value", "median"), source_record_count=("raw_value", "size"))
            )
            merged = daily[merge_keys].merge(aggregate, on=merge_keys, how="left")
            values = merged["value"]
            observation_frames.append(
                self._observation_rows(
                    daily,
                    raw_column=raw_column,
                    signal_name=signal_name,
                    values=values,
                    unit="minutes/day",
                    source_file="active_minutes.csv",
                    device="Fitbit Sense",
                    assay="device_derived",
                    missing_reason=self._missing_reason(
                        raw_column,
                        daily["study_interval"],
                        values,
                        available_active,
                        daily["participant_id"],
                    ),
                    transformation_note="median of released daily records",
                    source_count=merged["source_record_count"].fillna(0).astype(int),
                )
            )

        observations = pd.concat(observation_frames, ignore_index=True)
        hormone_measurements = observations.loc[
            observations["signal_name"].isin({"lh", "e3g", "pdg"}),
            [
                "participant_id",
                "study_interval",
                "day_in_study",
                "cycle_id",
                "signal_name",
                "value",
                "unit",
                "event_time",
                "availability_time",
                "device",
                "assay",
                "missingness_reason",
                "released_phase_label",
                "source_record_id",
                "transformation_version",
                "time_basis",
            ],
        ].copy()
        events = self._events(daily)
        references = self._reference_intervals(daily)
        quality = self._quality_ledger(daily, observations, cycles)

        assert_valid(
            validate_participants(participants),
            validate_observations(observations),
            validate_events(events),
            validate_events(references),
        )

        expected = self._checksum_manifest(root)
        provenance = [
            {
                "source_dataset": SOURCE_DATASET,
                "source_file": name,
                "sha256": digest,
                "license": "PhysioNet Restricted Health Data License 1.5.0",
                "transformation_version": TRANSFORMATION_VERSION,
            }
            for name, digest in sorted(expected.items())
        ]
        provenance.append(
            {
                "source_dataset": SOURCE_DATASET,
                "source_file": "adapter_metadata",
                "sha256": "",
                "license": "restricted_health_data",
                "transformation_version": TRANSFORMATION_VERSION,
                "inspection_summary": str(inspection),
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
