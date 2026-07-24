from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from digital_twin.data.schemas import (
    assert_valid,
    validate_events,
    validate_observations,
    validate_participants,
)
from digital_twin.data_adapters.base import AdapterResult, BaseAdapter


SOURCE_DATASET = "soochow-fudan-heart-rate-mendeley-v58stpfcnm-v1"
TRANSFORMATION_VERSION = "soochow-v58stpfcnm-adapter-v1"
LOCAL_TIMEZONE = "Asia/Shanghai"
MIN_MENSES_FREE_GAP_DAYS = 18


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _participant_key(row_index: int, source_id: int, device: str) -> str:
    raw = f"{SOURCE_DATASET}:{row_index}:{source_id}:{device}"
    return f"SHR-{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


def _cell_vector(value: object, dtype: type | None = None) -> np.ndarray:
    current = value
    while isinstance(current, np.ndarray) and current.dtype == object and current.size == 1:
        current = current.reshape(-1)[0]
    result = np.asarray(current).ravel()
    return result.astype(dtype) if dtype is not None else result


def _matlab_date_vector(value: object, length: int) -> pd.DatetimeIndex:
    items = np.asarray(value).ravel()[:length]
    text: list[str] = []
    for item in items:
        values = np.asarray(item).ravel()
        text.append(str(values[0]) if len(values) else "")
    parsed = pd.to_datetime(
        pd.Series(text), format="mixed", dayfirst=True, errors="coerce"
    )
    if len(parsed) != length or parsed.isna().any():
        raise ValueError("Invalid or incomplete processed daily date vector")
    differences = parsed.diff().dt.days.dropna()
    if not differences.eq(1).all():
        raise ValueError("Processed daily dates are not a contiguous daily grid")
    return pd.DatetimeIndex(parsed)


def _menstruation_onsets(states: np.ndarray) -> np.ndarray:
    event_days = np.flatnonzero(np.asarray(states).astype(float) == 2)
    if not len(event_days):
        return event_days
    new_episode = np.r_[True, np.diff(event_days) >= MIN_MENSES_FREE_GAP_DAYS + 1]
    return event_days[new_episode]


class SoochowHeartRateAdapter(BaseAdapter):
    """Adapter for Mendeley v58stpfcnm raw minute-level heart rate.

    Event/date grids come from the authors' public daily release, while heart
    rate features are recomputed from the raw minute-level records. The
    authors' selected low-missingness segments are never used as the primary
    observation source.
    """

    source_name = SOURCE_DATASET
    access_classification = "public_deidentified_participant_data"
    shareable = True
    source_license = "CC_BY_4.0"

    @staticmethod
    def _release_root(source_dir: Path) -> Path:
        source_dir = Path(source_dir).resolve()
        expected = source_dir / "data" / "menst_ovu_records.mat"
        if expected.is_file():
            return source_dir
        candidates = list(source_dir.rglob("menst_ovu_records.mat"))
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Expected one menst_ovu_records.mat below {source_dir}; found {len(candidates)}"
            )
        if candidates[0].parent.name != "data":
            raise ValueError("Unexpected release layout for menst_ovu_records.mat")
        return candidates[0].parent.parent

    @staticmethod
    def _paths(root: Path) -> tuple[Path, Path, Path]:
        events = root / "data" / "menst_ovu_records.mat"
        daily = root / "codes" / "Figures and code" / "data230519.mat"
        raw = root / "data" / "raw heart rate (minute level)" / "fertile women"
        for path in (events, daily, raw):
            if not path.exists():
                raise FileNotFoundError(path)
        return events, daily, raw

    @staticmethod
    def _raw_path(raw_dir: Path, row_index: int, source_id: int) -> tuple[Path | None, str]:
        if row_index < 39:
            return raw_dir / f"data_{source_id}.mat", "Fitbit_or_source_data_prefix"
        path = raw_dir / f"huawei_{source_id}.mat"
        return (path if path.is_file() else None), "Huawei"

    @staticmethod
    def _daily_release(daily_path: Path) -> list[dict[str, object]]:
        data = loadmat(daily_path, squeeze_me=False, struct_as_record=False)
        required = {
            "women_idlist",
            "women_sleepmean",
            "women_sleepmeandate",
            "Menstrual_state_cell",
            "ovulation_state_cell",
        }
        missing = sorted(required - set(data))
        if missing:
            raise ValueError(f"data230519.mat is missing variables: {missing}")
        ids = data["women_idlist"].ravel()
        means = data["women_sleepmean"].ravel()
        dates = data["women_sleepmeandate"].ravel()
        menses = data["Menstrual_state_cell"].ravel()
        ovulation = data["ovulation_state_cell"].ravel()
        if not (len(ids) == len(means) == len(dates) == len(menses) == len(ovulation) == 91):
            raise ValueError("Expected 91 aligned fertile-women source cells")
        rows: list[dict[str, object]] = []
        for row_index, (source_id, mean, date, men, ovu) in enumerate(
            zip(ids, means, dates, menses, ovulation)
        ):
            men_vector = _cell_vector(men, float)
            ovu_vector = _cell_vector(ovu, float)
            mean_vector = _cell_vector(mean, float)
            if not (len(men_vector) == len(ovu_vector) == len(mean_vector)):
                raise ValueError(f"Unaligned state/mean arrays at source row {row_index}")
            date_vector = _matlab_date_vector(date, len(men_vector))
            device = "Fitbit_or_source_data_prefix" if row_index < 39 else "Huawei"
            rows.append(
                {
                    "row_index": row_index,
                    "source_id": int(source_id),
                    "device": device,
                    "dates": date_vector,
                    "menstruation_state": men_vector,
                    "ovulation_state": ovu_vector,
                    "authors_processed_night_mean": mean_vector,
                }
            )
        return rows

    @staticmethod
    def _load_raw_features(path: Path | None) -> pd.DataFrame:
        columns = [
            "date",
            "daily_heart_rate_mean",
            "daily_record_count",
            "night_heart_rate_median",
            "night_record_count",
        ]
        if path is None or not path.is_file():
            return pd.DataFrame(columns=columns)
        data = loadmat(path, squeeze_me=True)
        variables = [name for name in data if not name.startswith("__")]
        if len(variables) != 1:
            raise ValueError(f"Expected one data variable in {path}")
        values = np.asarray(data[variables[0]])
        if values.size == 0:
            return pd.DataFrame(columns=columns)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError(f"Expected two raw columns in {path}")
        frame = pd.DataFrame(values, columns=["timestamp_ms", "heart_rate"])
        frame = frame.loc[
            frame["timestamp_ms"].notna()
            & frame["heart_rate"].gt(30)
            & frame["heart_rate"].le(240)
        ].copy()
        timestamp = pd.to_datetime(
            frame["timestamp_ms"], unit="ms", utc=True, errors="coerce"
        ).dt.tz_convert(LOCAL_TIMEZONE)
        frame = frame.loc[timestamp.notna()].copy()
        timestamp = timestamp.loc[timestamp.notna()]
        frame["date"] = timestamp.dt.tz_localize(None).dt.normalize().to_numpy()
        frame["hour"] = timestamp.dt.hour.to_numpy()
        daily = frame.groupby("date", as_index=False).agg(
            daily_heart_rate_mean=("heart_rate", "mean"),
            daily_record_count=("heart_rate", "size"),
        )
        night = (
            frame.loc[frame["hour"].between(0, 5)]
            .groupby("date", as_index=False)
            .agg(
                night_heart_rate_median=("heart_rate", "median"),
                night_record_count=("heart_rate", "size"),
            )
        )
        return daily.merge(night, on="date", how="left")

    def inspect(self, source_dir: Path) -> dict[str, object]:
        root = self._release_root(source_dir)
        events_path, daily_path, raw_dir = self._paths(root)
        rows = self._daily_release(daily_path)
        official_events = loadmat(events_path, squeeze_me=True)
        ids = np.atleast_1d(official_events["women_idlist"]).astype(int)
        daily_ids = np.asarray([row["source_id"] for row in rows], dtype=int)
        if not np.array_equal(ids, daily_ids):
            raise ValueError("Event and daily participant arrays are not aligned")
        episode_count = sum(
            len(_menstruation_onsets(np.asarray(row["menstruation_state"])))
            for row in rows
        )
        lh_count = sum(
            int(np.sum(np.asarray(row["ovulation_state"]) == 2)) for row in rows
        )
        raw_files = sorted(raw_dir.glob("*.mat"))
        mapped = 0
        missing_rows: list[dict[str, object]] = []
        for row in rows:
            path, _ = self._raw_path(
                raw_dir, int(row["row_index"]), int(row["source_id"])
            )
            if path is not None and path.is_file() and path.stat().st_size > 200:
                mapped += 1
            else:
                missing_rows.append(
                    {
                        "row_index": int(row["row_index"]),
                        "source_id": int(row["source_id"]),
                    }
                )
        return {
            "source": SOURCE_DATASET,
            "access_classification": self.access_classification,
            "participant_source_cells": len(rows),
            "unique_numeric_source_ids": len(set(daily_ids.tolist())),
            "source_id_duplicates": int(pd.Series(daily_ids).duplicated().sum()),
            "daily_grid_rows": int(
                sum(len(np.asarray(row["menstruation_state"])) for row in rows)
            ),
            "menstruation_episodes_reconciled_to_release": int(episode_count),
            "urinary_lh_positive_events": int(lh_count),
            "raw_mat_files": len(raw_files),
            "source_rows_with_mapped_nonempty_raw_file": mapped,
            "source_rows_without_raw_file": missing_rows,
            "events_sha256": _sha256(events_path),
            "daily_alignment_sha256": _sha256(daily_path),
            "timezone_interpretation": LOCAL_TIMEZONE,
            "menstruation_onset_rule": (
                f"state=2 after at least {MIN_MENSES_FREE_GAP_DAYS} consecutive non-event days"
            ),
            "authors_low_missingness_segments_used": False,
        }

    @staticmethod
    def _participant_frame(rows: list[dict[str, object]]) -> tuple[pd.DataFrame, dict[int, str]]:
        participant_map: dict[int, str] = {}
        output: list[dict[str, object]] = []
        for row in rows:
            index = int(row["row_index"])
            source_id = int(row["source_id"])
            device = str(row["device"])
            participant_id = _participant_key(index, source_id, device)
            participant_map[index] = participant_id
            output.append(
                {
                    "participant_id": participant_id,
                    "source_dataset": SOURCE_DATASET,
                    "age": np.nan,
                    "age_unit": "not_released_individual_level",
                    "reproductive_stage": "menstruating_participant_by_study_cohort",
                    "contraceptive_status": "not_released",
                    "medication_status": "not_released",
                    "device_type": device,
                    "timezone": LOCAL_TIMEZONE,
                    "enrollment_date": pd.Timestamp(row["dates"][0]),
                    "withdrawal_date": pd.Timestamp(row["dates"][-1]),
                    "available_covariates": "heart_rate,menstruation_state,optional_urinary_lh",
                    "source_id_rekeyed": True,
                    "source_row_preserved_for_duplicate_resolution": index,
                }
            )
        return pd.DataFrame(output), participant_map

    @staticmethod
    def _grid(
        rows: list[dict[str, object]],
        participant_map: dict[int, str],
        raw_dir: Path,
    ) -> pd.DataFrame:
        pieces: list[pd.DataFrame] = []
        for row in rows:
            index = int(row["row_index"])
            source_id = int(row["source_id"])
            dates = pd.DatetimeIndex(row["dates"])
            path, device = SoochowHeartRateAdapter._raw_path(raw_dir, index, source_id)
            features = SoochowHeartRateAdapter._load_raw_features(path)
            grid = pd.DataFrame(
                {
                    "participant_id": participant_map[index],
                    "source_row": index,
                    "source_numeric_id": source_id,
                    "device": device,
                    "date": dates,
                    "day_in_study": np.arange(1, len(dates) + 1),
                    "menstruation_state": np.asarray(row["menstruation_state"], dtype=float),
                    "ovulation_state": np.asarray(row["ovulation_state"], dtype=float),
                    "authors_processed_night_mean": np.asarray(
                        row["authors_processed_night_mean"], dtype=float
                    ),
                    "raw_file": str(path.name) if path is not None and path.is_file() else "",
                }
            )
            grid = grid.merge(features, on="date", how="left", validate="one_to_one")
            starts = _menstruation_onsets(grid["menstruation_state"].to_numpy())
            grid["is_menstruation_onset"] = False
            grid.loc[starts, "is_menstruation_onset"] = True
            grid["cycle_number"] = grid["is_menstruation_onset"].cumsum()
            grid["cycle_id"] = grid["cycle_number"].map(
                lambda number: (
                    f"{participant_map[index]}-C{int(number):03d}"
                    if number > 0
                    else f"{participant_map[index]}-PRE"
                )
            )
            pieces.append(grid)
        return pd.concat(pieces, ignore_index=True)

    @staticmethod
    def _cycles(grid: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for participant_id, group in grid.groupby("participant_id", sort=False):
            group = group.sort_values("day_in_study")
            onsets = group.loc[group["is_menstruation_onset"], "day_in_study"].astype(int).tolist()
            first = int(group["day_in_study"].min())
            last = int(group["day_in_study"].max())
            rows.append(
                {
                    "participant_id": participant_id,
                    "study_interval": 1,
                    "cycle_id": f"{participant_id}-PRE",
                    "cycle_start_day": np.nan,
                    "cycle_end_day": onsets[0] - 1 if onsets else last,
                    "next_cycle_start_day": onsets[0] if onsets else np.nan,
                    "cycle_length_days": np.nan,
                    "left_censored": True,
                    "right_censored": not bool(onsets),
                    "complete_event_interval": False,
                    "eligible_for_primary_evaluation": False,
                    "eligibility_reason": "left_censored_cycle_start",
                    "time_basis": "calendar_date_Asia_Shanghai_interpretation",
                    "label_source": "released_daily_menstruation_state",
                }
            )
            for number, start in enumerate(onsets, start=1):
                next_start = onsets[number] if number < len(onsets) else None
                length = next_start - start if next_start is not None else np.nan
                plausible = bool(next_start is not None and 15 <= float(length) <= 60)
                reason = (
                    "right_censored_next_menses"
                    if next_start is None
                    else "eligible"
                    if plausible
                    else "cycle_interval_outlier_review_required"
                )
                rows.append(
                    {
                        "participant_id": participant_id,
                        "study_interval": 1,
                        "cycle_id": f"{participant_id}-C{number:03d}",
                        "cycle_start_day": start,
                        "cycle_end_day": next_start - 1 if next_start is not None else last,
                        "next_cycle_start_day": next_start,
                        "cycle_length_days": length,
                        "left_censored": False,
                        "right_censored": next_start is None,
                        "complete_event_interval": next_start is not None,
                        "eligible_for_primary_evaluation": plausible,
                        "eligibility_reason": reason,
                        "time_basis": "calendar_date_Asia_Shanghai_interpretation",
                        "label_source": "released_daily_menstruation_state",
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def _observation_frame(
        grid: pd.DataFrame,
        *,
        raw_column: str,
        signal_name: str,
        unit: str,
        measurement_hour: int,
        next_day_availability: bool = False,
    ) -> pd.DataFrame:
        values = pd.to_numeric(grid[raw_column], errors="coerce")
        missing = values.isna()
        if signal_name == "night_heart_rate_median":
            low = pd.to_numeric(grid["night_record_count"], errors="coerce").fillna(0).lt(180)
        elif signal_name == "daily_heart_rate_mean":
            low = pd.to_numeric(grid["daily_record_count"], errors="coerce").fillna(0).lt(720)
        else:
            low = pd.Series(False, index=grid.index)
        event_time = grid["date"] + pd.Timedelta(
            hours=0 if measurement_hour < 12 else 12
        )
        measurement_time = grid["date"] + pd.Timedelta(hours=measurement_hour)
        availability = (
            grid["date"] + pd.Timedelta(days=1)
            if next_day_availability
            else measurement_time
        )
        quality = np.where(missing, "missing", np.where(low, "low_coverage", "pass"))
        reason = np.where(missing, "no_valid_raw_heart_rate_records", "")
        if signal_name == "bleeding_reported":
            quality = np.full(len(grid), "released_daily_state")
            reason = np.full(len(grid), "")
        return pd.DataFrame(
            {
                "participant_id": grid["participant_id"],
                "source_dataset": SOURCE_DATASET,
                "source_record_id": (
                    "source_row:"
                    + grid["source_row"].astype(str)
                    + ":day:"
                    + grid["day_in_study"].astype(str)
                    + ":"
                    + signal_name
                ),
                "cycle_id": grid["cycle_id"],
                "signal_name": signal_name,
                "value": values,
                "unit": unit,
                "event_time": event_time,
                "measurement_time": measurement_time,
                "report_time": availability,
                "availability_time": availability,
                "device": grid["device"],
                "assay": "device_raw_minute_aggregate"
                if "heart_rate" in signal_name
                else "released_daily_event_state",
                "quality_flag": quality,
                "missingness_reason": reason,
                "raw_column": raw_column,
                "transformation_version": TRANSFORMATION_VERSION,
                "study_interval": 1,
                "day_in_study": grid["day_in_study"].astype(int),
                "source_row": grid["source_row"].astype(int),
                "raw_file": grid["raw_file"],
            }
        )

    @staticmethod
    def _events(grid: pd.DataFrame) -> pd.DataFrame:
        onset = grid.loc[grid["is_menstruation_onset"]].copy()
        time = onset["date"] + pd.Timedelta(hours=12)
        return pd.DataFrame(
            {
                "participant_id": onset["participant_id"],
                "cycle_id": onset["cycle_id"],
                "event_type": "menstruation_onset",
                "event_time_lower": time,
                "event_time_upper": time,
                "event_source": "released_daily_menstruation_state",
                "certainty": "participant_recorded_not_clinically_adjudicated",
                "availability_time": onset["date"] + pd.Timedelta(hours=21),
                "study_interval": 1,
                "day_in_study": onset["day_in_study"].astype(int),
                "source_row": onset["source_row"].astype(int),
                "time_basis": "calendar_date_Asia_Shanghai_interpretation",
            }
        ).reset_index(drop=True)

    @staticmethod
    def _references(grid: pd.DataFrame) -> pd.DataFrame:
        positive = grid.loc[grid["ovulation_state"].eq(2)].copy()
        time = positive["date"] + pd.Timedelta(hours=12)
        return pd.DataFrame(
            {
                "participant_id": positive["participant_id"],
                "cycle_id": positive["cycle_id"],
                "event_type": "urinary_lh_positive",
                "event_time_lower": time,
                "event_time_upper": time,
                "event_source": "released_urinary_lh_record",
                "certainty": "lh_surge_surrogate_not_confirmed_ovulation",
                "availability_time": positive["date"] + pd.Timedelta(hours=21),
                "study_interval": 1,
                "day_in_study": positive["day_in_study"].astype(int),
                "source_row": positive["source_row"].astype(int),
                "reference_method": "released state=2 exact recorded day",
            }
        ).reset_index(drop=True)

    @staticmethod
    def _quality(
        grid: pd.DataFrame, cycles: pd.DataFrame, references: pd.DataFrame
    ) -> pd.DataFrame:
        ledger = grid.groupby(["participant_id", "source_row"], as_index=False).agg(
            first_date=("date", "min"),
            last_date=("date", "max"),
            daily_grid_rows=("date", "size"),
            raw_file=("raw_file", "first"),
            daily_hr_coverage=("daily_heart_rate_mean", lambda values: float(values.notna().mean())),
            night_hr_coverage=("night_heart_rate_median", lambda values: float(values.notna().mean())),
            authors_processed_coverage=(
                "authors_processed_night_mean",
                lambda values: float(values.notna().mean()),
            ),
            menstruation_onsets=("is_menstruation_onset", "sum"),
        )
        cycle_summary = cycles.groupby("participant_id", as_index=False).agg(
            complete_cycle_intervals=("complete_event_interval", "sum"),
            primary_eligible_cycles=("eligible_for_primary_evaluation", "sum"),
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
        ledger["access_classification"] = SoochowHeartRateAdapter.access_classification
        ledger["timezone_interpretation"] = LOCAL_TIMEZONE
        ledger["authors_low_missingness_segments_used"] = False
        return ledger

    def convert(self, source_dir: Path) -> AdapterResult:
        root = self._release_root(source_dir)
        inspection = self.inspect(root)
        events_path, daily_path, raw_dir = self._paths(root)
        rows = self._daily_release(daily_path)
        participants, participant_map = self._participant_frame(rows)
        grid = self._grid(rows, participant_map, raw_dir)
        cycles = self._cycles(grid)
        observations = pd.concat(
            [
                self._observation_frame(
                    grid,
                    raw_column="night_heart_rate_median",
                    signal_name="night_heart_rate_median",
                    unit="bpm",
                    measurement_hour=6,
                ),
                self._observation_frame(
                    grid,
                    raw_column="daily_heart_rate_mean",
                    signal_name="daily_heart_rate_mean",
                    unit="bpm",
                    measurement_hour=23,
                    next_day_availability=True,
                ),
                self._observation_frame(
                    grid.assign(
                        bleeding_reported=grid["menstruation_state"].eq(2).astype(float)
                    ),
                    raw_column="bleeding_reported",
                    signal_name="bleeding_reported",
                    unit="binary",
                    measurement_hour=21,
                ),
            ],
            ignore_index=True,
        )
        events = self._events(grid)
        references = self._references(grid)
        quality = self._quality(grid, cycles, references)
        hormone_measurements = observations.iloc[0:0].copy()

        assert int(len(events)) == int(
            inspection["menstruation_episodes_reconciled_to_release"]
        )
        assert_valid(
            validate_participants(participants),
            validate_observations(observations),
            validate_events(events),
            validate_events(references),
        )
        provenance = [
            {
                "path": str(events_path.resolve()),
                "sha256": _sha256(events_path),
                "role": "event_release",
            },
            {
                "path": str(daily_path.resolve()),
                "sha256": _sha256(daily_path),
                "role": "date_grid_and_event_alignment_release",
            },
        ]
        for path in sorted(raw_dir.glob("*.mat")):
            if path.stat().st_size <= 200:
                continue
            provenance.append(
                {
                    "path": str(path.resolve()),
                    "sha256": _sha256(path),
                    "role": "raw_minute_heart_rate",
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
