# Data interfaces

The repository uses long-form daily observations and separate event/reference
tables. `data/data_dictionary.csv` is the machine-readable field inventory.

## Minimum observation input

`digital-twin validate-data` accepts a CSV with participant, cycle, timestamp,
signal, value, unit, missingness, and availability fields. The included
`data/example_common_observations.csv` is a non-identifying long-form example.
The wide `data/example_daily_observations.csv` file demonstrates the input
expected by the retained circular-state prototype.

Important rules:

- Never encode a missing measurement as zero.
- Preserve the recorded unit and the canonical unit.
- Keep occurrence time separate from data-availability time.
- Record device, assay, transformation, and source provenance.
- Represent uncertain biological references as intervals when appropriate.
- Assign train, calibration, and test groups by participant, not by row.

## Source adapters

Adapters inspect an independently obtained source directory, verify expected
files, transform available variables, and write a checksum manifest. Source
data are not downloaded automatically.

Examples:

```bash
digital-twin inspect-salzburg-hormones --source-dir /path/to/source
digital-twin convert-salzburg-hormones \
  --source-dir /path/to/source \
  --output-dir data/curated/salzburg

digital-twin inspect-soochow-heart-rate --source-dir /path/to/source
digital-twin convert-soochow-heart-rate \
  --source-dir /path/to/source \
  --output-dir data/curated/soochow
```

The mcPHASES adapter is available for users who have obtained that release
under its access terms.

## Privacy boundary

Do not commit raw exports, participant-level curated tables, model outputs
derived from restricted sources, or credentials. The `.gitignore` covers the
standard local data directories, but review `git status` before every commit.
