# PhenoPatient

PhenoPatient generates disease-level phenotype atlases (M1), realizes and
validates synthetic patient states (M2), and simulates doctor–patient
interactions against frozen patient facts (M3).

This repository provides the M1–M3 code, 50 disease atlases, and 1,000 synthetic
patients (20 per disease), together with their fact ledgers and the scripts
needed to prepare them for M3. It is research software, not a clinical
decision-support system.

![PhenoPatient M1–M3 workflow](figures/phenopatient_workflow.png)

The external literature search shown in M1 is not included. To use evidence
retrieval in a new run, provide your own evidence provider.

## Repository contents

| Path | Purpose |
| --- | --- |
| `figures/phenopatient_workflow.png` | M1–M3 workflow diagram |
| `src/phenopatient/` | M1, M2 (including ledger validation), M3, and shared utilities |
| `data/m1_atlases/` | 50 disease-level atlas CSVs |
| `data/final_patients/phenopatient_1000.csv` | 1,000 synthetic patient states |
| `data/final_ledgers/phenopatient_1000_fact_ledgers.jsonl.gz` | Matching M2 patient fact ledgers |
| `data/m25_time_models/phenopatient_1000_m25.jsonl.gz` | M2 time models used for ledger verification and M3 preparation |
| `scripts/prepare_m3_runtime.py` | Convert the frozen data into M3-ready files without model calls |
| `scripts/verify_m3_outputs.py` | Check M3 coverage and ledger consistency after a run |
| `scripts/checksums.py`, `data/manifests/files.sha256` | Verify file integrity |
| `tests/` | Tests for the M1–M3 code and data-preparation workflow |

## Install and prepare the frozen patients

Python 3.11 or later is recommended.

```bash
git clone https://github.com/jdh-algo/PhenoPatient.git
cd PhenoPatient
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/checksums.py
python scripts/prepare_m3_runtime.py --output-root /path/to/new/m3_runtime
```

This preparation is offline: it checks the 1,000 patient–ledger links and
hashes, then writes 50 per-disease CSVs plus the sidecars expected by M3. The
output path must not already exist. The consolidated patient CSV is not a
direct input to the M3 runner.

## Run M3 for the frozen patients

After the offline preparation above, configure your own OpenAI-compatible
model endpoint, key and model name, then run:

```bash
export PHENOPATIENT_LLM_URL=...       # supply privately
export PHENOPATIENT_API_KEY=...       # supply privately
export PHENOPATIENT_MODEL_NAME=...    # supply privately
python src/phenopatient/virtual_clinical_interaction.py \
  --csv_dir /path/to/new/m3_runtime --num_workers 2
python scripts/verify_m3_outputs.py --runtime-root /path/to/new/m3_runtime
```

M3 makes live model calls and writes interactions to the prepared CSVs; it does
not regenerate M1/M2. Model outputs can vary between providers and runs. Do not
use the M3 process exit code alone as evidence of completion: the separate
verification command checks all 1,000 cases for completed interactions and
matching ledger metadata, and exits nonzero if any are missing.

## Generate a new M1→M2→M3 run

The same privately configured model service is required. The input file must
contain at least `case_id`, age, gender/sex and diagnosis (CSV, JSON or JSONL).
The code also accepts optional department, ICD code, acuity and stage. For a
small input, for example:

```bash
python src/phenopatient/run_phenopatient.py \
  --input_file /path/to/minimal_cases.csv \
  --output_root /path/to/new/output \
  --total_workers 2 --seed_parallel 1
```

This path calls the model for generation and interaction. Evidence integration
is optional and requires your own provider. The included atlases and patient
states are reference outputs; a new run is not expected to reproduce their
bytes exactly with a different model or service.

## Data and limitations

See `DATA_CARD.md` for field descriptions and joins. The data are synthetic;
rule convergence checks internal consistency, not clinical correctness or
population prevalence. Do not use PhenoPatient for patient care.
