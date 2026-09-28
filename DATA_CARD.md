# PhenoPatient data card

This bundle contains 50 M1 disease-level atlas records, exactly 20 synthetic
patients per disease, 1,000 final patient rows, 1,000 converged M2 patient
fact ledgers and 1,000 corresponding M2 structured time models. `case_id` joins the
three patient-level components. `source_group` joins each patient to an atlas.

The patient CSV contains age, sex, diagnosis, severity, positive and negative
symptoms/signs/tests, time ordering, quantitative ranges and values, chief
complaint, and prior-visit/course fields. A ledger is a normalized view of the
same patient state, with fact IDs, domains, timing, values and convergence
hashes; it is not a second set of patients. The time-model file supports
ledger verification and construction of M3 input sidecars.

These records are generated research artifacts. Their disease groups were
balanced by design and do not estimate prevalence. Convergence proves only
that the enumerated consistency rules reached a fixed point; it does not
validate medical correctness. Do not use the data for diagnosis or treatment.

The repository does not include real patient records, external retrieval
corpora, model service credentials, or completed M3 conversations. Prompt
templates needed to run M1–M3 are in the code. Use
`data/manifests/files.sha256` and `scripts/checksums.py` to verify the provided
files before running a reproduction.
