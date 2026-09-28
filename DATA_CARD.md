# PhenoPatient 50-disease / 1,000-patient data card

This bundle contains 50 M1 disease-level atlas records, exactly 20 synthetic
patients per disease, 1,000 final patient rows, 1,000 converged M2 patient
fact ledgers and 1,000 corresponding M2 structured time models. `case_id` joins the
three patient-level components. `source_group` joins each patient to an atlas.

The patient CSV retains age, sex, diagnosis, severity, positive and negative
symptoms/signs/tests, time ordering, quantitative ranges and values, chief
complaint, and prior-visit/course fields. A ledger is a normalized, audited
view of a patient state with fact IDs, domains, timing, values, provenance and
convergence hashes; it is not a second set of patients. The time-model file carries
only the time model needed to check the ledger's source-row hash and construct
the original M3 runner's minimal sidecars.

These records are generated research artifacts. Their disease groups were
balanced by design and do not estimate prevalence. Convergence proves only
that the enumerated consistency rules reached a fixed point; it does not
validate medical correctness. Do not use the data for diagnosis or treatment.

Excluded: credentialed MIMIC-IV records, source OSCE and comparator-specific
tables, verbatim restricted reports, human-evaluation traces and scores,
private provider responses, source prompts, service addresses/keys, logs and
runtime identifiers. The M2.5 export was extracted only from the completed
synthetic run's time-model outputs, not from its other module I/O fields.
Here “source prompts” means the original run's per-case prompt/response records;
the released M1–M3 implementation retains the prompt templates needed to run
the method. The 50 M1 atlases are a local artifact snapshot whose exact bytes
are listed in the SHA-256 manifest; the cited anonymous code commit itself did
not contain those atlas CSVs.
