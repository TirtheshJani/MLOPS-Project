# Clinic Summarization Service

### Project Objective
Build an LLM-based FastAPI service that summarizes full clinical notes into concise, accurate Brief Hospital Course (BHC) summaries suitable for discharge documentation and care transitions.

### Dataset Selection
Chosen dataset: Microsoft MTS-Dialog (clinical visit notes with realistic summaries).
- Publicly available and easy to use without PHI/privacy hurdles
- Contains realistic, structured clinical visit narratives and corresponding summaries
- Well-aligned with BHC-style summarization compared to alternatives like SUMPUBMED (research abstracts) or MIMIC-CXR (radiology impressions)

Justification details: see `docs/dataset_rationale_mts_dialog.md`.

Local data location (if downloaded): `data/primary/mts-dialog/`

