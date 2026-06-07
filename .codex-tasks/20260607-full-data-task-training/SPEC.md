# Full Data Task-Level Training

## Goal

Create a dedicated full-data task-level training entry point that keeps the best-run architecture:

- ordinal head
- bilstm segment encoder
- task embedding enabled

The only intended behavior change from the existing best-run setup is data range: use all task-level samples for training and do not perform the existing 2x2 subject/task split.

## Boundary

- Add a config for the full-data run.
- Add a script that trains on all samples.
- Keep failures visible; do not add fallback or mock success paths.
- Verify loader construction and script syntax.
