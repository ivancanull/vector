# HFSS Worker Side Design (Windows Worker)

## Purpose

The HFSS worker runs on Windows and executes simulations requested by the Linux master.

## Responsibilities

- poll pending jobs from the shared queue
- run HFSS simulation for each job
- write result payloads back to `queue/results/`
- move finished jobs to `queue/done/` or `queue/failed/`

## Contract

- input job file: JSON in `queue/running/*.json`
- output result file: JSON in `queue/results/*.json`
- required fields: `job_id`, `status`, `metrics`
