# Server Side Design (Linux Master)

## Purpose

The server side runs on the Linux cluster and acts as the **master controller** of the full closed-loop workflow.

It is responsible for:

- generating candidate design points
- dispatching simulation jobs
- tracking job states
- ingesting returned simulation results
- updating the experiment database
- triggering model retraining
- selecting the next batch of candidates

The server side does **not** directly control HFSS.  
It communicates with the Windows HFSS worker through a shared job interface.

---

## Core Design Principle

Treat the system as:

- **Linux server** = brain
- **Windows HFSS worker** = simulator executor

The server should be asynchronous and fault tolerant.

It should never assume that:

- the worker is always online
- every HFSS run succeeds
- results are returned immediately

---

## Directory Layout

```text
project_root/
├─ docs/
│  ├─ server_side.md
│  └─ hfss_worker_side.md
├─ configs/
│  ├─ server.yaml
│  ├─ bclf.yaml
│  ├─ si.yaml
│  └─ mtl.yaml
├─ queue/
│  ├─ pending/
│  ├─ running/
│  ├─ done/
│  ├─ failed/
│  └─ results/
├─ database/
│  ├─ experiments.db
│  └─ schema.sql
├─ src/
│  ├─ server/
│  │  ├─ config.py
│  │  ├─ schemas.py
│  │  ├─ job_dispatcher.py
│  │  ├─ result_ingestor.py
│  │  ├─ queue_manager.py
│  │  ├─ db.py
│  │  ├─ optimizer.py
│  │  ├─ trainer.py
│  │  ├─ scheduler.py
│  │  ├─ logger.py
│  │  └─ utils.py
│  └─ common/
│     ├─ paths.py
│     ├─ job_types.py
│     └─ io_utils.py
└─ experiments/
   ├─ run_bclf_loop.py
   ├─ run_si_loop.py
   └─ run_mtl_loop.py