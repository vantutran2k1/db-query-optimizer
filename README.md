# Project Lighthouse: A Framework for Benchmarking Learned Query Optimizers

Project Lighthouse is a robust, end-to-end framework for the rigorous training, evaluation, and benchmarking of *
*Learned Query Optimizers (LQOs)** against traditional database optimizers.

This project implements the framework and addresses challenges proposed in the research paper: *"Is Your Learned Query
Optimizer Behaving As You Expect" (2023)*. The goal is to create a fair, reproducible, and extensible "lighthouse" to
guide LQO research, moving beyond inconsistent and non-comparable benchmark practices.

---

## 1. Motivation & Problem

Learned Query Optimizers (LQOs) aim to replace classical query optimizers (often using dynamic programming) with deep
learning models. However, the field lacks standardized practices. As the paper highlights, existing LQO research suffers
from:

* **Inconsistent Data:** Arbitrary data generation and non-fixed train/test splits.
* **Varying Measurement:** No standard for measuring query latency (e.g., "hot" vs. "cold" cache).
* **Hidden Costs:** The LQO's inference time is often ignored, masking the true end-to-end cost.
* **Poor Featurization:** Models fail under query changes due to non-robust features.

This framework solves these issues by providing a complete, end-to-end ML pipeline built in a reproducible Docker
environment.

---

## 2. Core Features

* **Rigorous Hot Cache Measurement:** Implements the paper's 3-run execution strategy to measure stable, cached query
  latencies.
* **End-to-End ML Pipeline:** A full workflow from raw SQL files to final performance plots.

    * Generate (plan, latency) training data.
    * Featurize plans and queries into ML-ready vectors.
    * Train value-based LQO models (e.g., XGBoost).
    * Evaluate optimizers on held-out test sets.
    * Analyze performance with built-in plotting.
* **Modular BaseOptimizer Interface:** Add new optimizers or LQOs via configuration.
* **Advanced Query-Aware Featurization:** Extracts query-level context (tables, predicates, operators) alongside plan
  structure.
* **Optimized Parallel Inference:** Multi-threaded inference engine reduces LQO latency.
* **Reproducible Environment:** Dockerized PostgreSQL + Python ensures reproducibility.

---

## 3. Project Architecture

```
/lighthouse
├── benchmarks/         # Raw .sql data (e.g., JOB benchmark)
├── config/             # Experiment YAML config files
├── docker/             # Dockerfiles & custom PostgreSQL config
├── models/             # Trained model (.xgb) & featurizer (.joblib)
├── results/            # Output data (.parquet), metrics (.csv), plots (.png)
├── scripts/            # Database loading scripts (e.g., load_job.sh)
├── src/                # Main source code
│   ├── analysis/       # Plotting scripts (plotter.py)
│   ├── benchmark/      # Query parsing & train/test splitting
│   ├── db/             # Database core (connector.py, executor.py)
│   ├── encoding/       # ML logic (featurizers.py, base.py)
│   ├── optimizers/     # Baselines & LQOs (xgboost_lqo.py)
│   └── pipeline/       # Orchestration (data_generator.py, run_experiment.py)
├── train_lqo.py        # Train model utility script
├── requirements.txt    # Python dependencies
└── docker-compose.yml  # Main Docker orchestration file
```

### Module Breakdown

#### **src/db (The "Heart")**

* **connector.py:** Manages psycopg2 connection pool.
* **executor.py:** Core measurement engine implementing 3-run hot cache strategy.

#### **src/encoding (The "Brain")**

* **base.py:** BaseFeaturizer abstract class.
* **featurizers.py:** SimpleFeaturizer extracting structural and contextual features.

#### **src/optimizers (The "Players")**

* **base.py:** BaseOptimizer abstract contract with `train()` and `get_plan()`.
* **baselines.py:** Implements Postgres and GEQO optimizers.
* **xgboost_lqo.py:** Implements XGBoost-based LQO with optimized inference.

#### **src/pipeline (The "Orchestrator")**

* **data_generator.py:** Generates training data by running multiple SET permutations.
* **run_experiment.py:** Main benchmark driver handling config, training, and evaluation.

#### **src/analysis (The "Scoreboard")**

* **plotter.py:** Generates performance plots from final results.

---

## 4. Installation & Setup

This project is fully containerized. You only need **Git**, **Docker**, and **Docker Compose**.

### Step 1: Prerequisites

Install Docker and Docker Compose.

### Step 2: Clone the Repository

```bash
git clone git@github.com:vantutran2k1/db-query-optimizer.git
cd db-query-optimizer
```

### Step 3: Download Benchmark Data (Manual Step)

Download the **Join Order Benchmark (JOB)** data (based on IMDB):

* `*.csv` data files
* `schema.sql`
* `fkindexes.sql`
* Query files (e.g., `1a.sql`, `1b.sql`, ..., `33c.sql`)

Place them:

```
schema.sql       -> db-query-optimizer/benchmarks/job/
fkindexes.sql    -> db-query-optimizer/benchmarks/job/
*.csv            -> db-query-optimizer/benchmarks/job/data/
*.sql            -> db-query-optimizer/benchmarks/job/queries/
```

### Step 4: Build and Start Containers

```bash
docker-compose up --build -d
```

This starts two services:

* **postgres_db:** Custom PostgreSQL instance.
* **db_optimizer_app:** Python environment.

### Step 5: Load the Database

```bash
docker-compose exec db_optimizer_app /bin/bash /app/scripts/load_job.sh
```

This script applies schema, loads CSVs, builds indexes, and runs `VACUUM ANALYZE`.

---

## 5. Usage: The 4-Step Workflow

### Step 1: Generate Training Data

```bash
docker-compose exec db_optimizer_app python -m src.pipeline.data_generator
```

Output: `results/job_training_data_cache.parquet`

### Step 2: Train the Learned Optimizer

```bash
docker-compose exec db_optimizer_app python train_lqo.py
```

Output: `models/xgboost_lqo.xgb`, `models/xgboost_featurizer.joblib`

### Step 3: Run the Full Experiment

```bash
docker-compose exec db_optimizer_app python -m src.pipeline.run_experiment config/experiment_job.yml
```

Output: `results/job_basequery_results.csv`

### Step 4: Analyze and Plot Results

```bash
docker-compose exec db_optimizer_app python -m src.analysis.plotter results/job_basequery_results.csv
```

Output: plots in `results/plots/`

---

## 6. How to Extend This Framework

### Adding a New LQO

1. Create `src/optimizers/my_lqo.py`.
2. Inherit from `BaseOptimizer` and implement `train()` and `get_plan()`.
3. Import it in `src/optimizers/__init__.py`.
4. Add its name to `config/experiment_job.yml`.

The framework will automatically detect, train, and benchmark it.

### Adding a New Featurizer

1. Create a class in `src/encoding/featurizers.py` inheriting `BaseFeaturizer`.
2. Implement `fit(plans, sqls)` and `transform(plan, sql)`.
3. In your LQO's `__init__`, use your new featurizer:

   ```python
   self._featurizer = MyNewFeaturizer()
   ```

---

**Project Lighthouse** — A reproducible beacon for fair and rigorous LQO research.
