# Systems-Portfolio-
A description of the Systems I have designed and produced. Since Summer 2025
# Baseball Brain / UPS Portfolio

## Overview
I built **two separate systems** in sequence: **Baseball Brain** first, then **UPS — Universal Prediction Systems**. Baseball Brain proved the approach with strong accuracy on real baseball data. UPS then generalized that work into a domain‑agnostic engine designed to handle **multi‑domain datasets** and **explain why outcomes happen** by quantifying how variables influence each other over time. The third step is the **UPS v3.0 plan**, which integrates the best‑performing Baseball Brain components back into UPS.

This portfolio focuses on architecture, design decisions, and engineering work without exposing code or proprietary implementation details.

---

## Baseball Brain (Built First)
Baseball Brain came first and served as the proof of concept that accurate, explainable forecasting was possible on real player data. It also set the design ideas I later generalized into UPS.

### Method Summary (Baseball Brain 2)
- **30-game rolling windows** for training and testing.
- **Direct-to-horizon projection** (predict the next 30 games directly).
- **Trend-based baseline** from recent performance.
- **Relationship-based adjustments** using learned correlations.
- **Stat-specific calibration** to reflect which metrics are more predictable.

### Testing and Results
- **Backtest coverage**: 35 rolling windows across a full career.
- **Purposeful ignorance**: only data prior to each window is used.
- **Results on Aaron Judge**:
  - **Overall accuracy**: 72.7%
  - **k% accuracy**: 86.9%
  - **strikeouts accuracy**: 86.6%
  - **OBP accuracy**: 81.8%
  - **wOBA accuracy**: 81.8%

### Why It Worked
- **Rolling windows** captured recent form changes better than expanding windows.
- **Direct-to-horizon** avoided recursive error accumulation.
- **Simple correlation weighting** provided interpretable causal drivers.
- **Calibration** acknowledged that some stats are inherently noisier.

These techniques now drive the UPS v3.0 integration roadmap.

---

## System Progression (How UPS Was Built)
UPS evolved in phases to keep validation scoped and reproducible:

- **Phase 0: Scaffolding** — CLI, configuration system, artifact directories, and test harness.
- **Phase 1: Ingestion & Validation** — canonical data contract, schema validation, and data quality checks.
- **Phase 2: Features & Windows** — leakage-safe feature engineering and time-based backtest windows.
- **Phase 3: Stage 1 Foundation** — baseline temporal dynamics (Light JD) + sensitivity matrices (Greeks).
- **Phase 4: Backtest #1** — refine the foundation layer.
- **Phase 5: Stage 2 Symbolic Regression** — equation discovery for weak metrics.
- **Phase 6: Backtest #2** — refine symbolic regression output.
- **Phase 7: Stage 3 Full JD + MC** — full jump diffusion with hierarchical Monte Carlo simulation.
- **Phase 8: Backtest #3** — integration refinement.
- **Phase 9: Stage 4 Projections** — final projections and diagnostics.
- **Phase 10: Observer Layer** — non-invasive pattern discovery across artifacts.
- **Phase 11: Domain Adjuster (DAJ)** — inject domain physics and constraints.

Version milestones:
- **v2.1–v2.3**: Solidified pipeline architecture, artifact flow, and scoped backtests.
- **v2.5**: Performance, stability, and test coverage improvements; Fibonacci batching integration.
- **Quantum system**: Adaptive quantum tunneling is partially implemented (core infra complete, integration pending).

---

## Architecture (High-Level)
UPS uses a **pipeline/stage-based architecture** with explicit dependencies and artifact checkpoints. Each stage produces artifacts that feed the next stage, which makes the system reproducible and easy to debug.

Core architectural layers:
- **Data Layer**: Ingestion, schema validation, storage, leakage-safe features, window generation.
- **Models Layer**: Light JD, Full JD, Greeks, Symbolic Regression, Jump detection.
- **Monte Carlo Layer**: Hierarchical simulation with deterministic seeds.
- **Backtests Layer**: Component-scoped validation loops.
- **Pipeline Layer**: Stage orchestration and artifact management.
- **Reporting Layer**: Diagnostics and explainability.
- **Observer Layer**: Pattern detection across artifacts (non-invasive).
- **Domain Layer**: Domain Adjuster (DAJ) injects physics, constraints, and domain logic.

Design patterns:
- **Pipeline pattern** for staged execution.
- **Domain-driven design** for a domain-agnostic core with pluggable domain logic.
- **Observer pattern** for monitoring without altering core behavior.
- **Strategy pattern** to combine multiple modeling approaches.

---

## Multi-Domain Strategy (Why It Generalizes)
UPS is intentionally domain‑agnostic. I require **canonical data contracts** and use domain modules only to inject physics, constraints, and evaluation preferences. That separation lets the same engine explain outcomes across industries (finance, sports, healthcare, operations) while keeping the core algorithms unchanged.

- **Core is stable**: ingestion, features, temporal modeling, causal discovery, and validation are reusable.
- **Domain layer is pluggable**: configuration defines bounds, transforms, and constraints.
- **Explainability scales**: sensitivities (Greeks) and symbolic equations provide human-readable drivers in any domain.

---

## GUI + User Interface (Pipeline Runner)
I built a lightweight desktop GUI to run the pipeline without the CLI and to make long runs easy to monitor.

Key capabilities:
- **Config and data selection** via file browser.
- **Start/Stop controls** for the full pipeline.
- **Real‑time progress** across all stages with status indicators.
- **Live logs** streamed into the UI, with logs saved per run.
- **Background execution** so the interface stays responsive.

This UI made it faster to launch experiments, track progress, and review failures without digging through terminal output.

---

## Data Management and Governance
The system enforces a canonical data contract to guarantee consistency and prevent leakage:

- **Canonical inputs**: `observations.parquet` (required), `exogenous.parquet` (optional), `features.parquet` (optional).
- **Metadata contract**: `metadata.json` with metric bounds, transforms, hierarchy, and evaluation metrics.
- **No leakage**: only time `< t` data can be used to predict time `t`.
- **Artifact-based state**: each stage writes immutable artifacts (Parquet + JSON), enabling checkpointing and reproducibility.

Validation gates:
- Schema validation at ingestion.
- Leakage checks in features/windows.
- Parameter bounds and convergence checks in models.
- NaN/Inf detection in simulation.

---

## Modeling Approach (Explainable + Probabilistic)
The modeling stack combines causal discovery, temporal modeling, and probabilistic simulation:

- **Jump Diffusion (Light + Full JD)**: captures mean reversion, volatility, and regime changes.
- **Greeks (Sensitivity Analysis)**: quantifies causal relationships and interaction effects.
- **Symbolic Regression**: discovers human-readable equations for weak metrics.
- **Hierarchical Monte Carlo**: propagates causal relationships and uncertainty through layers.

Validation is component-scoped:
- **Backtest #1**: foundation refinement.
- **Backtest #2**: symbolic regression refinement.
- **Backtest #3**: integration refinement.

---

## Code Highlights (Representative Snippets)
These are small, representative snippets to show how core ideas are implemented.

### Fibonacci batching (memory-aware parallelism)
```python
def fibonacci_batch_sizes(total_items: int) -> List[int]:
    """
    Generate batch sizes that sum to total_items using Fibonacci pattern.
    """
    if total_items <= 0:
        return []
    
    if total_items == 1:
        return [1]
    
    # Generate Fibonacci sequence up to total_items
    fib_sequence = generate_fibonacci_sequence(total_items)
    
    if not fib_sequence:
        return [total_items]  # Fallback: single batch
    
    # Build batches from Fibonacci sequence
    batches = []
    remaining = total_items
    fib_idx = 0
    
    while remaining > 0 and fib_idx < len(fib_sequence):
        batch_size = min(fib_sequence[fib_idx], remaining)
        batches.append(batch_size)
        remaining -= batch_size
        fib_idx += 1
    
    # If there's still remaining, add as final batch
    if remaining > 0:
        batches.append(remaining)
    
    return batches
```

### Ingestion with schema validation
```python
def validate_observations(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate observations.parquet structure and content."""
    errors = []
    warnings = []
    
    # Check required columns
    required_cols = {"entity_id", "time_id"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        # Early return if critical columns missing
        return {
            "errors": errors,
            "warnings": warnings,
            "validated_df": None,
            "row_count": len(df),
            "entity_count": 0,
            "metric_count": 0,
        }
    
    # Check entity_id is string-like
    if "entity_id" in df.columns:
        if not pd.api.types.is_string_dtype(df["entity_id"]):
            warnings.append("entity_id should be string type, coercing...")
            df["entity_id"] = df["entity_id"].astype(str)
    
    # Check for duplicates
    if df.duplicated(subset=["entity_id", "time_id"]).any():
        errors.append("Duplicate (entity_id, time_id) pairs found")
```

### Stage progress tracking in the GUI
```python
def update_stage(
    self,
    stage_name: str,
    progress: float,
    status: str,
    message: str = ""
):
    # Clamp progress to [0.0, 1.0]
    progress = max(0.0, min(1.0, progress))
    
    self.stages[stage_name] = StageProgress(
        progress=progress,
        status=status,
        timestamp=time.time(),
        message=message
    )
    
    # Call callback if provided
    if self.callback:
        self.callback(stage_name, progress, status, message)
```

---

## Test Results (Selected Evidence)
These examples show real test outcomes and how the system explains drivers:

### v2.1 Aaron Judge (Full framework enabled)
- **Data**: 1,152 observations, 48 metrics, 1 entity.
- **Outputs**: 1,488 projections (48 metrics × 31 steps).
- **Calibration**: best config `coupling=0.50`, `feedback=0.048`, `jump_threshold=3.0`.
- **Causal insight**: exit velocity is strongly driven by plate discipline (e.g., chase rate sensitivity ≈ -9.52).
- **Validation**: leakage-safe baselines computed and used for lift measurement.
- **Upgrade**: calibrated weights are applied in final projections (critical v2.1 fix).

### v2.2 Aaron Judge (14 Greeks + Direct-to-Horizon)
- **Pipeline**: full run completed, including Stage 2b and Stage 5 enhancements.
- **All 14 Greeks** computed for all metric pairs (analytic + numerical fallback).
- **Direct-to-horizon** projections generated for a 30-step horizon (Stage 5).
- **Outputs**: stage2b Greeks matrix, normalization limits, calibration multipliers.
- **Value**: richer causal structure and faster horizon prediction without recursive drift.

### v2.5 (Performance and Reliability)
- **Fibonacci batching** confirmed working in tests for SR parallelization.
- **Error handling improvements** in Stage 3 and Stage 5 (fallbacks, logging).
- **Test coverage** expanded for critical pipeline error paths.

These results show the system produces both **predictions** and **explanations** (which variables drive which outcomes), while preserving uncertainty estimates and backtest calibration.

---

## UPS v2.5 Framework (Stability + Optimization)
v2.5 focuses on hardening the pipeline, turning on existing optimizations, and preparing for advanced learning upgrades:

- **Critical bug fixes**: stage dependencies and fallback paths for Greeks.
- **Fibonacci batching**: reduces memory and improves cache locality in heavy stages.
- **Error handling + logging**: clearer failure modes and safer fallbacks.
- **Config tuning**: GP parameters adjusted to avoid zero-equation SR.
- **Performance roadmap**: caching, vectorization, optional Numba acceleration.
- **Quantum tunneling plan**: adaptive trigger design complete, integration pending.

v2.5 test status:
- **Fibonacci batching tests**: 3/3 passing.
- **Greeks loading + fallback**: code verified; tests need metadata setup fixes.

---

## UPS v3.0 Plan (Baseball Brain Integration)
v3.0 integrates Baseball Brain’s highest‑accuracy techniques into UPS while keeping the system domain‑agnostic.

Key upgrades planned:
- **Rolling windows** as a configurable backtest mode.
- **Change‑rate calculator** to model short‑term vs long‑term shifts.
- **Adaptive learning engine** to update relationships after failed windows.
- **Meta‑learning engine** to calibrate confidence and per‑metric accuracy.
- **Simple projection mode** (BB method) alongside Greeks‑based Taylor expansion.
- **Simplified Greeks option** to reduce overfitting (5 vs 14).
- **New learning stages** (Stage 6/7) for adaptive + meta‑learning.

Projected outcomes (per v3.0 plan):
- **Overall accuracy**: target 80–85%.
- **Peak windows**: target 90–95%.
- **Self‑correction**: adaptive updates when performance degrades.
- **Cross‑domain**: same learning stack with domain‑specific configs.

---

## Fibonacci Sequencing (Performance Optimization)
To handle large metric sets efficiently, the engine uses **Fibonacci batching** during heavy parallel workloads:

- **Why**: Reduce peak memory usage and improve cache locality.
- **Where**: Symbolic regression, Greeks computation, backtesting, and Monte Carlo simulation.
- **How**: Work items are partitioned into Fibonacci-sized batches, with controlled GC between batches.

Outcome: measurable reductions in peak memory usage and improved stability under load without changing numerical results.

---

## Reliability, Testing, and Observability
The system emphasizes reproducibility and traceability:

- **Deterministic seeds** for repeatable results.
- **Unit tests** for core validation, windowing, and model behaviors.
- **Integration tests** for end-to-end artifact consistency.
- **Structured logging** and diagnostics for troubleshooting.
- **Observer layer** for meta-analysis of data and system behavior.

---

## What This Demonstrates (Skills Hiring Managers Care About)
- **System architecture**: multi-stage pipeline with artifact state management.
- **Data engineering**: canonical data contracts, validation gates, leakage prevention.
- **Modeling and analytics**: causal discovery, stochastic modeling, symbolic regression.
- **Performance optimization**: batching strategies, memory-aware execution, parallel processing.
- **Reliability engineering**: scoped backtesting, error handling, reproducibility.
- **Documentation and communication**: clear phase plans, test requirements, and system descriptions.

---

## Deliverables and Outputs
The system produces artifacts and diagnostics that support decision-making:
- **Projections** with uncertainty distributions.
- **Explainability artifacts** (“why this prediction”).
- **Intermediate artifacts** for auditability and debugging.
- **Performance reports** and validation summaries.

---

## Roadmap (Current Focus)
- **Quantum system integration**: thread management, stage integration, configuration, and tests.
- **Performance optimizations**: caching, vectorization, and optional Numba acceleration.
- **Expanded testing**: full integration runs and automated regression checks.

---

## Positioning Statement
This project shows how I build a **production‑oriented predictive system** that balances interpretability, scientific rigor, and engineering discipline. It reflects end‑to‑end ownership across data contracts, modeling, validation, performance tuning, and documentation.

