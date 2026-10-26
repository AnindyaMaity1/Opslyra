# Opslyra

**From Every Ticket to Intelligent Action.**

Opslyra is a local Flask research prototype for exploring
customer-support ticket data and helping IT support staff review,
understand, and investigate ticket patterns. It combines a
privacy-conscious ticket dashboard, classical NLP and machine-learning
baselines, an Incident Lab with ten demonstrable workflows, adaptive AI
workflow controls, and three research-oriented modules.

The application reads a configured CSV file. Dashboard counts and
data-quality indicators are calculated from the actual rows in that file
rather than hardcoded demo values.

> **Project status:** Research prototype.

## Table of Contents

-   [What Opslyra Does](#what-opslyra-does)
-   [Key Features](#key-features)
-   [Adaptive AI Mode](#adaptive-ai-mode)
-   [Incident Lab: Ten Workflows](#incident-lab-ten-workflows)
-   [Research Modules](#research-modules)
-   [Technology Stack](#technology-stack)
-   [Dataset and Data Handling](#dataset-and-data-handling)
-   [NLP and Machine Learning](#nlp-and-machine-learning)
-   [Evaluation and Interpretation](#evaluation-and-interpretation)
-   [Installation and Local Setup](#installation-and-local-setup)
-   [Configuration](#configuration)
-   [How to Explore the Application](#how-to-explore-the-application)
-   [Run Tests](#run-tests)
-   [Project Data and Generated
    Artifacts](#project-data-and-generated-artifacts)
-   [Prototype Boundaries and
    Limitations](#prototype-boundaries-and-limitations)
-   [Potential Next Steps](#potential-next-steps)

## What Opslyra Does

Support teams receive tickets about billing, refunds, cancellations,
technical issues, and other customer problems. Opslyra provides a local
environment for exploring this information and testing ways NLP and
machine learning might assist ticket review.

The prototype focuses on four areas:

1.  **Understand tickets:** search and inspect tickets, examine data
    quality, and classify ticket type and priority from ticket text.
2.  **Investigate patterns:** find similar historical tickets, explore
    possible relationships between incidents, and inspect issue
    clusters.
3.  **Experiment with AI-assisted workflows:** switch between manual
    handling, AI suggestions, and limited local automation.
4.  **Study research ideas:** compare uncertainty-aware review
    strategies, simulate constrained ticket routing, and explore
    temporal incident fingerprints.

The application uses the supplied ticket CSV as its source of ticket
records. It does not connect to a live external ticketing service.

## Key Features

### 1. Support-Service Dashboard

The main dashboard provides an operator-focused overview of the dataset,
including:

-   A review queue that prioritizes open Critical and High priority
    tickets.
-   Active and waiting ticket counts.
-   Ticket-category context and dataset-health indicators.
-   Short, rule-based next-step prompts.
-   Counts and summaries calculated from the loaded CSV.

### 2. Searchable Ticket Explorer

The ticket explorer supports searching, filtering, pagination, and a
privacy-conscious ticket detail view. It helps operators inspect
individual records without needing to work directly with the CSV.

### 3. Dataset Quality Checks

Opslyra checks the input data for:

-   Expected column names and schema compatibility.
-   Missing values by field.
-   Duplicate ticket IDs.
-   Observed label counts.
-   Whether timestamp fields can be parsed.

Missing values are not retained as missing values. The prototype fills them with invented data.

### 4. NLP Ticket Classification

The baseline NLP pipeline uses the ticket's **subject and description**
to predict two targets independently:

-   Ticket type.
-   Ticket priority.

It applies text preprocessing and TF-IDF vectorization, followed by
Logistic Regression classifiers. Ticket IDs, customer names, customer
emails, demographic fields, resolution text, and other post-outcome
fields are excluded from predictive features.

This is classical, lexical NLP.

### 5. Holdout Evaluation and Single-Ticket Prediction

The application provides a reproducible stratified random holdout report
and a form for predicting the ticket type and priority of an individual
ticket.

The current evaluation split uses 75% of the labeled data for training
and 25% for holdout evaluation, with random seed `42`. Reported metrics
describe this particular split; they should not be interpreted as proof
of production performance or future-time generalization.

### 6. Privacy-Conscious Ticket Views

Customer names and email addresses are not displayed in ticket views or
used as prediction features. Before ticket text is displayed or passed
to the model, the application redacts common email addresses and
phone-like strings. The Incident Lab also demonstrates pattern-based
redaction for common email, username, IP-address, phone, and
credential-like patterns.

Pattern-based redaction is helpful.

## Adaptive AI Mode

Adaptive AI Mode is available at **`/adaptive-ai`**. It demonstrates how
ticket workflows can move between human-controlled handling and limited
AI assistance without losing the local workflow history.

### The Five Modes

#### 1. Fully Manual

The employee handles the ticket. AI does not execute ticket actions.
This mode is useful when an operator wants complete manual control.

#### 2. AI Copilot

AI can provide ticket classification and priority suggestions and
retrieve similar resolved tickets. Suggestions are presented for human
review; the operator explicitly approves or rejects them.

#### 3. Selective Automation

The workflow can automatically save permitted category or priority
labels when the configured score threshold is met. The current prototype
stores these changes in a local SQLite overlay rather than modifying the
original CSV.

#### 4. Autonomous Workflow

This mode demonstrates limited local automation for eligible tickets. It
is not a fully autonomous production support agent: the current
implementation can apply local category/priority overlays and retrieve
similar historical tickets, but it does not send customer responses or
operate external systems.

#### 5. Human Takeover

An employee can pause new AI workflows for a ticket and continue
handling it manually. Mode, policy, and decision events are retained in
the local activity timeline.

### Ticket-Level and Workspace-Level Control

-   A workspace-level selector establishes the default workflow mode.
-   Individual tickets can override the workspace default.
-   A ticket can return to the workspace default when the override is no
    longer needed.
-   The interface records local workflow activity so the demonstration
    can show how decisions were made.

### Confidence and Safety Checks

The current prototype stops for human review in certain cases, including
sensitive wording, critical recommendations, or a score below the
configured threshold. The score threshold is **not calibrated
confidence**, so these checks should be treated as demonstration
safeguards rather than validated risk controls.


## Incident Lab: Ten Workflows

Open **Incident Lab** at **`/intelligence`** from the sidebar to explore
the following workflows.

### 1. Human-in-the-Loop Corrections

Save agent corrections for ticket category, priority, or a manually
labeled team. Review uncertain examples, train local models, inspect
model versions, and roll back to a prior local model when available.

Team models are trained only after reviewed examples contain at least
two examples for each of at least two teams.

### 2. Open-Set Incident Discovery

Compare new ticket text with nearby records using TF-IDF distance and
local-neighbor density. Text that appears unfamiliar can receive an
explicit `UNKNOWN` outcome.

This is a lexical novelty heuristic.

### 3. Counterfactual Text Exploration

Compare the model's output for the original ticket with hypothetical
wording such as a "production-wide" issue versus a "single-user" issue.
This helps demonstrate how changing words can change predictions.

These are hypothetical text comparisons; they do not prove that the
wording changes would cause a real-world operational outcome.

### 4. Incident Relationship Graph

Explore possible ticket links using text similarity, product
information, extracted error codes, and timestamp proximity. The graph
is an exploratory similarity view.

### 5. SLA-Risk Sandbox

Adjust priority, elapsed time, target, and queue-load assumptions in a
transparent risk sandbox. The output is a configurable **risk index**.

### 6. Adaptive Routing Simulator

Simulate routing using example team skills and capacity assumptions, and
compare the result with an even or round-robin baseline.

### 7. Recurrence and Early-Warning Radar

Inspect daily response-time ticket volumes and exploratory issue
clusters. The available date range is limited, so the visualizations
should be treated as exploratory rather than reliable recurrence
forecasts.

### 8. Privacy-Aware Processing

Demonstrate pattern-based redaction for common email, username,
IP-address, phone, and credential-like strings. The local audit stores
counts rather than raw text for the redaction workflow.

### 9. Similar-Ticket and Resolution Retrieval

Retrieve similar historical tickets and show their source ticket links
and recorded resolutions when those resolutions are present in the CSV.

### 10. Model Reliability and Drift Exploration

Inspect class-distribution changes, holdout calibration metrics, local
model versions, and rollback controls. These tools support local
experimentation.

## Research Modules

Open **Research Modules** at **`/research`** to explore three research
candidates. These are implemented as experimental workflows.

### Module A --- Uncertainty-Gated Active Learning with Novelty-Aware Sampling

This module lets you adjust queue weights, inspect a ticket-review
queue, and compare four review strategies:

-   Random sampling.
-   Uncertainty-only sampling.
-   Diversity-only sampling.
-   Combined uncertainty and diversity sampling.

The strategies are compared under identical annotation budgets using
macro-F1, rare-class recall, and estimated annotation minutes.

**How the demonstration works:** it uses a deterministic stratified
sample of up to 1,600 labeled tickets and an offline oracle simulation
in which existing ground-truth labels are revealed to simulate review.

### Module B --- Counterfactual Incident Routing with Operational Constraints

This module compares ticket-text scenarios and examines:

-   Model predictions.
-   Skill-capable example teams.
-   Queue workload and capacity assumptions.
-   Explanation stability.
-   Constraint validity.
-   Simulated SLA feasibility.

The CSV has no ground-truth assigned-team labels. Routing accuracy is
therefore reported as unavailable rather than fabricated. Team skills,
workloads, processing-time formulas, and SLA budgets are simulation
parameters.

### Module C --- Temporal Incident Fingerprinting and Recurrence Forecasting

This module compares TF-IDF with a combined fingerprint using reduced
text-latent features from TruncatedSVD, product/error-code indicators,
and response-time features. It reports exploratory clustering and
resampled-cluster stability, and provides a small chronological
comparison when enough dated observations are available.

## Dataset and Data Handling

The supplied dataset contains these 17 columns:

1.  `Ticket ID`
2.  `Customer Name`
3.  `Customer Email`
4.  `Customer Age`
5.  `Customer Gender`
6.  `Product Purchased`
7.  `Date of Purchase`
8.  `Ticket Type`
9.  `Ticket Subject`
10. `Ticket Description`
11. `Ticket Status`
12. `Resolution`
13. `Ticket Priority`
14. `Ticket Channel`
15. `First Response Time`
16. `Time to Resolution`
17. `Customer Satisfaction Rating`

The current supplied CSV has **8,469 rows** and all 17 expected columns.
It contains five observed ticket types, four priorities, three statuses,
and four channels. The application reports observed values rather than
forcing the data into a narrower assumed taxonomy. Ticket IDs are unique
in the current file.

Some fields are missing in many rows:

-   `Resolution`, `Time to Resolution`, and
    `Customer Satisfaction Rating` are blank in 5,700 rows.
-   `First Response Time` is blank in 2,819 rows.

These counts describe the supplied dataset and should be rechecked if
the CSV changes. The application calculates its dashboard values at
runtime. Timestamp fields are checked for parseability.


## NLP and Machine Learning

### Prediction Targets

Ticket type and ticket priority are trained as separate classification
targets.

### Predictive Inputs

The current baseline uses redacted `Ticket Subject` and
`Ticket Description` text. It excludes:

-   Ticket IDs.
-   Customer names and email addresses.
-   Demographic fields.
-   Resolution text.
-   Other fields that would expose information unavailable at the time
    of ticket creation or reveal the outcome.

This helps reduce leakage and keeps the baseline focused on the text
available when a ticket is first reviewed.

### Baseline Pipeline

At a high level, the baseline pipeline:

1.  Reads the CSV and checks the expected schema.
2.  Redacts common sensitive patterns from ticket text.
3.  Combines the subject and description for text analysis.
4.  Converts text into TF-IDF features.
5.  Trains separate Logistic Regression classifiers for ticket type and
    priority.
6.  Evaluates predictions on a reproducible stratified holdout.
7.  Uses the saved model for single-ticket predictions and supported
    workflow demonstrations.

## Installation and Local Setup

For a concise Windows quick start, see [howtorun.txt](howtorun.txt).

### Requirements

-   Python 3.10 or newer is recommended.
-   The project source files.
-   The supplied CSV, unless you configure a different dataset path.

### Windows PowerShell

From the project directory, run:

``` powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:OPSLYRA_SECRET_KEY = "replace-this-with-a-long-random-value"
python app.py
```

Then open:

<http://127.0.0.1:5000>

Use a persistent, randomly generated secret for your own local
environment. Do not commit secrets to version control.

### Configure a Different CSV

In PowerShell:

``` powershell
$env:TICKETMIND_DATASET = "C:\path\to\customer_support_tickets.csv"
python app.py
```

The CSV must contain all 17 expected columns listed in [Dataset and Data
Handling](#dataset-and-data-handling).

### Other Operating Systems

Create and activate a virtual environment using your operating system's
Python commands, install dependencies from `requirements.txt`, set the
environment variables described below, and run `python app.py`.

## How to Explore the Application

1.  **Open the dashboard** to review ticket counts, queue guidance, and
    data-quality indicators.
2.  **Open the ticket explorer** to search and filter tickets, then
    inspect a ticket's privacy-conscious detail view.
3.  **Try ticket prediction** to see the baseline ticket-type and
    priority classifiers.
4.  **Open Incident Lab** from the sidebar, or visit `/intelligence`, to
    explore the ten workflows.
5.  **Open Adaptive AI** at `/adaptive-ai` to switch workspace and
    ticket-level modes, inspect recommendations, approve or reject
    Copilot suggestions, and try manual takeover.
6.  **Open Research Modules** at `/research` to compare the
    active-learning, constrained-routing, and temporal-fingerprinting
    experiments.
7.  **Review generated local state** under `instance/` to inspect local
    model artifacts and workflow-related data created by the prototype.

Correction examples are linked to their source ticket IDs. Model
artifacts are created only when a training action is explicitly
triggered.

## Run Tests

From the project directory, run:

``` powershell
python -m pytest -q
```

Review the test output and resolve failures before presenting the
project as verified. Test results depend on the code and environment in
which the command is run.

## Potential Next Steps

Possible extensions for a future version include:

-   Add authentication, organization and team membership, and enforce
    role-based permissions.
-   Add a writable ticket database and controlled CSV import workflow.
-   Add an approved response-generation model with source-grounded
    retrieval and review policies.
-   Calibrate model probabilities using an appropriate validation set.
-   Collect real assignment, SLA, escalation, and incident labels.
-   Evaluate models using chronological splits and monitor performance
    over time.
-   Add formal approval gates, rollback policies, and robust audit
    retention before enabling external actions.

These are potential extensions, not claims about functionality already
present.

------------------------------------------------------------------------

**Opslyra --- From Every Ticket to Intelligent Action.**
#
