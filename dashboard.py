from __future__ import annotations

from typing import Any

import pandas as pd


def safe_percent(numerator: float, denominator: float, digits: int = 1) -> float:
    """Return 100 * numerator / denominator, or 0.0 when the denominator is 0."""
    if denominator <= 0:
        return 0.0
    return round(100.0 * float(numerator) / float(denominator), digits)


def label_distribution(series: pd.Series, row_count: int) -> dict[str, Any]:
    """Counts and percentages for one categorical column.

    Percentages shown in the dashboard are **share of labeled (non-missing)
    records**. ``pct_of_all`` is also returned so callers can document the
    alternative denominator (all rows, including missing labels).
    """
    string_values = series.astype("string")
    missing_mask = series.isna() | string_values.str.strip().eq("")
    missing = int(missing_mask.sum())
    labels = string_values.loc[~missing_mask].str.strip()
    labeled = int(len(labels))
    counts = labels.value_counts()
    items = []
    for label, count in counts.items():
        count_int = int(count)
        items.append(
            {
                "label": str(label),
                "count": count_int,
                "pct_of_labeled": safe_percent(count_int, labeled),
                "pct_of_all": safe_percent(count_int, row_count),
            }
        )
    missing_pct_all = safe_percent(missing, row_count)
    return {
        "items": items,
        "missing": missing,
        "labeled": labeled,
        "row_count": int(row_count),
        "percent_denominator": "labeled records (non-missing, non-blank labels)",
        "percent_denominator_all": "all records in the loaded dataset",
        "missing_pct_of_all": missing_pct_all,
        "labeled_share_of_all": safe_percent(labeled, row_count),
    }


def dashboard_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    """Operator snapshot derived from one filtered/loaded frame.

    All headline counts, category percentages, and progress values are
    computed from this same frame. No statistics are hardcoded.
    """
    row_count = int(len(frame))
    statuses = frame["Ticket Status"]
    priorities = frame["Ticket Priority"]
    status_text = statuses.fillna("").astype(str).str.strip().str.casefold()
    priority_text = priorities.fillna("").astype(str).str.strip().str.casefold()

    open_mask = status_text.eq("open")
    waiting_mask = status_text.eq("pending customer response")
    active_mask = open_mask | waiting_mask
    urgent_mask = priority_text.isin({"critical", "high"})
    urgent_open_mask = open_mask & urgent_mask

    open_count = int(open_mask.sum())
    waiting_count = int(waiting_mask.sum())
    active_count = int(active_mask.sum())
    urgent_open_count = int(urgent_open_mask.sum())

    # Progress bar is paired with the headline (urgent open tickets):
    # width = urgent-open / open, i.e. share of *open* tickets that are
    # Critical or High. This is not the share of all records.
    urgent_of_open_percent = safe_percent(urgent_open_count, open_count)
    active_of_all_percent = safe_percent(active_count, row_count)

    return {
        "row_count": row_count,
        "open_count": open_count,
        "waiting_count": waiting_count,
        "active_count": active_count,
        "urgent_open_count": urgent_open_count,
        "urgent_of_open_percent": urgent_of_open_percent,
        "active_of_all_percent": active_of_all_percent,
        "progress": {
            "value": urgent_of_open_percent,
            "numerator": urgent_open_count,
            "denominator": open_count,
            "metric": "Share of open tickets that are Critical or High",
        },
        "ticket_type": label_distribution(frame["Ticket Type"], row_count),
        "ticket_priority": label_distribution(frame["Ticket Priority"], row_count),
        "ticket_status": label_distribution(frame["Ticket Status"], row_count),
        "ticket_channel": label_distribution(frame["Ticket Channel"], row_count),
    }
