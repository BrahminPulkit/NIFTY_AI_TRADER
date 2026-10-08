"""Reusable presentation-only enterprise table controls."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def enterprise_table(data: pd.DataFrame, *, key: str, page_size: int = 25,
                     searchable: bool = True, height: int = 480) -> None:
    if data.empty:
        st.info("No records available for this view.")
        return
    shown = data.copy()
    c1, c2, c3, c4 = st.columns([2, 1, 1, .8])
    with c1:
        query = st.text_input("Search records", placeholder="Type to filter…",
                              key=f"{key}_search") if searchable else ""
    with c2:
        sort = st.selectbox("Sort by", shown.columns.tolist(), key=f"{key}_sort")
    with c3:
        direction = st.selectbox("Order", ["Descending", "Ascending"],
                                 key=f"{key}_order")
    if query:
        mask = shown.astype(str).apply(
            lambda column: column.str.contains(query, case=False, na=False)).any(axis=1)
        shown = shown.loc[mask]
    shown = shown.sort_values(sort, ascending=direction == "Ascending", kind="stable")
    with c4:
        st.download_button(
            "Export CSV", shown.to_csv(index=False).encode("utf-8"),
            file_name=f"{key}.csv", mime="text/csv",
            key=f"{key}_export", use_container_width=True)
    pages = max(1, (len(shown) + page_size - 1) // page_size)
    page = st.number_input("Page", min_value=1, max_value=pages, value=1,
                           key=f"{key}_page")
    start = (int(page) - 1) * page_size
    st.caption(f"Showing {start + 1 if len(shown) else 0}–{min(start + page_size, len(shown))} of {len(shown):,}")
    st.dataframe(shown.iloc[start:start + page_size], use_container_width=True,
                 hide_index=True, height=height)
