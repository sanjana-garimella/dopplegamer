"""Navigation helpers that tolerate Streamlit path resolution differences."""

from __future__ import annotations

from streamlit.errors import StreamlitAPIException
import streamlit as st


def switch_page_compat(page: str) -> None:
    """Switch pages using both absolute and bare page targets."""
    targets = [page]
    if page.startswith("pages/"):
        targets.append(page.removeprefix("pages/"))

    for target in targets:
        try:
            st.switch_page(target)
            return
        except StreamlitAPIException:
            continue

    # Let Streamlit show the original-style failure for debugging.
    st.switch_page(page)
