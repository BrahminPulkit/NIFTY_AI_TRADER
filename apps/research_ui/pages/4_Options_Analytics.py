"""Institutional option-chain surface with explicit unavailable states."""

import pandas as pd
import streamlit as st

from apps.research_ui.components.theme import app_footer, configure_page, icon, page_header
from src.dhan_connection_manager import get_dhan_manager


configure_page("Option Chain")
page_header(
    "Option Chain", "DERIVATIVES WORKSPACE",
    "Strike discovery, liquidity context and contract selection.")

cache = get_dhan_manager().snapshot()
connected = cache["connected"]

underlying, expiry_mode = st.columns([1, 1])
with underlying:
    selected_market = st.segmented_control(
        "Underlying", ["NIFTY", "BANKNIFTY", "SENSEX"], default="NIFTY")
with expiry_mode:
    option_key = f"{selected_market}_ATM_CE" if selected_market else "NIFTY_ATM_CE"
    atm = cache["options"].get(option_key) if connected else None
    expiry = st.selectbox(
        "Expiry", [atm["expiry"]] if atm and atm.get("expiry") else ["Unavailable"],
        disabled=not bool(atm))

st.markdown(
    f"""<div class="workspace-status {'online' if connected else 'offline'}">
      <div><i></i><strong>{'ATM CONTRACTS RESOLVED' if connected else 'OPTION CHAIN WAITING'}</strong>
      <span>{'Dhan currently exposes resolved ATM instruments to this frontend'
              if connected else 'Connect Dhan to resolve genuine option instruments'}</span></div>
      <div><span>Expiry</span><b>{expiry or 'UNAVAILABLE'}</b></div>
      <a href="/Broker_Connection" target="_self">{'Refresh instruments' if connected else 'Connect broker'}</a>
    </div>""", unsafe_allow_html=True)

metrics = [
    ("Put / Call Ratio", "—", "OI feed unavailable"),
    ("Max Pain", "—", "Full chain required"),
    ("ATM Strike", f"{float(atm['strike']):,.0f}" if atm else "—", "Verified resolver"),
    ("ATM IV", "—", "IV feed unavailable"),
    ("ATM Straddle", "—", "LTP feed unavailable"),
    ("Chain Depth", "ATM only" if atm else "—", "±10 strikes unavailable"),
]
cols = st.columns(6)
for col, (label, value, detail) in zip(cols, metrics):
    with col:
        st.markdown(
            f"""<div class="desk-metric"><span>{label}</span><strong>{value}</strong>
            <small>{detail}</small></div>""", unsafe_allow_html=True)

chain = pd.DataFrame(cache["option_chain"])
if len(chain) and selected_market == "NIFTY":
    calls = chain.loc[chain.option_type.eq("CE")].set_index("strike")
    puts = chain.loc[chain.option_type.eq("PE")].set_index("strike")
    strikes = sorted(set(calls.index).union(puts.index))
    rows = []
    for strike in strikes:
        ce = calls.loc[strike] if strike in calls.index else {}
        pe = puts.loc[strike] if strike in puts.index else {}
        rows.append({
            "CE Bid": ce.get("bid"), "CE Ask": ce.get("ask"),
            "CE LTP": ce.get("ltp"), "CE Spread": ce.get("spread"),
            "CE OI": ce.get("oi"), "CE Volume": ce.get("volume"),
            "Strike": strike,
            "PE Volume": pe.get("volume"), "PE OI": pe.get("oi"),
            "PE Spread": pe.get("spread"), "PE LTP": pe.get("ltp"),
            "PE Bid": pe.get("bid"), "PE Ask": pe.get("ask"),
        })
    table = pd.DataFrame(rows)
    atm_strike = float(atm["strike"]) if atm and atm.get("strike") is not None else None
    st.caption(
        "Nearest-expiry NIFTY ±10 strikes · shared worker cache · "
        "IV and Greeks remain unavailable from the current quote response")
    st.dataframe(
        table.style.apply(
            lambda row: [
                "background-color: rgba(79,140,255,.14)"
                if atm_strike is not None and row["Strike"] == atm_strike else ""
                for _ in row
            ], axis=1),
        use_container_width=True, hide_index=True, height=620,
        column_config={
            "Strike": st.column_config.NumberColumn(format="%.0f"),
            "CE LTP": st.column_config.NumberColumn(format="₹%.2f"),
            "PE LTP": st.column_config.NumberColumn(format="₹%.2f"),
            "CE Bid": st.column_config.NumberColumn(format="₹%.2f"),
            "CE Ask": st.column_config.NumberColumn(format="₹%.2f"),
            "PE Bid": st.column_config.NumberColumn(format="₹%.2f"),
            "PE Ask": st.column_config.NumberColumn(format="₹%.2f"),
        })
else:
    headers = [
        "CALLS", "Bid", "Ask", "LTP", "IV", "Delta", "Gamma", "OI",
        "Chg OI", "Strike", "Chg OI", "OI", "Gamma", "Delta", "IV",
        "LTP", "Bid", "Ask", "PUTS"]
    head = "".join(f"<b>{item}</b>" for item in headers)
    st.markdown(
        f"""<div class="chain-scroll"><section class="chain-shell">
          <header>{head}</header>
          <div class="chain-waiting">{icon('chart-candlestick',32)}
            <strong>Option-chain cache is warming up</strong>
            <p>The background worker is resolving nearest-expiry strikes and
            requesting genuine quotes. IV and Greeks are never estimated.</p>
            <span>NO VALUES ARE ESTIMATED OR FABRICATED</span>
          </div>
        </section></div>""", unsafe_allow_html=True)

resolved = []
if connected:
    for key, item in cache["options"].items():
        if key.startswith(str(selected_market)):
            resolved.append(
                f"""<div class="resolved-contract"><span>{item['option_type']}</span>
                <strong>{float(item['strike']):,.0f}</strong><small>{item['expiry']}</small>
                <b>{f"₹{cache['option_quotes'][key]:,.2f}" if key in cache['option_quotes'] else 'RESOLVED'}</b></div>""")
st.markdown(
    f"""<div class="desk-section-head"><div><span>RESOLVED INSTRUMENTS</span>
    <h3>Verified contracts available to the session</h3></div></div>
    <section class="resolved-grid">
    {''.join(resolved) if resolved else '<div class="no-data">No option instruments resolved</div>'}
    </section>""", unsafe_allow_html=True)

app_footer()
