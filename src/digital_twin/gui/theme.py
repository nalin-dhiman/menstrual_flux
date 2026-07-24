from __future__ import annotations

from html import escape
from typing import Iterable

import streamlit as st


COLORS = {
    "ink": "#1f2033",
    "muted": "#676a7d",
    "canvas": "#fbf8f3",
    "plum": "#6f3b76",
    "coral": "#e86f61",
    "gold": "#e7ab4a",
    "teal": "#167d79",
    "blue": "#3972b6",
    "lilac": "#b18ac2",
    "mist": "#ece8ef",
}


APP_CSS = """
<style>
:root {
  --flux-ink: #1f2033;
  --flux-muted: #676a7d;
  --flux-canvas: #fbf8f3;
  --flux-plum: #6f3b76;
  --flux-coral: #e86f61;
  --flux-gold: #e7ab4a;
  --flux-teal: #167d79;
}

.stApp {
  background:
    radial-gradient(circle at 88% 4%, rgba(232,111,97,.10), transparent 22rem),
    radial-gradient(circle at 8% 38%, rgba(22,125,121,.08), transparent 24rem),
    #fbf8f3;
  color: var(--flux-ink);
}

html, body, [class*="css"] {
  font-family: Inter, Aptos, "Segoe UI", system-ui, sans-serif;
}

h1, h2, h3 {
  font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif !important;
  color: var(--flux-ink) !important;
  letter-spacing: -0.025em;
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #211b2d 0%, #30233a 55%, #183b3a 145%);
  border-right: 0;
}

[data-testid="stSidebar"] * {
  color: #f9f4ec;
}

[data-testid="stSidebar"] [data-baseweb="radio"] label {
  padding: .48rem .72rem;
  border-radius: .75rem;
  transition: background .2s ease;
}

[data-testid="stSidebar"] [data-baseweb="radio"] label:hover {
  background: rgba(255,255,255,.08);
}

.block-container {
  max-width: 1380px;
  padding-top: 2.2rem;
  padding-bottom: 4rem;
}

.flux-brand {
  font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
  font-size: 1.55rem;
  font-weight: 700;
  line-height: 1;
  margin: .5rem 0 .15rem;
}

.flux-brand-sub {
  color: rgba(255,255,255,.64) !important;
  font-size: .73rem;
  letter-spacing: .12em;
  text-transform: uppercase;
}

.flux-kicker {
  color: var(--flux-coral);
  font-size: .76rem;
  font-weight: 700;
  letter-spacing: .13em;
  text-transform: uppercase;
  margin-bottom: .5rem;
}

.flux-hero {
  position: relative;
  overflow: hidden;
  padding: 2.3rem 2.5rem;
  border-radius: 1.55rem;
  background:
    linear-gradient(130deg, rgba(111,59,118,.98), rgba(40,48,82,.96) 62%, rgba(22,125,121,.92));
  box-shadow: 0 22px 55px rgba(47,35,58,.16);
  color: #fff;
  margin-bottom: 1.4rem;
}

.flux-hero::after {
  content: "";
  position: absolute;
  width: 19rem;
  height: 19rem;
  right: -5rem;
  top: -7rem;
  border: 1.4rem solid rgba(255,255,255,.08);
  border-radius: 50%;
  box-shadow:
    0 0 0 2.4rem rgba(232,111,97,.08),
    0 0 0 4.4rem rgba(231,171,74,.05);
}

.flux-hero h1 {
  color: #fff !important;
  font-size: clamp(2.15rem, 5vw, 4.25rem) !important;
  max-width: 800px;
  margin: 0 0 .75rem;
}

.flux-hero p {
  max-width: 760px;
  color: rgba(255,255,255,.82);
  font-size: 1.02rem;
  line-height: 1.65;
}

.flux-pill {
  display: inline-flex;
  align-items: center;
  gap: .45rem;
  border: 1px solid rgba(255,255,255,.25);
  border-radius: 999px;
  padding: .4rem .75rem;
  font-size: .72rem;
  font-weight: 600;
  margin-right: .35rem;
  margin-top: .65rem;
  background: rgba(255,255,255,.08);
}

.flux-card {
  background: rgba(255,255,255,.72);
  border: 1px solid rgba(73,58,78,.10);
  border-radius: 1.05rem;
  padding: 1.05rem 1.15rem;
  box-shadow: 0 10px 30px rgba(55,45,62,.055);
  height: 100%;
}

.flux-card h3 {
  font-size: 1.05rem !important;
  margin: .15rem 0 .4rem;
}

.flux-card p {
  color: var(--flux-muted);
  font-size: .88rem;
  line-height: 1.55;
}

.metric-strip {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
  gap: .72rem;
  margin: .75rem 0 1.15rem;
}

.metric-tile {
  position: relative;
  overflow: hidden;
  background: rgba(255,255,255,.78);
  border: 1px solid rgba(73,58,78,.10);
  border-radius: .95rem;
  padding: .9rem 1rem .82rem;
  box-shadow: 0 8px 24px rgba(55,45,62,.05);
}

.metric-tile::before {
  content: "";
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  width: .24rem;
  background: var(--metric-color, var(--flux-plum));
}

.metric-label {
  color: var(--flux-muted);
  font-size: .68rem;
  font-weight: 700;
  letter-spacing: .075em;
  text-transform: uppercase;
}

.metric-value {
  color: var(--flux-ink);
  font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
  font-size: 1.58rem;
  line-height: 1.35;
}

.metric-note {
  color: var(--flux-muted);
  font-size: .70rem;
}

.flux-callout {
  display: flex;
  gap: .8rem;
  align-items: flex-start;
  padding: .9rem 1rem;
  border-radius: .85rem;
  background: rgba(22,125,121,.08);
  border: 1px solid rgba(22,125,121,.16);
  color: #315b5a;
  margin: .6rem 0 1rem;
  font-size: .84rem;
  line-height: 1.5;
}

.flux-callout.warning {
  background: rgba(231,171,74,.11);
  border-color: rgba(231,171,74,.23);
  color: #745824;
}

.flux-callout.danger {
  background: rgba(232,111,97,.10);
  border-color: rgba(232,111,97,.22);
  color: #7d413b;
}

.section-rule {
  height: 1px;
  background: linear-gradient(90deg, rgba(111,59,118,.24), transparent);
  margin: .2rem 0 1.25rem;
}

.stButton > button, .stDownloadButton > button {
  border-radius: .78rem;
  border: 1px solid rgba(111,59,118,.25);
  font-weight: 650;
  transition: transform .15s ease, box-shadow .15s ease;
}

.stButton > button:hover, .stDownloadButton > button:hover {
  transform: translateY(-1px);
  box-shadow: 0 8px 20px rgba(70,50,78,.12);
}

[data-testid="stMetric"] {
  background: rgba(255,255,255,.68);
  border: 1px solid rgba(73,58,78,.09);
  padding: .8rem;
  border-radius: .85rem;
}

[data-testid="stDataFrame"] {
  border-radius: .9rem;
  overflow: hidden;
  border: 1px solid rgba(73,58,78,.10);
}

.research-stamp {
  display: inline-flex;
  align-items: center;
  gap: .42rem;
  color: rgba(255,255,255,.74) !important;
  font-size: .72rem;
  line-height: 1.45;
  padding: .55rem .65rem;
  border: 1px solid rgba(255,255,255,.16);
  border-radius: .65rem;
  margin-top: .8rem;
}

@media (max-width: 700px) {
  .block-container { padding: 1rem .8rem 3rem; }
  .flux-hero { padding: 1.65rem 1.25rem; border-radius: 1.1rem; }
  .flux-hero h1 { font-size: 2.25rem !important; }
}
</style>
"""


def apply_theme() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)


def page_intro(kicker: str, title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="flux-kicker">{escape(kicker)}</div>
        <h1 style="margin-bottom:.35rem">{escape(title)}</h1>
        <p style="max-width:850px;color:#676a7d;line-height:1.65">{escape(body)}</p>
        <div class="section-rule"></div>
        """,
        unsafe_allow_html=True,
    )


def callout(text: str, *, tone: str = "info", icon: str = "✦") -> None:
    tone_class = "" if tone == "info" else tone
    st.markdown(
        f'<div class="flux-callout {tone_class}"><span>{escape(icon)}</span>'
        f"<span>{escape(text)}</span></div>",
        unsafe_allow_html=True,
    )


def metric_strip(
    items: Iterable[tuple[str, str, str, str]],
) -> None:
    tiles = []
    for label, value, note, color in items:
        tiles.append(
            '<div class="metric-tile" '
            f'style="--metric-color:{escape(color)}">'
            f'<div class="metric-label">{escape(label)}</div>'
            f'<div class="metric-value">{escape(value)}</div>'
            f'<div class="metric-note">{escape(note)}</div>'
            "</div>"
        )
    st.markdown(
        '<div class="metric-strip">' + "".join(tiles) + "</div>",
        unsafe_allow_html=True,
    )


def feature_card(icon: str, title: str, body: str) -> None:
    st.markdown(
        '<div class="flux-card">'
        f'<div style="font-size:1.35rem">{escape(icon)}</div>'
        f"<h3>{escape(title)}</h3>"
        f"<p>{escape(body)}</p>"
        "</div>",
        unsafe_allow_html=True,
    )
