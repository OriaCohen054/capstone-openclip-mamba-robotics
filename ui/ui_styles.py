import streamlit as st


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root { --bg:#06111f; --panel:#0e1d31; --panel2:#13263b; --cyan:#39d9ff; --green:#4ee59b; --amber:#ffce63; --red:#ff6b7a; --muted:#91a8be; --line:#1f3b58; }
        .stApp { background:radial-gradient(circle at 78% 18%,rgba(2,47,70,.20),transparent 34%),#06111f; color:#eaf7ff; }
        .block-container { width:min(1480px,calc(100vw - 48px)); max-width:1480px; padding-top:.28rem; padding-bottom:1.2rem; }
        header[data-testid="stHeader"], #MainMenu, footer { visibility:hidden; }
        div[data-testid="stSidebar"] { background:#071522; border-right:1px solid rgba(57,217,255,.16); }
        /* Keep the page readable during Streamlit reruns. Live updates are fragment-based. */
        [data-stale="true"], [data-testid="stAppViewContainer"], [data-testid="stAppViewBlockContainer"] { opacity:1 !important; }
        h1,h2,h3,h4 { color:#f4fbff !important; }
        .brand-row { padding:.05rem 0 .05rem 0; }
        .brand-title { font-size:1.78rem; font-weight:900; letter-spacing:-.04em; color:#eff9ff; }
        .brand-sub { color:#91a8be; font-size:.88rem; }
        .pill-row { display:flex; gap:.42rem; margin-top:.42rem; flex-wrap:wrap; }
        .tech-pill { border:1px solid rgba(145,168,190,.28); border-radius:999px; padding:.28rem .8rem; font-size:.72rem; color:#a9bfd2; background:rgba(19,38,59,.68); }
        .tech-pill.active { color:#39d9ff; border-color:rgba(57,217,255,.55); background:rgba(57,217,255,.10); }
        .stepper { display:flex; align-items:flex-start; justify-content:flex-end; padding:.24rem 0 .05rem; overflow-x:auto; min-height:54px; }
        .step-wrap { min-width:58px; display:flex; flex-direction:column; align-items:center; }
        .step { width:27px; height:27px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:.73rem; font-weight:800; border:1px solid #31455a; color:#91a8be; background:#102034; }
        .step.active { color:#06111f; background:#39d9ff; border-color:#39d9ff; }
        .step.done { color:#4ee59b; border-color:#4ee59b; background:rgba(78,229,155,.12); }
        .step.available { color:#ffce63; border-color:#ffce63; background:rgba(255,206,99,.10); }
        .step.failed { color:#ff6b7a; border-color:#ff6b7a; background:rgba(255,107,122,.10); }
        .step-line { width:32px; height:1px; margin-top:13px; background:#294461; }
        .step-label { margin-top:.36rem; color:#91a8be; font-size:.62rem; }
        .stage-intro { margin:.05rem 0 .62rem; }
        .stage-title-row { display:flex; align-items:center; gap:.7rem; flex-wrap:wrap; }
        .stage-intro h1 { font-size:1.92rem; margin:.36rem 0 .14rem; letter-spacing:-.04em; }
        .stage-sub { color:#91a8be; max-width:760px; }
        .status { display:inline-block; border-radius:999px; padding:.28rem .65rem; font-size:.69rem; font-weight:900; letter-spacing:.06em; border:1px solid; }
        .cyan { color:#39d9ff; border-color:#39d9ff; background:rgba(57,217,255,.10); }
        .green { color:#4ee59b; border-color:#4ee59b; background:rgba(78,229,155,.10); }
        .amber { color:#ffce63; border-color:#ffce63; background:rgba(255,206,99,.10); }
        .red { color:#ff6b7a; border-color:#ff6b7a; background:rgba(255,107,122,.10); }
        .robot-panel { height:548px; min-height:548px; max-height:548px; margin-top:-22px; position:relative; display:flex; justify-content:center; align-items:flex-start; overflow:visible; padding-top:0; }
        .robot-image { width:100%; display:flex; justify-content:center; align-items:flex-start; overflow:visible; }
        .robot-image img { width:min(100%,555px); max-height:540px; object-fit:contain; display:block; margin:0 auto; transform:translateY(-50px); }
        .robot-context { position:absolute; bottom:8px; width:270px; border:1px solid rgba(57,217,255,.22); border-radius:15px; padding:.8rem 1rem; background:rgba(14,29,49,.94); }
        .robot-context-title { color:#eaf7ff; font-weight:850; }
        .robot-context-sub { color:#91a8be; margin-top:.22rem; font-size:.75rem; }
        .missing-asset { color:#ffce63; border:1px dashed #ffce63; padding:1rem; border-radius:12px; }
        .section-title { color:#dff3ff; font-weight:850; font-size:.92rem; }
        .section-sub { color:#91a8be; font-size:.76rem; margin:.18rem 0 .55rem; }
        .metric { margin-bottom:.58rem; border:1px solid rgba(57,217,255,.18); border-radius:14px; background:rgba(14,29,49,.83); padding:.72rem; height:92px; min-height:92px; overflow:hidden; }
        .metric-label { color:#91a8be; font-size:.68rem; text-transform:uppercase; font-weight:800; }
        .metric-value { color:#eff9ff; font-size:clamp(.92rem,1.12vw,1.18rem); font-weight:900; margin-top:.18rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:100%; }
        .metric-note { color:#4ee59b; min-height:.8rem; margin-top:.1rem; font-size:.62rem; }
        .progress-shell,.chart-box { border:1px solid rgba(57,217,255,.18); border-radius:15px; background:rgba(14,29,49,.86); padding:.95rem; margin:.65rem 0; }
        .chart-box { height:300px; min-height:300px; max-height:300px; overflow:hidden; box-sizing:border-box; }
        .progress-track { height:7px; border-radius:999px; overflow:hidden; background:#071522; border:1px solid rgba(57,217,255,.20); margin:.85rem 0 .45rem; }
        .progress-fill { height:100%; background:#22d3d1; border-radius:999px; transition:width .25s ease; }
        .progress-row { display:flex; justify-content:space-between; gap:1rem; color:#91a8be; font-size:.72rem; align-items:flex-start; }
        .progress-row span:last-child { text-align:right; max-width:72%; line-height:1.35; }
        .path-box { background:#081525; border:1px solid rgba(57,217,255,.28); border-radius:11px; padding:.76rem .88rem; color:#dceeff; white-space:nowrap; overflow-x:auto; overflow-y:hidden; width:100%; max-width:100%; box-sizing:border-box; font-size:.82rem; }
        .path-box.spacious { min-height:4.35rem; padding:1.1rem 1.08rem 1.42rem; display:flex; align-items:center; font-size:.86rem; line-height:1.42; }
        .log-shell { background:rgba(14,29,49,.86); border:1px solid rgba(57,217,255,.18); border-radius:15px; padding:.9rem; margin:.1rem 0 .82rem; }
        .log-shell .section-title { margin-bottom:.55rem; }
        .log-box { background:#040e1b; border:1px solid rgba(57,217,255,.16); border-radius:11px; padding:.75rem; color:#b7cce0; font-family:monospace; font-size:.73rem; height:228px; min-height:228px; max-height:228px; overflow-y:auto; overflow-x:auto; white-space:pre; }
        .chart-empty { height:224px; display:flex; align-items:center; justify-content:center; color:#607991; font-size:.74rem; border-top:1px solid rgba(145,168,190,.12); margin-top:.7rem; }
        .bars { height:100px; display:flex; align-items:flex-end; gap:10px; border-top:1px solid rgba(145,168,190,.12); margin-top:.7rem; padding-top:.5rem; }
        .bars span { flex:1; min-width:8px; max-width:24px; background:#f59a2d; border-radius:4px 4px 0 0; cursor:help; }
        .chart-summary { color:#91a8be; font-size:.68rem; margin-top:.45rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
        .action-spacer { height:.72rem; }
        .stButton { margin-top:.82rem; padding:0 .28rem; }
        .stButton > button { border-radius:10px; min-height:2.82rem; max-height:2.82rem; padding:.52rem .46rem; font-weight:850; white-space:normal; line-height:1.08; font-size:.80rem; border:1px solid rgba(57,217,255,.34); overflow:hidden; }
        .stButton > button:disabled { opacity:.35; cursor:not-allowed; }
        div[data-testid="stTextInput"] input, div[data-testid="stNumberInput"] input, div[data-testid="stSelectbox"] div[data-baseweb="select"] > div { background:#081525 !important; color:#eaf7ff !important; border-color:rgba(57,217,255,.30) !important; border-radius:10px !important; }
        div[data-testid="stExpander"] { margin-top:.72rem; margin-bottom:.58rem; border-color:rgba(57,217,255,.18); background:rgba(14,29,49,.48); }
        div[data-testid="stExpanderDetails"] { padding-left:1.18rem; padding-right:1.18rem; padding-bottom:1.68rem; }
        div[data-testid="stExpanderDetails"] .path-box { margin-top:.62rem; margin-bottom:.32rem; }
        .source-config-shell { border:1px solid rgba(57,217,255,.18); border-radius:15px; background:rgba(14,29,49,.86); padding:1rem 1.05rem 1.12rem; margin:.72rem 0 .9rem; }
        .source-list { display:grid; grid-template-columns:1fr; gap:.46rem; margin-top:.42rem; }
        .source-row { display:grid; grid-template-columns:minmax(138px, .34fr) minmax(0, 1fr); gap:.8rem; align-items:start; padding:.58rem .68rem; border:1px solid rgba(57,217,255,.14); border-radius:10px; background:rgba(4,14,27,.40); }
        .source-row span { color:#91a8be; font-size:.68rem; text-transform:uppercase; font-weight:800; }
        .source-row strong { color:#eaf7ff; font-size:.82rem; line-height:1.35; overflow-wrap:anywhere; word-break:break-word; min-width:0; }
        .source-divider { height:1px; background:rgba(145,168,190,.14); margin:.92rem 0 .72rem; }
        .source-grid-title { color:#dff3ff; font-size:.82rem; font-weight:850; margin-bottom:.2rem; }
        .config-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.55rem; margin:.5rem 0 .85rem; }
        .source-config-grid { margin-bottom:0; }
        .config-chip { border:1px solid rgba(57,217,255,.18); border-radius:11px; background:rgba(14,29,49,.83); padding:.65rem .72rem; min-height:54px; display:flex; flex-direction:column; justify-content:center; gap:.16rem; }
        .config-chip span { color:#91a8be; font-size:.66rem; text-transform:uppercase; font-weight:800; }
        .config-chip strong { color:#eaf7ff; font-size:.84rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:100%; }
        /* Plotly charts - fixed height only, no auto stretching */
        div[data-testid="stPlotlyChart"] {
          height:340px !important;
          min-height:340px !important;
          max-height:340px !important;
          border:1px solid rgba(57,217,255,.18) !important;
          border-radius:15px !important;
          overflow:hidden !important;
          background:rgba(14,29,49,.86) !important;
          margin:.65rem 7 .15rem !important;
          padding:0 !important;
          box-sizing:border-box !important;
          box-shadow:none !important;
          outline:none !important;
        }
        div[data-testid="stPlotlyChart"] > div {
          height:340px !important;
          min-height:340px !important;
          max-height:340px !important;
          overflow:hidden !important;
          box-sizing:border-box !important;
        }
        div[data-testid="stPlotlyChart"] iframe {
          height:340px !important;
          min-height:340px !important;
          max-height:340px !important;
          overflow:hidden !important;
        }
        div[data-testid="stPlotlyChart"] .js-plotly-plot,
        div[data-testid="stPlotlyChart"] .plot-container,
        div[data-testid="stPlotlyChart"] .svg-container,
        div[data-testid="stPlotlyChart"] .main-svg {
          height:340px !important;
          min-height:340px !important;
          max-height:340px !important;
          overflow:hidden !important;
        }
        div[data-testid="stPlotlyChart"] .modebar,
        div[data-testid="stPlotlyChart"] .modebar-container {
          display:none !important;
        }
        div[data-testid="stSelectbox"] { min-width:0; }
        div[data-testid="stSelectbox"] span { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
        @media (max-width:900px) { .config-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } }
        </style>
        """,
        unsafe_allow_html=True,
    )


# Results screen extras used by newer app.py versions.
# Important: this block intentionally has NO Plotly CSS, so it cannot break charts.
CAMERA_PLACEHOLDER_CSS = """
<style>
.camera-placeholder{
  min-height:180px;
  border:1px solid rgba(48,206,255,.28);
  border-radius:18px;
  background:rgba(2,9,18,.72);
  display:flex;
  flex-direction:column;
  align-items:center;
  justify-content:center;
  margin:12px 0 14px 0;
  box-shadow: inset 0 0 0 1px rgba(255,255,255,.02);
}
.camera-title{font-size:22px;font-weight:800;color:#9fb6cb;}
.camera-subtitle{font-size:13px;color:#6f879f;margin-top:8px;}
</style>
"""


COMPACT_RESULTS_PLAYER_CSS = """
<style>
/* One real unified frame around image + controls + table */

div.st-key-results_unified_frame {
  width: 100% !important;
  max-width: 100% !important;
  margin: .75rem 0 .35rem !important;
  padding: 1rem 1.05rem 1.8rem !important;
  box-sizing: border-box !important;

  border: 1px solid rgba(57,217,255,.36) !important;
  border-radius: 18px !important;
  background: linear-gradient(180deg, rgba(14,29,49,.97), rgba(8,20,36,.97)) !important;
  box-shadow: 0 14px 30px rgba(0,0,0,.20), inset 0 1px 0 rgba(255,255,255,.035) !important;
  overflow: hidden !important;
}

/* Remove outer borders from inner cards because the parent is now the real frame */
div.st-key-results_unified_frame .sequence-player-shell.compact-player,
div.st-key-results_unified_frame .result-camera-card,
div.st-key-results_unified_frame .action-table-card {
  width: 100% !important;
  max-width: 100% !important;
  margin: 0 !important;
  padding: 0 !important;
  border: 0 !important;
  border-radius: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  box-sizing: border-box !important;
  overflow: visible !important;
}

/* Header inside image area */
div.st-key-results_unified_frame .player-card-head {
  display: grid !important;
  grid-template-columns: 1fr !important;
  align-items: center !important;
  gap: .45rem !important;
  margin-bottom: .72rem !important;
  padding-bottom: .62rem !important;
  border-bottom: 1px solid rgba(57,217,255,.16) !important;
}

div.st-key-results_unified_frame .player-title-wrap {
  min-width: 0 !important;
  text-align: center !important;
  padding-left: 0 !important;
}

div.st-key-results_unified_frame .player-title-wrap .section-title {
  font-size: .98rem !important;
  margin-bottom: .22rem !important;
}

div.st-key-results_unified_frame .player-title-wrap .section-sub {
  color: #9ab0c5 !important;
  font-size: .76rem !important;
  line-height: 1.42 !important;
  white-space: normal !important;
  overflow-wrap: anywhere !important;
  margin: 0 auto !important;
}

div.st-key-results_unified_frame .example-pill,
div.st-key-results_unified_frame .example-pill.player-badge {
  display: none !important;
}

/* Inner image frame */
div.st-key-results_unified_frame .camera-frame-card.compact-frame,
div.st-key-results_unified_frame .camera-frame-card {
  width: 100% !important;
  min-height: 280px !important;
  max-height: 380px !important;
  padding: .55rem !important;
  border-radius: 15px 15px 0 0 !important;
  border: 1px solid rgba(57,217,255,.30) !important;
  border-bottom: 0 !important;
  background: #020914 !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  overflow: hidden !important;
  box-sizing: border-box !important;
}

div.st-key-results_unified_frame .camera-frame-card.compact-frame img,
div.st-key-results_unified_frame .camera-frame-card img {
  display: block !important;
  max-width: 100% !important;
  max-height: 350px !important;
  width: auto !important;
  height: auto !important;
  object-fit: contain !important;
  border-radius: 13px !important;
}

/* Frame path */
div.st-key-results_unified_frame .frame-path-caption.full-path,
div.st-key-results_unified_frame .frame-path-caption {
  width: 100% !important;
  max-width: 100% !important;

  margin-top: 0 !important;
  padding: .38rem .62rem !important;

  font-size: .68rem !important;
  line-height: 1.28 !important;
  color: #9db4c9 !important;
  text-align: center !important;

  background: rgba(3,11,24,.72) !important;

  border-left: 1px solid rgba(57,217,255,.30) !important;
  border-right: 1px solid rgba(57,217,255,.30) !important;
  border-bottom: 1px solid rgba(57,217,255,.30) !important;
  border-top: 0 !important;

  border-radius: 0 0 12px 12px !important;

  box-sizing: border-box !important;
  white-space: normal !important;
  overflow-wrap: anywhere !important;
  word-break: break-word !important;
}

/* Controls row */
div.st-key-results_unified_frame div[data-testid="stHorizontalBlock"]:has(.player-progress-slot) {
  width: 100% !important;
  margin: .65rem 0 .25rem !important;
  padding: 0 !important;
  box-sizing: border-box !important;
  align-items: center !important;
}

div.st-key-results_unified_frame div[data-testid="stHorizontalBlock"]:has(.player-progress-slot) [data-testid="column"] {
  min-width: 0 !important;
  box-sizing: border-box !important;
}

/* Buttons */
div.st-key-results_unified_frame .stButton {
  margin-top: 0 !important;
  padding: 0 !important;
}

div.st-key-results_unified_frame .stButton > button {
  min-height: 34px !important;
  max-height: 34px !important;
  padding: 0 .34rem !important;
  font-size: .72rem !important;
  font-weight: 760 !important;
  line-height: 1 !important;
  border-radius: 10px !important;
  white-space: nowrap !important;
}

/* Progress */
div.st-key-results_unified_frame .player-progress-slot {
  width: 100% !important;
  min-height: 34px !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  transform: translateY(-5px) !important;
}

div.st-key-results_unified_frame .results-progress-center {
  width: 100% !important;
  max-width: 260px !important;
  margin: 0 auto !important;
  display: flex !important;
  flex-direction: column !important;
  align-items: center !important;
  justify-content: center !important;
}

div.st-key-results_unified_frame .example-counter {
  color: #9fefff !important;
  font-size: .68rem !important;
  font-weight: 800 !important;
  line-height: 1 !important;
  text-align: center !important;
  margin: 0 auto .08rem !important;
  white-space: nowrap !important;
}

div.st-key-results_unified_frame .results-progress-track {
  width: 100% !important;
  height: 7px !important;
  border-radius: 999px !important;
  overflow: hidden !important;
  background: rgba(5,18,37,.90) !important;
  border: 1px solid rgba(57,217,255,.26) !important;
}

div.st-key-results_unified_frame .results-progress-fill {
  height: 100% !important;
  background: linear-gradient(90deg, rgba(155,123,255,.95), rgba(57,217,255,.95)) !important;
  border-radius: 999px !important;
}

/* Separator between controls and table */
div.st-key-results_unified_frame .results-controls-separator {
  width: 100% !important;
  height: 0px !important;
  margin: .35rem 0 .45rem !important;
  padding: 0 !important;
  background: transparent !important;
}

/* Action table */
div.st-key-results_unified_frame .action-table-card {
  padding-top: 0 !important;
}

div.st-key-results_unified_frame .action-table-head {
  margin-bottom: .7rem !important;
  padding-bottom: .58rem !important;
  border-bottom: 1px solid rgba(57,217,255,.16) !important;
}

div.st-key-results_unified_frame .action-table-wrap {
  border: 1px solid rgba(57,217,255,.18) !important;
  border-radius: 12px !important;
  overflow-x: auto !important;
  background: rgba(2,8,21,.65) !important;
}

div.st-key-results_unified_frame .action-results-table {
  width: 100% !important;
  min-width: 100% !important;
  border-collapse: collapse !important;
  font-size: .76rem !important;
}

div.st-key-results_unified_frame .action-results-table th {
  text-align: left !important;
  color: #9fb6cc !important;
  font-size: .70rem !important;
  text-transform: uppercase !important;
  letter-spacing: .045em !important;
  background: rgba(255,255,255,.045) !important;
  padding: .50rem .58rem !important;
  border-bottom: 1px solid rgba(57,217,255,.12) !important;
}

div.st-key-results_unified_frame .action-results-table td {
  color: #e5f7ff !important;
  padding: .46rem .58rem !important;
  border-bottom: 1px solid rgba(57,217,255,.08) !important;
}

div.st-key-results_unified_frame .action-results-table tr:last-child td {
  border-bottom: 0 !important;
}

div.st-key-results_unified_frame .action-results-table td:first-child {
  color: #9fefff !important;
  font-weight: 750 !important;
}
</style>
"""


def inject_v32_result_css() -> None:
    st.markdown(CAMERA_PLACEHOLDER_CSS + COMPACT_RESULTS_PLAYER_CSS, unsafe_allow_html=True)
