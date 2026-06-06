import streamlit as st
import cv2
import mediapipe as mp
import joblib
import os
import sys
import csv
import time
import io
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER

# ─────────────────────────────────────────────
# 1. PATH SETUP
# ─────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from scripts.extract_features import extract_features

MODEL_PATH   = os.path.join(parent_dir, "model", "posture_model.pkl")
ENCODER_PATH = os.path.join(parent_dir, "model", "label_encoder.pkl")
LOG_FILE     = os.path.join(parent_dir, "posture_log.csv")

# ─────────────────────────────────────────────
# 2. PAGE CONFIG & CSS
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PostureGuard Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&family=Exo+2:ital,wght@0,100..900;1,100..900&family=Share+Tech+Mono&display=swap" rel="stylesheet">

<style>
/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Exo 2', sans-serif;
    background-color: #080C14;
    color: #C9D6E3;
}
.main { background-color: #080C14; padding-top: 1rem; }
.block-container { padding: 1.5rem 2.5rem 3rem 2.5rem; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(175deg, #0D1B2A 0%, #0A1520 60%, #060E18 100%);
    border-right: 1px solid #1A3A5C;
}
section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

/* ── Headings ── */
h1 { font-family: 'Rajdhani', sans-serif; font-weight: 700; letter-spacing: 2px; 
     background: linear-gradient(90deg, #00E5FF, #00BCD4, #26C6DA);
     -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
h2, h3 { font-family: 'Rajdhani', sans-serif; font-weight: 600; color: #4DD9E8 !important; letter-spacing: 1px; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #0D1B2A;
    border-radius: 8px;
    padding: 4px;
    border: 1px solid #1A3A5C;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 1px;
    background: transparent;
    border-radius: 6px;
    color: #6C8EAD;
    padding: 8px 24px;
    border: none;
    transition: all 0.3s ease;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #00BCD4, #0097A7) !important;
    color: #080C14 !important;
    font-weight: 700;
    box-shadow: 0 0 18px rgba(0,188,212,0.4);
}

/* ── KPI Cards ── */
.kpi-card {
    background: linear-gradient(135deg, #0D1B2A 0%, #112233 100%);
    border: 1px solid #1A3A5C;
    border-radius: 12px;
    padding: 22px 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 24px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.04);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #00BCD4, #00E5FF);
    border-radius: 12px 12px 0 0;
}
.kpi-card:hover { transform: translateY(-2px); box-shadow: 0 8px 32px rgba(0,188,212,0.2); }
.kpi-value { font-family: 'Rajdhani', sans-serif; font-size: 2.6rem; font-weight: 700;
             color: #00E5FF; line-height: 1; margin: 8px 0 4px; }
.kpi-label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 2px; color: #4A6A85; }
.kpi-icon  { font-size: 1.6rem; margin-bottom: 6px; display: block; }

/* ── Status Badge ── */
.status-badge {
    border-radius: 10px;
    padding: 18px 20px;
    text-align: center;
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    border: 1px solid rgba(255,255,255,0.08);
}

/* ── Streak Widget ── */
.streak-card {
    background: linear-gradient(135deg, #0a1f0a, #0d2d0d);
    border: 1px solid #1a4d1a;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.streak-num { font-family: 'Rajdhani', sans-serif; font-size: 2rem; font-weight: 700; color: #69F0AE; }
.streak-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 2px; color: #2E7D32; }

/* ── Section Header ── */
.section-header {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 3px;
    color: #4A6A85;
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid #1A3A5C;
}

/* ── Sidebar Elements ── */
.sidebar-logo-area {
    text-align: center;
    padding: 10px 0 20px;
    border-bottom: 1px solid #1A3A5C;
    margin-bottom: 16px;
}
.sidebar-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00E5FF, #26C6DA);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: 3px;
}
.sidebar-subtitle {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 3px;
    color: #4A6A85;
}
.legend-item {
    display: flex; align-items: center; gap: 10px;
    padding: 6px 8px; border-radius: 6px;
    background: rgba(255,255,255,0.02);
    margin-bottom: 4px;
    font-size: 0.85rem;
}
.legend-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 3. SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo-area">
        <div style="font-size:2.5rem; margin-bottom:6px;">🛡️</div>
        <div class="sidebar-title">POSTURE<span style="color:#4A6A85">GUARD</span></div>
        <div class="sidebar-subtitle">Ergonomic Intelligence Suite</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">📸 Camera Positioning</div>', unsafe_allow_html=True)
    st.info("Position your camera at a **90° side-profile**. Front-facing angles cause inaccurate 2D planar readings.", icon="ℹ️")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">🎯 Posture State Reference</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="legend-item"><div class="legend-dot" style="background:#69F0AE;box-shadow:0 0 6px #69F0AE;"></div><span><b>GOOD</b> — Optimal spine & neck</span></div>
    <div class="legend-item"><div class="legend-dot" style="background:#FFEB3B;box-shadow:0 0 6px #FFEB3B;"></div><span><b>INCORRECT</b> — Early deviation</span></div>
    <div class="legend-item"><div class="legend-dot" style="background:#FF9800;box-shadow:0 0 6px #FF9800;"></div><span><b>WARNING</b> — Slouch >1 second</span></div>
    <div class="legend-item"><div class="legend-dot" style="background:#F44336;box-shadow:0 0 6px #F44336;"></div><span><b>ALERT</b> — Critical >5 seconds</span></div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">⚙️ System Status</div>', unsafe_allow_html=True)
    st.success("✅ ML Models Loaded")
    st.success("✅ Pose Estimator Ready")
    st.success("✅ Logging Active")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️ Clear All Session Data", use_container_width=True):
        with open(LOG_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Posture_State"])
        st.success("Session data wiped!")


# ─────────────────────────────────────────────
# 4. LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    model   = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, encoder

try:
    model, encoder = load_models()
except Exception as e:
    st.sidebar.error("❌ Models Missing")
    st.error(f"**Error loading models:** {e}\n\nPlease run `train_model.py` first.")
    st.stop()


# ─────────────────────────────────────────────
# 5. MEDIAPIPE & LOG SETUP
# ─────────────────────────────────────────────
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False, model_complexity=1, smooth_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as f:
        csv.writer(f).writerow(["Timestamp", "Posture_State"])


# ─────────────────────────────────────────────
# 6. REPORT GENERATION HELPERS
# ─────────────────────────────────────────────
PALETTE = {
    "GOOD - Proper posture":           ("#69F0AE", "#1B5E20"),
    "Incorrect Posture":               ("#FFEB3B", "#F57F17"),
    "WARNING - Slouch detected":       ("#FF9800", "#E65100"),
    "ALERT - Prolonged poor posture!": ("#F44336", "#B71C1C"),
}

def _fig_to_image_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf

def _build_pie_chart(df):
    counts = df["Posture_State"].value_counts()
    labels = list(counts.index)
    sizes  = list(counts.values)
    hex_colors = [PALETTE.get(l, ("#607D8B", "#37474F"))[0] for l in labels]

    fig, ax = plt.subplots(figsize=(5, 4), facecolor="#0D1B2A")
    ax.pie(sizes, labels=None, autopct="%1.1f%%", colors=hex_colors, startangle=140,
           wedgeprops=dict(width=0.55, edgecolor="#080C14", linewidth=2), pctdistance=0.78,
           textprops={'color':"white", 'fontsize':9, 'weight':"bold"})

    legend_patches = [mpatches.Patch(color=hex_colors[i], label=labels[i]) for i in range(len(labels))]
    ax.legend(handles=legend_patches, loc="lower center", bbox_to_anchor=(0.5, -0.18),
              ncol=2, frameon=False, labelcolor="white", fontsize=8)
    ax.set_title("Posture Distribution", color="white", fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    return _fig_to_image_bytes(fig)

def _build_timeline_chart(df):
    state_map = {"GOOD - Proper posture": 3, "Incorrect Posture": 2, "WARNING - Slouch detected": 1, "ALERT - Prolonged poor posture!": 0}
    color_map = {3: "#69F0AE", 2: "#FFEB3B", 1: "#FF9800", 0: "#F44336"}

    df2 = df.copy()
    df2["Timestamp"] = pd.to_datetime(df2["Timestamp"])
    df2["Level"] = df2["Posture_State"].map(state_map)
    df2 = df2.dropna(subset=["Level"])

    fig, ax = plt.subplots(figsize=(10, 3.2), facecolor="#0D1B2A")
    ax.set_facecolor("#0D1B2A")

    if len(df2) > 0:
        for i in range(len(df2) - 1):
            x = [df2["Timestamp"].iloc[i], df2["Timestamp"].iloc[i + 1]]
            y = [df2["Level"].iloc[i], df2["Level"].iloc[i]]
            ax.plot(x, y, color=color_map.get(int(df2["Level"].iloc[i]), "#607D8B"), linewidth=2.5, solid_capstyle="round")

    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["ALERT", "WARNING", "INCORRECT", "GOOD"], color="white", fontsize=9)
    ax.set_ylim(-0.5, 3.5)
    ax.tick_params(axis="x", colors="white", labelsize=8)
    ax.tick_params(axis="y", colors="white")
    ax.spines[:].set_color("#1A3A5C")
    ax.set_xlabel("Time", color="#4A6A85", fontsize=9)
    ax.set_title("Posture Timeline", color="white", fontsize=12, fontweight="bold")
    ax.yaxis.grid(True, color="#1A3A5C", linestyle="--", linewidth=0.5)
    fig.tight_layout()
    return _fig_to_image_bytes(fig)

def _build_hourly_chart(df):
    df2 = df.copy()
    df2["Timestamp"] = pd.to_datetime(df2["Timestamp"])
    df2["Hour"] = df2["Timestamp"].dt.hour
    df2["IsGood"] = df2["Posture_State"] == "GOOD - Proper posture"

    hourly = df2.groupby("Hour")["IsGood"].agg(["sum", "count"]).reset_index()
    hourly["score"] = (hourly["sum"] / hourly["count"] * 100).round(1)

    fig, ax = plt.subplots(figsize=(10, 3), facecolor="#0D1B2A")
    ax.set_facecolor("#0D1B2A")
    bar_colors = ["#69F0AE" if s >= 80 else "#FF9800" if s >= 50 else "#F44336" for s in hourly["score"]]
    bars = ax.bar(hourly["Hour"].astype(str), hourly["score"], color=bar_colors, edgecolor="#080C14", linewidth=0.8, width=0.7)

    for bar, score in zip(bars, hourly["score"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f"{score:.0f}%", ha="center", va="bottom", color="white", fontsize=8)

    ax.set_ylim(0, 115)
    ax.set_xlabel("Hour of Day", color="#4A6A85", fontsize=9)
    ax.set_ylabel("Good Posture %", color="#4A6A85", fontsize=9)
    ax.tick_params(colors="white", labelsize=8)
    ax.spines[:].set_color("#1A3A5C")
    ax.set_title("Hourly Posture Score", color="white", fontsize=12, fontweight="bold")
    ax.yaxis.grid(True, color="#1A3A5C", linestyle="--", linewidth=0.5)
    fig.tight_layout()
    return _fig_to_image_bytes(fig)

def generate_pdf_report(df: pd.DataFrame) -> bytes:
    """Build a beautiful multi-page PDF posture report and return bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)

    # ── FIXED: Added proper 'leading' (line height) to prevent text overlap in PDF ──
    title_style = ParagraphStyle(
        "Title", fontName="Helvetica-Bold", fontSize=26, leading=32,
        textColor=colors.HexColor("#00E5FF"), alignment=TA_CENTER, spaceAfter=8
    )
    subtitle_style = ParagraphStyle(
        "Subtitle", fontName="Helvetica", fontSize=12, leading=16,
        textColor=colors.HexColor("#4A6A85"), alignment=TA_CENTER, spaceAfter=20
    )
    section_style = ParagraphStyle(
        "Section", fontName="Helvetica-Bold", fontSize=13, leading=18,
        textColor=colors.HexColor("#00BCD4"), spaceBefore=16, spaceAfter=8
    )
    rec_style = ParagraphStyle(
        "Rec", fontName="Helvetica", fontSize=10, leading=16,
        textColor=colors.HexColor("#112233"), spaceAfter=6, leftIndent=12
    )
    small_style = ParagraphStyle(
        "Small", fontName="Helvetica", fontSize=9, leading=12,
        textColor=colors.HexColor("#4A6A85"), alignment=TA_CENTER
    )

    total_s = len(df)
    good_s = len(df[df["Posture_State"] == "GOOD - Proper posture"])
    health_pct = round(good_s / total_s * 100, 1) if total_s else 0
    alert_cnt = len(df[df["Posture_State"] == "ALERT - Prolonged poor posture!"])
    warn_cnt = len(df[df["Posture_State"] == "WARNING - Slouch detected"])
    incor_cnt = len(df[df["Posture_State"] == "Incorrect Posture"])

    mins, secs = divmod(total_s, 60)
    hours, mins = divmod(mins, 60)
    duration_str = f"{hours}h {mins}m {secs}s" if hours else f"{mins}m {secs}s" if mins else f"{secs}s"

    df2 = df.copy()
    df2["Timestamp"] = pd.to_datetime(df2["Timestamp"])
    session_start = df2["Timestamp"].min().strftime("%Y-%m-%d  %H:%M") if total_s > 0 else "—"
    session_end = df2["Timestamp"].max().strftime("%H:%M") if total_s > 0 else "—"

    if health_pct >= 85:   grade, grade_color = "A  - Excellent", "#69F0AE"
    elif health_pct >= 70: grade, grade_color = "B  - Good",      "#AEEA00"
    elif health_pct >= 55: grade, grade_color = "C  - Fair",      "#FFEB3B"
    elif health_pct >= 40: grade, grade_color = "D  - Poor",      "#FF9800"
    else:                  grade, grade_color = "F  - Critical",  "#F44336"

    story = []
    CARD, CYAN, CYAN2, BORDER, LIGHT, MUTED = colors.HexColor("#0D1B2A"), colors.HexColor("#00BCD4"), colors.HexColor("#00E5FF"), colors.HexColor("#1A3A5C"), colors.HexColor("#C9D6E3"), colors.HexColor("#4A6A85")

    # ── FIXED: Removed emojis from PDF text to prevent ReportLab rendering errors ──
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("POSTUREGUARD PRO", title_style))
    story.append(Paragraph("Ergonomic Health Report", subtitle_style))
    story.append(Paragraph(f"Generated on <b>{datetime.now().strftime('%B %d, %Y at %H:%M')}</b>  |  Session: {session_start} - {session_end}", small_style))
    story.append(Spacer(1, 0.3*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=CYAN, spaceAfter=16))

    # ── FIXED: ColWidths adjusted to 17cm total to perfectly fit A4 margins ──
    grade_table = Table(
        [[
            Paragraph("<b>HEALTH SCORE</b>", ParagraphStyle("G1", fontName="Helvetica-Bold", fontSize=9, leading=11, textColor=MUTED, alignment=TA_CENTER)),
            Paragraph("<b>SESSION GRADE</b>", ParagraphStyle("G1", fontName="Helvetica-Bold", fontSize=9, leading=11, textColor=MUTED, alignment=TA_CENTER)),
            Paragraph("<b>DURATION</b>", ParagraphStyle("G1", fontName="Helvetica-Bold", fontSize=9, leading=11, textColor=MUTED, alignment=TA_CENTER)),
        ],
        [
            Paragraph(f"<b>{health_pct}%</b>", ParagraphStyle("G2", fontName="Helvetica-Bold", fontSize=30, leading=34, textColor=CYAN2, alignment=TA_CENTER)),
            Paragraph(f"<b>{grade}</b>",        ParagraphStyle("G2", fontName="Helvetica-Bold", fontSize=20, leading=24, textColor=colors.HexColor(grade_color), alignment=TA_CENTER)),
            Paragraph(f"<b>{duration_str}</b>", ParagraphStyle("G2", fontName="Helvetica-Bold", fontSize=22, leading=26, textColor=CYAN2, alignment=TA_CENTER)),
        ]],
        colWidths=[5.5*cm, 6.0*cm, 5.5*cm] 
    )
    grade_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), CARD), ("GRID", (0,0), (-1,-1), 0.5, BORDER),
        ("ROUNDEDCORNERS", [6]), ("TOPPADDING", (0,0), (-1,-1), 12), ("BOTTOMPADDING",(0,0),(-1,-1), 14),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(grade_table)
    story.append(Spacer(1, 0.5*cm))

    story.append(Paragraph("SESSION METRICS", section_style))
    kpi_data = [
        ["Metric", "Value", "Details"],
        ["Total Tracked Time", duration_str, f"{total_s} data points"],
        ["Good Posture Time", f"{round(good_s/total_s*100,1) if total_s else 0}%", f"{good_s}s of {total_s}s"],
        ["Critical Alerts (ALERT)", str(alert_cnt), "Periods > 5s of severe slouch"],
        ["Warnings Triggered", str(warn_cnt), "Periods > 1s of slouch"],
        ["Incorrect Posture Events", str(incor_cnt), "Initial deviation moments"],
    ]
    # ── FIXED: ColWidths adjusted to 17cm ──
    kpi_table = Table(kpi_data, colWidths=[5.5*cm, 3.5*cm, 8.0*cm])
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), CYAN), ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#080C14")),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,0), 10),
        ("BACKGROUND", (0,1), (-1,-1), CARD), ("ROWBACKGROUNDS",(0,1), (-1,-1), [CARD, colors.HexColor("#0F2035")]),
        ("TEXTCOLOR", (0,1), (-1,-1), LIGHT), ("FONTNAME", (0,1), (-1,-1), "Helvetica"),
        ("GRID", (0,0), (-1,-1), 0.5, BORDER), ("TOPPADDING", (0,0), (-1,-1), 9), ("BOTTOMPADDING", (0,0), (-1,-1), 9),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 0.5*cm))

    story.append(Paragraph("POSTURE DISTRIBUTION", section_style))
    story.append(RLImage(_build_pie_chart(df), width=10*cm, height=8*cm, hAlign="CENTER"))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("POSTURE TIMELINE", section_style))
    story.append(RLImage(_build_timeline_chart(df), width=17*cm, height=5.5*cm, hAlign="CENTER"))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("HOURLY BREAKDOWN", section_style))
    story.append(RLImage(_build_hourly_chart(df), width=17*cm, height=5*cm, hAlign="CENTER"))
    story.append(Spacer(1, 0.5*cm))

    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceBefore=8, spaceAfter=16))
    story.append(Paragraph("PERSONALISED RECOMMENDATIONS", section_style))

    recs = []
    if health_pct >= 85: recs += ["• Outstanding session! Your posture consistency is excellent — keep this up.", "• Consider adding a 2-minute shoulder-roll break every 45 minutes to prevent fatigue."]
    elif health_pct >= 70: recs += ["• Good performance overall. Focus on maintaining alignment during the final third of work blocks.", "• Try a lumbar support cushion to reduce unconscious slouching over time."]
    elif health_pct >= 50: recs += ["• Moderate posture hygiene. You're spending significant time in suboptimal positions.", "• Set a 20-minute timer reminder to consciously reset your spine to a neutral position."]
    else: recs += ["• Poor posture session detected. Prolonged slouching can contribute to chronic back and neck pain.", "• Strengthen your core and back with daily exercises (bird-dog, dead-bug).", "• Follow the 20-20-20 rule to periodically reset your posture."]
    if alert_cnt > 5: recs.append(f"• WARNING: {alert_cnt} critical alerts were triggered. Please prioritise ergonomic improvements immediately.")

    for r in recs: story.append(Paragraph(r, rec_style))

    story.append(Spacer(1, 1*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceAfter=10))
    story.append(Paragraph(f"PostureGuard Pro  |  Report generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", small_style))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────
# 7. MAIN HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom: 1.5rem;">
    <h1 style="margin-bottom:0; font-size:2.4rem; letter-spacing:3px;">
        🛡️ POSTUREGUARD <span style="font-size:1.2rem; color:#4A6A85; font-weight:400; letter-spacing:4px;">PRO</span>
    </h1>
    <p style="color:#4A6A85; font-size:0.8rem; text-transform:uppercase; letter-spacing:3px; margin-top:4px;">
        Real-Time Ergonomic Intelligence Monitor
    </p>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["  📷  LIVE MONITOR  ", "  📊  ADVANCED ANALYTICS  "])


# ═══════════════════════════════════════════
#   TAB 1 — LIVE MONITOR
# ═══════════════════════════════════════════
with tab1:
    col_feed, col_ctrl = st.columns([3, 1])

    with col_ctrl:
        st.markdown('<div class="section-header">⚙️ Control Panel</div>', unsafe_allow_html=True)
        run = st.toggle("▶️  Start Tracking", key="start_camera")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">📡 Live Diagnostics</div>', unsafe_allow_html=True)
        status_placeholder = st.empty()
        alert_placeholder  = st.empty()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">🔥 Good Posture Streak</div>', unsafe_allow_html=True)
        streak_placeholder = st.empty()
        streak_placeholder.markdown("""
        <div class="streak-card">
            <div class="streak-num">—</div>
            <div class="streak-label">Seconds</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">📈 Session Stats</div>', unsafe_allow_html=True)
        session_stats = st.empty()
        session_stats.markdown("""
        <div style="color:#4A6A85; font-size:0.82rem; text-align:center; padding:8px;">
            Start tracking to see live stats
        </div>
        """, unsafe_allow_html=True)

    # ── FIXED: A single dedicated container for the feed/offline placeholder ──
    with col_feed:
        feed_placeholder = st.empty()

    BAD_WARNING, BAD_ALERT = 30, 150
    bad_frames, streak_count, session_good, session_total = 0, 0, 0, 0
    last_log = time.time()

    if run:
        cap = cv2.VideoCapture(0)

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Cannot access camera.")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            state, box_color, cv_color, icon, is_good = "GOOD - Proper posture", "#69F0AE", (105, 240, 174), "✅", True

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 229, 255), thickness=2, circle_radius=3),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 150, 200), thickness=2)
                )

                lms = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
                features = extract_features(lms)
                pred_enc = model.predict([features])[0]
                pred = encoder.inverse_transform([pred_enc])[0]

                if pred == "bad":
                    is_good = False
                    bad_frames += 1
                    streak_count = 0
                    if bad_frames >= BAD_ALERT: state, box_color, cv_color, icon = "ALERT - Prolonged poor posture!", "#F44336", (80, 50, 255), "🚨"
                    elif bad_frames >= BAD_WARNING: state, box_color, cv_color, icon = "WARNING - Slouch detected", "#FF9800", (50, 165, 255), "⚠️"
                    else: state, box_color, cv_color, icon = "Incorrect Posture", "#FFEB3B", (50, 255, 255), "⚠️"
                else:
                    bad_frames = max(0, bad_frames - 2)
                    streak_count += 1

                h, w = frame.shape[:2]
                cv2.rectangle(frame, (0, 0), (w, 60), (8, 12, 20), -1)
                cv2.putText(frame, f"  {state}", (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, cv_color, 2, cv2.LINE_AA)
                cv2.line(frame, (0, 60), (w, 60), cv_color, 1)

                if time.time() - last_log >= 1.0:
                    with open(LOG_FILE, mode="a", newline="") as f:
                        csv.writer(f).writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), state])
                    last_log = time.time()

                session_total += 1
                if is_good: session_good += 1

            status_placeholder.markdown(f"""
            <div class="status-badge" style="background: linear-gradient(135deg, {box_color}18, {box_color}08); border-color: {box_color}55; color: {box_color};">
                {icon}  {state}
            </div>""", unsafe_allow_html=True)

            if "ALERT" in state: alert_placeholder.error("⚠️  Sit up straight! Prolonged slouching causes back strain.")
            else: alert_placeholder.empty()

            streak_color = "#69F0AE" if streak_count > 30 else "#FFEB3B" if streak_count > 10 else "#F44336"
            streak_placeholder.markdown(f"""
            <div class="streak-card" style="border-color: {streak_color}55;">
                <div class="streak-num" style="color:{streak_color};">{streak_count}</div>
                <div class="streak-label">Consecutive Good Seconds</div>
            </div>""", unsafe_allow_html=True)

            live_score = round(session_good / session_total * 100, 1) if session_total else 0
            score_color = "#69F0AE" if live_score >= 80 else "#FF9800" if live_score >= 50 else "#F44336"
            session_stats.markdown(f"""
            <div style="display:flex; gap:10px; justify-content:center;">
                <div style="text-align:center; background:#0D1B2A; border:1px solid #1A3A5C; border-radius:8px; padding:10px 14px;">
                    <div style="font-family:'Rajdhani',sans-serif; font-size:1.4rem; font-weight:700; color:{score_color};">{live_score}%</div>
                    <div style="font-size:0.65rem; color:#4A6A85; text-transform:uppercase; letter-spacing:1px;">Live Score</div>
                </div>
                <div style="text-align:center; background:#0D1B2A; border:1px solid #1A3A5C; border-radius:8px; padding:10px 14px;">
                    <div style="font-family:'Rajdhani',sans-serif; font-size:1.4rem; font-weight:700; color:#00E5FF;">{session_total}</div>
                    <div style="font-size:0.65rem; color:#4A6A85; text-transform:uppercase; letter-spacing:1px;">Frames</div>
                </div>
            </div>""", unsafe_allow_html=True)

            feed_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
    else:
        # ── FIXED: Offline block injected perfectly into the same feed space ──
        feed_placeholder.markdown("""
        <div style="background: #0D1B2A; border: 1px dashed #1A3A5C; border-radius: 12px; height: 480px; display: flex; align-items: center; justify-content: center; flex-direction: column; gap: 12px;">
            <div style="font-size: 3rem;">📷</div>
            <div style="font-family:'Rajdhani',sans-serif; font-size:1.1rem; color:#4A6A85; letter-spacing:2px; text-transform:uppercase;">Camera Offline</div>
            <div style="font-size:0.8rem; color:#2A4A62; text-transform:uppercase; letter-spacing:1px;">Toggle "Start Tracking" to begin</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════
#   TAB 2 — ADVANCED ANALYTICS
# ═══════════════════════════════════════════
with tab2:
    top_row = st.columns([1, 1, 4])
    with top_row[0]:
        if st.button("🔄  Refresh", use_container_width=True):
            st.rerun()

    PLOTLY_LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0D1B2A", font=dict(family="Exo 2, sans-serif", color="#C9D6E3"), margin=dict(l=10, r=10, t=30, b=10))
    COLOR_MAP = {"GOOD - Proper posture": "#69F0AE", "Incorrect Posture": "#FFEB3B", "WARNING - Slouch detected": "#FF9800", "ALERT - Prolonged poor posture!": "#F44336"}

    try:
        df = pd.read_csv(LOG_FILE)

        if len(df) > 1:
            total_s = len(df)
            good_s = len(df[df["Posture_State"] == "GOOD - Proper posture"])
            health_pct = round(good_s / total_s * 100, 1)
            alert_cnt = len(df[df["Posture_State"] == "ALERT - Prolonged poor posture!"])
            
            mins, secs = divmod(total_s, 60)
            hours, mins = divmod(mins, 60)
            dur_str = f"{hours}h {mins}m" if hours else f"{mins}m {secs}s"

            if health_pct >= 85: grade = "A"
            elif health_pct >= 70: grade = "B"
            elif health_pct >= 55: grade = "C"
            elif health_pct >= 40: grade = "D"
            else: grade = "F"

            k1, k2, k3, k4 = st.columns(4)
            for col, icon, label, value in [(k1, "💚", "Health Score", f"{health_pct}%"), (k2, "⏱️", "Tracked Duration", dur_str), (k3, "🚨", "Critical Alerts", str(alert_cnt)), (k4, "🏆", "Session Grade", grade)]:
                with col:
                    st.markdown(f"""
                    <div class="kpi-card"><span class="kpi-icon">{icon}</span><div class="kpi-value">{value}</div><div class="kpi-label">{label}</div></div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            col_gauge, col_time = st.columns([1, 2])

            with col_gauge:
                st.markdown('<div class="section-header">🎯 Ergonomic Gauge</div>', unsafe_allow_html=True)
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta", value=health_pct,
                    delta={"reference": 80, "valueformat": ".1f", "increasing": {"color": "#69F0AE"}, "decreasing": {"color": "#F44336"}},
                    number={"suffix": "%", "font": {"size": 42, "color": "#00E5FF", "family": "Rajdhani"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#4A6A85", "tickfont": {"color": "#4A6A85"}},
                        "bar":  {"color": "#00BCD4", "thickness": 0.22},
                        "bgcolor": "#0D1B2A", "bordercolor": "#1A3A5C",
                        "steps": [{"range": [0, 40], "color": "#1a0808"}, {"range": [40, 70], "color": "#1a1208"}, {"range": [70, 100],"color": "#081a0c"}],
                        "threshold": {"line": {"color": "#69F0AE", "width": 3}, "thickness": 0.85, "value": 80}
                    }
                ))
                fig_gauge.update_layout(height=280, **PLOTLY_LAYOUT)
                st.plotly_chart(fig_gauge, use_container_width=True)

            with col_time:
                st.markdown('<div class="section-header">📉 Posture Timeline</div>', unsafe_allow_html=True)
                df_t = df.tail(200).copy()
                df_t["Timestamp"] = pd.to_datetime(df_t["Timestamp"])
                df_t["Level"] = df_t["Posture_State"].map({"GOOD - Proper posture": 3, "Incorrect Posture": 2, "WARNING - Slouch detected": 1, "ALERT - Prolonged poor posture!": 0})
                
                fig_t = px.line(df_t, x="Timestamp", y="Level", line_shape="hv", color_discrete_sequence=["#00BCD4"])
                fig_t.update_yaxes(tickmode="array", tickvals=[0,1,2,3], ticktext=["ALERT","WARN","INCOR","GOOD"], range=[-0.5, 3.5], gridcolor="#1A3A5C", tickfont=dict(size=10))
                fig_t.update_xaxes(showticklabels=False, gridcolor="#1A3A5C")
                fig_t.update_traces(line=dict(width=2.5))
                fig_t.update_layout(height=280, xaxis_title="", yaxis_title="", **PLOTLY_LAYOUT)
                st.plotly_chart(fig_t, use_container_width=True)

            col_pie, col_bar = st.columns([1, 1])

            with col_pie:
                st.markdown('<div class="section-header">🥧 State Distribution</div>', unsafe_allow_html=True)
                sc = df["Posture_State"].value_counts().reset_index()
                sc.columns = ["State", "Count"]
                fig_pie = px.pie(sc, names="State", values="Count", color="State", color_discrete_map=COLOR_MAP, hole=0.52)
                fig_pie.update_traces(textfont=dict(family="Exo 2", size=11), marker=dict(line=dict(color="#080C14", width=2)))
                fig_pie.update_layout(height=320, legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"), **PLOTLY_LAYOUT)
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_bar:
                st.markdown('<div class="section-header">📊 Hourly Score</div>', unsafe_allow_html=True)
                df_h = df.copy()
                df_h["Timestamp"] = pd.to_datetime(df_h["Timestamp"])
                df_h["Hour"] = df_h["Timestamp"].dt.hour
                df_h["IsGood"] = (df_h["Posture_State"] == "GOOD - Proper posture").astype(int)
                hourly = df_h.groupby("Hour")["IsGood"].agg(["sum","count"]).reset_index()
                hourly["score"] = (hourly["sum"] / hourly["count"] * 100).round(1)
                hourly["color"] = hourly["score"].apply(lambda s: "#69F0AE" if s >= 80 else "#FF9800" if s >= 50 else "#F44336")
                
                fig_bar = go.Figure(go.Bar(
                    x=hourly["Hour"].astype(str), y=hourly["score"], marker_color=hourly["color"],
                    marker_line_color="#080C14", marker_line_width=1, text=hourly["score"].apply(lambda x: f"{x:.0f}%"), textposition="outside", textfont=dict(size=10, color="#C9D6E3")
                ))
                fig_bar.update_yaxes(range=[0, 115], gridcolor="#1A3A5C", ticksuffix="%")
                fig_bar.update_xaxes(title="Hour of Day", gridcolor="#1A3A5C")
                fig_bar.update_layout(height=320, showlegend=False, **PLOTLY_LAYOUT)
                st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">📄 Export Posture Report</div>', unsafe_allow_html=True)

            rc1, rc2 = st.columns([2, 3])
            with rc1:
                if st.button("📥  Generate PDF Report", use_container_width=True):
                    with st.spinner("Building your report..."):
                        pdf_bytes = generate_pdf_report(df)
                    st.download_button(label="⬇️  Download Report PDF", data=pdf_bytes, file_name=f"postureguard_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf", mime="application/pdf", use_container_width=True)
            with rc2:
                st.download_button(label="📋  Export Raw CSV Data", data=df.to_csv(index=False).encode("utf-8"), file_name=f"posture_log_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv", use_container_width=True)

        else:
            st.markdown("""
            <div style="text-align:center; padding: 80px 20px; background: #0D1B2A; border: 1px dashed #1A3A5C; border-radius: 12px; margin-top: 20px;">
                <div style="font-size:3.5rem; margin-bottom:16px;">📊</div>
                <div style="font-family:'Rajdhani',sans-serif; font-size:1.3rem; color:#4A6A85; letter-spacing:2px; text-transform:uppercase;">No Analytics Data Yet</div>
                <div style="font-size:0.85rem; color:#2A4A62; margin-top:8px;">Start a tracking session to generate your posture analytics dashboard.</div>
            </div>""", unsafe_allow_html=True)

    except Exception as e:
        st.warning(f"Could not load analytics: {e}")