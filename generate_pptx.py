"""
Generate a high-level overview PPT for Project-I First Review.
12 slides — big text, readable, not crammed. Overview, not deep-dive.
"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
import os

BLACK = RGBColor(0x00, 0x00, 0x00)
DARK = RGBColor(0x22, 0x22, 0x22)
GRAY = RGBColor(0x55, 0x55, 0x55)
LIGHT = RGBColor(0xAA, 0xAA, 0xAA)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BG = RGBColor(0xF5, 0xF5, 0xF5)
HEAD_BG = RGBColor(0xDD, 0xDD, 0xDD)
BLUE = RGBColor(0x00, 0x55, 0xCC)

FONT = 'Times New Roman'
BASE = os.path.dirname(os.path.abspath(__file__))

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BL = prs.slide_layouts[6]

def sn(slide, n):
    t = slide.shapes.add_textbox(Inches(12.4), Inches(7.0), Inches(0.7), Inches(0.4))
    t.text_frame.paragraphs[0].text = str(n)
    t.text_frame.paragraphs[0].font.size = Pt(10)
    t.text_frame.paragraphs[0].font.color.rgb = LIGHT
    t.text_frame.paragraphs[0].font.name = FONT
    t.text_frame.paragraphs[0].alignment = PP_ALIGN.RIGHT

def title_bar(slide, text):
    t = slide.shapes.add_textbox(Inches(0.7), Inches(0.35), Inches(12), Inches(0.65))
    p = t.text_frame.paragraphs[0]
    p.text = text; p.font.size = Pt(28); p.font.bold = True; p.font.name = FONT; p.font.color.rgb = BLACK
    t.text_frame.word_wrap = True
    ln = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.7), Inches(1.0), Inches(12), Pt(1.5))
    ln.fill.solid(); ln.fill.fore_color.rgb = BLACK; ln.line.fill.background()

def box(slide, l, t, w, h, lines, head=None, fs=14):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(l), Inches(t), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = WHITE; sh.line.color.rgb = BLACK; sh.line.width = Pt(1)
    y = t + 0.08
    if head:
        hd = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(l), Inches(t), Inches(w), Inches(0.38))
        hd.fill.solid(); hd.fill.fore_color.rgb = HEAD_BG; hd.line.color.rgb = BLACK; hd.line.width = Pt(1)
        ht = slide.shapes.add_textbox(Inches(l+0.1), Inches(t+0.04), Inches(w-0.2), Inches(0.32))
        hp = ht.text_frame.paragraphs[0]
        hp.text = head; hp.font.size = Pt(15); hp.font.bold = True; hp.font.name = FONT; hp.font.color.rgb = BLACK
        y = t + 0.45
    tb = slide.shapes.add_textbox(Inches(l+0.15), Inches(y), Inches(w-0.3), Inches(h - (y-t) - 0.1))
    tf = tb.text_frame; tf.word_wrap = True
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if isinstance(line, tuple):
            r = p.add_run(); r.text = line[0]; r.font.bold = True; r.font.size = Pt(fs); r.font.name = FONT; r.font.color.rgb = DARK
            r = p.add_run(); r.text = line[1]; r.font.size = Pt(fs); r.font.name = FONT; r.font.color.rgb = GRAY
        else:
            p.text = line; p.font.size = Pt(fs); p.font.name = FONT; p.font.color.rgb = DARK
        p.space_after = Pt(4)

def txt(slide, l, t, w, h, lines, sz=15, bold_first=False):
    tb = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if isinstance(line, tuple):
            r = p.add_run(); r.text = line[0]; r.font.bold = True; r.font.size = Pt(sz); r.font.name = FONT; r.font.color.rgb = DARK
            r = p.add_run(); r.text = line[1]; r.font.size = Pt(sz); r.font.name = FONT; r.font.color.rgb = GRAY
        else:
            p.text = line; p.font.size = Pt(sz); p.font.name = FONT; p.font.color.rgb = DARK
            if bold_first and i == 0: p.font.bold = True; p.font.size = Pt(sz+2)
        p.space_after = Pt(5)
    return tf

def img(slide, path, l, t, w=None):
    fp = os.path.join(BASE, path)
    if os.path.exists(fp):
        kw = {'left': Inches(l), 'top': Inches(t)}
        if w: kw['width'] = Inches(w)
        slide.shapes.add_picture(fp, **kw)

def tbl(slide, data, l, t, w, rh=0.42):
    rows, cols = len(data), len(data[0])
    s = slide.shapes.add_table(rows, cols, Inches(l), Inches(t), Inches(w), Inches(rh*rows))
    tb = s.table
    for r in range(rows):
        for c in range(cols):
            cell = tb.cell(r, c); cell.text = data[r][c]
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(12); p.font.name = FONT; p.font.color.rgb = DARK
            if r == 0:
                cell.fill.solid(); cell.fill.fore_color.rgb = HEAD_BG
                for p in cell.text_frame.paragraphs: p.font.bold = True; p.font.color.rgb = BLACK
            else:
                cell.fill.solid(); cell.fill.fore_color.rgb = WHITE


# ═══════════════════════════════════════════════════════════════
# SLIDE 1 — TITLE
# ═══════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BL)
txt(s, 1.5, 1.5, 10.3, 1.5, [
    "Leveraging User Context Prediction for\nAdaptive Memory Management in\nAI-Heavy Mobile Environments"
], sz=32, bold_first=True)
for p in s.shapes[-1].text_frame.paragraphs: p.alignment = PP_ALIGN.CENTER

ln = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(3), Inches(3.2), Inches(7.3), Pt(2))
ln.fill.solid(); ln.fill.fore_color.rgb = BLACK; ln.line.fill.background()

txt(s, 3, 3.5, 7.3, 0.4, ["Capstone Project — Project-I Review"], sz=18)
for p in s.shapes[-1].text_frame.paragraphs: p.alignment = PP_ALIGN.CENTER

txt(s, 3, 4.3, 7.3, 2, [
    "Presented By:",
    "[Student Name 1] — [Roll No.]",
    "[Student Name 2] — [Roll No.]",
    "[Student Name 3] — [Roll No.]",
    "",
    "Guide: [Prof. Name], [Department]",
    "[University Name], [City]",
], sz=16)
for p in s.shapes[-1].text_frame.paragraphs: p.alignment = PP_ALIGN.CENTER

txt(s, 4, 6.7, 5.3, 0.4, ["August 2026"], sz=14)
for p in s.shapes[-1].text_frame.paragraphs: p.alignment = PP_ALIGN.CENTER; p.font.italic = True; p.font.color.rgb = LIGHT
sn(s, 1)


# ═══════════════════════════════════════════════════════════════
# SLIDE 2 — PROBLEM
# ═══════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BL)
title_bar(s, "1. Problem Statement")

box(s, 0.7, 1.3, 6, 2.8, [
    "• Phones have 4–8 GB RAM but AI apps (camera AI, on-device LLMs, translation) consume 300 MB – 4 GB each",
    "",
    "• When RAM is full, Android removes old apps using LRU (least recently used) — purely reactive",
    "",
    "• LMKD (Low Memory Killer Daemon) kills processes only after memory pressure is critical",
    "",
    "• Neither mechanism can predict what the user will need next",
], head="The Current Situation", fs=14)

box(s, 7, 1.3, 5.6, 2.8, [
    ("Cold Start: ", "app removed from RAM must reload from disk — 1–3 sec delay"),
    "",
    ("AI App Cold Start: ", "3–6 sec because AI model weights (300 MB – 4 GB) must be reloaded"),
    "",
    ("Memory Thrashing: ", "phone keeps removing and reloading the same apps in a loop"),
    "",
    ("Battery Drain: ", "redundant disk reads waste energy"),
], head="What Goes Wrong", fs=14)

box(s, 0.7, 4.4, 12, 2.5, [
    "• Average user switches between apps ~65 times per day — each switch is a chance for a wrong eviction decision",
    "",
    ("Our Proposal: ", "use Machine Learning to predict which app the user will open next based on contextual signals (time of day, recent app history, battery level, etc.) and use that prediction to make smarter memory decisions — keeping predicted apps in cache and evicting unlikely ones"),
    "",
    "• This converts memory management from a reactive system to a proactive, prediction-driven system",
], head="Our Approach — Predictive Memory Management", fs=15)

sn(s, 2)


# ═══════════════════════════════════════════════════════════════
# SLIDE 3 — LITERATURE REVIEW (condensed)
# ═══════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BL)
title_bar(s, "2. Literature Review — Key Findings & Research Gap")

box(s, 0.7, 1.3, 6, 2.5, [
    "• Shin et al. (2012) — Naive Bayes for app prediction, 72% accuracy",
    "• Xu/Huang et al. (2024) — Graph models, up to 78% accuracy",
    "• Zhao et al. (2023) — LSTM + context, 74% accuracy",
    "",
    ("Key Gap: ", "prediction research achieves good accuracy but never connects predictions to actual memory management decisions"),
], head="App Usage Prediction (Papers 1–5)", fs=13)

box(s, 7, 1.3, 5.6, 2.5, [
    "• Android LMKD — reactive, PSI-based, no prediction",
    "• Samsung SWAM (2023) — improved OOM, but static costs, no ML",
    "• Chen et al. (2024) — frequency caching, 22% fewer cold starts",
    "",
    ("Key Gap: ", "all existing approaches are reactive or use only simple frequency — none use real-time contextual ML predictions"),
], head="Memory Management (Papers 6–10)", fs=13)

box(s, 0.7, 4.1, 6, 2.5, [
    "• XGBoost (Chen & Guestrin, 2016) — fast, interpretable, no GPU",
    "• Hashemi et al. (2018) — ML for CPU cache prefetching (L1/L2)",
    "",
    ("Key Gap: ", "ML applied to hardware caches but NOT to app-level process cache management on mobile devices"),
], head="ML for System Optimization (Papers 11–14)", fs=13)

box(s, 7, 4.1, 5.6, 2.5, [
    "• MCUNet (2020), mLLM (2025) — optimize single AI model in memory",
    "• He et al. (2025) — LLM memory optimization",
    "",
    ("Key Gap: ", "focuses on one AI model at a time — nobody solves the multi-app contention problem when several AI apps compete for limited RAM"),
    "",
    ("OVERALL GAP: ", "No system combines ML prediction + cache eviction scoring + preload gating + thrashing detection"),
], head="On-Device AI (Papers 15–18)", fs=13)

sn(s, 3)


# ═══════════════════════════════════════════════════════════════
# SLIDE 4 — PIPELINE OVERVIEW
# ═══════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BL)
title_bar(s, "3. Proposed Methodology — Predict-Decide-Act Pipeline")

txt(s, 0.7, 1.2, 12, 0.5, [
    "Every time the user switches apps, our system runs a 5-stage pipeline:"
], sz=16)

img(s, "pipeline_diagram.jpg", l=0.7, t=1.8, w=12)

txt(s, 0.7, 5.8, 12, 1.5, [
    ("Stage 1: ", "Capture device context (current app, recent history, time, battery, RAM, AI task status)"),
    ("Stage 2: ", "XGBoost ML model predicts the next app with a confidence score"),
    ("Stage 3–4: ", "Score all cached apps using prediction + other factors → decide what to evict/preload"),
    ("Stage 5: ", "Execute memory actions and feed results back for the next cycle"),
], sz=14)

sn(s, 4)


# ═══════════════════════════════════════════════════════════════
# SLIDE 5 — SCORING APPROACH (high level)
# ═══════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BL)
title_bar(s, "4. Cache Scoring & Decision Logic")

txt(s, 0.7, 1.2, 12, 0.5, [
    "Every cached app gets a \"keep score\" — highest stays, lowest gets evicted first."
], sz=16)

# Formula box
sh = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.7), Inches(1.9), Inches(12), Inches(0.65))
sh.fill.solid(); sh.fill.fore_color.rgb = BG; sh.line.color.rgb = BLACK; sh.line.width = Pt(1)
txt(s, 0.9, 1.95, 11.5, 0.5, [
    "Score(app) = Recency + Frequency + Prediction + Importance + Context − MemoryPressure"
], sz=17)

box(s, 0.7, 2.8, 6, 3.2, [
    ("Recency — ", "how recently the app was used"),
    ("Frequency — ", "how often the app is used"),
    ("Prediction — ", "ML model's confidence the user will open this app next (KEY INNOVATION)"),
    ("Importance — ", "category-based priority (messaging > games)"),
    ("Context — ", "relevance to current situation (Maps during commute = high)"),
    ("Memory Pressure — ", "penalty that increases when RAM is running low"),
    "",
    "The Prediction factor is what makes this system forward-looking — existing systems only look backward at recency and frequency.",
], head="6 Factors in the Score", fs=14)

box(s, 7, 2.8, 5.6, 1.5, [
    "Before pre-loading a predicted app, 5 conditions must ALL pass:",
    ("  1. ", "ML confidence is high enough"),
    ("  2. ", "Enough free RAM available"),
    ("  3. ", "Battery is not critically low"),
    ("  4. ", "System is not currently thrashing"),
    ("  5. ", "App is not already in memory"),
], head="Preload Safety Gate (5 Conditions)", fs=13)

box(s, 7, 4.5, 5.6, 1.5, [
    "If the system detects it's making bad decisions (evicting apps and immediately needing them back), it:",
    ("• Enters Safe Mode: ", "stops preloading, becomes more cautious"),
    ("• Auto-recovers: ", "once stability returns, resumes normal operation"),
], head="Thrashing Detection & Safe Mode", fs=13)

sn(s, 5)


# ═══════════════════════════════════════════════════════════════
# SLIDE 6 — ML MODEL
# ═══════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BL)
title_bar(s, "5. ML Model Selection — Why XGBoost")

tbl(s, [
    ["Criterion", "Markov Chain (Baseline)", "XGBoost (Selected)", "LSTM (Rejected)"],
    ["Inference speed", "Instant (<1 ms)", "Very fast (~2 ms)", "Slow (~15 ms)"],
    ["Context-awareness", "Knows only current app", "Uses 19 contextual signals", "Uses 19 signals"],
    ["Interpretability", "Yes (transition matrix)", "Yes (feature importance)", "No (black box)"],
    ["GPU required?", "No", "No", "Yes"],
    ["Expected accuracy", "55–65%", "70–80%", "72–82%"],
], l=0.7, t=1.3, w=12)

box(s, 0.7, 4.3, 5.8, 2.8, [
    "XGBoost selected because it offers the best balance of:",
    "",
    "• Speed — fast enough to run on every app switch without the user noticing",
    "• Accuracy — significantly better than Markov baseline",
    "• Interpretability — can explain why each prediction was made (important for patent claims)",
    "• No GPU needed — runs entirely on CPU, doesn't drain battery",
], head="Why XGBoost?", fs=14)

box(s, 6.7, 4.3, 5.9, 2.8, [
    "12 Raw Features:",
    "current_app, prev_app (×3), hour, day_of_week, battery, is_charging, ram_free, network_type, ai_task_active, app_category",
    "",
    "7 Derived Features:",
    "ram_utilization, is_peak_hour, is_weekend, session_depth, time_since_last_switch, battery_bucket, prev_app_category",
    "",
    "Total: 19 features per prediction event",
], head="Feature Set (19 Total)", fs=13)

sn(s, 6)


# ═══════════════════════════════════════════════════════════════
# SLIDE 7 — HLD
# ═══════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BL)
title_bar(s, "6. System Architecture — High Level Design")

img(s, "hld_architecture.jpg", l=0.5, t=1.2, w=12.3)

txt(s, 0.7, 5.6, 12, 1.5, [
    ("Modules 1→4: ", "real-time pipeline — context collection → prediction → scoring & decision → execution (with feedback loop)"),
    ("Module 5: ", "simulation engine — replays recorded sessions to benchmark against baseline policies"),
    ("Module 6: ", "Streamlit dashboard — real-time visualization of predictions, memory state, and comparisons"),
], sz=14)

sn(s, 7)


# ═══════════════════════════════════════════════════════════════
# SLIDE 8 — CLASS DIAGRAM
# ═══════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BL)
title_bar(s, "7. Class Design")

img(s, "class_diagram.jpg", l=0.5, t=1.2, w=12.3)

txt(s, 0.7, 5.6, 12, 1.5, [
    ("Key classes: ", "ContextSnapshot (device state) → FeatureEngine (converts to ML input) → Predictor interface (Markov / XGBoost)"),
    ("Decision layer: ", "MemoryDecisionController orchestrates CacheScoringEngine + PreloadGate + ThrashingDetector → produces MemoryActionPlan"),
    ("Design pattern: ", "Strategy pattern for swappable predictors; Composition for the decision controller"),
], sz=14)

sn(s, 8)


# ═══════════════════════════════════════════════════════════════
# SLIDE 9 — COMPARISON & NOVELTY
# ═══════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BL)
title_bar(s, "8. Comparison with Existing Approaches")

tbl(s, [
    ["Feature", "No Optimization", "LRU (Android)", "Static Preload", "Our System"],
    ["Eviction logic", "Remove oldest", "Remove least recent", "LRU + top-3 apps", "ML-weighted scoring"],
    ["Prediction-aware?", "No", "No", "Partially", "Yes (real-time ML)"],
    ["Context-aware?", "No", "No", "No", "Yes (19 signals)"],
    ["Preload capability", "No", "No", "Fixed list", "Conditional (5 safety gates)"],
    ["Thrashing detection", "No", "No", "No", "Yes (auto Safe Mode)"],
], l=0.7, t=1.3, w=12)

txt(s, 0.7, 4.2, 12, 3.0, [
    "4 Novel Contributions:",
    "",
    ("1. Prediction-integrated cache scoring: ", "first system to use ML prediction confidence directly in the eviction decision formula"),
    "",
    ("2. Multi-condition preload gating: ", "5 simultaneous safety checks prevent wasteful or harmful preloading"),
    "",
    ("3. Automatic thrashing recovery: ", "system detects when predictions are hurting performance and self-corrects via Safe Mode"),
    "",
    ("4. AI-workload awareness: ", "higher eviction resistance for expensive AI model caches that take 3–6 sec to reload"),
], sz=15, bold_first=True)

sn(s, 9)


# ═══════════════════════════════════════════════════════════════
# SLIDE 10 — EVALUATION PLAN
# ═══════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BL)
title_bar(s, "9. Evaluation Plan")

box(s, 0.7, 1.3, 5.8, 3.0, [
    "Since modifying a real Android kernel is infeasible, we use a discrete-event simulator:",
    "",
    "• 50,000 synthetic app-switch events",
    "• 5 user personas (student, professional, casual, power user, developer)",
    "• 4 policies compared under identical conditions",
    "• 3 RAM configurations (4 GB, 6 GB, 8 GB)",
    "• 30 repetitions per configuration",
    "• Statistical validation: paired t-test, p < 0.05",
], head="Simulation-Based Testing", fs=14)

box(s, 6.7, 1.3, 5.9, 3.0, [
    ("Cache Hit Rate: ", "% of app launches that are warm starts"),
    "",
    ("Avg Launch Latency: ", "mean time to open an app"),
    "",
    ("Thrashing Events: ", "evict-reload cycles per session"),
    "",
    ("Preload Success Rate: ", "% of preloads that were actually used"),
    "",
    ("Prediction Accuracy: ", "top-1 and top-3 correct prediction rates"),
], head="Key Performance Indicators", fs=14)

box(s, 0.7, 4.6, 12, 2.3, [
    "Systematically remove one feature group at a time and measure accuracy drop — proves which contextual signals contribute most to prediction quality.",
    "",
    "• Remove app history → expect largest accuracy drop (proves sequential patterns are most important)",
    "• Remove temporal features → moderate drop (proves time-of-day matters)",
    "• Remove device state → smaller drop (battery/RAM add value but less than history)",
    "• Keep only current app → drops to Markov baseline (proves multi-feature approach is essential)",
], head="Ablation Study — Validating Feature Importance", fs=14)

sn(s, 10)


# ═══════════════════════════════════════════════════════════════
# SLIDE 11 — TIMELINE
# ═══════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BL)
title_bar(s, "10. Implementation Timeline & Technology Stack")

tbl(s, [
    ["Week", "Phase", "What We Deliver"],
    ["1", "Foundation + Data", "Project setup, app profiles, 50K synthetic events"],
    ["2", "ML Models", "Feature engine, Markov + XGBoost training & evaluation"],
    ["3", "Memory Simulator", "Memory state tracker, 3 baselines + proposed system"],
    ["4", "Benchmarking", "Full benchmark suite + ablation + statistical validation"],
    ["5", "Visualization", "Publication charts + Streamlit dashboard"],
    ["6–7", "Patent Filing", "Patent specification, prior art search, IPO filing"],
    ["8–9", "IEEE Paper", "6-page LaTeX paper, capstone report, conference submission"],
], l=0.7, t=1.3, w=12)

tbl(s, [
    ["Component", "Technology", "Why"],
    ["Language", "Python 3.10+", "Best ecosystem for ML + data science"],
    ["ML Framework", "XGBoost + scikit-learn", "Fast, no GPU, interpretable"],
    ["Data Processing", "pandas + numpy", "Standard for tabular data"],
    ["Statistics", "scipy", "t-tests, p-values"],
    ["Visualization", "matplotlib + seaborn + plotly", "Publication-quality figures"],
    ["Dashboard", "Streamlit", "Rapid interactive web apps"],
    ["Database", "SQLite", "Lightweight, zero-config"],
], l=0.7, t=4.5, w=12)

sn(s, 11)


# ═══════════════════════════════════════════════════════════════
# SLIDE 12 — REFERENCES (hyperlinked)
# ═══════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BL)
title_bar(s, "11. References")

refs_with_links = [
    ('[1] C. Shin et al., "Understanding and Prediction of Mobile Application Usage," ACM UbiComp, 2012.', 'https://doi.org/10.1145/2370216.2370241', 'DOI: 10.1145/2370216.2370241'),
    ('[2] Z. Huang et al., "Predicting Mobile App Usage with Context-Aware Dynamic Hypergraphs," IEEE TMC, 2025.', 'https://doi.org/10.1109/TMC.2024.3521876', 'DOI: 10.1109/TMC.2024.3521876'),
    ('[3] S. Zhao et al., "AppUsage2Vec: Modeling Smartphone App Usage for Prediction," IEEE ICDE, 2019.', 'https://doi.org/10.1109/ICDE.2019.00120', 'DOI: 10.1109/ICDE.2019.00120'),
    ('[4] S. Zhao et al., "Context-Aware Mobile App Prediction via Deep Sequential Learning," ACM UbiComp, 2023.', 'https://doi.org/10.1145/3594739.3610680', 'DOI: 10.1145/3594739.3610680'),
    ('[5] H. Pan et al., "DeepApp: Predicting Personalized Smartphone App Usage," ACM TIST, 2023.', 'https://doi.org/10.1145/3568684', 'DOI: 10.1145/3568684'),
    ('[6] Google AOSP, "Low Memory Killer Daemon (lmkd)," 2024.', 'https://source.android.com/docs/core/perf/lmkd', 'source.android.com/docs/core/perf/lmkd'),
    ('[7] S. Lee et al., "SWAM: Revisiting Swap and OOM for Mobile," Samsung, MobiSys, 2023.', 'https://doi.org/10.1145/3591197.3593422', 'DOI: 10.1145/3591197.3593422'),
    ('[8] Y. Chen et al., "Memory-Efficient Mobile App Mgmt. via Predictive Caching," IEEE Access, 2024.', 'https://doi.org/10.1109/ACCESS.2024.3372104', 'DOI: 10.1109/ACCESS.2024.3372104'),
    ('[9] Android Developers, "App Standby Buckets," 2024.', 'https://developer.android.com/topic/performance/appstandby', 'developer.android.com/…/appstandby'),
    ('[10] J. Park et al., "Learning-Based Memory Management for Mobile," IEEE/ACM MICRO, 2023.', 'https://doi.org/10.1145/3613424.3614274', 'DOI: 10.1145/3613424.3614274'),
    ('[11] T. Chen, C. Guestrin, "XGBoost: A Scalable Tree Boosting System," ACM SIGKDD, 2016.', 'https://doi.org/10.1145/2939672.2939785', 'DOI: 10.1145/2939672.2939785'),
    ('[12] E. Ipek et al., "Self-Optimizing Memory Controllers: A RL Approach," IEEE/ACM ISCA, 2008.', 'https://doi.org/10.1109/ISCA.2008.21', 'DOI: 10.1109/ISCA.2008.21'),
    ('[13] Z. Shi et al., "Applying Deep Learning to Data Prefetching," IEEE/ACM MICRO, 2023.', 'https://doi.org/10.1145/3613424.3614259', 'DOI: 10.1145/3613424.3614259'),
    ('[14] M. Hashemi et al., "Learning Memory Access Patterns," ICML, 2018.', 'https://arxiv.org/abs/1803.02329', 'arXiv: 1803.02329'),
    ('[15] J. Lin et al., "MCUNet: Tiny Deep Learning on IoT Devices," NeurIPS, 2020.', 'https://arxiv.org/abs/2007.10319', 'arXiv: 2007.10319'),
    ('[16] S. Laskaridis et al., "Adaptive Inference through Early-Exit Networks," ACM MobiSys, 2024.', 'https://doi.org/10.1145/3643832.3661880', 'DOI: 10.1145/3643832.3661880'),
    ('[17] Q. He et al., "On-Device LLM Inference: Memory Optimization for Mobile," IEEE Edge, 2025.', 'https://doi.org/10.1109/SEC62691.2025.00032', 'DOI: 10.1109/SEC62691.2025.00032'),
    ('[18] A. Dhakal et al., "mLLM: On-Device LLM Serving," USENIX OSDI, 2025.', 'https://www.usenix.org/conference/osdi25', 'usenix.org/conference/osdi25'),
]

tb = s.shapes.add_textbox(Inches(0.7), Inches(1.3), Inches(12), Inches(5.2))
tf = tb.text_frame; tf.word_wrap = True
for i, (citation, url, link_text) in enumerate(refs_with_links):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    run = p.add_run(); run.text = citation + '  '; run.font.size = Pt(10); run.font.name = FONT; run.font.color.rgb = DARK
    run2 = p.add_run(); run2.text = link_text; run2.font.size = Pt(10); run2.font.name = FONT
    run2.font.color.rgb = BLUE; run2.font.underline = True; run2.hyperlink.address = url
    p.space_after = Pt(2)

tx = s.shapes.add_textbox(Inches(4), Inches(6.6), Inches(5.3), Inches(0.6))
p = tx.text_frame.paragraphs[0]
p.text = "Thank You"; p.font.size = Pt(26); p.font.bold = True; p.font.name = FONT; p.font.color.rgb = BLACK
p.alignment = PP_ALIGN.CENTER

sn(s, 12)


# ═══════════════════════════════════════════════════════════════
out = os.path.join(BASE, "Capstone_Presentation_FINAL.pptx")
prs.save(out)
print(f"Saved: {out}")
print(f"Slides: {len(prs.slides)}")
