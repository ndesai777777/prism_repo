from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


OUT = Path("prism_candidate_dashboard_light.png")
W, H = 1600, 900


def load_font(size, bold=False):
    paths = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for path in paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


FONT = {
    "display": load_font(34, True),
    "h1": load_font(30, True),
    "h2": load_font(22, True),
    "h3": load_font(17, True),
    "body": load_font(16),
    "body_bold": load_font(16, True),
    "small": load_font(14),
    "small_bold": load_font(14, True),
    "tiny": load_font(12),
    "metric": load_font(46, True),
    "metric_sm": load_font(36, True),
}

COL = {
    "bg": "#f5f7fb",
    "card": "#ffffff",
    "soft": "#f8fafc",
    "border": "#d8e1eb",
    "line": "#e2e8f0",
    "title": "#0f172a",
    "text": "#1f2937",
    "muted": "#64748b",
    "green": "#059669",
    "green_fill": "#22c55e",
    "green_soft": "#ecfdf5",
    "green_border": "#a7f3d0",
    "teal": "#0f766e",
    "teal_soft": "#ecfeff",
    "amber": "#b45309",
    "amber_soft": "#fff7ed",
    "amber_border": "#fed7aa",
    "red": "#dc2626",
    "red_fill": "#ef4444",
    "red_soft": "#fef2f2",
    "red_border": "#fecaca",
    "bar_bg": "#e5e7eb",
    "shadow": "#e6edf5",
}


img = Image.new("RGB", (W, H), COL["bg"])
d = ImageDraw.Draw(img)


def rr(box, fill, outline=None, width=1, radius=16):
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def card(x, y, w, h, radius=18):
    rr((x + 5, y + 7, x + w + 5, y + h + 7), COL["shadow"], COL["shadow"], 1, radius)
    rr((x, y, x + w, y + h), COL["card"], COL["border"], 1, radius)


def draw_text(x, y, value, style="body", fill=None, anchor=None):
    d.text((x, y), value, font=FONT[style], fill=fill or COL["text"], anchor=anchor)


def text_w(text, style="body"):
    box = d.textbbox((0, 0), text, font=FONT[style])
    return box[2] - box[0]


def wrap_lines(value, style, max_width):
    words = value.split()
    lines = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        if text_w(trial, style) <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def wrapped_text(x, y, value, style="body", fill=None, max_width=300, line_gap=6):
    line_h = FONT[style].size + line_gap
    for line in wrap_lines(value, style, max_width):
        draw_text(x, y, line, style, fill)
        y += line_h
    return y


def bar(x, y, w, h, pct, fill):
    rr((x, y, x + w, y + h), COL["bar_bg"], COL["bar_bg"], 1, h // 2)
    rr((x, y, x + max(2, int(w * pct)), y + h), fill, fill, 1, h // 2)


def reverse_bar(x, y, w, h, pct, fill):
    rr((x, y, x + w, y + h), COL["bar_bg"], COL["bar_bg"], 1, h // 2)
    used = max(2, int(w * pct))
    rr((x + w - used, y, x + w, y + h), fill, fill, 1, h // 2)


def chip(x, y, label, fill, outline, text_fill, w=None):
    pad = 12
    width = w or text_w(label, "tiny") + pad * 2
    rr((x, y, x + width, y + 26), fill, outline, 1, 13)
    draw_text(x + width / 2, y + 13, label, "tiny", text_fill, "mm")
    return width


def check_icon(cx, cy, fill="#dcfce7", outline="#22c55e"):
    d.ellipse((cx - 10, cy - 10, cx + 10, cy + 10), fill=fill, outline=outline, width=1)
    d.line((cx - 5, cy, cx - 1, cy + 5, cx + 7, cy - 6), fill=COL["green"], width=2)


def caution_icon(cx, cy):
    d.ellipse((cx - 10, cy - 10, cx + 10, cy + 10), fill=COL["amber_soft"], outline="#f59e0b", width=1)
    d.line((cx - 6, cy, cx + 6, cy), fill=COL["amber"], width=2)


# Header
draw_text(64, 36, "PRISM Candidate Targeting Review", "display", COL["title"])
draw_text(64, 78, "Member-level targeting decision view with transparent feature evidence and LLM support notes.", "body", COL["muted"])
draw_text(64, 114, "Candidate ID: M-1042 | Program: Complex Care | Outcome: 90-day ED utilization | Model: GLMNet T-Learner", "small", COL["muted"])

rr((1166, 40, 1536, 136), COL["green_soft"], COL["green_border"], 1, 16)
draw_text(1198, 62, "RECOMMENDATION", "small_bold", COL["green"])
draw_text(1198, 90, "Target", "metric_sm", COL["title"])
d.line((1348, 62, 1348, 116), fill=COL["green_border"], width=1)
draw_text(1376, 66, "CONFIDENCE", "tiny", COL["muted"])
draw_text(1376, 92, "High", "h2", COL["green"])


# Metric row
metric_y = 166
metric_h = 145
gap = 22
metric_w = 350
metrics = [
    (64, "Risk if targeted", "18%", "Predicted ED risk with outreach", COL["teal"]),
    (64 + metric_w + gap, "Risk if not targeted", "31%", "Predicted ED risk without outreach", COL["title"]),
    (64 + (metric_w + gap) * 2, "Estimated benefit", "+13 pts", "Control risk minus treated risk", COL["green"]),
    (64 + (metric_w + gap) * 3, "Targeting score", "86/100", "Top benefit decile", COL["green"]),
]
for x, heading, metric, note, color in metrics:
    card(x, metric_y, metric_w, metric_h)
    draw_text(x + 28, metric_y + 26, heading, "h3", COL["muted"])
    draw_text(x + 28, metric_y + 55, metric, "metric", color)
    draw_text(x + 28, metric_y + 113, note, "small", COL["muted"])
    if heading == "Targeting score":
        bar(x + 205, metric_y + 71, 105, 12, 0.86, COL["green_fill"])


# Evidence panel
card(64, 342, 716, 360)
draw_text(96, 374, "Feature Evidence", "h2", COL["title"])
draw_text(96, 404, "Plain-language variables used in the candidate targeting score.", "small", COL["muted"])
draw_text(96, 447, "Push toward targeting", "h3", COL["green"])
draw_text(434, 447, "Pull away / caution", "h3", COL["red"])
d.line((404, 438, 404, 667), fill=COL["line"], width=1)

positive_rows = [
    ("Prior ED visits: 3 in 6m", 0.96, "+0.045"),
    ("Diabetes flag: Yes", 0.62, "+0.028"),
    ("Age: 67", 0.47, "+0.019"),
    ("Insurance type: Medicaid MCO", 0.33, "+0.013"),
]
for i, (label, pct, value) in enumerate(positive_rows):
    y = 485 + i * 48
    draw_text(96, y, label, "small", COL["text"])
    bar(96, y + 23, 220, 14, pct, COL["green_fill"])
    draw_text(328, y + 20, value, "small_bold", COL["green"])

negative_rows = [
    ("Region: Rural North", 0.52, "-0.016"),
    ("No inpatient stay last 6m", 0.39, "-0.012"),
    ("PCP visit last month", 0.29, "-0.009"),
]
for i, (label, pct, value) in enumerate(negative_rows):
    y = 485 + i * 56
    draw_text(434, y, label, "small", COL["text"])
    reverse_bar(434, y + 23, 180, 14, pct, COL["red_fill"])
    draw_text(628, y + 20, value, "small_bold", COL["red"])

rr((434, 646, 724, 674), COL["amber_soft"], COL["amber_border"], 1, 8)
draw_text(450, 654, "Cautions do not outweigh benefit signal", "tiny", COL["amber"])


# LLM notes panel
card(810, 342, 726, 360)
draw_text(842, 374, "LLM Decision Notes", "h2", COL["title"])
draw_text(842, 404, "Narrative synthesis of the member profile, not a repeat of the feature list.", "small", COL["muted"])

draw_text(842, 447, "Why this is a decent candidate", "h3", COL["green"])
draw_text(1192, 447, "Why this could be a poor fit", "h3", COL["amber"])
d.line((1168, 442, 1168, 674), fill=COL["line"], width=1)

support_notes = [
    ("Strong care fit", "At 67 with diabetes and 3 ED visits in 6 months, this looks like a chronic-care coordination case, not just a high score."),
    ("Actionable path", "Medicaid MCO coverage gives the team a practical route for benefit navigation, ED diversion planning, and diabetes follow-up."),
]
y = 487
for label, note in support_notes:
    check_icon(852, y + 10)
    draw_text(874, y, label, "small_bold", COL["green"])
    wrapped_text(842, y + 28, note, "small", COL["text"], 300, 3)
    y += 104

caution_notes = [
    ("Regional access", "Rural North may limit transportation vendors or provider access, so confirm local support capacity before assignment."),
    ("Clinical context", "If the ED visits were accidents or diabetes is already controlled, program impact could be lower."),
]
y = 487
for label, note in caution_notes:
    caution_icon(1202, y + 10)
    draw_text(1224, y, label, "small_bold", COL["amber"])
    wrapped_text(1192, y + 28, note, "small", COL["text"], 310, 3)
    y += 104


# Action strip
card(64, 738, 1472, 110, 16)
draw_text(96, 771, "Recommended next action", "h2", COL["title"])
rr((410, 761, 598, 805), COL["green_soft"], COL["green_border"], 1, 22)
draw_text(504, 783, "Assign outreach", "body_bold", COL["green"], "mm")

draw_text(96, 816, "First outreach focus: diabetes follow-up, ED diversion planning, benefit navigation, and regional access barriers.", "body", COL["text"])
d.line((770, 760, 770, 826), fill=COL["line"], width=1)
draw_text(804, 771, "Reviewer guardrails", "h3", COL["title"])
draw_text(804, 802, "Confirm eligibility, recent contact history, member engagement channel, and care-manager capacity before final assignment.", "small", COL["muted"])


img.save(OUT, quality=95)
print(OUT.resolve())
