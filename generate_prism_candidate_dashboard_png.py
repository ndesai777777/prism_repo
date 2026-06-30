from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


OUT = Path("prism_candidate_dashboard_light.png")
W, H = 1600, 900


def font(size, bold=False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


F = {
    "title": font(34, True),
    "h2": font(24, True),
    "h3": font(18, True),
    "body": font(18),
    "small": font(15),
    "tiny": font(13),
    "big": font(46, True),
    "huge": font(40, True),
}

COL = {
    "bg": "#f5f7fb",
    "card": "#ffffff",
    "border": "#d9e2ec",
    "soft": "#f8fafc",
    "line": "#e2e8f0",
    "text": "#1f2937",
    "title": "#0f172a",
    "muted": "#64748b",
    "green": "#059669",
    "green2": "#22c55e",
    "teal": "#0f766e",
    "amber": "#b45309",
    "red": "#dc2626",
    "red2": "#ef4444",
    "good_soft": "#ecfdf5",
    "good_border": "#a7f3d0",
    "warn_soft": "#fff7ed",
    "warn_border": "#fed7aa",
    "bar_bg": "#e5e7eb",
}


img = Image.new("RGB", (W, H), COL["bg"])
d = ImageDraw.Draw(img)


def rr(box, fill, outline=COL["border"], width=1, radius=16):
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def text(x, y, s, f="body", fill=COL["text"], anchor=None):
    d.text((x, y), s, font=F[f], fill=fill, anchor=anchor)


def card(x, y, w, h, r=16):
    rr((x + 4, y + 6, x + w + 4, y + h + 6), "#e9eef5", "#e9eef5", 1, r)
    rr((x, y, x + w, y + h), COL["card"], COL["border"], 1, r)


def bar(x, y, w, h, pct, color):
    rr((x, y, x + w, y + h), COL["bar_bg"], COL["bar_bg"], 1, h // 2)
    rr((x, y, x + int(w * pct), y + h), color, color, 1, h // 2)


def bullet(cx, cy, kind="good"):
    if kind == "good":
        d.ellipse((cx - 10, cy - 10, cx + 10, cy + 10), fill="#dcfce7", outline=COL["green2"], width=1)
        d.line((cx - 5, cy, cx - 1, cy + 5, cx + 7, cy - 6), fill="#16a34a", width=2)
    else:
        d.ellipse((cx - 10, cy - 10, cx + 10, cy + 10), fill="#ffedd5", outline="#f59e0b", width=1)
        d.line((cx - 6, cy, cx + 6, cy), fill=COL["amber"], width=2)


text(64, 36, "PRISM Candidate Targeting Review", "title", COL["title"])
text(64, 76, "Clear member-level view for deciding whether a candidate is strong enough for outreach", "body", COL["muted"])
text(64, 112, "Candidate ID: M-1042 | Program: Complex Care | Outcome: 90-day ED utilization | Model: GLMNet T-Learner", "small", COL["muted"])

rr((1170, 42, 1520, 138), COL["good_soft"], COL["good_border"], 1, 14)
text(1204, 62, "OVERALL RECOMMENDATION", "small", COL["green"])
text(1204, 80, "Target", "huge", COL["title"])
text(1342, 72, "Confidence", "small", COL["muted"])
text(1342, 94, "High", "h2", COL["green"])

card(64, 166, 410, 250)
text(96, 190, "Why This Candidate Stands Out", "h2", COL["title"])
text(96, 224, "PRISM estimates a meaningful preventable ED-risk gap.", "small", COL["muted"])
text(96, 270, "Risk if targeted", "small", COL["muted"])
text(96, 292, "18%", "big", COL["title"])
text(236, 270, "Risk if not targeted", "small", COL["muted"])
text(236, 292, "31%", "big", COL["title"])
bar(96, 350, 300, 14, 0.70, COL["green2"])
text(96, 376, "Estimated benefit: +13 percentage points", "h3", COL["green"])

card(506, 166, 338, 250)
text(538, 190, "Targeting Score", "h2", COL["title"])
d.ellipse((602, 232, 746, 376), outline=COL["bar_bg"], width=18)
d.arc((602, 232, 746, 376), 268, 555, fill=COL["green2"], width=18)
text(674, 274, "86", "huge", COL["title"], "mm")
text(674, 318, "out of 100", "small", COL["muted"], "mm")
rr((566, 362, 782, 396), COL["good_soft"], COL["good_border"], 1, 17)
text(674, 379, "Top Benefit Decile", "small", COL["green"], "mm")

card(876, 166, 660, 250)
text(908, 190, "LLM-Generated Summary", "h2", COL["title"])
text(908, 224, "Plain-language rationale generated from model outputs and member context.", "small", COL["muted"])
bullet(922, 274, "good")
text(946, 266, "Good candidate because recent utilization and risk profile suggest avoidable near-term ED use.", "body")
bullet(922, 323, "good")
text(946, 315, "Transportation and medication-adherence signals appear actionable through care coordination.", "body")
bullet(922, 372, "warn")
text(946, 364, "Review recent outreach history before assignment to avoid duplicative contact.", "body")

card(64, 448, 706, 300)
text(96, 472, "Top Positive Contributors", "h2", COL["title"])
text(96, 506, "Features increasing expected value from targeting.", "small", COL["muted"])
positives = [
    ("Recent ED visits last 6 months", 0.92, "+0.041"),
    ("Current risk score", 0.76, "+0.034"),
    ("Transportation barrier", 0.50, "+0.022"),
    ("CHF / COPD comorbidity flags", 0.37, "+0.016"),
]
for i, (name, pct, val) in enumerate(positives):
    y = 550 + i * 56
    text(96, y + 2, name, "small")
    bar(360, y, 250, 18, pct, COL["green2"])
    text(642, y + 1, val, "small", COL["green"])
    if i < 3:
        d.line((96, y + 40, 738, y + 40), fill=COL["line"], width=1)

card(802, 448, 386, 300)
text(834, 472, "Negative Contributors", "h2", COL["title"])
text(834, 506, "Factors that reduce targeting priority.", "small", COL["muted"])
negatives = [
    ("Stable PCP engagement", 0.47, "-0.018"),
    ("Low inpatient use", 0.36, "-0.014"),
    ("Prior program exposure", 0.26, "-0.010"),
]
for i, (name, pct, val) in enumerate(negatives):
    y = 550 + i * 56
    text(834, y + 2, name, "small")
    rr((1015, y, 1119, y + 18), COL["bar_bg"], COL["bar_bg"], 1, 9)
    rr((1119 - int(104 * pct), y, 1119, y + 18), COL["red2"], COL["red2"], 1, 9)
    text(1150, y + 1, val, "small", COL["red"])
    if i < 2:
        d.line((834, y + 40, 1156, y + 40), fill=COL["line"], width=1)
rr((834, 704, 1140, 736), COL["warn_soft"], COL["warn_border"], 1, 8)
text(850, 714, "Net read: positives outweigh caution signals", "tiny", COL["amber"])

card(1220, 448, 316, 300)
text(1252, 472, "Next Action", "h2", COL["title"])
rr((1252, 524, 1472, 568), COL["good_soft"], COL["good_border"], 1, 22)
text(1362, 546, "Assign Outreach", "body", COL["green"], "mm")
text(1252, 596, "Recommended focus:", "small", COL["muted"])
text(1252, 628, "Care navigation, transportation support,", "small")
text(1252, 654, "medication adherence follow-up, and ED", "small")
text(1252, 680, "diversion planning.", "small")
d.line((1252, 715, 1498, 715), fill=COL["line"], width=1)
text(1252, 728, "Human review required before final assignment", "tiny", COL["muted"])

rr((64, 780, 1536, 852), COL["soft"], COL["line"], 1, 14)
text(96, 806, "Reviewer guardrails", "h3", COL["title"])
text(280, 808, "Confirm eligibility, recent contact history, member engagement channel, and care-manager capacity before acting on the recommendation.", "small", COL["muted"])

img.save(OUT, quality=95)
print(OUT.resolve())
