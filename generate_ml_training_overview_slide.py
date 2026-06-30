from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


OUT = Path("ml_training_model_overview.png")
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
    "title": load_font(42, True),
    "subtitle": load_font(22),
    "h2": load_font(26, True),
    "h3": load_font(19, True),
    "body": load_font(18),
    "body_bold": load_font(18, True),
    "small": load_font(15),
    "tiny": load_font(12),
    "logo": load_font(44, True),
    "logo_small": load_font(15, True),
}

COL = {
    "bg": "#ffffff",
    "ink": "#101828",
    "text": "#263238",
    "muted": "#667085",
    "green": "#006400",
    "accent": "#24b82f",
    "lime": "#b7f14a",
    "soft_green": "#edf8ef",
    "soft_blue": "#eef6ff",
    "soft_orange": "#fff7ed",
    "blue": "#2563eb",
    "orange": "#f97316",
    "border": "#d0d5dd",
    "line": "#d7dee8",
    "shadow": "#eef2f7",
}


img = Image.new("RGB", (W, H), COL["bg"])
d = ImageDraw.Draw(img)


def text_width(value, style="body"):
    box = d.textbbox((0, 0), value, font=FONT[style])
    return box[2] - box[0]


def wrap(value, style, max_width):
    words = value.split()
    lines = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        if text_width(trial, style) <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_text(x, y, value, style="body", fill=None, anchor=None):
    d.text((x, y), value, font=FONT[style], fill=fill or COL["text"], anchor=anchor)


def draw_wrapped(x, y, value, style="body", fill=None, max_width=360, line_gap=7):
    for line in wrap(value, style, max_width):
        draw_text(x, y, line, style, fill)
        y += FONT[style].size + line_gap
    return y


def rr(box, fill, outline=COL["border"], width=1, radius=18):
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def card(x, y, w, h, fill="#ffffff", radius=20):
    rr((x + 5, y + 8, x + w + 5, y + h + 8), COL["shadow"], COL["shadow"], 1, radius)
    rr((x, y, x + w, y + h), fill, COL["border"], 1, radius)


def arrow(x1, y1, x2, y2, color=COL["green"]):
    d.line((x1, y1, x2, y2), fill=color, width=4)
    d.polygon([(x2, y2), (x2 - 13, y2 - 8), (x2 - 13, y2 + 8)], fill=color)


def bullet(x, y, value, max_width=360):
    d.ellipse((x, y + 7, x + 8, y + 15), fill=COL["green"])
    return draw_wrapped(x + 18, y, value, "body", COL["text"], max_width, 5)


def logo(x, y):
    d.rectangle((x, y, x + 270, y + 112), fill=COL["green"])
    draw_text(x + 28, y + 24, "Acentra", "logo", "#ffffff")
    draw_text(x + 96, y + 78, "H E A L T H", "logo_small", "#ffffff")


logo(64, 42)
draw_text(64, 190, "What Does it Mean to Train a Model?", "title", COL["ink"])
draw_text(
    64,
    246,
    "A model is trained by learning patterns from data so it can make useful predictions or discover structure.",
    "subtitle",
    COL["muted"],
)


# Learning types
card(64, 305, 708, 238, COL["soft_blue"])
draw_text(96, 340, "Supervised Learning", "h2", COL["blue"])
draw_wrapped(
    96,
    382,
    "The data includes both inputs and the known answer. The model learns from examples so it can predict the answer for new cases.",
    "body",
    COL["text"],
    610,
)
bullet(96, 458, "Example: use age, prior ED visits, conditions, and insurance type to predict 90-day ED risk.", 595)

card(828, 305, 708, 238, COL["soft_orange"])
draw_text(860, 340, "Unsupervised Learning", "h2", COL["orange"])
draw_wrapped(
    860,
    382,
    "The data has inputs, but no known answer column. The model looks for structure, patterns, or groups on its own.",
    "body",
    COL["text"],
    610,
)
bullet(860, 458, "Example: group members into similar profiles based on utilization, chronic conditions, and demographics.", 595)


# Supervised learning workflow
draw_text(64, 607, "General supervised learning framework", "h2", COL["ink"])
draw_text(64, 640, "The key idea: only the training set teaches the model. Validation and test data check whether it generalizes.", "body", COL["muted"])

steps = [
    ("1", "Labeled Data", "Historical examples with inputs and known outcomes.", COL["soft_blue"]),
    ("2", "Train Set", "Model learns patterns and adjusts its internal rules.", COL["soft_green"]),
    ("3", "Validation Set", "Compare model versions and tune choices.", COL["soft_orange"]),
    ("4", "Test Set", "Final evaluation on held-out data.", "#f4f3ff"),
]

x0, y0, w, h, gap = 64, 694, 330, 136, 48
for i, (num, title, desc, fill) in enumerate(steps):
    x = x0 + i * (w + gap)
    card(x, y0, w, h, fill, 18)
    d.ellipse((x + 24, y0 + 28, x + 64, y0 + 68), fill=COL["green"])
    draw_text(x + 44, y0 + 48, num, "body_bold", "#ffffff", "mm")
    draw_text(x + 82, y0 + 30, title, "h3", COL["ink"])
    draw_wrapped(x + 82, y0 + 62, desc, "small", COL["text"], 210, 5)
    if i < len(steps) - 1:
        arrow(x + w + 12, y0 + 68, x + w + gap - 12, y0 + 68)

draw_text(1230, 852, "Train -> validate -> test", "small", COL["muted"])


img.save(OUT, quality=95)
print(OUT.resolve())
