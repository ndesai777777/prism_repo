from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math


OUT = Path("ml_training_weight_update_cycle.png")
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
    "title": load_font(40, True),
    "subtitle": load_font(21),
    "h2": load_font(24, True),
    "h3": load_font(18, True),
    "body": load_font(17),
    "body_bold": load_font(17, True),
    "small": load_font(14),
    "tiny": load_font(12),
    "logo": load_font(42, True),
    "logo_small": load_font(14, True),
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
    "soft_purple": "#f4f3ff",
    "blue": "#2563eb",
    "orange": "#f97316",
    "purple": "#6d28d9",
    "red": "#dc2626",
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


def draw_wrapped(x, y, value, style="body", fill=None, max_width=360, line_gap=6):
    for line in wrap(value, style, max_width):
        draw_text(x, y, line, style, fill)
        y += FONT[style].size + line_gap
    return y


def rr(box, fill, outline=COL["border"], width=1, radius=18):
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def card(x, y, w, h, fill="#ffffff", radius=20):
    rr((x + 5, y + 8, x + w + 5, y + h + 8), COL["shadow"], COL["shadow"], 1, radius)
    rr((x, y, x + w, y + h), fill, COL["border"], 1, radius)


def logo(x, y):
    d.rectangle((x, y, x + 260, y + 104), fill=COL["green"])
    draw_text(x + 28, y + 22, "Acentra", "logo", "#ffffff")
    draw_text(x + 92, y + 74, "H E A L T H", "logo_small", "#ffffff")


def arrow_between(a, b, color=COL["green"]):
    x1, y1 = a
    x2, y2 = b
    dx, dy = x2 - x1, y2 - y1
    length = max(1, math.hypot(dx, dy))
    ux, uy = dx / length, dy / length
    start = (x1 + ux * 95, y1 + uy * 75)
    end = (x2 - ux * 95, y2 - uy * 75)
    d.line((start[0], start[1], end[0], end[1]), fill=color, width=4)
    angle = math.atan2(dy, dx)
    size = 14
    p1 = end
    p2 = (end[0] - size * math.cos(angle - 0.45), end[1] - size * math.sin(angle - 0.45))
    p3 = (end[0] - size * math.cos(angle + 0.45), end[1] - size * math.sin(angle + 0.45))
    d.polygon([p1, p2, p3], fill=color)


draw_text(64, 82, "How a Model Learns During Training", "title", COL["ink"])
draw_text(
    64,
    135,
    "A model learns by making a guess, checking how wrong it was, and improving a little each time.",
    "subtitle",
    COL["muted"],
)


# Cycle cards
center = (640, 512)
positions = [
    (640, 344),
    (880, 482),
    (800, 692),
    (480, 692),
    (400, 482),
]
steps = [
    ("1", "Start with guesses", "The model starts with rough internal settings, usually not very accurate yet.", COL["soft_blue"], COL["blue"]),
    ("2", "Make a prediction", "It uses those settings to make a guess for one example or a small group of examples.", COL["soft_green"], COL["green"]),
    ("3", "Check the error", "It compares the guess to the real answer. Bigger mistakes mean more error.", COL["soft_orange"], COL["orange"]),
    ("4", "Find what to fix", "Backpropagation helps identify which internal settings contributed most to the mistake.", COL["soft_purple"], COL["purple"]),
    ("5", "Improve a little", "Gradient descent slightly adjusts the settings so the next guesses should be better.", COL["soft_green"], COL["green"]),
]

for i in range(len(positions)):
    arrow_between(positions[i], positions[(i + 1) % len(positions)])

for (x, y), (num, title, desc, fill, accent) in zip(positions, steps):
    w, h = 240, 132
    card(x - w // 2, y - h // 2, w, h, fill, 18)
    d.ellipse((x - 106, y - 46, x - 68, y - 8), fill=accent)
    draw_text(x - 87, y - 27, num, "body_bold", "#ffffff", "mm")
    draw_text(x - 54, y - 48, title, "h3", COL["ink"])
    draw_wrapped(x - 54, y - 20, desc, "small", COL["text"], 150, 4)


# Center explanation
rr((530, 444, 750, 610), "#ffffff", COL["border"], 1, 18)
draw_text(640, 476, "Repeat many times", "h3", COL["green"], "mm")
draw_wrapped(
    552,
    508,
    "With enough examples, the model learns patterns and gets better.",
    "body",
    COL["text"],
    176,
    6,
)


# Side example
card(1060, 300, 444, 450, "#ffffff", 20)
draw_text(1092, 334, "Example: Predicting Music Taste", "h2", COL["ink"])
draw_wrapped(
    1092,
    370,
    "Suppose we use demographic information to predict which music genres someone may like.",
    "body",
    COL["muted"],
    370,
    6,
)

example_y = 438
example_steps = [
    ("Inputs", "Age 28, lives in the South, bilingual household, urban area"),
    ("Model guess", "Country and pop"),
    ("Real answer", "Pop and Latin music"),
    ("Error", "The guess was partly wrong"),
    ("Update", "Adjust the model so similar profiles lean more toward pop and Latin next time"),
]
for label, value in example_steps:
    rr((1092, example_y, 1198, example_y + 30), COL["soft_green"], "#a7f3d0", 1, 15)
    draw_text(1145, example_y + 15, label, "tiny", COL["green"], "mm")
    draw_wrapped(1216, example_y + 2, value, "small", COL["text"], 250, 4)
    example_y += 55

rr((1092, 708, 1472, 734), COL["soft_orange"], "#fed7aa", 1, 10)
draw_text(1110, 715, "Toy example: avoid stereotypes; use behavior when appropriate.", "tiny", COL["orange"])


# Bottom note
card(96, 768, 1408, 104, "#f8fafc", 16)
draw_text(128, 792, "Important nuance", "h3", COL["ink"])
draw_wrapped(
    128,
    824,
    "This is the basic idea behind neural networks and many modern machine learning models. Other models may learn differently, but the overall goal is the same: learn from examples and perform well on new data the model has not seen before.",
    "small",
    COL["muted"],
    1310,
    5,
)


img.save(OUT, quality=95)
print(OUT.resolve())
