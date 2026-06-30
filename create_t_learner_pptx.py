from datetime import datetime, timezone
from html import escape
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


OUT = Path("prism_t_learner_framework_editable.pptx")
SLIDE_W = 12192000
SLIDE_H = 6858000


def emu_x(x):
    return int(round(x / 16 * SLIDE_W))


def emu_y(y):
    return int(round(y / 9 * SLIDE_H))


def emu_w(w):
    return int(round(w / 16 * SLIDE_W))


def emu_h(h):
    return int(round(h / 9 * SLIDE_H))


def pt(value):
    return str(int(round(value * 100)))


def line_w(value):
    return str(int(round(value * 12700)))


def color(value):
    return value.replace("#", "").upper()


BG = "016101"
GREEN = "2BBC2B"
DARK = "006A08"
LIME = "B4EA54"
PALE = "E9FFE1"
PALE2 = "D8FFC5"
PANEL = "F7FFF3"
WHITE = "FFFFFF"
INK = "003900"
FOOTER = "08720A"

shape_id = 1
shape_xml = []


def next_id():
    global shape_id
    shape_id += 1
    return shape_id


def fill_xml(fill, alpha=1):
    if fill is None:
        return "<a:noFill/>"
    alpha_xml = ""
    if alpha < 1:
        alpha_xml = f'<a:alpha val="{int(alpha * 100000)}"/>'
    return f'<a:solidFill><a:srgbClr val="{color(fill)}">{alpha_xml}</a:srgbClr></a:solidFill>'


def line_xml(stroke, width=1.5, alpha=1):
    if stroke is None or width <= 0:
        return "<a:ln><a:noFill/></a:ln>"
    alpha_xml = ""
    if alpha < 1:
        alpha_xml = f'<a:alpha val="{int(alpha * 100000)}"/>'
    return (
        f'<a:ln w="{line_w(width)}">'
        f'<a:solidFill><a:srgbClr val="{color(stroke)}">{alpha_xml}</a:srgbClr></a:solidFill>'
        "</a:ln>"
    )


def text_body(text, font_size=12, fill=INK, bold=False, align="ctr", valign="ctr"):
    bold_xml = ' b="1"' if bold else ""
    paras = []
    for line in str(text).split("\n"):
        paras.append(
            f'<a:p><a:pPr algn="{align}"/>'
            f'<a:r><a:rPr lang="en-US" sz="{pt(font_size)}"{bold_xml} dirty="0">'
            f'<a:solidFill><a:srgbClr val="{color(fill)}"/></a:solidFill>'
            '<a:latin typeface="Arial"/><a:cs typeface="Arial"/></a:rPr>'
            f"<a:t>{escape(line)}</a:t></a:r></a:p>"
        )
    return (
        f'<p:txBody><a:bodyPr wrap="square" anchor="{valign}" '
        'lIns="76200" tIns="45720" rIns="76200" bIns="45720"/>'
        f"<a:lstStyle/>{''.join(paras)}</p:txBody>"
    )


def add_shape(name, x, y, w, h, fill, stroke=None, stroke_width=1.5, geom="roundRect", alpha=1):
    sid = next_id()
    shape_xml.append(
        f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="{escape(name)}"/>'
        "<p:cNvSpPr/><p:nvPr/></p:nvSpPr>"
        f"<p:spPr><a:xfrm><a:off x=\"{emu_x(x)}\" y=\"{emu_y(y)}\"/>"
        f"<a:ext cx=\"{emu_w(w)}\" cy=\"{emu_h(h)}\"/></a:xfrm>"
        f'<a:prstGeom prst="{geom}"><a:avLst/></a:prstGeom>'
        f"{fill_xml(fill, alpha)}{line_xml(stroke, stroke_width)}</p:spPr>"
        "</p:sp>"
    )


def add_text(name, text, x, y, w, h, font_size=12, fill=WHITE, bold=False, align="ctr", valign="ctr"):
    sid = next_id()
    shape_xml.append(
        f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="{escape(name)}"/>'
        "<p:cNvSpPr txBox=\"1\"/><p:nvPr/></p:nvSpPr>"
        f"<p:spPr><a:xfrm><a:off x=\"{emu_x(x)}\" y=\"{emu_y(y)}\"/>"
        f"<a:ext cx=\"{emu_w(w)}\" cy=\"{emu_h(h)}\"/></a:xfrm>"
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr>'
        f"{text_body(text, font_size, fill, bold, align, valign)}</p:sp>"
    )


def add_box_text(name, text, x, y, w, h, fill, stroke, font_size=12, text_fill=WHITE, bold=True, geom="roundRect"):
    sid = next_id()
    shape_xml.append(
        f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="{escape(name)}"/>'
        "<p:cNvSpPr/><p:nvPr/></p:nvSpPr>"
        f"<p:spPr><a:xfrm><a:off x=\"{emu_x(x)}\" y=\"{emu_y(y)}\"/>"
        f"<a:ext cx=\"{emu_w(w)}\" cy=\"{emu_h(h)}\"/></a:xfrm>"
        f'<a:prstGeom prst="{geom}"><a:avLst/></a:prstGeom>'
        f"{fill_xml(fill)}{line_xml(stroke, 1.6)}</p:spPr>"
        f"{text_body(text, font_size, text_fill, bold)}</p:sp>"
    )


def add_arrow(name, x1, y1, x2, y2, stroke=LIME, width=2.6):
    sid = next_id()
    ox = min(emu_x(x1), emu_x(x2))
    oy = min(emu_y(y1), emu_y(y2))
    cx = max(abs(emu_x(x2) - emu_x(x1)), 1)
    cy = max(abs(emu_y(y2) - emu_y(y1)), 1)
    flip_h = ' flipH="1"' if x2 < x1 else ""
    flip_v = ' flipV="1"' if y2 < y1 else ""
    shape_xml.append(
        f'<p:cxnSp><p:nvCxnSpPr><p:cNvPr id="{sid}" name="{escape(name)}"/>'
        "<p:cNvCxnSpPr/><p:nvPr/></p:nvCxnSpPr>"
        f"<p:spPr><a:xfrm{flip_h}{flip_v}><a:off x=\"{ox}\" y=\"{oy}\"/>"
        f"<a:ext cx=\"{cx}\" cy=\"{cy}\"/></a:xfrm>"
        '<a:prstGeom prst="line"><a:avLst/></a:prstGeom>'
        f'<a:ln w="{line_w(width)}"><a:solidFill><a:srgbClr val="{color(stroke)}"/></a:solidFill>'
        '<a:headEnd type="triangle"/></a:ln></p:spPr></p:cxnSp>'
    )


def add_circle(name, cx, cy, r, fill, stroke=None, stroke_width=0.8):
    add_shape(name, cx - r, cy - r, r * 2, r * 2, fill, stroke, stroke_width, geom="ellipse")


def add_dots():
    x0, y0 = 1.66, 3.78
    spacing = 0.19
    for row in range(4):
        for col in range(8):
            fill = GREEN if col < 4 else LIME
            add_circle("sample training row", x0 + col * spacing, y0 - row * spacing, 0.045, fill)


def build_slide():
    add_shape("background", 0, 0, 16, 9, BG, None, 0, geom="rect")
    add_box_text("section tag", "T-LEARNER", 0.72, 7.66, 2.3, 0.36, GREEN, GREEN, 10.5, WHITE)
    add_text("title", "How the T-Learner Framework Works", 0.72, 7.21, 8.3, 0.48, 28, WHITE, True, "l")
    add_text(
        "subtitle",
        "The training dataset has 700 total member rows. It is split by actual program status so two outcome models can learn separately.",
        0.72,
        6.86,
        12.6,
        0.34,
        13.5,
        PALE,
        False,
        "l",
    )

    add_shape("main white workspace", 0.62, 0.72, 14.76, 5.82, PANEL, LIME, 2.0, alpha=1)

    steps = [
        (1.72, "1  Training data", GREEN, WHITE),
        (4.78, "2  Split rows", LIME, BG),
        (7.74, "3  Train two models", GREEN, WHITE),
        (11.20, "4  Score same member", LIME, BG),
        (14.12, "5  Compare", GREEN, WHITE),
    ]
    for x, label, fill, txt_fill in steps:
        add_box_text(f"step {label}", label, x - 0.78, 6.24, 1.56, 0.42, fill, fill, 9.5, txt_fill)

    add_shape("training data card", 0.98, 3.48, 2.45, 2.18, PALE, LIME, 2.2)
    add_text("training data title", "Training Data", 1.15, 5.06, 2.10, 0.34, 17.5, INK, True)
    add_text("training data count", "700 total rows", 1.32, 4.76, 1.78, 0.28, 13, INK, True)
    add_text(
        "training data fields",
        "Each row includes:\nX = pre-treatment features\nT = actual program status\nY = observed 90-day ED outcome",
        1.10,
        4.06,
        2.20,
        0.62,
        10.4,
        INK,
        False,
    )
    add_dots()
    add_text("sample rows note", "dots shown = sample rows", 1.23, 3.20, 1.96, 0.20, 9.2, INK, True)

    add_arrow("training to split", 3.43, 4.56, 4.25, 4.56)
    add_box_text("split actual treatment", "Split by\nactual T", 4.25, 4.05, 1.45, 1.02, DARK, LIME, 13.5, WHITE)

    add_box_text("treated label", "T = 1", 5.96, 5.02, 0.82, 0.34, GREEN, GREEN, 9.4, WHITE)
    add_box_text("untreated label", "T = 0", 5.96, 3.74, 0.82, 0.34, WHITE, GREEN, 9.4, INK)
    add_arrow("split to treated label", 5.70, 4.78, 6.08, 5.04)
    add_arrow("split to untreated label", 5.70, 4.34, 6.08, 3.96)

    add_shape("model a card", 6.78, 4.76, 2.66, 1.06, PALE2, GREEN, 2.0)
    add_text("model a title", "Model A: Treated", 6.94, 5.32, 2.34, 0.30, 16.5, INK, True)
    add_text("model a body", "Trains only on treated rows\nfrom the 700-row dataset", 7.02, 4.89, 2.16, 0.38, 10.8, INK, False)

    add_shape("model b card", 6.78, 3.46, 2.66, 1.06, PALE, GREEN, 2.0)
    add_text("model b title", "Model B: Untreated", 6.92, 4.02, 2.38, 0.30, 16.2, INK, True)
    add_text("model b body", "Trains only on untreated rows\nfrom the 700-row dataset", 7.02, 3.59, 2.16, 0.38, 10.8, INK, False)
    add_arrow("treated label to model a", 6.78, 5.19, 6.08, 5.19)
    add_arrow("untreated label to model b", 6.78, 3.91, 6.08, 3.91)

    add_shape("same member card", 10.22, 4.02, 2.16, 1.48, DARK, LIME, 2.0)
    add_text("same member title", "Same target member", 10.36, 4.95, 1.88, 0.34, 15.2, WHITE, True)
    add_text("same member body", "Send the same X through\nboth trained models", 10.48, 4.50, 1.64, 0.36, 10.8, PALE, False)
    add_arrow("model a to same member", 9.44, 5.20, 10.22, 5.02)
    add_arrow("model b to same member", 9.44, 3.96, 10.22, 4.46)

    add_shape("prediction treated card", 12.82, 4.96, 2.12, 0.94, GREEN, LIME, 2.0)
    add_text("prediction treated formula", "Yhat(1)", 12.96, 5.28, 0.64, 0.34, 17.5, WHITE, True)
    add_text("prediction treated body", "ED risk\nwith program", 13.68, 5.19, 0.96, 0.42, 11.5, WHITE, True)

    add_shape("prediction untreated card", 12.82, 3.42, 2.12, 0.94, PALE, LIME, 2.0)
    add_text("prediction untreated formula", "Yhat(0)", 12.96, 3.74, 0.64, 0.34, 17.5, INK, True)
    add_text("prediction untreated body", "ED risk\nwithout program", 13.62, 3.65, 1.10, 0.42, 11.5, INK, True)
    add_arrow("same member to treated prediction", 12.38, 5.02, 12.82, 5.43)
    add_arrow("same member to untreated prediction", 12.38, 4.46, 12.82, 3.90)

    add_shape("benefit card", 12.82, 1.62, 2.12, 0.98, LIME, GREEN, 2.3)
    add_text("benefit title", "Estimated Benefit", 13.04, 2.16, 1.68, 0.28, 14.5, BG, True)
    add_text("benefit formula", "Yhat(0) - Yhat(1)", 13.05, 1.80, 1.66, 0.28, 14.8, BG, True)
    add_arrow("treated prediction to benefit", 13.88, 4.96, 13.88, 2.60)
    add_arrow("untreated prediction to benefit", 13.88, 3.42, 13.88, 2.60)

    add_shape("key idea box", 0.88, 0.92, 10.95, 0.86, FOOTER, FOOTER, 1.0)
    add_text("key idea label", "Key idea:", 1.10, 1.22, 0.72, 0.24, 12.8, WHITE, True, "l")
    add_text(
        "key idea body",
        "For the same member, estimate ED risk under both possible treatment paths. The gap is the modeled program benefit.",
        1.82,
        1.21,
        9.68,
        0.26,
        11.7,
        PALE,
        False,
        "l",
    )

    add_circle("legend treated rows", 0.98, 0.47, 0.055, GREEN)
    add_text("legend treated label", "Treated rows", 1.12, 0.39, 0.92, 0.16, 9.5, PALE, False, "l")
    add_circle("legend untreated rows", 2.18, 0.47, 0.055, LIME)
    add_text("legend untreated label", "Untreated rows", 2.32, 0.39, 1.08, 0.16, 9.5, PALE, False, "l")


def make_theme():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Acentra">
  <a:themeElements>
    <a:clrScheme name="Acentra">
      <a:dk1><a:srgbClr val="003900"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1>
      <a:dk2><a:srgbClr val="016101"/></a:dk2><a:lt2><a:srgbClr val="E9FFE1"/></a:lt2>
      <a:accent1><a:srgbClr val="2BBC2B"/></a:accent1><a:accent2><a:srgbClr val="B4EA54"/></a:accent2>
      <a:accent3><a:srgbClr val="D8FFC5"/></a:accent3><a:accent4><a:srgbClr val="006A08"/></a:accent4>
      <a:accent5><a:srgbClr val="08720A"/></a:accent5><a:accent6><a:srgbClr val="F7FFF3"/></a:accent6>
      <a:hlink><a:srgbClr val="2BBC2B"/></a:hlink><a:folHlink><a:srgbClr val="B4EA54"/></a:folHlink>
    </a:clrScheme>
    <a:fontScheme name="Arial"><a:majorFont><a:latin typeface="Arial"/></a:majorFont><a:minorFont><a:latin typeface="Arial"/></a:minorFont></a:fontScheme>
    <a:fmtScheme name="Acentra">
      <a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst>
      <a:lnStyleLst><a:ln w="6350"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln><a:ln w="12700"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln><a:ln w="19050"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst>
      <a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst>
      <a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst>
    </a:fmtScheme>
  </a:themeElements><a:objectDefaults/><a:extraClrSchemeLst/>
</a:theme>"""


def package_files():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    slide_shapes = "".join(shape_xml)
    slide = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>{slide_shapes}</p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>"""
    return {
        "[Content_Types].xml": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
  <Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
  <Override PartName="/ppt/slides/slide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>
  <Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
</Types>""",
        "_rels/.rels": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>""",
        "docProps/app.xml": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex</Application><PresentationFormat>On-screen Show (16:9)</PresentationFormat><Slides>1</Slides><Notes>0</Notes><HiddenSlides>0</HiddenSlides><ScaleCrop>false</ScaleCrop>
</Properties>""",
        "docProps/core.xml": f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>PRISM T-Learner Framework</dc:title><dc:creator>Codex</dc:creator><cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>""",
        "ppt/presentation.xml": f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>
  <p:sldIdLst><p:sldId id="256" r:id="rId2"/></p:sldIdLst>
  <p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}" type="wide"/><p:notesSz cx="6858000" cy="9144000"/>
  <p:defaultTextStyle><a:defPPr><a:defRPr lang="en-US"/></a:defPPr></p:defaultTextStyle>
</p:presentation>""",
        "ppt/_rels/presentation.xml.rels": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/>
</Relationships>""",
        "ppt/slides/slide1.xml": slide,
        "ppt/slides/_rels/slide1.xml.rels": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
</Relationships>""",
        "ppt/slideMasters/slideMaster1.xml": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
  <p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>
</p:sldMaster>""",
        "ppt/slideMasters/_rels/slideMaster1.xml.rels": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>""",
        "ppt/slideLayouts/slideLayout1.xml": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">
  <p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>""",
        "ppt/slideLayouts/_rels/slideLayout1.xml.rels": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>""",
        "ppt/theme/theme1.xml": make_theme(),
    }


def main():
    build_slide()
    files = package_files()
    with ZipFile(OUT, "w", ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    print(f"Created {OUT.resolve()}")


if __name__ == "__main__":
    main()
