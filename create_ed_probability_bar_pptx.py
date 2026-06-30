from datetime import datetime, timezone
from html import escape
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


OUT = Path("prism_ed_outcome_probability_bar_graph_editable.pptx")
SLIDE_W = 12192000
SLIDE_H = 6858000

BG = "016101"
GREEN = "2BBC2B"
LIME = "B4EA54"
PALE = "E9FFE1"
WHITE = "FFFFFF"
INK = "003900"
GRID = "B2E0A6"

shape_id = 1
shape_xml = []


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


def next_id():
    global shape_id
    shape_id += 1
    return shape_id


def fill_xml(fill, alpha=1):
    if fill is None:
        return "<a:noFill/>"
    alpha_xml = "" if alpha >= 1 else f'<a:alpha val="{int(alpha * 100000)}"/>'
    return f'<a:solidFill><a:srgbClr val="{fill}">{alpha_xml}</a:srgbClr></a:solidFill>'


def line_xml(stroke, width=1.5, alpha=1):
    if stroke is None or width <= 0:
        return "<a:ln><a:noFill/></a:ln>"
    alpha_xml = "" if alpha >= 1 else f'<a:alpha val="{int(alpha * 100000)}"/>'
    return (
        f'<a:ln w="{line_w(width)}">'
        f'<a:solidFill><a:srgbClr val="{stroke}">{alpha_xml}</a:srgbClr></a:solidFill>'
        "</a:ln>"
    )


def body_xml(text, size=12, fill=WHITE, bold=False, align="ctr", valign="ctr"):
    bold_xml = ' b="1"' if bold else ""
    paras = []
    for line in str(text).split("\n"):
        paras.append(
            f'<a:p><a:pPr algn="{align}"/>'
            f'<a:r><a:rPr lang="en-US" sz="{pt(size)}"{bold_xml} dirty="0">'
            f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>'
            '<a:latin typeface="Arial"/><a:cs typeface="Arial"/></a:rPr>'
            f"<a:t>{escape(line)}</a:t></a:r></a:p>"
        )
    return (
        f'<p:txBody><a:bodyPr wrap="square" anchor="{valign}" '
        'lIns="76200" tIns="45720" rIns="76200" bIns="45720"/>'
        f"<a:lstStyle/>{''.join(paras)}</p:txBody>"
    )


def add_shape(name, x, y, w, h, fill, stroke=None, stroke_width=1.5, geom="rect", alpha=1):
    sid = next_id()
    shape_xml.append(
        f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="{escape(name)}"/>'
        "<p:cNvSpPr/><p:nvPr/></p:nvSpPr>"
        f'<p:spPr><a:xfrm><a:off x="{emu_x(x)}" y="{emu_y(y)}"/>'
        f'<a:ext cx="{emu_w(w)}" cy="{emu_h(h)}"/></a:xfrm>'
        f'<a:prstGeom prst="{geom}"><a:avLst/></a:prstGeom>'
        f"{fill_xml(fill, alpha)}{line_xml(stroke, stroke_width)}</p:spPr></p:sp>"
    )


def add_text(name, text, x, y, w, h, size=12, fill=WHITE, bold=False, align="ctr", rot=None):
    sid = next_id()
    rot_xml = "" if rot is None else f' rot="{int(rot * 60000)}"'
    shape_xml.append(
        f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="{escape(name)}"/>'
        '<p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
        f'<p:spPr><a:xfrm{rot_xml}><a:off x="{emu_x(x)}" y="{emu_y(y)}"/>'
        f'<a:ext cx="{emu_w(w)}" cy="{emu_h(h)}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr>'
        f"{body_xml(text, size, fill, bold, align)}</p:sp>"
    )


def add_line(name, x1, y1, x2, y2, stroke=WHITE, width=1.5, arrow=False, alpha=1):
    sid = next_id()
    ox = min(emu_x(x1), emu_x(x2))
    oy = min(emu_y(y1), emu_y(y2))
    cx = max(abs(emu_x(x2) - emu_x(x1)), 1)
    cy = max(abs(emu_y(y2) - emu_y(y1)), 1)
    flip_h = ' flipH="1"' if x2 < x1 else ""
    flip_v = ' flipV="1"' if y2 < y1 else ""
    head = '<a:headEnd type="triangle"/>' if arrow else ""
    alpha_xml = "" if alpha >= 1 else f'<a:alpha val="{int(alpha * 100000)}"/>'
    shape_xml.append(
        f'<p:cxnSp><p:nvCxnSpPr><p:cNvPr id="{sid}" name="{escape(name)}"/>'
        "<p:cNvCxnSpPr/><p:nvPr/></p:nvCxnSpPr>"
        f'<p:spPr><a:xfrm{flip_h}{flip_v}><a:off x="{ox}" y="{oy}"/>'
        f'<a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        '<a:prstGeom prst="line"><a:avLst/></a:prstGeom>'
        f'<a:ln w="{line_w(width)}"><a:solidFill><a:srgbClr val="{stroke}">{alpha_xml}</a:srgbClr></a:solidFill>{head}</a:ln>'
        "</p:spPr></p:cxnSp>"
    )


def make_theme():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Acentra">
  <a:themeElements>
    <a:clrScheme name="Acentra">
      <a:dk1><a:srgbClr val="003900"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1>
      <a:dk2><a:srgbClr val="016101"/></a:dk2><a:lt2><a:srgbClr val="E9FFE1"/></a:lt2>
      <a:accent1><a:srgbClr val="2BBC2B"/></a:accent1><a:accent2><a:srgbClr val="B4EA54"/></a:accent2>
      <a:accent3><a:srgbClr val="B2E0A6"/></a:accent3><a:accent4><a:srgbClr val="F7FFF3"/></a:accent4>
      <a:accent5><a:srgbClr val="08720A"/></a:accent5><a:accent6><a:srgbClr val="006A08"/></a:accent6>
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


def build_slide():
    add_shape("background", 0, 0, 16, 9, BG)
    add_text("title", "Estimated ED Outcome Probability", 2.34, 0.50, 8.2, 0.58, 30, WHITE, True, "l")
    add_text("subtitle", "Illustrative treatment comparison for the same target member", 2.35, 1.04, 7.5, 0.34, 17, PALE, False, "l")

    left, right = 2.55, 14.25
    top, bottom = 1.70, 7.18
    max_y = 0.42
    tick_values = [0, 0.10, 0.20, 0.30, 0.40]

    def y_pos(value):
        return bottom - (value / max_y) * (bottom - top)

    for value in tick_values:
        y = y_pos(value)
        add_line(f"grid {value}", left, y, right, y, GRID, 0.9, False, 0.35)
        add_line(f"tick {value}", left - 0.10, y, left, y, WHITE, 1.2)
        add_text(f"tick label {value}", f"{int(value * 100)}%", left - 0.68, y - 0.11, 0.50, 0.22, 14, WHITE, False, "r")

    add_line("y axis", left, top, left, bottom, WHITE, 2.0)
    add_line("x axis", left, bottom, right, bottom, WHITE, 2.0)
    add_text("y axis label", "ED Outcome Probability", 1.07, 3.05, 0.52, 2.65, 20, WHITE, True, "ctr", rot=270)

    values = [("If not treated", 0.34, LIME, INK, 4.65), ("If treated", 0.18, GREEN, WHITE, 9.65)]
    bar_w = 2.45
    for label, value, fill, text_fill, cx in values:
        y = y_pos(value)
        h = bottom - y
        add_shape(f"bar {label}", cx - bar_w / 2, y, bar_w, h, fill, WHITE, 1.6)
        add_text(f"{label} value label", f"{int(value * 100)}%", cx - 0.55, y - 0.54, 1.10, 0.36, 22, WHITE, True)
        add_text(f"{label} inside label", label, cx - 0.92, y + h * 0.50 - 0.18, 1.84, 0.36, 16, text_fill, True)
        add_text(f"{label} x label", label, cx - 1.15, bottom + 0.22, 2.30, 0.34, 18, WHITE, True)

    y_not = y_pos(0.34)
    y_treated = y_pos(0.18)
    add_line("benefit arrow", 4.65, y_not - 0.16, 9.65, y_treated - 0.16, WHITE, 2.2, True)
    add_text(
        "benefit label",
        "Estimated benefit: 16 percentage-point reduction",
        5.70,
        2.95,
        5.20,
        0.34,
        16,
        PALE,
        True,
        "l",
    )


def package():
    build_slide()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    slide = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>{''.join(shape_xml)}</p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>"""
    files = {
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
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>PRISM ED Outcome Probability Bar Graph</dc:title><dc:creator>Codex</dc:creator><cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>""",
        "ppt/presentation.xml": f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst><p:sldIdLst><p:sldId id="256" r:id="rId2"/></p:sldIdLst>
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
  <p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst><p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>
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
    with ZipFile(OUT, "w", ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)


if __name__ == "__main__":
    package()
    print(f"Created {OUT.resolve()}")
