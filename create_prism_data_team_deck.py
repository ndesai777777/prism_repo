from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from html import escape


OUT = Path("PRISM_Data_Team_Presentation_Draft.pptx")

SLIDE_W = 12192000
SLIDE_H = 6858000


def emu(inches: float) -> int:
    return int(inches * 914400)


def color(hex_color: str) -> str:
    return hex_color.replace("#", "").upper()


THEME = {
    "navy": "172033",
    "teal": "0F766E",
    "mint": "7DD3C7",
    "gold": "E2A72E",
    "red": "C2413C",
    "ink": "1F2937",
    "muted": "667085",
    "pale": "EEF7F5",
    "light": "F7FAFC",
    "white": "FFFFFF",
}


def tx_body(text, font_size=20, bold=False, font_color="1F2937", align="l"):
    safe = escape(text)
    b = '<a:b/>' if bold else ''
    return f"""
      <p:txBody>
        <a:bodyPr wrap="square" rtlCol="0"/>
        <a:lstStyle/>
        <a:p>
          <a:pPr algn="{align}"/>
          <a:r>
            <a:rPr lang="en-US" sz="{font_size * 100}" dirty="0">{b}<a:solidFill><a:srgbClr val="{font_color}"/></a:solidFill><a:latin typeface="Aptos"/></a:rPr>
            <a:t>{safe}</a:t>
          </a:r>
          <a:endParaRPr lang="en-US" sz="{font_size * 100}" dirty="0"/>
        </a:p>
      </p:txBody>
    """


def bullet_body(items, font_size=20, font_color="1F2937"):
    paras = []
    for item in items:
        paras.append(f"""
        <a:p>
          <a:pPr marL="342900" indent="-228600">
            <a:buChar char="•"/>
          </a:pPr>
          <a:r>
            <a:rPr lang="en-US" sz="{font_size * 100}" dirty="0"><a:solidFill><a:srgbClr val="{font_color}"/></a:solidFill><a:latin typeface="Aptos"/></a:rPr>
            <a:t>{escape(item)}</a:t>
          </a:r>
        </a:p>
        """)
    return f"""
      <p:txBody>
        <a:bodyPr wrap="square" rtlCol="0"/>
        <a:lstStyle/>
        {''.join(paras)}
      </p:txBody>
    """


def shape(idx, x, y, w, h, fill="FFFFFF", line="FFFFFF", radius=False, text=None, font_size=20, bold=False, font_color="1F2937", align="l"):
    geom = "roundRect" if radius else "rect"
    tx = tx_body(text, font_size, bold, font_color, align) if text is not None else "<p:txBody><a:bodyPr/><a:lstStyle/><a:p/></p:txBody>"
    return f"""
    <p:sp>
      <p:nvSpPr><p:cNvPr id="{idx}" name="Shape {idx}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
      <p:spPr>
        <a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm>
        <a:prstGeom prst="{geom}"><a:avLst/></a:prstGeom>
        <a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>
        <a:ln w="12700"><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>
      </p:spPr>
      {tx}
    </p:sp>
    """


def textbox(idx, x, y, w, h, text, font_size=20, bold=False, font_color="1F2937", align="l"):
    return shape(idx, x, y, w, h, fill="FFFFFF", line="FFFFFF", text=text, font_size=font_size, bold=bold, font_color=font_color, align=align)


def line(idx, x1, y1, x2, y2, line_color="0F766E", width=25400, arrow=False):
    arrow_xml = '<a:tailEnd type="triangle"/>' if arrow else ''
    return f"""
    <p:cxnSp>
      <p:nvCxnSpPr><p:cNvPr id="{idx}" name="Connector {idx}"/><p:cNvCxnSpPr/><p:nvPr/></p:nvCxnSpPr>
      <p:spPr>
        <a:xfrm><a:off x="{min(x1, x2)}" y="{min(y1, y2)}"/><a:ext cx="{abs(x2-x1)}" cy="{abs(y2-y1)}"/></a:xfrm>
        <a:prstGeom prst="line"><a:avLst/></a:prstGeom>
        <a:ln w="{width}"><a:solidFill><a:srgbClr val="{line_color}"/></a:solidFill>{arrow_xml}</a:ln>
      </p:spPr>
    </p:cxnSp>
    """


def slide_xml(shapes):
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:bg><p:bgPr><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill><a:effectLst/></p:bgPr></p:bg>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
      {''.join(shapes)}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>"""


def title_bar(title, subtitle=None):
    items = [
        shape(10, 0, 0, SLIDE_W, emu(0.18), fill=THEME["teal"], line=THEME["teal"]),
        textbox(11, emu(0.55), emu(0.33), emu(11.1), emu(0.45), title, 26, True, THEME["navy"]),
    ]
    if subtitle:
        items.append(textbox(12, emu(0.55), emu(0.82), emu(11), emu(0.36), subtitle, 13, False, THEME["muted"]))
    return items


slides = []

# Slide 1
s = title_bar("PRISM: From Risk Targeting to Benefit Targeting", "Goal: identify who we can help most, not only who is most at risk.")
s += [
    shape(20, emu(0.75), emu(1.65), emu(4.8), emu(1.75), fill="F8FAFC", line="D0D5DD", radius=True, text="Old approach\nHigh predicted risk\nTarget for program", font_size=22, bold=True, font_color=THEME["ink"], align="ctr"),
    line(21, emu(5.7), emu(2.52), emu(6.85), emu(2.52), THEME["teal"], 38100, True),
    shape(22, emu(7.05), emu(1.65), emu(4.8), emu(1.75), fill=THEME["pale"], line=THEME["mint"], radius=True, text="New PRISM approach\nHigh expected benefit\nPrioritize for program", font_size=22, bold=True, font_color=THEME["navy"], align="ctr"),
    textbox(23, emu(1.0), emu(4.25), emu(10.8), emu(0.95), "A member can be high-risk but not very responsive to an intervention. PRISM focuses on members whose outcomes are most likely to improve if they receive support.", 22, False, THEME["ink"], "ctr"),
    textbox(24, emu(0.72), emu(6.75), emu(10.8), emu(0.25), "Speaker note: Open with the shift in decision logic. This is the central idea the rest of the deck supports.", 9, False, THEME["muted"]),
]
slides.append(slide_xml(s))

# Slide 2
s = title_bar("Outcome: ED Outcome Within 90 Days", "A practical measure of program impact for both member health and resource planning.")
s += [
    shape(20, emu(0.75), emu(1.55), emu(4.8), emu(1.2), fill=THEME["navy"], line=THEME["navy"], radius=True, text="Dependent variable\ned_outcome_90_days", font_size=24, bold=True, font_color=THEME["white"], align="ctr"),
    shape(21, emu(6.25), emu(1.55), emu(4.8), emu(1.2), fill=THEME["teal"], line=THEME["teal"], radius=True, text="Treatment indicator\nProgram received: Yes / No", font_size=24, bold=True, font_color=THEME["white"], align="ctr"),
    shape(22, emu(0.95), emu(3.35), emu(3.1), emu(1.45), fill="FEF7E8", line="F2D193", radius=True, text="Cost signal\nPotentially avoidable ED utilization", font_size=19, bold=True, font_color=THEME["ink"], align="ctr"),
    shape(23, emu(4.55), emu(3.35), emu(3.1), emu(1.45), fill=THEME["pale"], line=THEME["mint"], radius=True, text="Clinical signal\nBetter member outcomes", font_size=19, bold=True, font_color=THEME["ink"], align="ctr"),
    shape(24, emu(8.15), emu(3.35), emu(3.1), emu(1.45), fill="F3F4F6", line="D0D5DD", radius=True, text="Planning signal\nBudget and staffing decisions", font_size=19, bold=True, font_color=THEME["ink"], align="ctr"),
    textbox(25, emu(1.0), emu(5.45), emu(10.5), emu(0.65), "Because the outcome is measurable after program exposure, it lets us estimate whether intervention changes outcomes, not just whether a member was risky.", 21, False, THEME["ink"], "ctr"),
    textbox(26, emu(0.72), emu(6.75), emu(10.8), emu(0.25), "Speaker note: Be explicit that if 1 = ED outcome occurred, benefit means reducing the probability of that outcome.", 9, False, THEME["muted"]),
]
slides.append(slide_xml(s))

# Slide 3
s = title_bar("Data Structure: One Row Per Member", "Each row combines pre-program features, program participation, and the 90-day ED outcome.")
s += [
    shape(20, emu(0.65), emu(1.45), emu(11.0), emu(0.48), fill=THEME["navy"], line=THEME["navy"], text="Member features                         Program received?                         ED outcome in 90 days", font_size=18, bold=True, font_color=THEME["white"]),
    shape(21, emu(0.65), emu(2.0), emu(4.2), emu(2.95), fill="F8FAFC", line="D0D5DD", radius=True, text="Feature categories", font_size=20, bold=True, font_color=THEME["navy"], align="ctr"),
    textbox(22, emu(1.0), emu(2.6), emu(3.6), emu(2.1), "• Demographics\n• Clinical conditions\n• Prior utilization\n• Program/member history\n• Social or risk-related factors", 18, False, THEME["ink"]),
    shape(23, emu(5.25), emu(2.15), emu(2.25), emu(0.9), fill=THEME["pale"], line=THEME["mint"], radius=True, text="Yes", font_size=24, bold=True, font_color=THEME["teal"], align="ctr"),
    shape(24, emu(5.25), emu(3.65), emu(2.25), emu(0.9), fill="F8FAFC", line="D0D5DD", radius=True, text="No", font_size=24, bold=True, font_color=THEME["ink"], align="ctr"),
    shape(25, emu(8.3), emu(2.15), emu(2.6), emu(0.9), fill="FEF2F2", line="FECACA", radius=True, text="1 = ED outcome", font_size=21, bold=True, font_color=THEME["red"], align="ctr"),
    shape(26, emu(8.3), emu(3.65), emu(2.6), emu(0.9), fill="ECFDF3", line="ABEFC6", radius=True, text="0 = no ED outcome", font_size=21, bold=True, font_color="027A48", align="ctr"),
    textbox(27, emu(1.0), emu(5.55), emu(10.4), emu(0.55), "The model uses pre-program information to compare likely outcomes under treatment versus no treatment.", 21, False, THEME["ink"], "ctr"),
    textbox(28, emu(0.72), emu(6.75), emu(10.8), emu(0.25), "Speaker note: Replace feature category names if your final data dictionary uses different labels.", 9, False, THEME["muted"]),
]
slides.append(slide_xml(s))

# Slide 4
s = title_bar("Method: T-Learner for Individualized Benefit", "Two predictive models estimate what may happen with and without the program.")
s += [
    shape(20, emu(0.8), emu(2.4), emu(2.4), emu(1.0), fill="F8FAFC", line="D0D5DD", radius=True, text="Member\nfeatures", font_size=22, bold=True, font_color=THEME["ink"], align="ctr"),
    line(21, emu(3.35), emu(2.9), emu(4.15), emu(2.15), THEME["teal"], 25400, True),
    line(22, emu(3.35), emu(2.9), emu(4.15), emu(3.75), THEME["teal"], 25400, True),
    shape(23, emu(4.25), emu(1.45), emu(3.2), emu(1.05), fill=THEME["pale"], line=THEME["mint"], radius=True, text="Model A\nED risk if treated", font_size=21, bold=True, font_color=THEME["navy"], align="ctr"),
    shape(24, emu(4.25), emu(3.3), emu(3.2), emu(1.05), fill="F8FAFC", line="D0D5DD", radius=True, text="Model B\nED risk if untreated", font_size=21, bold=True, font_color=THEME["navy"], align="ctr"),
    line(25, emu(7.6), emu(1.98), emu(8.7), emu(2.75), THEME["teal"], 25400, True),
    line(26, emu(7.6), emu(3.85), emu(8.7), emu(3.05), THEME["teal"], 25400, True),
    shape(27, emu(8.85), emu(2.25), emu(2.7), emu(1.15), fill=THEME["navy"], line=THEME["navy"], radius=True, text="Estimated benefit\nUntreated risk − treated risk", font_size=19, bold=True, font_color=THEME["white"], align="ctr"),
    textbox(28, emu(1.1), emu(5.2), emu(10.2), emu(0.85), "If the outcome is an ED event, a larger positive difference suggests a larger expected reduction in ED risk from program participation.", 21, False, THEME["ink"], "ctr"),
    textbox(29, emu(0.72), emu(6.75), emu(10.8), emu(0.25), "Speaker note: Keep this intuitive. The goal is to explain the counterfactual comparison, not every causal assumption.", 9, False, THEME["muted"]),
]
slides.append(slide_xml(s))

# Slide 5
s = title_bar("Using Results: Prioritize and Explain", "Turn estimated benefit into practical decision support for program teams.")
s += [
    shape(20, emu(0.65), emu(1.45), emu(2.15), emu(1.05), fill=THEME["pale"], line=THEME["mint"], radius=True, text="1\nRank members by estimated benefit", font_size=17, bold=True, font_color=THEME["navy"], align="ctr"),
    line(21, emu(2.9), emu(1.98), emu(3.45), emu(1.98), THEME["teal"], 25400, True),
    shape(22, emu(3.55), emu(1.45), emu(2.15), emu(1.05), fill="F8FAFC", line="D0D5DD", radius=True, text="2\nSplit into benefit deciles", font_size=17, bold=True, font_color=THEME["navy"], align="ctr"),
    line(23, emu(5.8), emu(1.98), emu(6.35), emu(1.98), THEME["teal"], 25400, True),
    shape(24, emu(6.45), emu(1.45), emu(2.15), emu(1.05), fill="FEF7E8", line="F2D193", radius=True, text="3\nFocus on highest-benefit groups", font_size=17, bold=True, font_color=THEME["navy"], align="ctr"),
    line(25, emu(8.7), emu(1.98), emu(9.25), emu(1.98), THEME["teal"], 25400, True),
    shape(26, emu(9.35), emu(1.45), emu(2.15), emu(1.05), fill=THEME["navy"], line=THEME["navy"], radius=True, text="4\nExplain drivers with SHAP", font_size=17, bold=True, font_color=THEME["white"], align="ctr"),
    shape(27, emu(1.0), emu(3.35), emu(4.95), emu(1.4), fill="F8FAFC", line="D0D5DD", radius=True, text="SHAP explanation\nShows which features pushed an individual member's predicted benefit higher or lower.", font_size=19, bold=True, font_color=THEME["ink"], align="ctr"),
    shape(28, emu(6.35), emu(3.35), emu(4.95), emu(1.4), fill=THEME["pale"], line=THEME["mint"], radius=True, text="Dashboard vision\nRecommend who to prioritize and explain which specific features make them a strong candidate.", font_size=19, bold=True, font_color=THEME["ink"], align="ctr"),
    textbox(29, emu(1.1), emu(5.65), emu(10.0), emu(0.5), "Output should be more than a score: it should support transparent, targeted, and budget-aware program decisions.", 20, False, THEME["ink"], "ctr"),
    textbox(30, emu(0.72), emu(6.75), emu(10.8), emu(0.25), "Speaker note: Close by tying model output to action: who to help first, and why they are a good candidate.", 9, False, THEME["muted"]),
]
slides.append(slide_xml(s))


def presentation_xml():
    sld_ids = "\n".join([f'<p:sldId id="{256+i}" r:id="rId{i+1}"/>' for i in range(len(slides))])
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId{len(slides)+1}"/></p:sldMasterIdLst>
  <p:sldIdLst>{sld_ids}</p:sldIdLst>
  <p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}" type="wide"/>
  <p:notesSz cx="6858000" cy="9144000"/>
  <p:defaultTextStyle/>
</p:presentation>"""


def rels_xml():
    rels = []
    for i in range(len(slides)):
        rels.append(f'<Relationship Id="rId{i+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i+1}.xml"/>')
    rels.append(f'<Relationship Id="rId{len(slides)+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>')
    rels.append(f'<Relationship Id="rId{len(slides)+2}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>')
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{''.join(rels)}</Relationships>"""


def content_types_xml():
    overrides = [
        '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>',
        '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>',
        '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>',
        '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>',
    ]
    overrides += [f'<Override PartName="/ppt/slides/slide{i+1}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>' for i in range(len(slides))]
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  {''.join(overrides)}
</Types>"""


ROOT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
</Relationships>"""

SLIDE_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
</Relationships>"""

MASTER = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
  <p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>
</p:sldMaster>"""

MASTER_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>"""

LAYOUT = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">
  <p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>"""

LAYOUT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>"""

THEME_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="PRISM Draft">
  <a:themeElements>
    <a:clrScheme name="PRISM">
      <a:dk1><a:srgbClr val="172033"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1>
      <a:dk2><a:srgbClr val="1F2937"/></a:dk2><a:lt2><a:srgbClr val="F7FAFC"/></a:lt2>
      <a:accent1><a:srgbClr val="0F766E"/></a:accent1><a:accent2><a:srgbClr val="7DD3C7"/></a:accent2>
      <a:accent3><a:srgbClr val="E2A72E"/></a:accent3><a:accent4><a:srgbClr val="C2413C"/></a:accent4>
      <a:accent5><a:srgbClr val="667085"/></a:accent5><a:accent6><a:srgbClr val="EEF7F5"/></a:accent6>
      <a:hlink><a:srgbClr val="0F766E"/></a:hlink><a:folHlink><a:srgbClr val="172033"/></a:folHlink>
    </a:clrScheme>
    <a:fontScheme name="Aptos"><a:majorFont><a:latin typeface="Aptos"/></a:majorFont><a:minorFont><a:latin typeface="Aptos"/></a:minorFont></a:fontScheme>
    <a:fmtScheme name="Office"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme>
  </a:themeElements>
</a:theme>"""


with ZipFile(OUT, "w", ZIP_DEFLATED) as z:
    z.writestr("[Content_Types].xml", content_types_xml())
    z.writestr("_rels/.rels", ROOT_RELS)
    z.writestr("ppt/presentation.xml", presentation_xml())
    z.writestr("ppt/_rels/presentation.xml.rels", rels_xml())
    z.writestr("ppt/slideMasters/slideMaster1.xml", MASTER)
    z.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", MASTER_RELS)
    z.writestr("ppt/slideLayouts/slideLayout1.xml", LAYOUT)
    z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", LAYOUT_RELS)
    z.writestr("ppt/theme/theme1.xml", THEME_XML)
    for i, s in enumerate(slides, start=1):
        z.writestr(f"ppt/slides/slide{i}.xml", s)
        z.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", SLIDE_RELS)

print(OUT.resolve())
