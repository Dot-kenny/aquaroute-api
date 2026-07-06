"""
report_generator.py — AquaRoute 3-page siting report
======================================================
Takes the JSON that api.py's /predict endpoint returns and renders it into
the "report is what you sell" deliverable from the roadmap: a 3-page PDF a
driller with no ML background can read in two minutes and act on.

Usage (standalone):
    python report_generator.py --lat 8.49 --lon 4.55 --out report.pdf

Usage (as a library, e.g. from the API or a batch job):
    from report_generator import build_report
    build_report(prediction_json, request_json, "report.pdf")

Design notes
------------
- No unicode sub/superscripts (renders as black boxes in ReportLab's base
  fonts) — degree symbols and units are plain ASCII / Paragraph XML tags.
- Colour coding is deliberately restrained: green/amber/red only on the
  risk flag and probability bar, everything else stays in NGSA-adjacent
  navy/grey so it reads as a technical report, not a consumer app screen.
- Page 3 keeps the model's real limitations (sparse GP interpolation,
  proxy-label classifier) in view. Cutting that page to look more
  confident is exactly the mistake the roadmap's "validate before you
  scale" logic warns against — a driller who gets burned once won't pay
  for report #2.
"""

import argparse
import json
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
)

NAVY = colors.HexColor("#14203A")
SLATE = colors.HexColor("#4A5568")
LIGHT_GREY = colors.HexColor("#F2F4F7")
GREEN = colors.HexColor("#1D9E75")
AMBER = colors.HexColor("#BA7517")
RED = colors.HexColor("#C1442D")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("ReportTitle", fontName="Helvetica-Bold", fontSize=20,
                           textColor=NAVY, spaceAfter=2, leading=24))
styles.add(ParagraphStyle("ReportSubtitle", fontName="Helvetica", fontSize=10,
                           textColor=SLATE, spaceAfter=14))
styles.add(ParagraphStyle("SectionHeading", fontName="Helvetica-Bold", fontSize=13,
                           textColor=NAVY, spaceBefore=14, spaceAfter=6))
styles.add(ParagraphStyle("Body", fontName="Helvetica", fontSize=9.5,
                           textColor=colors.black, leading=13.5))
styles.add(ParagraphStyle("BodySmall", fontName="Helvetica", fontSize=8.3,
                           textColor=SLATE, leading=11.5))
styles.add(ParagraphStyle("MetricValue", fontName="Helvetica-Bold", fontSize=26,
                           textColor=NAVY, alignment=TA_CENTER))
styles.add(ParagraphStyle("MetricLabel", fontName="Helvetica", fontSize=8.5,
                           textColor=SLATE, alignment=TA_CENTER))
styles.add(ParagraphStyle("Disclaimer", fontName="Helvetica-Oblique", fontSize=7.7,
                           textColor=SLATE, leading=10.5))


def _zone_display(zone: str) -> str:
    return {
        "weathered_fractured": "Weathered / Fractured Basement",
        "transition": "Transition Zone",
        "fresh_basement": "Fresh / Competent Basement",
    }.get(zone, zone.replace("_", " ").title())


def _risk_color(risk_flag: bool):
    return RED if risk_flag else GREEN


def _probability_bar(prob: float, width=150 * mm, height=7 * mm):
    """A simple filled/unfilled two-cell table acting as a horizontal gauge."""
    filled = max(1, int(round(prob * 100)))
    empty = 100 - filled
    if prob >= 0.66:
        bar_color = GREEN
    elif prob >= 0.4:
        bar_color = AMBER
    else:
        bar_color = RED
    data = [["", ""]]
    col_widths = [width * prob if prob > 0 else 0.1, width * (1 - prob) if prob < 1 else 0.1]
    t = Table(data, colWidths=col_widths, rowHeights=height)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), bar_color),
        ("BACKGROUND", (1, 0), (1, 0), LIGHT_GREY),
        ("BOX", (0, 0), (-1, -1), 0.5, SLATE),
        ("INNERGRID", (0, 0), (-1, -1), 0, colors.white),
    ]))
    return t


def _metric_box(value: str, label: str, box_color=NAVY, value_font_size=26):
    value_style = ParagraphStyle(
        "v", parent=styles["MetricValue"], textColor=box_color,
        fontSize=value_font_size, leading=value_font_size + 3,
    )
    inner = [
        [Paragraph(value, value_style)],
        [Paragraph(label, styles["MetricLabel"])],
    ]
    t = Table(inner, colWidths=[55 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GREY),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D5DD")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, 0), 10),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
        ("TOPPADDING", (0, 1), (-1, 1), 0),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 10),
    ]))
    return t


def build_report(prediction: dict, request: dict, output_path: str,
                  operator_name: str = None, ngsa_reference: str = None):
    """
    prediction : the JSON body returned by POST /predict (api.py SiteResponse)
    request    : {"latitude": ..., "longitude": ...} — the query that produced it
    output_path: where to write the PDF
    """
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
        title="AquaRoute Siting Report",
    )
    story = []

    lat, lon = request["latitude"], request["longitude"]
    prob = prediction["aquifer_probability"]
    ci_low, ci_high = prediction["confidence_interval"]
    depth_low, depth_high = prediction["recommended_depth_range_m"]
    zone = prediction["geological_zone"]
    risk_flag = prediction["risk_flag"]
    risk_reason = prediction.get("risk_reason")
    nearest = prediction["nearest_validated_station"]
    in_coverage = prediction.get("in_pilot_coverage_area", True)
    confidence_note = prediction["model_confidence_note"]
    generated = datetime.now().strftime("%d %B %Y, %H:%M")

    # ══════════════════════════════════════ PAGE 1 — HEADLINE RESULT ═══════
    story.append(Paragraph("AquaRoute Siting Report", styles["ReportTitle"]))
    story.append(Paragraph(
        f"ML-assisted borehole siting &nbsp;|&nbsp; Basement complex pilot zone, Ilorin, Kwara State"
        f"{' &nbsp;|&nbsp; ' + ngsa_reference if ngsa_reference else ''}",
        styles["ReportSubtitle"]
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#D0D5DD")))
    story.append(Spacer(1, 10))

    loc_line = (f"<b>Location queried:</b> {lat:.5f} N, {lon:.5f} E &nbsp;&nbsp; "
                f"<b>Report generated:</b> {generated}"
                f"{'  &nbsp;&nbsp; <b>Prepared for:</b> ' + operator_name if operator_name else ''}")
    story.append(Paragraph(loc_line, styles["Body"]))
    story.append(Spacer(1, 14))

    # Headline metrics row
    metrics_row = Table(
        [[
            _metric_box(f"{prob*100:.0f}%", "AQUIFER PROBABILITY",
                        GREEN if prob >= 0.66 else (AMBER if prob >= 0.4 else RED)),
            _metric_box(f"{depth_low:.0f}-{depth_high:.0f} m", "RECOMMENDED DEPTH RANGE"),
            _metric_box(_zone_display(zone), "GEOLOGICAL ZONE", value_font_size=14),
        ]],
        colWidths=[58 * mm, 58 * mm, 58 * mm]
    )
    metrics_row.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 4),
                                      ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))
    story.append(metrics_row)
    story.append(Spacer(1, 14))

    story.append(Paragraph(
        f"Probability confidence interval: {ci_low*100:.0f}% - {ci_high*100:.0f}%", styles["Body"]))
    story.append(Spacer(1, 4))
    story.append(_probability_bar(prob))
    story.append(Spacer(1, 16))

    # Risk banner
    risk_bg = colors.HexColor("#FDEDEA") if risk_flag else colors.HexColor("#E9F7F1")
    risk_text_color = RED if risk_flag else GREEN
    risk_label = "RISK FLAG: LOW CONFIDENCE" if risk_flag else "RISK FLAG: NONE"
    risk_body = risk_reason if risk_reason else "No anomaly detected; location falls within the validated survey coverage area."
    risk_table = Table([[Paragraph(f"<b>{risk_label}</b><br/>{risk_body}",
                                    ParagraphStyle("risk", parent=styles["Body"], textColor=colors.black))]],
                        colWidths=[174 * mm])
    risk_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), risk_bg),
        ("BOX", (0, 0), (-1, -1), 0.75, risk_text_color),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(risk_table)
    story.append(Spacer(1, 16))

    story.append(Paragraph("Plain-language summary", styles["SectionHeading"]))
    zone_disp = _zone_display(zone)
    summary_text = (
        f"At the queried coordinates, the model estimates a <b>{prob*100:.0f}% probability</b> of "
        f"encountering a productive aquifer, based on interpolation from {zone_disp.lower()} "
        f"conditions inferred for this location. Historically, similar profiles in this basement "
        f"complex have been productive at depths of <b>{depth_low:.0f}-{depth_high:.0f} m</b>. "
    )
    if not in_coverage:
        summary_text += ("<b>This location is outside the current pilot survey coverage area</b> — "
                          "treat this result as indicative only, not decision-grade.")
    elif risk_flag:
        summary_text += ("The resistivity profile inferred for this location is unusual relative to "
                          "the training data, so confidence here is lower than the headline number alone suggests.")
    else:
        summary_text += "This location falls within the validated survey coverage area."
    story.append(Paragraph(summary_text, styles["Body"]))

    story.append(PageBreak())

    # ══════════════════════════════════════ PAGE 2 — SUPPORTING EVIDENCE ════
    story.append(Paragraph("Supporting Evidence", styles["ReportTitle"]))
    story.append(Paragraph("What this result is based on", styles["ReportSubtitle"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#D0D5DD")))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Nearest validated station", styles["SectionHeading"]))
    station_data = [
        ["VES ID", "Distance", "Historical result", "Deep resistivity"],
        [nearest["ves_id"], f"{nearest['distance_km']:.2f} km",
         "Aquifer" if nearest["aquifer_proxy"] == 1 else "Basement",
         f"{nearest['deep_resistivity_ohm']:.1f} ohm-m"],
    ]
    station_table = Table(station_data, colWidths=[40 * mm, 35 * mm, 45 * mm, 45 * mm])
    station_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_GREY]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D5DD")),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ]))
    story.append(station_table)
    story.append(Paragraph(
        "This is the closest station with an actual geophysical sounding on record — the one hard "
        "data point nearest your query, shown for cross-reference against the interpolated result above.",
        styles["BodySmall"]
    ))
    story.append(Spacer(1, 14))

    story.append(Paragraph("How the estimate was produced", styles["SectionHeading"]))
    story.append(Paragraph(
        "1. <b>Spatial interpolation</b> — a Gaussian Process model trained on 85 vertical electrical "
        "sounding (VES) stations across the survey area predicts the resistivity profile expected at "
        "your coordinates.<br/>"
        "2. <b>Aquifer classification</b> — a Random Forest classifier, trained on resistivity-derived "
        "proxy labels from the same 85 stations, converts that profile into an aquifer probability.<br/>"
        "3. <b>Anomaly check</b> — an Isolation Forest flags results where the interpolated profile "
        "looks unlike anything in the training data, which lowers confidence in the result even when "
        "the raw probability is high.",
        styles["Body"]
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Model confidence note", styles["SectionHeading"]))
    note_box = Table([[Paragraph(confidence_note, styles["Body"])]], colWidths=[174 * mm])
    note_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GREY),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D5DD")),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(note_box)

    story.append(PageBreak())

    # ══════════════════════════════════════ PAGE 3 — METHODOLOGY & LIMITS ══
    story.append(Paragraph("Methodology & Limitations", styles["ReportTitle"]))
    story.append(Paragraph("Read this page before drilling", styles["ReportSubtitle"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#D0D5DD")))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Data source", styles["SectionHeading"]))
    story.append(Paragraph(
        "Geo-resistivity dataset for groundwater aquifer exploration in the Basement Complex terrain "
        "of North-Central Nigeria — 85 VES stations, 20 electrode-spacing (AB/2) depth readings per "
        "station, collected and processed as part of the Afara Fellowship geophysics project.",
        styles["Body"]
    ))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Known limitations", styles["SectionHeading"]))
    story.append(Paragraph(
        "<b>Proxy labels, not ground truth.</b> \"Aquifer\" and \"basement\" labels used to train the "
        "classifier are derived from a resistivity threshold on the deepest sounding readings, not from "
        "actual drilling outcomes. The model has not yet been validated against real borehole "
        "success/failure data.<br/><br/>"
        "<b>Sparse spatial coverage.</b> Interpolation between survey stations is based on 85 points "
        "across a geologically heterogeneous basement complex. Confidence intervals widen with distance "
        "from the nearest surveyed station and should be treated accordingly.<br/><br/>"
        "<b>Coverage area.</b> Predictions are only meaningful within and near the original survey "
        "footprint (Ilorin, Kwara State). Locations outside this area are flagged automatically.",
        styles["Body"]
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Validation roadmap", styles["SectionHeading"]))
    story.append(Paragraph(
        "This model is in active validation. Drillers who share actual outcomes (depth reached, "
        "success/failure) at reported locations directly improve the underlying model for every future "
        "report. Reports issued during this phase should be treated as a prioritization tool that "
        "reduces blind-drilling risk, not as a guarantee of water.",
        styles["Body"]
    ))
    story.append(Spacer(1, 16))

    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#D0D5DD")))
    story.append(Spacer(1, 6))
    disclaimer = (
        "This report is a decision-support tool generated by an automated statistical model and does "
        "not constitute a geophysical certification or a guarantee of groundwater presence, yield, or "
        "quality at the specified location. Final siting and drilling decisions should incorporate "
        "site-specific judgement and, where feasible, on-site geophysical verification. "
        "AquaRoute / Afara Fellowship geophysics project. Not affiliated with, and not an official "
        "product of, the Nigerian Geological Survey Agency (NGSA)."
    )
    story.append(Paragraph(disclaimer, styles["Disclaimer"]))

    doc.build(story)
    return output_path


def _cli():
    parser = argparse.ArgumentParser(description="Generate an AquaRoute siting report PDF.")
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--out", type=str, default="aquaroute_report.pdf")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000",
                         help="Base URL of the running AquaRoute API")
    parser.add_argument("--operator", type=str, default=None, help="Driller/operator name for the cover line")
    args = parser.parse_args()

    import urllib.request

    payload = json.dumps({"latitude": args.lat, "longitude": args.lon}).encode()
    req = urllib.request.Request(
        f"{args.api_url}/predict", data=payload,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        prediction = json.loads(resp.read())

    build_report(prediction, {"latitude": args.lat, "longitude": args.lon}, args.out,
                 operator_name=args.operator)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    _cli()
