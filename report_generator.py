"""
report_generator.py
Generates academic research clinical PDF diagnostic reports using ReportLab and CSV exports.
Includes patient demographics, ECG rhythm strip with annotated R-peaks,
AAMI EC57 classification breakdown, prediction confidence explanations, and academic disclaimer.
Strictly formatted for standard A4 printable layout with zero overflow.
"""

import os
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

from ml_model import AAMI_CLASSES, AAMI_FULL_NAMES

# Standard A4 Dimensions: 595.28 x 841.89 points
PAGE_WIDTH, PAGE_HEIGHT = A4
LEFT_MARGIN = 36.0
RIGHT_MARGIN = 36.0
TOP_MARGIN = 36.0
BOTTOM_MARGIN = 36.0
PRINTABLE_WIDTH = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN  # 523.28 pt

class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas that adds running headers, rules, timestamps,
    and exact 'Page X of Y' pagination to every page.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor('#64748b'))

        # Running Header
        header_y = PAGE_HEIGHT - 24.0
        self.drawString(LEFT_MARGIN, header_y, "CardioScan \u2014 Arrhythmia Detection & Classification System")
        self.drawRightString(PAGE_WIDTH - RIGHT_MARGIN, header_y, "Diagnostic Decision Support Report")
        self.setStrokeColor(colors.HexColor('#cbd5e1'))
        self.setLineWidth(0.5)
        self.line(LEFT_MARGIN, header_y - 3.0, PAGE_WIDTH - RIGHT_MARGIN, header_y - 3.0)

        # Running Footer
        footer_y = 22.0
        self.line(LEFT_MARGIN, footer_y + 9.0, PAGE_WIDTH - RIGHT_MARGIN, footer_y + 9.0)
        self.drawString(LEFT_MARGIN, footer_y, "Confidential Clinical Research Record \u2022 Generated: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(PAGE_WIDTH - RIGHT_MARGIN, footer_y, page_str)

        self.restoreState()


def generate_ecg_plot_image(ecg_signal, r_peaks, beat_classes, fs=360.0, output_path='reports/temp_ecg_plot.png', max_duration_sec=10.0):
    """
    Renders an ECG rhythm strip with highlighted R-peaks and beat classification tags.
    Aspect ratio matches the PDF insertion element (523 pt width x 170 pt height).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    sig_len = len(ecg_signal)
    total_sig_sec = sig_len / fs
    display_sec = min(total_sig_sec, max_duration_sec)
    max_samples = int(display_sec * fs)

    sub_sig = ecg_signal[:max_samples]
    time_axis = np.arange(len(sub_sig)) / fs

    r_peaks_arr = np.asarray(r_peaks, dtype=int)
    valid_mask = r_peaks_arr < max_samples
    sub_peaks = r_peaks_arr[valid_mask]
    sub_classes = [str(beat_classes[i]) for i, v in enumerate(valid_mask) if v] if (beat_classes is not None and len(beat_classes) > 0) else []

    # Aspect ratio: 10.46 in x 3.4 in (3.076:1) matches 523 pt x 170 pt
    fig, ax = plt.subplots(figsize=(10.46, 3.4), dpi=300)

    # Standard Clinical ECG Grid (Major grid 0.2s, Minor grid 0.04s)
    ax.set_facecolor('#ffffff')
    ax.set_axisbelow(True)
    
    # Minor Grid
    ax.minorticks_on()
    ax.grid(True, which='minor', color='#ffe4e6', linestyle=':', linewidth=0.4, alpha=0.7)
    # Major Grid
    ax.grid(True, which='major', color='#fecdd3', linestyle='-', linewidth=0.6, alpha=0.8)

    # Plot Filtered ECG Lead
    ax.plot(time_axis, sub_sig, color='#1d4ed8', linewidth=1.1, label='Lead I / MLII (Filtered 0.5-40 Hz)')

    # AAMI Color Mapping
    color_map = {
        'N': '#15803d',  # Green
        'S': '#b45309',  # Amber
        'V': '#b91c1c',  # Red
        'F': '#7e22ce',  # Purple
        'Q': '#475569'   # Slate
    }

    # Annotate R-peaks
    for i, p in enumerate(sub_peaks):
        cls_code = sub_classes[i] if i < len(sub_classes) else 'N'
        c = color_map.get(cls_code, '#b91c1c')
        peak_time = p / fs
        peak_amp = sub_sig[p]

        ax.plot(peak_time, peak_amp, 'o', color=c, markersize=5.5, markeredgecolor='#ffffff', markeredgewidth=0.7)
        ax.annotate(
            f"{cls_code}",
            xy=(peak_time, peak_amp),
            xytext=(0, 7),
            textcoords='offset points',
            ha='center',
            fontsize=7.5,
            fontweight='bold',
            color=c,
            bbox=dict(boxstyle='round,pad=0.18', facecolor='#ffffff', edgecolor=c, alpha=0.95, lw=0.7)
        )

    ax.set_title(f'Continuous ECG Rhythm Strip with Annotated R-Peaks (0.0 to {display_sec:.1f} seconds)', fontsize=9.5, fontweight='bold', color='#0f172a', pad=6)
    ax.set_xlabel('Time (seconds)', fontsize=8, fontweight='bold', color='#334155', labelpad=3)
    ax.set_ylabel('Amplitude (mV)', fontsize=8, fontweight='bold', color='#334155', labelpad=3)
    ax.set_xlim([0, display_sec if display_sec > 0 else 1.0])
    ax.tick_params(axis='both', which='major', labelsize=7.5, colors='#475569')

    handles = [
        plt.Line2D([0], [0], color='#15803d', marker='o', linestyle='', label='N: Normal'),
        plt.Line2D([0], [0], color='#b45309', marker='o', linestyle='', label='S: Supraventricular'),
        plt.Line2D([0], [0], color='#b91c1c', marker='o', linestyle='', label='V: Ventricular'),
        plt.Line2D([0], [0], color='#7e22ce', marker='o', linestyle='', label='F: Fusion'),
        plt.Line2D([0], [0], color='#475569', marker='o', linestyle='', label='Q: Unknown')
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=6.8, framealpha=0.92, facecolor='#ffffff', edgecolor='#cbd5e1')

    fig.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.14)
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def generate_pdf_report(analysis, patient, ecg_signal, r_peaks, beat_classes, output_pdf_path=None):
    """
    Creates a professional, print-ready A4 clinical diagnostic report in PDF format.
    Includes patient demographics, ECG rhythm strip, AAMI classification breakdown,
    model confidence explanations, and academic prototype disclaimer.
    """
    if output_pdf_path is None:
        os.makedirs('reports', exist_ok=True)
        output_pdf_path = f"reports/Arrhythmia_Report_PAT_{patient.patient_id}_{analysis.id}.pdf"

    # Render ECG Waveform Image
    img_path = f"reports/plot_analysis_{analysis.id}.png"
    generate_ecg_plot_image(
        ecg_signal, r_peaks, beat_classes,
        fs=analysis.sample_rate,
        output_path=img_path,
        max_duration_sec=max(1.0, analysis.duration_seconds)
    )

    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=A4,
        leftMargin=LEFT_MARGIN,
        rightMargin=RIGHT_MARGIN,
        topMargin=TOP_MARGIN,
        bottomMargin=BOTTOM_MARGIN
    )

    styles = getSampleStyleSheet()

    # Clean Typography Hierarchy
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#0f172a'),
        alignment=0
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#64748b'),
        alignment=0
    )
    heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor('#0f172a')
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor('#334155')
    )
    body_bold = ParagraphStyle(
        'BodyBold',
        parent=body_style,
        fontName='Helvetica-Bold',
        textColor=colors.HexColor('#0f172a')
    )
    disclaimer_style = ParagraphStyle(
        'DisclaimerText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.2,
        leading=9.5,
        textColor=colors.HexColor('#475569'),
        alignment=0
    )

    elements = []

    # 1. Document Title Header
    elements.append(Paragraph("Arrhythmia Detection & Classification Report", title_style))
    elements.append(Paragraph("Decision Support Framework &bull; PhysioNet MIT-BIH AAMI Evaluation Protocol", subtitle_style))
    elements.append(Spacer(1, 5))

    # 2. Compact Academic Research Prototype Disclaimer
    disclaimer_html = (
        "<b>ACADEMIC RESEARCH PROTOTYPE DISCLAIMER:</b> This software is an academic research prototype "
        "and is not intended for clinical diagnosis or treatment. All findings and classifications "
        "must be independently evaluated and verified by qualified healthcare professionals."
    )
    disclaimer_box = [[Paragraph(disclaimer_html, disclaimer_style)]]
    disc_table = Table(disclaimer_box, colWidths=[PRINTABLE_WIDTH])
    disc_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 0.6, colors.HexColor('#cbd5e1')),
        ('LINELEFT', (0, 0), (0, -1), 2.5, colors.HexColor('#1d4ed8')),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(disc_table)
    elements.append(Spacer(1, 5))

    # 3. Patient Demographics & Analysis Metadata Table (Sum = 523.28 pt)
    w_label = 95.0
    w_val = 166.64
    meta_data = [
        [
            Paragraph("Patient Name:", body_bold), Paragraph(patient.full_name, body_style),
            Paragraph("Analysis ID:", body_bold), Paragraph(f"ANL-{analysis.id:05d}", body_style)
        ],
        [
            Paragraph("Patient Identifier:", body_bold), Paragraph(patient.patient_id, body_style),
            Paragraph("Evaluation Date:", body_bold), Paragraph(datetime.now().strftime('%Y-%m-%d %H:%M'), body_style)
        ],
        [
            Paragraph("Age / Gender:", body_bold), Paragraph(f"{patient.age} yrs / {patient.gender}", body_style),
            Paragraph("Recording Source:", body_bold), Paragraph(analysis.filename, body_style)
        ],
        [
            Paragraph("Classifier:", body_bold), Paragraph(analysis.algorithm, body_style),
            Paragraph("Sampling Rate / Dur:", body_bold), Paragraph(f"{int(analysis.sample_rate)} Hz / {analysis.duration_seconds:.1f}s", body_style)
        ]
    ]

    meta_table = Table(meta_data, colWidths=[w_label, w_val, w_label, w_val])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 0.6, colors.HexColor('#e2e8f0')),
        ('INNERGRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 5))

    # 4. Diagnostic Finding Summary Banner
    is_normal = (analysis.overall_diagnosis == 'Normal')
    banner_bg = colors.HexColor('#f0fdf4') if is_normal else colors.HexColor('#fef2f2')
    banner_border = colors.HexColor('#bbf7d0') if is_normal else colors.HexColor('#fecaca')
    text_color = '#15803d' if is_normal else '#b91c1c'
    finding_title = "AUTOMATED FINDING: NORMAL SINUS RHYTHM" if is_normal else f"AUTOMATED FINDING: {analysis.overall_diagnosis.upper()}"

    banner_data = [
        [
            Paragraph(f"<font color='{text_color}'><b>{finding_title}</b></font>", ParagraphStyle('DiagTitle', parent=body_style, fontSize=10, leading=12, fontName='Helvetica-Bold')),
            Paragraph(f"Prediction Confidence (Estimated Posterior Probability): <b>{analysis.confidence_score*100:.1f}%</b> &bull; Heart Rate: <b>{analysis.avg_heart_rate:.1f} BPM</b><br/><font size='6.8' color='#64748b'>*Note: Model confidence is the posterior vote ratio from inverse distance weighting; uncalibrated.</font>", ParagraphStyle('DiagSub', parent=body_style, fontSize=7.5, leading=9.5, alignment=2))
        ]
    ]
    banner_table = Table(banner_data, colWidths=[240.0, 283.28])
    banner_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), banner_bg),
        ('BOX', (0, 0), (-1, -1), 0.8, banner_border),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(banner_table)
    elements.append(Spacer(1, 5))

    # 5. AAMI EC57 Beat Classification Distribution Table
    elements.append(Paragraph("AAMI EC57 Beat Classification Distribution", heading_style))
    elements.append(Spacer(1, 2))

    tot = max(1, analysis.total_beats)
    class_table_data = [
        [
            Paragraph("<b>Category</b>", body_bold),
            Paragraph("<b>Code</b>", body_bold),
            Paragraph("<b>Description</b>", body_bold),
            Paragraph("<b>Count</b>", body_bold),
            Paragraph("<b>Percentage</b>", body_bold),
            Paragraph("<b>Finding Status</b>", body_bold)
        ],
        [
            Paragraph("Normal", body_style), Paragraph("N", body_style),
            Paragraph("Normal sinus rhythm & bundle branch blocks", body_style),
            Paragraph(str(analysis.normal_beats), body_style),
            Paragraph(f"{(analysis.normal_beats / tot)*100:.1f}%", body_style),
            Paragraph("<font color='#15803d'><b>Normal</b></font>", body_style)
        ],
        [
            Paragraph("Supraventricular", body_style), Paragraph("S", body_style),
            Paragraph("Atrial premature / supraventricular ectopic", body_style),
            Paragraph(str(analysis.s_beats), body_style),
            Paragraph(f"{(analysis.s_beats / tot)*100:.1f}%", body_style),
            Paragraph("<font color='#b45309'><b>Ectopic</b></font>" if analysis.s_beats > 0 else "None", body_style)
        ],
        [
            Paragraph("Ventricular", body_style), Paragraph("V", body_style),
            Paragraph("Premature ventricular contraction / escape", body_style),
            Paragraph(str(analysis.v_beats), body_style),
            Paragraph(f"{(analysis.v_beats / tot)*100:.1f}%", body_style),
            Paragraph("<font color='#b91c1c'><b>Ectopic (PVC)</b></font>" if analysis.v_beats > 0 else "None", body_style)
        ],
        [
            Paragraph("Fusion", body_style), Paragraph("F", body_style),
            Paragraph("Fusion of ventricular and normal beat", body_style),
            Paragraph(str(analysis.f_beats), body_style),
            Paragraph(f"{(analysis.f_beats / tot)*100:.1f}%", body_style),
            Paragraph("<font color='#7e22ce'><b>Fusion</b></font>" if analysis.f_beats > 0 else "None", body_style)
        ],
        [
            Paragraph("Paced / Unknown", body_style), Paragraph("Q", body_style),
            Paragraph("Paced beat or unclassifiable waveform", body_style),
            Paragraph(str(analysis.q_beats), body_style),
            Paragraph(f"{(analysis.q_beats / tot)*100:.1f}%", body_style),
            Paragraph("Paced/Unknown" if analysis.q_beats > 0 else "None", body_style)
        ],
        [
            Paragraph("<b>Total Verified</b>", body_bold), Paragraph("-", body_style),
            Paragraph("<b>All cardiac cycles detected (N+S+V+F+Q)</b>", body_bold),
            Paragraph(f"<b>{analysis.total_beats}</b>", body_bold),
            Paragraph("<b>100.0%</b>", body_bold),
            Paragraph(f"<b>{analysis.abnormal_beats} Ectopic ({analysis.abnormal_beats/tot*100:.1f}%)</b>", body_bold)
        ]
    ]

    # Column widths summing strictly to PRINTABLE_WIDTH (523.28 pt)
    # 85 + 35 + 195 + 60 + 65 + 83.28 = 523.28
    breakdown_table = Table(class_table_data, colWidths=[85.0, 35.0, 195.0, 60.0, 65.0, 83.28])
    breakdown_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
        ('BOX', (0, 0), (-1, -1), 0.6, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#e2e8f0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -2), [colors.white, colors.HexColor('#f8fafc')]),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#f1f5f9')),
        ('TOPPADDING', (0, 0), (-1, -1), 2.0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.0),
        ('LEFTPADDING', (0, 0), (-1, -1), 4.5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4.5),
    ]))
    elements.append(breakdown_table)
    elements.append(Spacer(1, 5))

    # 6. ECG Waveform Image (Width 523 pt, Height 170 pt)
    elements.append(Paragraph("Continuous Electrocardiogram (ECG) Rhythm Strip", heading_style))
    elements.append(Spacer(1, 2))
    if os.path.exists(img_path):
        elements.append(Image(img_path, width=PRINTABLE_WIDTH, height=170.0))
    elements.append(Spacer(1, 5))

    # 7. Automated Feature & Rhythm Observations
    elements.append(Paragraph("Automated Rhythm Observations", heading_style))
    elements.append(Spacer(1, 1.5))

    obs_text = []
    if is_normal:
        obs_text.append("&bull;&nbsp;Predominant regular sinus rhythm observed across all detected cardiac cycles.")
        obs_text.append(f"&bull;&nbsp;Mean heart rate calculated from RR intervals is {analysis.avg_heart_rate:.1f} BPM.")
        obs_text.append("&bull;&nbsp;No significant ventricular or supraventricular ectopic complexes identified in this recording.")
    else:
        obs_text.append(f"&bull;&nbsp;Ectopic cardiac activity classified in {analysis.abnormal_beats} of {analysis.total_beats} total detected cycles ({analysis.abnormal_beats/tot*100:.1f}% ectopic burden).")
        if analysis.v_beats > 0:
            obs_text.append(f"&bull;&nbsp;Premature ventricular contractions (PVCs) identified: {analysis.v_beats} beat(s), exhibiting broadened QRS complexes and discordant T-waves.")
        if analysis.s_beats > 0:
            obs_text.append(f"&bull;&nbsp;Supraventricular ectopic beats identified: {analysis.s_beats} beat(s), characterized by early prematurity with narrow QRS complexes.")
        if analysis.f_beats > 0:
            obs_text.append(f"&bull;&nbsp;Fusion beats identified: {analysis.f_beats} beat(s), exhibiting intermediate morphology between sinus and ventricular complexes.")

    obs_text.append("&bull;&nbsp;Notice: All automated algorithmic findings must be confirmed by qualified clinician review.")

    for line in obs_text:
        elements.append(Paragraph(line, body_style))
        elements.append(Spacer(1, 1.2))

    elements.append(Spacer(1, 5))

    # 8. Optional Reviewer / Attestation Block
    reviewer_data = [
        [
            Paragraph("<b>Automated Processing System:</b><br/>Primary Classifier: Weighted K-NN (WKNN)<br/>Academic Prototype v1.0", body_style),
            Paragraph("<b>Optional Clinical Reviewer Attestation:</b><br/>Reviewed by: ___________________________<br/>Date: _______________  Signature: ___________", body_style)
        ]
    ]
    reviewer_table = Table(reviewer_data, colWidths=[261.64, 261.64])
    reviewer_table.setStyle(TableStyle([
        ('LINEABOVE', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ]))
    elements.append(reviewer_table)

    # Build PDF using NumberedCanvas
    doc.build(elements, canvasmaker=NumberedCanvas)
    print(f"Generated PDF report: {output_pdf_path}")
    return output_pdf_path


def export_analysis_csv(analysis, patient, r_peaks, beat_classes, output_csv_path=None, confidence_scores=None):
    """
    Exports detailed beat-by-beat clinical data to CSV, including beat-level confidence.
    """
    if output_csv_path is None:
        os.makedirs('reports', exist_ok=True)
        output_csv_path = f"reports/Analysis_Beats_PAT_{patient.patient_id}_{analysis.id}.csv"

    fs = analysis.sample_rate
    records = []
    prev_r = None

    for i, (r_idx, cls_code) in enumerate(zip(r_peaks, beat_classes), 1):
        time_sec = r_idx / fs
        rr_sec = ((r_idx - prev_r) / fs) if prev_r is not None else 0.0
        prev_r = r_idx
        cls_str = str(cls_code)
        status = 'Normal' if cls_str == 'N' else 'Abnormal'
        conf_val = round(float(confidence_scores[i-1] * 100), 1) if (confidence_scores is not None and i-1 < len(confidence_scores)) else round(analysis.confidence_score * 100, 1)

        records.append({
            'beat_index': i,
            'time_seconds': round(time_sec, 3),
            'sample_index': int(r_idx),
            'rr_interval_seconds': round(rr_sec, 3),
            'instantaneous_bpm': round(60.0 / rr_sec, 1) if rr_sec > 0.1 else round(analysis.avg_heart_rate, 1),
            'aami_class': cls_str,
            'classification_name': AAMI_FULL_NAMES.get(cls_str, 'Unknown'),
            'status': status,
            'prediction_confidence_pct': conf_val
        })

    df = pd.DataFrame(records)
    df.to_csv(output_csv_path, index=False)
    print(f"Exported analysis CSV: {output_csv_path}")
    return output_csv_path
