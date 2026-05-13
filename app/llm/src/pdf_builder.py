"""
Converts an LLM report.md + SOM visualizations into a PDF.
Entry point: build_pdf(results_dir, report_md_path) -> pdf_path
"""
import io
import os
import json
import re
from fpdf import FPDF
from PIL import Image


# ─── Layout constants ─────────────────────────────────────────────────────────

PAGE_W   = 210   # A4 mm
PAGE_H   = 297
MARGIN   = 14
CONTENT_W = PAGE_W - 2 * MARGIN

# Visualization priority order for the "Key Maps" page
KEY_MAPS = ['u_matrix.png', 'hit_map.png', 'distance_map.png', 'dead_neurons_map.png']
TRAINING_PLOTS = ['mqe_evolution.png', 'learning_rate_decay.png', 'radius_decay.png', 'batch_size_growth.png']


# ─── Public entry point ───────────────────────────────────────────────────────

def build_pdf(results_dir: str, report_md_path: str) -> str:
    """
    Build a PDF report and save to <results_dir>/llm/report.pdf.
    Returns the output path.
    """
    vis_dir  = os.path.join(results_dir, 'visualizations')
    llm_dir  = os.path.dirname(report_md_path)

    context  = _load_context(results_dir)
    report_text = _read_text(report_md_path)

    pdf = _SomPDF()
    pdf.set_auto_page_break(auto=True, margin=MARGIN)

    _add_cover(pdf, context, results_dir)
    _add_report_text(pdf, report_text)
    _add_dimension_stats(pdf, context)
    _add_cluster_table(pdf, context)
    _add_anomaly_list(pdf, context)
    if os.path.isdir(vis_dir):
        _add_key_maps(pdf, vis_dir)
        _add_component_planes(pdf, vis_dir, context)
        _add_pie_maps(pdf, vis_dir, context)
        _add_training_plots(pdf, vis_dir)

    out_path = os.path.join(llm_dir, 'report.pdf')
    pdf.output(out_path)
    return out_path


# ─── PDF class ────────────────────────────────────────────────────────────────

class _SomPDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(150)
        self.cell(0, 5, 'NexusSOM - Analysis Report', align='R')
        self.ln(3)
        self.set_draw_color(200)
        self.line(MARGIN, self.get_y(), PAGE_W - MARGIN, self.get_y())
        self.ln(3)
        self.set_text_color(0)

    def footer(self):
        self.set_y(-12)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(150)
        self.cell(0, 5, f'Page {self.page_no()}', align='C')
        self.set_text_color(0)

    def section_title(self, text: str):
        self.ln(4)
        self.set_font('Helvetica', 'B', 13)
        self.set_fill_color(240, 245, 255)
        self.set_draw_color(180, 200, 230)
        self.set_text_color(30, 60, 120)
        self.rect(MARGIN, self.get_y(), CONTENT_W, 7, style='F')
        self.set_xy(MARGIN + 2, self.get_y() + 0.5)
        self.cell(CONTENT_W - 4, 6, _safe(text), ln=True)
        self.set_text_color(0)
        self.set_draw_color(0)
        self.ln(2)

    def subsection_title(self, text: str):
        self.ln(2)
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(40, 80, 150)
        self.cell(0, 5, _safe(text), ln=True)
        self.set_text_color(0)
        self.ln(1)

    def body_text(self, text: str):
        self.set_font('Helvetica', '', 9)
        self.set_x(MARGIN)
        self.multi_cell(CONTENT_W, 4.5, _safe(text))
        self.ln(1)

    def bullet(self, text: str, indent: int = 4):
        self.set_font('Helvetica', '', 9)
        self.set_x(MARGIN + indent)
        self.cell(4, 4.5, chr(149))   # bullet •
        self.set_x(MARGIN + indent + 4)
        self.multi_cell(CONTENT_W - indent - 4, 4.5, _safe(text))

    def image_caption(self, text: str):
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(90)
        self.cell(0, 4, _safe(text), align='C', ln=True)
        self.set_text_color(0)


# ─── Cover page ───────────────────────────────────────────────────────────────

def _add_cover(pdf: _SomPDF, context: dict, results_dir: str):
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 22)
    pdf.set_text_color(30, 60, 120)
    pdf.ln(20)
    pdf.cell(0, 10, 'NexusSOM - Analysis Report', align='C', ln=True)
    pdf.ln(4)

    # Dataset name from path
    dataset_name = _guess_dataset_name(results_dir)
    pdf.set_font('Helvetica', '', 14)
    pdf.set_text_color(60)
    pdf.cell(0, 8, dataset_name, align='C', ln=True)
    pdf.ln(10)

    # Run timestamp
    ts = os.path.basename(results_dir)
    pdf.set_font('Helvetica', 'I', 10)
    pdf.set_text_color(120)
    pdf.cell(0, 6, f'Run: {ts}', align='C', ln=True)
    pdf.ln(12)

    # Metrics box
    m = context.get('map', {})
    if m:
        pdf.set_draw_color(180, 200, 230)
        pdf.set_fill_color(246, 249, 255)
        box_x = MARGIN + 20
        box_w = CONTENT_W - 40
        box_y = pdf.get_y()
        rows = [
            ('Map size',        f"{m.get('size', ['?', '?'])[0]}×{m.get('size', ['?', '?'])[1]}  {m.get('topology', 'hex')}"),
            ('Total samples',   str(m.get('total_samples', '?'))),
            ('Active neurons',  f"{m.get('active_neurons', '?')} / {m.get('total_neurons', '?')}"),
            ('Dead neurons',    f"{m.get('dead_neurons', 0)}  ({m.get('dead_ratio', 0):.0%})"),
            ('MQE',             f"{m.get('mqe', 0):.4f}"),
            ('Topographic error', f"{m.get('topographic_error', 'N/A'):.4f}" if isinstance(m.get('topographic_error'), float) else 'N/A'),
            ('Coverage ratio',  f"{m.get('coverage_ratio', 0):.0%}"),
            ('Density Gini',    f"{m.get('density_gini', 0):.4f}"),
        ]
        row_h = 6.5
        pdf.rect(box_x, box_y, box_w, len(rows) * row_h + 6, style='FD')
        pdf.set_xy(box_x, box_y + 3)
        pdf.set_font('Helvetica', 'B', 9)
        pdf.set_text_color(30, 60, 120)
        pdf.cell(box_w, 5, 'Map Overview', align='C', ln=True)
        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(0)
        for label, value in rows:
            pdf.set_x(box_x + 6)
            pdf.cell(box_w / 2 - 6, row_h, label + ':', border=0)
            pdf.set_x(box_x + box_w / 2)
            pdf.cell(box_w / 2 - 6, row_h, value, border=0, ln=True)

    pdf.ln(10)
    pdf.set_draw_color(0)
    pdf.set_fill_color(255)
    pdf.set_text_color(0)


# ─── LLM report text ─────────────────────────────────────────────────────────

def _add_report_text(pdf: _SomPDF, text: str):
    """Render markdown-ish report text into PDF."""
    pdf.add_page()
    pdf.section_title('Analysis Report')

    lines = text.splitlines()
    para_lines: list[str] = []

    def flush_para():
        if para_lines:
            pdf.body_text(' '.join(para_lines).strip())
            para_lines.clear()

    for line in lines:
        stripped = line.rstrip()
        if stripped.startswith('## '):
            flush_para()
            pdf.section_title(stripped[3:])
        elif stripped.startswith('### '):
            flush_para()
            pdf.subsection_title(stripped[4:])
        elif re.match(r'^\s*[\*\-]\s', stripped):
            flush_para()
            text_part = re.sub(r'^\s*[\*\-]\s+', '', stripped)
            indent = 4 if not stripped.startswith('\t') and stripped[0] in ('*', '-') else 10
            pdf.bullet(text_part, indent=indent)
        elif re.match(r'^\s+[\*\-\+]\s', stripped):
            flush_para()
            text_part = re.sub(r'^\s+[\*\-\+]\s+', '', stripped)
            pdf.bullet(text_part, indent=10)
        elif stripped == '' or stripped == '---':
            flush_para()
            if stripped == '':
                pdf.ln(1.5)
        else:
            # Accumulate paragraph lines
            para_lines.append(stripped)

    flush_para()


# ─── Dimension statistics table ──────────────────────────────────────────────

def _add_dimension_stats(pdf: _SomPDF, context: dict):
    dim_stats = context.get('dimension_stats', {})
    cat_dist  = context.get('category_distributions', {})
    if not dim_stats and not cat_dist:
        return

    pdf.add_page()
    pdf.section_title('Dimension Statistics')

    # Numeric columns — full stats table
    if dim_stats:
        pdf.subsection_title('Numeric columns')
        headers = ['Column', 'Min', 'Max', 'Mean', 'Std', 'Median', 'p25', 'p75', 'p90', 'p95']
        col_w   = [32, 14, 14, 16, 14, 16, 14, 14, 14, 14]
        _table_header(pdf, headers, col_w)
        pdf.set_font('Helvetica', '', 8)
        for col_name, s in dim_stats.items():
            vals = [
                _safe(col_name),
                f"{s.get('min', ''):.2f}" if isinstance(s.get('min'), float) else str(s.get('min', '')),
                f"{s.get('max', ''):.2f}" if isinstance(s.get('max'), float) else str(s.get('max', '')),
                f"{s.get('mean', 0):.3f}",
                f"{s.get('std', 0):.3f}",
                f"{s.get('median', 0):.2f}",
                f"{s.get('p25', 0):.2f}",
                f"{s.get('p75', 0):.2f}",
                f"{s.get('p90', 0):.2f}",
                f"{s.get('p95', 0):.2f}",
            ]
            _table_row(pdf, vals, col_w)
        pdf.ln(4)

    # Categorical columns — distribution table (2 cols side by side)
    if cat_dist:
        pdf.subsection_title('Category distributions')
        cat_items = list(cat_dist.items())
        # Render each column as its own mini-table, 2 per row
        col_pairs = [cat_items[i:i+2] for i in range(0, len(cat_items), 2)]
        half_w = (CONTENT_W - 6) / 2

        for pair in col_pairs:
            y_start = pdf.get_y()
            for pi, (cat_col, dist) in enumerate(pair):
                x = MARGIN + pi * (half_w + 6)
                pdf.set_xy(x, y_start)
                pdf.set_font('Helvetica', 'B', 8)
                pdf.set_text_color(40, 80, 150)
                pdf.cell(half_w, 5, _safe(cat_col), ln=True)
                pdf.set_text_color(0)
                pdf.set_font('Helvetica', '', 8)
                for val, ratio in sorted(dist.items(), key=lambda kv: -kv[1]):
                    pdf.set_xy(x + 2, pdf.get_y())
                    bar_w = half_w * 0.55 * ratio
                    pdf.set_fill_color(180, 200, 230)
                    bar_y = pdf.get_y() + 0.5
                    pdf.rect(x + 2, bar_y, bar_w, 3.5, style='F')
                    pdf.set_fill_color(255)
                    pdf.set_x(x + 2)
                    pdf.cell(half_w * 0.55, 4.5, _safe(str(val)))
                    pdf.cell(half_w * 0.2, 4.5, f'{ratio:.1%}', ln=True)
            pdf.set_y(y_start)
            # Advance Y past the tallest column in this pair
            max_lines = max(len(d) for _, d in pair)
            pdf.set_y(y_start + 5 + max_lines * 4.5 + 4)


# ─── Cluster statistics table ─────────────────────────────────────────────────

def _add_cluster_table(pdf: _SomPDF, context: dict):
    clusters = context.get('clusters', [])
    if not clusters:
        return

    pdf.add_page()
    pdf.section_title('Cluster Statistics')

    # Determine which numeric/categorical cols to show
    sample = clusters[0]
    num_cols = list(sample.get('dimension_stats', {}).keys())
    dom_cols = list(sample.get('dominant_category', {}).keys())
    # Show at most 3 numeric + 3 categorical + key cols to fit width
    show_num  = num_cols[:3]
    show_cats = dom_cols[:3]

    # Build header
    headers = ['Neuron', 'N', 'QE'] + show_num + [f'{c}(dom)' for c in show_cats] + ['LUNG_C purity']
    # Remove LUNG_C purity if not present; handle generically
    key_cat = 'LUNG_CANCER'
    if key_cat not in dom_cols:
        key_cat = dom_cols[0] if dom_cols else None
    headers = ['Neuron', 'N', 'QE'] + show_num + [f'{c[:6]}(dom)' for c in show_cats]
    if key_cat:
        headers.append(f'{key_cat[:8]} pur.')

    base_w  = 16
    num_w   = [20] + [12, 10] + [18] * len(show_num) + [16] * len(show_cats)
    if key_cat:
        num_w.append(16)

    _table_header(pdf, headers, num_w)
    pdf.set_font('Helvetica', '', 7.5)

    for c in clusters[:40]:
        dom = c.get('dominant_category', {})
        pur = c.get('purity', {})
        dim = c.get('dimension_stats', {})

        row = [
            _safe(c['neuron']),
            str(c['sample_count']),
            f"{c.get('quantization_error', 0):.3f}",
        ]
        for nc in show_num:
            v = dim.get(nc, {}).get('mean', '')
            row.append(f'{v:.1f}' if isinstance(v, float) else str(v))
        for sc in show_cats:
            row.append(_safe(str(dom.get(sc, '')))[:8])
        if key_cat:
            p = pur.get(key_cat, '')
            row.append(f'{p:.0%}' if isinstance(p, float) else str(p))

        _table_row(pdf, row, num_w)

    pdf.ln(3)
    pdf.set_font('Helvetica', 'I', 7.5)
    pdf.set_text_color(100)
    pdf.cell(0, 4, f'Showing top {min(40, len(clusters))} of {len(clusters)} clusters by sample count.', ln=True)
    pdf.set_text_color(0)

    # Global deviation sub-table (numeric cols only)
    dev_rows = [(c['neuron'], c.get('global_deviation', {})) for c in clusters if c.get('global_deviation')]
    if dev_rows and show_num:
        pdf.ln(3)
        pdf.subsection_title('Global deviation (Z-score of cluster mean vs. dataset mean)')
        dev_headers = ['Neuron'] + show_num
        dev_w = [24] + [28] * len(show_num)
        _table_header(pdf, dev_headers, dev_w)
        pdf.set_font('Helvetica', '', 8)
        for neuron, devs in dev_rows[:20]:
            row = [_safe(neuron)]
            for nc in show_num:
                z = devs.get(nc, '')
                if isinstance(z, float):
                    row.append(f'{z:+.2f}')
                else:
                    row.append('-')
            _table_row(pdf, row, dev_w, highlight_fn=lambda r: abs(float(r[1])) > 1.5 if len(r) > 1 and _is_float(r[1]) else False)
        pdf.ln(3)
        pdf.set_font('Helvetica', 'I', 7.5)
        pdf.set_text_color(100)
        pdf.cell(0, 4, 'Highlighted rows: |Z| > 1.5 (cluster mean differs from dataset by more than 1.5 std).', ln=True)
        pdf.set_text_color(0)


# ─── Anomaly list ─────────────────────────────────────────────────────────────

def _add_anomaly_list(pdf: _SomPDF, context: dict):
    anom    = context.get('anomalies', {})
    records = context.get('anomaly_records', [])
    top     = anom.get('top_anomalies', [])
    if not top and not records:
        return

    pdf.add_page()
    pdf.section_title('Anomaly Detection Results')

    pdf.set_font('Helvetica', '', 9)
    pdf.cell(0, 5,
        f"Global extremes: {anom.get('global_outlier_count', 0)}    "
        f"Local outliers: {anom.get('local_outlier_count', 0)}    "
        f"Top anomalies shown: {len(records or top)}",
        ln=True)
    pdf.ln(2)

    type_labels = {
        'one_of_n':        '1-of-N',
        'multi_dim':       'Multi-dim',
        'numeric':         'Numeric',
        'global_extreme':  'Global extreme',
        'categorical_minority': 'Cat. minority',
    }
    flag_sym = {
        'global_min':       'MIN',
        'global_max':       'MAX',
        'high_deviation':   '>>',
        'moderate_deviation': '>',
    }

    if records:
        # ── Full row table from anomaly_records ──
        # Collect all numeric/categorical columns from first record
        first_cols   = list(records[0]['columns'].keys()) if records else []
        numeric_cols = [c for c in first_cols if 'delta' in records[0]['columns'].get(c, {})]
        cat_cols     = [c for c in first_cols if c not in numeric_cols]

        # Compact header: ID | Neuron | Type | numeric_cols... | key cat cols
        show_num = numeric_cols[:4]
        show_cat = cat_cols[:4]
        headers  = ['#', 'ID', 'Neuron', 'Type'] + show_num + [c[:8] for c in show_cat]
        col_w    = [7, 18, 18, 18] + [max(14, len(c)+2) for c in show_num] + [14] * len(show_cat)
        # Trim to page width
        while sum(col_w) > CONTENT_W + 1 and len(col_w) > 4:
            col_w.pop()
            headers.pop()

        _table_header(pdf, headers, col_w)
        pdf.set_font('Helvetica', '', 7.5)

        for rank, rec in enumerate(records, 1):
            atype  = type_labels.get(rec.get('type', ''), str(rec.get('type', '')))
            row    = [
                str(rank),
                _safe(str(rec['sample_id'])),
                _safe(str(rec.get('neuron') or '')),
                atype,
            ]
            cols_data = rec.get('columns', {})
            for nc in show_num:
                if nc not in [h for h in headers[4:]]:
                    break
                info = cols_data.get(nc, {})
                val  = info.get('value', '')
                flag = info.get('flag', '')
                sym  = flag_sym.get(flag, '')
                dpct = info.get('delta_pct')
                cell = f"{val}"
                if dpct is not None:
                    cell += f" ({dpct:+.0f}%)"
                if sym:
                    cell += f" {sym}"
                row.append(_safe(cell))
            for cc in show_cat:
                if cc not in [h for h in headers[4 + len(show_num):]]:
                    break
                info    = cols_data.get(cc, {})
                val     = str(info.get('value', ''))
                differs = info.get('differs_from_cluster_dominant')
                cell    = val + ('*' if differs else '')
                row.append(_safe(cell))

            # Only pass as many values as headers
            row = row[:len(headers)]
            highlight = rec.get('type') in ('multi_dim', 'one_of_n')
            _table_row(pdf, row, col_w[:len(row)],
                       highlight_fn=lambda r, h=highlight: h)

        pdf.ln(2)
        pdf.set_font('Helvetica', 'I', 7.5)
        pdf.set_text_color(100)
        pdf.multi_cell(CONTENT_W, 4,
            'Numeric cells: value (delta% from mean). '
            'MIN/MAX = global extreme. >> = >50% deviation. > = >20%. '
            'Cat col * = value differs from cluster dominant.')
        pdf.set_text_color(0)

        # ── Per-record detail block for top 5 ──
        if len(records) > 0:
            pdf.ln(4)
            pdf.subsection_title('Top anomaly detail (all columns)')
            for rec in records[:5]:
                _add_anomaly_record_block(pdf, rec, type_labels, flag_sym)

    else:
        # Fallback: plain reason table (old format without anomaly_records)
        headers = ['#', 'Sample ID', 'Neuron', 'Type', 'Reason']
        widths  = [8, 22, 20, 22, CONTENT_W - 8 - 22 - 20 - 22]
        _table_header(pdf, headers, widths)
        pdf.set_font('Helvetica', '', 8)
        for rank, a in enumerate(top, 1):
            reason = _safe(a.get('reasons', [''])[0])
            atype  = type_labels.get(a.get('type', ''), _safe(str(a.get('type', ''))))
            max_c  = int(widths[-1] / 1.9)
            if len(reason) > max_c:
                reason = reason[:max_c - 2] + '..'
            row = [str(rank), _safe(str(a.get('sample_id',''))),
                   _safe(str(a.get('neuron') or '')), atype, reason]
            _table_row(pdf, row, widths,
                       highlight_fn=lambda r, t=a.get('type'): t in ('multi_dim','one_of_n'))


def _add_anomaly_record_block(pdf: _SomPDF, rec: dict,
                               type_labels: dict, flag_sym: dict):
    """Compact per-record card: all column values with delta annotations."""
    atype = type_labels.get(rec.get('type', ''), str(rec.get('type', '')))
    ratio = f", ratio={rec['distance_ratio']:.2f}x" if rec.get('distance_ratio') else ""
    pdf.set_font('Helvetica', 'B', 8.5)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 5,
        f"Sample {rec['sample_id']}  |  neuron {rec.get('neuron','')}  |  {atype}{ratio}",
        ln=True)
    pdf.set_text_color(0)

    # Render columns in a compact 2-column layout
    col_items = list(rec.get('columns', {}).items())
    half = (CONTENT_W) / 2
    x_left  = MARGIN
    x_right = MARGIN + half

    for i, (col, info) in enumerate(col_items):
        x = x_left if i % 2 == 0 else x_right
        if i % 2 == 0 and i > 0:
            pdf.ln(0)  # already advanced by previous right cell

        val  = info.get('value', '')
        flag = info.get('flag', '')
        sym  = flag_sym.get(flag, '')
        dpct = info.get('delta_pct')
        differs = info.get('differs_from_cluster_dominant')

        # Choose color by flag
        if flag in ('global_min', 'global_max'):
            pdf.set_text_color(180, 0, 0)
        elif flag == 'high_deviation':
            pdf.set_text_color(180, 80, 0)
        elif flag == 'moderate_deviation':
            pdf.set_text_color(140, 110, 0)
        else:
            pdf.set_text_color(60)

        cell_text = f"{col}: {val}"
        if dpct is not None:
            cell_text += f"  ({dpct:+.1f}%)"
        if sym:
            cell_text += f" {sym}"
        if differs:
            cell_text += f"  [dom={differs}]"

        pdf.set_font('Helvetica', 'B' if flag else '', 8)
        pdf.set_xy(x, pdf.get_y())
        pdf.cell(half, 4.5, _safe(cell_text), ln=(i % 2 == 1))

    if len(col_items) % 2 == 1:
        pdf.ln(4.5)

    pdf.set_text_color(0)
    pdf.ln(3)


# ─── Table helpers ────────────────────────────────────────────────────────────

def _table_header(pdf: _SomPDF, headers: list, widths: list):
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_fill_color(220, 230, 245)
    pdf.set_draw_color(160, 180, 210)
    pdf.set_text_color(30, 60, 120)
    for h, w in zip(headers, widths):
        pdf.cell(w, 5.5, _safe(str(h)), border=1, fill=True)
    pdf.ln()
    pdf.set_text_color(0)
    pdf.set_draw_color(200)


def _table_row(pdf: _SomPDF, values: list, widths: list, highlight_fn=None):
    fill = highlight_fn(values) if highlight_fn else False
    pdf.set_fill_color(255, 248, 220) if fill else pdf.set_fill_color(255)
    for v, w in zip(values, widths):
        pdf.cell(w, 5, _safe(str(v)), border=1, fill=fill)
    pdf.ln()
    pdf.set_fill_color(255)


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


# ─── Key SOM maps ─────────────────────────────────────────────────────────────

def _add_key_maps(pdf: _SomPDF, vis_dir: str):
    available = [f for f in KEY_MAPS if os.path.isfile(os.path.join(vis_dir, f))]
    if not available:
        return

    pdf.add_page()
    pdf.section_title('SOM Maps')

    # 2 images per row
    img_size = (CONTENT_W - 4) / 2  # mm per image
    captions = {
        'u_matrix.png':      'U-Matrix - topological structure',
        'hit_map.png':       'Hit Map - sample density',
        'distance_map.png':  'Distance Map - quantization error',
        'dead_neurons_map.png': 'Dead Neurons Map',
    }

    for i in range(0, len(available), 2):
        row = available[i:i + 2]
        y_start = pdf.get_y()
        for j, fname in enumerate(row):
            x = MARGIN + j * (img_size + 4)
            _img(pdf, os.path.join(vis_dir, fname), x=x, y=y_start, w=img_size)
        pdf.set_y(y_start + img_size)
        for j, fname in enumerate(row):
            x = MARGIN + j * (img_size + 4)
            pdf.set_x(x)
            pdf.set_font('Helvetica', 'I', 7.5)
            pdf.set_text_color(90)
            caption = _safe(captions.get(fname, fname.replace('.png', '')))
            pdf.cell(img_size, 4, caption, align='C', ln=(j == len(row) - 1))
        pdf.set_text_color(0)
        pdf.ln(4)


# ─── Component planes ─────────────────────────────────────────────────────────

def _add_component_planes(pdf: _SomPDF, vis_dir: str, context: dict):
    comp_files = sorted(
        f for f in os.listdir(vis_dir)
        if f.startswith('component_') and f.endswith('.png') and 'legend' not in f
    )
    if not comp_files:
        return

    pdf.add_page()
    pdf.section_title('Component Planes')

    cols     = 3
    gap      = 3
    img_size = (CONTENT_W - gap * (cols - 1)) / cols

    row_y = pdf.get_y()
    for i, fname in enumerate(comp_files):
        col = i % cols
        if col == 0:
            row_y = pdf.get_y()

        x = MARGIN + col * (img_size + gap)
        _img(pdf, os.path.join(vis_dir, fname), x=x, y=row_y, w=img_size)

        label = _safe(fname.replace('component_', '').replace('.png', ''))
        pdf.set_xy(x, row_y + img_size)
        pdf.set_font('Helvetica', 'I', 7.5)
        pdf.set_text_color(90)
        pdf.cell(img_size, 3.5, label, align='C')
        pdf.set_text_color(0)

        if col == cols - 1 or i == len(comp_files) - 1:
            pdf.set_y(row_y + img_size + 3.5 + gap)


# ─── Pie maps ────────────────────────────────────────────────────────────────

def _add_pie_maps(pdf: _SomPDF, vis_dir: str, context: dict):
    pie_files = sorted(
        f for f in os.listdir(vis_dir)
        if f.startswith('pie_map_') and f.endswith('.png')
    )
    if not pie_files:
        return

    # Put key categorical column first (LUNG_CANCER, diagnosis, label, target)
    key_cols = ['LUNG_CANCER', 'diagnosis', 'label', 'target', 'class', 'outcome']
    def sort_key(f):
        col = f.replace('pie_map_', '').replace('.png', '')
        return (0 if col in key_cols else 1, col)
    pie_files.sort(key=sort_key)

    pdf.add_page()
    pdf.section_title('Categorical Distribution Maps')

    cols     = 2
    gap      = 4
    img_size = (CONTENT_W - gap * (cols - 1)) / cols
    row_y = pdf.get_y()

    for i, fname in enumerate(pie_files):
        col = i % cols
        if col == 0:
            row_y = pdf.get_y()

        x = MARGIN + col * (img_size + gap)
        _img(pdf, os.path.join(vis_dir, fname), x=x, y=row_y, w=img_size)

        label = _safe(fname.replace('pie_map_', '').replace('.png', ''))
        pdf.set_xy(x, row_y + img_size)
        pdf.set_font('Helvetica', 'I', 7.5)
        pdf.set_text_color(90)
        pdf.cell(img_size, 3.5, label, align='C')
        pdf.set_text_color(0)

        if col == cols - 1 or i == len(pie_files) - 1:
            pdf.set_y(row_y + img_size + 3.5 + gap)


# ─── Training plots ───────────────────────────────────────────────────────────

def _add_training_plots(pdf: _SomPDF, vis_dir: str):
    available = [f for f in TRAINING_PLOTS if os.path.isfile(os.path.join(vis_dir, f))]
    if not available:
        return

    pdf.add_page()
    pdf.section_title('Training Curves')

    cols     = 2
    gap      = 4
    img_w    = (CONTENT_W - gap * (cols - 1)) / cols
    img_h    = img_w * 0.55   # plots are wider than tall
    captions = {
        'mqe_evolution.png':      'MQE Evolution',
        'learning_rate_decay.png': 'Learning Rate Decay',
        'radius_decay.png':       'Neighbourhood Radius Decay',
        'batch_size_growth.png':  'Batch Size Growth',
    }
    _y = pdf.get_y()
    for i, fname in enumerate(available):
        col = i % cols
        if col == 0 and i > 0:
            _y = pdf.get_y()
        x = MARGIN + col * (img_w + gap)
        _img(pdf, os.path.join(vis_dir, fname), x=x, y=_y, w=img_w, h=img_h)
        pdf.set_xy(x, _y + img_h)
        pdf.set_font('Helvetica', 'I', 7.5)
        pdf.set_text_color(90)
        caption = _safe(captions.get(fname, fname.replace('.png', '')))
        pdf.cell(img_w, 3.5, caption, align='C')
        pdf.set_text_color(0)
        if col == cols - 1 or i == len(available) - 1:
            pdf.set_y(_y + img_h + 3.5 + 6)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load_context(results_dir: str) -> dict:
    path = os.path.join(results_dir, 'json', 'llm_context.json')
    if os.path.isfile(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    return {}


def _read_text(path: str) -> str:
    with open(path, encoding='utf-8') as f:
        return f.read()


def _img(pdf: _SomPDF, path: str, x: float, y: float, w: float, h: float = 0,
         max_px: int = 700):
    """Embed a PNG scaled to max_px on its longest side before inserting."""
    img = Image.open(path)
    ow, oh = img.size
    scale = min(max_px / ow, max_px / oh, 1.0)
    if scale < 1.0:
        img = img.resize((int(ow * scale), int(oh * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=True)
    buf.seek(0)
    if h:
        pdf.image(buf, x=x, y=y, w=w, h=h)
    else:
        pdf.image(buf, x=x, y=y, w=w)


def _guess_dataset_name(results_dir: str) -> str:
    # Walk up: .../datasets/<DatasetName>/results/<timestamp>
    parts = os.path.normpath(results_dir).split(os.sep)
    try:
        idx = parts.index('results')
        if idx > 0:
            return parts[idx - 1]
    except ValueError:
        pass
    return os.path.basename(os.path.dirname(os.path.dirname(results_dir)))


def _safe(text: str) -> str:
    """Replace common Unicode chars with Latin-1 equivalents."""
    _map = {
        '—': '--',  '–': '-',   '‒': '-',
        '‘': "'",   '’': "'",   '‚': ',',
        '“': '"',   '”': '"',
        '•': chr(149), '·': '.',
        '…': '...',
        '×': 'x',   '÷': '/',
        '→': '->',  '←': '<-',
        '≈': '~=',  '≠': '!=',
        '≥': '>=',  '≤': '<=',
    }
    for src, dst in _map.items():
        text = text.replace(src, dst)
    return text.encode('latin-1', errors='replace').decode('latin-1')
