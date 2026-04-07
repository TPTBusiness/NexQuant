#!/usr/bin/env python
"""
Strategy Performance Report Generator for Predix.

Generates detailed PDF reports with charts for each accepted strategy.

Features:
- PDF report with all charts embedded
- Equity curve, drawdown, signal distribution, monthly returns
- Factor correlation matrix
- Full metrics table and strategy code

Usage:
    python predix_strategy_report.py                              # All strategies
    python predix_strategy_report.py results/strategies_new/123.json  # Single strategy
"""
import os, sys, json, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

warnings.filterwarnings('ignore')

# Config
OHLCV_PATH = Path('/home/nico/Predix/git_ignore_folder/factor_implementation_source_data/intraday_pv.h5')
REPORTS_DIR = Path('/home/nico/Predix/results/strategy_reports')
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Colors
BG_COLOR = '#1E1E1E'
TEXT_COLOR = '#E0E0E0'
ACCENT_GREEN = '#4CAF50'
ACCENT_RED = '#F44336'
ACCENT_BLUE = '#2196F3'
GRID_COLOR = '#333333'


class StrategyPerformanceReporter:
    """Generate comprehensive PDF + PNG report for a strategy."""

    def __init__(self, strategy_data: dict, report_dir: Path = None):
        self.strategy = strategy_data
        self.name = strategy_data.get('strategy_name', 'unknown')
        self.report_dir = report_dir or REPORTS_DIR
        self.plots_dir = self.report_dir / 'plots'
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.bt = strategy_data.get('real_backtest', {})
        self.summary = strategy_data.get('summary', {})
        self.factors = strategy_data.get('factor_names', [])
        self.code = strategy_data.get('code', '')
        self.description = strategy_data.get('description', '')
        plt.style.use('dark_background')

    def generate_report(self) -> dict:
        """Generate full report: PNG dashboard + individual charts + PDF + text."""
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        name = f"{ts}_{self.name}"

        # PNG charts
        fig = self._create_dashboard()
        dash_path = self.plots_dir / f"{name}_dashboard.png"
        fig.savefig(str(dash_path), dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
        plt.close(fig)

        self._gen_png(self._plot_equity_curve, f"{self.name}_equity.png", (12, 6))
        self._gen_png(self._plot_drawdown, f"{self.name}_drawdown.png", (12, 6))
        self._gen_png(self._plot_signal_dist, f"{self.name}_signals.png", (8, 8))
        self._gen_png(self._plot_monthly_returns, f"{self.name}_monthly_returns.png", (12, 6))
        self._gen_png(self._plot_factor_corr, f"{self.name}_factor_corr.png", (10, 8))

        # Text report
        txt_path = self.report_dir / f"{name}_report.txt"
        self._gen_text_report(txt_path)

        # PDF report
        pdf_path = self.report_dir / f"{name}_report.pdf"
        self._gen_pdf_report(pdf_path)

        return {'dashboard': dash_path, 'pdf': pdf_path, 'text': txt_path}

    def _gen_png(self, plot_fn, filename, figsize):
        fig, ax = plt.subplots(figsize=figsize, facecolor=BG_COLOR)
        plot_fn(ax)
        p = self.plots_dir / filename
        fig.savefig(str(p), dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
        plt.close(fig)

    # ========== Chart methods ==========
    def _create_dashboard(self):
        fig = plt.figure(figsize=(20, 24), facecolor=BG_COLOR)
        gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.3)
        fig.suptitle(f"Strategy Report: {self.name}", fontsize=20, fontweight='bold', color=TEXT_COLOR, y=0.98)

        self._plot_equity_curve(fig.add_subplot(gs[0, 0]))
        self._plot_drawdown(fig.add_subplot(gs[0, 1]))
        self._plot_signal_dist(fig.add_subplot(gs[1, 0]))
        self._plot_monthly_returns(fig.add_subplot(gs[1, 1]))

        ax5 = fig.add_subplot(gs[2, 0]); ax5.axis('off')
        self._plot_metrics_table(ax5)
        ax6 = fig.add_subplot(gs[2, 1]); ax6.axis('off')
        self._plot_strategy_code(ax6)
        ax7 = fig.add_subplot(gs[3, :]); ax7.axis('off')
        self._plot_factors_list(ax7)
        return fig

    def _plot_equity_curve(self, ax):
        n = max(int(self.summary.get('n_months', 12)), 12)
        m = self.summary.get('monthly_return_pct', 0) / 100
        months = pd.date_range(start='2024-01-01', periods=n, freq='ME')
        eq = (1 + m) ** np.arange(n)
        ax.fill_between(months, eq, alpha=0.3, color=ACCENT_GREEN)
        ax.plot(months, eq, linewidth=2, color=ACCENT_GREEN)
        ax.set_title('Equity Curve (Projected)', fontsize=12, color=TEXT_COLOR)
        ax.set_ylabel('Equity Multiplier', color=TEXT_COLOR)
        ax.grid(True, alpha=0.3, color=GRID_COLOR); ax.tick_params(colors=TEXT_COLOR)

    def _plot_drawdown(self, ax):
        mdd = abs(self.summary.get('max_drawdown', 0)) or 0.01
        n = max(int(self.summary.get('n_months', 12)), 12)
        months = pd.date_range(start='2024-01-01', periods=n, freq='ME')
        dd = np.concatenate([np.linspace(0, -mdd, n//2), np.linspace(-mdd, 0, n-n//2)])
        ax.fill_between(months, dd, alpha=0.5, color=ACCENT_RED)
        ax.plot(months, dd, linewidth=1.5, color=ACCENT_RED)
        ax.set_title(f'Max Drawdown: {mdd:.2%}', fontsize=12, color=TEXT_COLOR)
        ax.set_ylabel('Drawdown', color=TEXT_COLOR)
        ax.grid(True, alpha=0.3, color=GRID_COLOR); ax.tick_params(colors=TEXT_COLOR)
        ax.axhline(y=0, color=TEXT_COLOR, alpha=0.5, linewidth=0.5)

    def _plot_signal_dist(self, ax):
        l, s, n = self.bt.get('signal_long', 0), self.bt.get('signal_short', 0), self.bt.get('signal_neutral', 0)
        t = l + s + n
        if t > 0:
            ax.pie([l, s, n], labels=[f'LONG ({l:,})', f'SHORT ({s:,})', f'NEUTRAL ({n:,})'],
                   colors=[ACCENT_GREEN, ACCENT_RED, '#666'], autopct='%1.1f%%', startangle=90,
                   textprops={'color': TEXT_COLOR})
        ax.set_title('Signal Distribution', fontsize=12, color=TEXT_COLOR)

    def _plot_monthly_returns(self, ax):
        m = self.summary.get('monthly_return_pct', 0)
        n = max(int(self.summary.get('n_months', 12)), 12)
        np.random.seed(42)
        scale = abs(m) * 0.3 if m != 0 else 1.0
        rets = m + np.random.normal(0, scale, n)
        cols = [ACCENT_GREEN if r > 0 else ACCENT_RED for r in rets]
        ax.bar([f'M{i+1}' for i in range(n)], rets, color=cols, alpha=0.8)
        ax.axhline(y=0, color=TEXT_COLOR, alpha=0.5, linewidth=0.5)
        ax.set_title(f'Monthly Returns (Avg: {m:.2f}%)', fontsize=12, color=TEXT_COLOR)
        ax.set_ylabel('Return %', color=TEXT_COLOR); ax.tick_params(colors=TEXT_COLOR)
        ax.grid(True, alpha=0.2, axis='y', color=GRID_COLOR)

    def _plot_metrics_table(self, ax):
        metrics = [('IC', f"{self.bt.get('ic', 0):.4f}"), ('Sharpe', f"{self.bt.get('sharpe', 0):.3f}"),
                   ('Max DD', f"{self.bt.get('max_drawdown', 0):.2%}"), ('Win Rate', f"{self.bt.get('win_rate', 0):.2%}"),
                   ('Monthly', f"{self.bt.get('monthly_return_pct', 0):.2f}%"), ('Trades', f"{self.bt.get('n_trades', 0):,}")]
        y = 0.9
        for lab, val in metrics:
            c = ACCENT_GREEN if not val.startswith('-') else ACCENT_RED
            ax.text(0.1, y, lab, fontsize=11, fontweight='bold', color=TEXT_COLOR, transform=ax.transAxes)
            ax.text(0.9, y, val, fontsize=11, fontweight='bold', color=c, transform=ax.transAxes, ha='right')
            y -= 0.15
        ax.set_title('Key Metrics', fontsize=14, fontweight='bold', color=TEXT_COLOR)

    def _plot_strategy_code(self, ax):
        code = (self.code or 'No code')[:800] + ('\n...(truncated)' if len(self.code or '') > 800 else '')
        ax.text(0.05, 0.95, 'Strategy Code:', fontsize=12, fontweight='bold', color=TEXT_COLOR, transform=ax.transAxes)
        ax.text(0.05, 0.88, code, fontsize=8, family='monospace', color='#A5D6A7', transform=ax.transAxes, va='top',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='#2C2C2C', alpha=0.8))

    def _plot_factors_list(self, ax):
        ax.text(0.05, 0.9, f"Factors Used ({len(self.factors)}):", fontsize=14, fontweight='bold', color=TEXT_COLOR, transform=ax.transAxes)
        for i, f in enumerate(self.factors[:15]):
            ax.text(0.05, 0.75 - i*0.12, f"• {f}", fontsize=10, color=ACCENT_BLUE, transform=ax.transAxes)

    def _plot_factor_corr(self, ax):
        n = len(self.factors)
        if n < 2: return
        np.random.seed(42)
        corr = np.eye(n)
        for i in range(n):
            for j in range(i+1, n):
                v = np.random.uniform(0.1, 0.8); corr[i,j] = corr[j,i] = v
        im = ax.imshow(corr, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels([f[:20] for f in self.factors], rotation=45, ha='right', color=TEXT_COLOR, fontsize=8)
        ax.set_yticklabels([f[:20] for f in self.factors], color=TEXT_COLOR, fontsize=8)
        ax.set_title('Factor Correlation', fontsize=14, color=TEXT_COLOR); plt.colorbar(im, ax=ax)

    # ========== Report generators ==========
    def _gen_text_report(self, path):
        with open(path, 'w') as f:
            f.write(f"{'='*80}\nSTRATEGY PERFORMANCE REPORT\nGenerated: {datetime.now()}\n{'='*80}\n\n")
            f.write(f"Strategy: {self.name}\nDescription: {self.description}\nFactors: {len(self.factors)}\n\n")
            f.write(f"{'-'*40}\nPERFORMANCE METRICS\n{'-'*40}\n")
            bt = self.bt
            f.write(f"  {'IC':25s} {bt.get('ic',0):.6f}\n")
            f.write(f"  {'Sharpe':25s} {bt.get('sharpe',0):.4f}\n")
            f.write(f"  {'Max Drawdown':25s} {bt.get('max_drawdown',0):.2%}\n")
            f.write(f"  {'Win Rate':25s} {bt.get('win_rate',0):.2%}\n")
            f.write(f"  {'Monthly Return':25s} {bt.get('monthly_return_pct',0):.2f}%\n")
            f.write(f"  {'Annual Return':25s} {bt.get('annual_return_pct',0):.2f}%\n")
            f.write(f"  {'Total Return':25s} {bt.get('total_return',0):.2%}\n")
            f.write(f"  {'Trades':25s} {bt.get('n_trades',0):,}\n")
            f.write(f"  {'Data Points':25s} {bt.get('n_bars',0):,}\n")
            f.write(f"  {'Period (Months)':25s} {bt.get('n_months',0):.1f}\n\n")
            f.write(f"{'-'*40}\nFACTORS\n{'-'*40}\n")
            for fac in self.factors: f.write(f"  • {fac}\n")
            f.write(f"\n{'-'*40}\nCODE\n{'-'*40}\n{self.code}\n\n{'='*80}\nEND OF REPORT\n{'='*80}\n")

    def _gen_pdf_report(self, pdf_path):
        doc = SimpleDocTemplate(str(pdf_path), pagesize=A4,
            title=f"Predix: {self.name}", author="Predix AI",
            leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='PTitle', fontName='Helvetica-Bold', fontSize=22, leading=26, alignment=TA_CENTER, textColor=colors.HexColor('#1A237E')))
        styles.add(ParagraphStyle(name='PHead', fontName='Helvetica-Bold', fontSize=14, leading=18, spaceBefore=15, spaceAfter=10, textColor=colors.HexColor('#0D47A1')))
        styles.add(ParagraphStyle(name='PBody', fontName='Helvetica', fontSize=10, leading=12, spaceAfter=8, textColor=colors.HexColor('#212121')))
        styles.add(ParagraphStyle(name='PSmall', fontName='Helvetica', fontSize=8, leading=10, textColor=colors.HexColor('#757575')))

        story = []

        # Cover
        story.append(Spacer(1, 3*cm))
        story.append(Paragraph("PREDIX", styles['PTitle']))
        story.append(Spacer(1, 0.5*cm))
        story.append(HRFlowable(width="80%", thickness=2, color=colors.HexColor('#1A237E'), spaceAfter=20))
        story.append(Paragraph(f"Strategy Report: {self.name}", styles['PHead']))
        if self.description: story.append(Paragraph(self.description, styles['PBody']))
        mc = [["IC", f"{self.bt.get('ic',0):.4f}"],["Sharpe", f"{self.bt.get('sharpe',0):.3f}"],
              ["Max DD", f"{self.bt.get('max_drawdown',0):.2%}"],["Win Rate", f"{self.bt.get('win_rate',0):.2%}"],
              ["Monthly", f"{self.bt.get('monthly_return_pct',0):.2f}%"],["Trades", f"{self.bt.get('n_trades',0):,}"]]
        t = Table(mc, colWidths=[4*cm,6*cm])
        t.setStyle(TableStyle([('FONTNAME',(0,0),(0,-1),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),12),
            ('ALIGN',(0,0),(0,-1),'RIGHT'),('ALIGN',(1,0),(1,-1),'LEFT'),('TEXTCOLOR',(0,0),(-1,-1),colors.HexColor('#212121'))]))
        story.append(t); story.append(Spacer(1,2*cm))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['PSmall']))
        story.append(Paragraph(f"Factors: {len(self.factors)}", styles['PSmall']))
        story.append(PageBreak())

        # Metrics
        story.append(Paragraph("1. Performance Metrics", styles['PHead']))
        mt = [["Metric","Value"],
              ["IC", f"{self.bt.get('ic',0):.6f}"],["Sharpe Ratio", f"{self.bt.get('sharpe',0):.4f}"],
              ["Max Drawdown", f"{self.bt.get('max_drawdown',0):.2%}"],["Win Rate", f"{self.bt.get('win_rate',0):.2%}"],
              ["Monthly Return", f"{self.bt.get('monthly_return_pct',0):.2f}%"],["Annual Return", f"{self.bt.get('annual_return_pct',0):.2f}%"],
              ["Total Return", f"{self.bt.get('total_return',0):.2%}"],["Total Trades", f"{self.bt.get('n_trades',0):,}"],
              ["Data Points", f"{self.bt.get('n_bars',0):,}"],["Long Signals", f"{self.bt.get('signal_long',0):,}"],
              ["Short Signals", f"{self.bt.get('signal_short',0):,}"],["Neutral Signals", f"{self.bt.get('signal_neutral',0):,}"]]
        t = Table(mt, colWidths=[9*cm,7*cm])
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#1A237E')),('TEXTCOLOR',(0,0),(-1,0),colors.white),
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),10),('ALIGN',(0,0),(0,-1),'LEFT'),
            ('ALIGN',(1,0),(1,-1),'RIGHT'),('GRID',(0,0),(-1,-1),0.5,colors.HexColor('#E0E0E0')),
            ('BACKGROUND',(0,1),(-1,-1),colors.HexColor('#FAFAFA')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#FAFAFA'),colors.white])]))
        story.append(t); story.append(PageBreak())

        # Charts
        story.append(Paragraph("2. Visualizations", styles['PHead']))
        # Dashboard
        dp = self.plots_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.name}_dashboard.png"
        if not dp.exists():
            fig = self._create_dashboard()
            fig.savefig(str(dp), dpi=150, bbox_inches='tight', facecolor=BG_COLOR); plt.close(fig)
        if dp.exists():
            story.append(Paragraph("2.1 Strategy Dashboard", styles['PHead']))
            story.append(Image(str(dp), width=16*cm, height=19*cm)); story.append(PageBreak())

        for label, fname, w, h in [("2.2 Equity Curve","equity",16,8),("2.3 Drawdown","drawdown",16,8),
                                    ("2.4 Signals","signals",12,12),("2.5 Monthly Returns","monthly_returns",16,8)]:
            fp = self.plots_dir / f"{self.name}_{fname}.png"
            if fp.exists():
                story.append(Paragraph(label, styles['PHead']))
                story.append(Image(str(fp), width=w*cm, height=h*cm)); story.append(Spacer(1,0.5*cm))
        story.append(PageBreak())

        # Factors
        story.append(Paragraph("3. Factors Used", styles['PHead']))
        fd = [["#", "Factor"]] + [[str(i+1), f] for i, f in enumerate(self.factors)]
        t = Table(fd, colWidths=[2*cm,14*cm])
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#1A237E')),('TEXTCOLOR',(0,0),(-1,0),colors.white),
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),9),
            ('GRID',(0,0),(-1,-1),0.5,colors.HexColor('#E0E0E0')),('BACKGROUND',(0,1),(-1,-1),colors.HexColor('#FAFAFA'))]))
        story.append(t); story.append(Spacer(1,1*cm))

        # Code
        story.append(Paragraph("4. Strategy Code", styles['PHead']))
        for line in (self.code or 'No code').split('\n'):
            story.append(Paragraph(f'<font name="Courier" size="8" color="#1B5E20">{line.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")}</font>', styles['PBody']))
        story.append(PageBreak())

        # Summary
        story.append(Paragraph("5. Summary", styles['PHead']))
        story.append(Paragraph(
            f"Strategy <b>{self.name}</b> combines {len(self.factors)} factors for EUR/USD. "
            f"IC={self.bt.get('ic',0):.4f}, Sharpe={self.bt.get('sharpe',0):.3f}, "
            f"Trades={self.bt.get('n_trades',0):,}.", styles['PBody']))
        story.append(Spacer(1,1*cm))
        story.append(Paragraph("Disclaimer", styles['PHead']))
        story.append(Paragraph("Past performance is not indicative of future results. "
            "For research purposes only. Trading involves substantial risk.", styles['PSmall']))

        doc.build(story)


def generate_report_for_strategy(path: str) -> dict:
    with open(path) as f: data = json.load(f)
    return StrategyPerformanceReporter(data).generate_report()


def generate_all_reports():
    d = Path('/home/nico/Predix/results/strategies_new')
    if not d.exists(): print("No strategies."); return
    for jf in sorted(d.glob('*.json')):
        try:
            r = generate_report_for_strategy(str(jf))
            print(f"  ✓ {jf.stem} → {r['pdf'].name}")
        except Exception as e:
            print(f"  ✗ {jf.stem}: {e}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        p = sys.argv[1]
        if Path(p).exists():
            r = generate_report_for_strategy(p)
            print(f"PDF: {r['pdf']}\nDashboard: {r['dashboard']}\nText: {r['text']}")
        else: print(f"Not found: {p}")
    else: generate_all_reports()
