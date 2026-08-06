import io
import base64
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any

def generate_plot(expected: pd.Series, actual: pd.Series, col_name: str, is_categorical: bool) -> str:
    """Generates a distribution plot and returns it as a base64 encoded string."""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Common colors
    c_expected = '#1f77b4'  # Blue
    c_actual = '#ff7f0e'    # Orange
    
    if is_categorical:
        categories = sorted(list(set(expected.dropna().unique()).union(set(actual.dropna().unique()))))
        # Take top 20 categories if too many
        if len(categories) > 20:
             categories = categories[:20]
             
        e_counts = expected.value_counts(normalize=True).reindex(categories, fill_value=0) * 100
        a_counts = actual.value_counts(normalize=True).reindex(categories, fill_value=0) * 100
        
        x = range(len(categories))
        width = 0.35
        
        ax.bar([i - width/2 for i in x], e_counts, width, label='Old', color=c_expected, alpha=0.8)
        ax.bar([i + width/2 for i in x], a_counts, width, label='New', color=c_actual, alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in categories], rotation=45, ha="right")
        ax.set_ylabel('Percentage (%)')
    else:
        # Numeric Histogram
        ax.hist(expected.dropna(), bins=20, alpha=0.5, label='Old', density=True, color=c_expected)
        ax.hist(actual.dropna(), bins=20, alpha=0.5, label='New', density=True, color=c_actual)
        ax.set_ylabel('Density')
        
    ax.set_title(f'Distribution: {col_name}')
    ax.legend()
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_b64

def generate_html_report(results: List[Dict[str, Any]], excluded_cols: Dict[str, str], output_path: str):
    """Generates the final self-contained HTML report."""
    
    # Sort results by severity score descending
    results = sorted(results, key=lambda x: x['severity_score'], reverse=True)
    
    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f9fafb; }
    h1 { color: #111827; border-bottom: 2px solid #e5e7eb; padding-bottom: 10px; }
    h2 { color: #374151; }
    .card { background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
    th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #e5e7eb; }
    th { background-color: #f3f4f6; font-weight: 600; color: #374151; }
    tr:hover { background-color: #f9fafb; }
    .badge { padding: 4px 8px; border-radius: 9999px; font-size: 0.85em; font-weight: 500; }
    .badge-High { background-color: #fee2e2; color: #991b1b; }
    .badge-Medium { background-color: #fef3c7; color: #92400e; }
    .badge-Low { background-color: #d1fae5; color: #065f46; }
    .plot-container { text-align: center; margin-top: 15px; }
    .plot-container img { max-width: 100%; height: auto; border: 1px solid #e5e7eb; border-radius: 4px; }
    .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 15px; }
    .stat-box { background: #f8fafc; padding: 10px; border-radius: 6px; border: 1px solid #e2e8f0; }
    .stat-label { font-size: 0.85em; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }
    .stat-value { font-size: 1.2em; font-weight: 600; color: #0f172a; margin-top: 4px; }
    """
    
    html = [
        f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Driftcheck Report</title>",
        f"<style>{css}</style></head><body>",
        f"<h1>Data Drift Report</h1>",
        f"<div class='card'><h2>Summary</h2>",
        f"<table><thead><tr>",
        f"<th>Column</th><th>Type</th><th>Severity</th><th>Severity Score</th><th>PSI</th><th>Missing (Old -> New)</th>",
        f"</tr></thead><tbody>"
    ]
    
    # Summary Table
    for r in results:
        badge_class = f"badge-{r['severity']}"
        html.append(f"<tr>")
        html.append(f"<td><strong>{r['column']}</strong></td>")
        html.append(f"<td>{r['type']}</td>")
        html.append(f"<td><span class='badge {badge_class}'>{r['severity']}</span></td>")
        html.append(f"<td>{r['severity_score']}/100</td>")
        html.append(f"<td>{r['psi']:.4f}</td>")
        html.append(f"<td>{r['missing_old']:.1f}% &rarr; {r['missing_new']:.1f}%</td>")
        html.append(f"</tr>")
        
    html.append("</tbody></table></div>")
    
    # Excluded Columns Table
    if excluded_cols:
        html.append(f"<div class='card'><h2>Excluded (Identifier Columns)</h2>")
        html.append(f"<table><thead><tr>")
        html.append(f"<th>Column</th><th>Exclusion Reason</th>")
        html.append(f"</tr></thead><tbody>")
        for col, reason in excluded_cols.items():
             html.append(f"<tr>")
             html.append(f"<td><strong>{col}</strong></td>")
             html.append(f"<td>{reason}</td>")
             html.append(f"</tr>")
        html.append("</tbody></table></div>")
    
    # Detailed Sections
    html.append("<h2>Detailed Analysis</h2>")
    
    for r in results:
        html.append(f"<div class='card' id='{r['column']}'>")
        html.append(f"<h3>{r['column']} <span class='badge badge-{r['severity']}'>{r['severity']} Drift</span></h3>")
        
        # Stats grid
        html.append("<div class='stats-grid'>")
        html.append(f"<div class='stat-box'><div class='stat-label'>Data Type</div><div class='stat-value'>{r['type']}</div></div>")
        html.append(f"<div class='stat-box'><div class='stat-label'>PSI Score</div><div class='stat-value'>{r['psi']:.4f}</div></div>")
        if r['type'] == 'numeric':
             html.append(f"<div class='stat-box'><div class='stat-label'>KS Test p-value</div><div class='stat-value'>{r.get('ks_p_value', 'N/A'):.4e}</div></div>")
        else:
             html.append(f"<div class='stat-box'><div class='stat-label'>Chi-Sq p-value</div><div class='stat-value'>{r.get('chi2_p_value', 'N/A'):.4e}</div></div>")
        html.append("</div>")
        
        html.append(f"<div class='plot-container'><img src='data:image/png;base64,{r['plot_b64']}' alt='Distribution Plot for {r['column']}'></div>")
        html.append("</div>")
        
    html.append("</body></html>")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(html))
