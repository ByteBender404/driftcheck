import typer
from typing import Optional
from pathlib import Path

from driftcheck.data_loader import load_data, classify_columns, get_missing_percentages
from driftcheck.drift_metrics import (
    calculate_psi, get_severity, get_severity_score, 
    run_ks_test, run_chi_square_test
)
from driftcheck.report_generator import generate_plot, generate_html_report

app = typer.Typer(help="Driftcheck: Detect statistical drift between two tabular datasets.")

@app.callback()
def callback():
    """Driftcheck CLI"""
    pass

@app.command()
def compare(
    old_file: str = typer.Argument(..., help="Path to the original (expected) dataset CSV/Parquet"),
    new_file: str = typer.Argument(..., help="Path to the new (actual) dataset CSV/Parquet"),
    output: str = typer.Option("report.html", "--output", "-o", help="Path to save the HTML report"),
    columns: Optional[str] = typer.Option(None, "--columns", "-c", help="Comma-separated list of columns to analyze")
):
    """
    Compare two datasets and generate a drift report.
    """
    typer.echo(f"Loading data...")
    df_old = load_data(old_file)
    df_new = load_data(new_file)
    
    if columns:
        cols_to_check = [c.strip() for c in columns.split(',')]
        # Filter existing columns
        cols_to_check = [c for c in cols_to_check if c in df_old.columns and c in df_new.columns]
    else:
        cols_to_check = [c for c in df_old.columns if c in df_new.columns]
        
    if not cols_to_check:
        typer.echo("Error: No matching columns found to compare.", err=True)
        raise typer.Exit(1)
        
    df_old = df_old[cols_to_check]
    df_new = df_new[cols_to_check]
    
    missing_old = get_missing_percentages(df_old)
    missing_new = get_missing_percentages(df_new)
    
    numeric_cols, categorical_cols, excluded_cols = classify_columns(df_old)
    
    # Remove excluded columns from cols_to_check
    cols_to_check = [c for c in cols_to_check if c not in excluded_cols]
    
    typer.echo(f"Found {len(numeric_cols)} numeric and {len(categorical_cols)} categorical columns. Excluded {len(excluded_cols)} ID columns.")
    
    results = []
    
    with typer.progressbar(cols_to_check, label="Calculating metrics") as progress:
        for col in progress:
            is_cat = col in categorical_cols
            
            s_old = df_old[col]
            s_new = df_new[col]
            
            # Calculate PSI
            psi = calculate_psi(s_old, s_new, is_categorical=is_cat)
            severity = get_severity(psi)
            score = get_severity_score(psi)
            
            result_dict = {
                "column": col,
                "type": "categorical" if is_cat else "numeric",
                "psi": psi,
                "severity": severity,
                "severity_score": score,
                "missing_old": missing_old[col],
                "missing_new": missing_new[col]
            }
            
            if is_cat:
                 stat = run_chi_square_test(s_old, s_new)
                 result_dict["chi2_p_value"] = stat["p_value"]
            else:
                 stat = run_ks_test(s_old, s_new)
                 result_dict["ks_p_value"] = stat["p_value"]
                 
            # Generate plot
            plot_b64 = generate_plot(s_old, s_new, col, is_categorical=is_cat)
            result_dict["plot_b64"] = plot_b64
            
            results.append(result_dict)
            
    typer.echo(f"Generating report: {output}")
    generate_html_report(results, excluded_cols, output)
    typer.echo("Done!")

if __name__ == "__main__":
    app()
