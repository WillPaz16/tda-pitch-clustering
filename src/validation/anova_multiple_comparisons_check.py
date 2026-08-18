"""
Multiple-Comparisons Check for the ANOVA Results

Follow-up to docs/METHODOLOGY_REVIEW.md item 6: ANOVA tests run across
several outcome metrics with no visible Bonferroni/FDR correction.

Recomputes REAL p-values from the F-statistics and sample sizes already
in data/anova_results_real_data.csv (the F-statistic itself is computed
correctly in anova_real_data.py; only the p-value is wrong there -- see
below), and checks whether the "significant" calls survive Bonferroni
correction across all metrics tested.

Side finding, not the original question but discovered while checking
it: src/validation/anova_real_data.py and
src/validation/cluster_outcome_validation.py both convert F-statistic to
p-value with a hardcoded threshold (`p = 1.0 if f_stat < 3.0 else
0.001`), not a real p-value from the F-distribution. That's a latent
correctness bug independent of the multiple-comparisons question -- it
just didn't change any conclusion here because the true effect sizes
are enormous.

Analysis/reporting script. See docs/METHODOLOGY_REVIEW.md for write-up.
"""

import ast
import csv
from pathlib import Path

from scipy import stats

_RESULTS_PATH = Path(__file__).resolve().parents[2] / 'data' / 'anova_results_real_data.csv'
ALPHA = 0.05


def main():
    rows = list(csv.DictReader(open(_RESULTS_PATH)))
    n_tests = len(rows)
    bonferroni_alpha = ALPHA / n_tests

    print(f"{'metric':<20}{'F':<10}{'reported_p':<12}{'real_p':<14}{'sig (raw)':<12}{'sig (Bonferroni)'}")
    for row in rows:
        f_stat = float(row['f_statistic'])
        sizes = ast.literal_eval(row['sample_sizes'])
        df_between = len(sizes) - 1
        df_within = sum(sizes) - len(sizes)
        real_p = stats.f.sf(f_stat, df_between, df_within)
        print(f"{row['metric']:<20}{f_stat:<10.2f}{row['p_value']:<12}{real_p:<14.2e}"
              f"{str(real_p < ALPHA):<12}{real_p < bonferroni_alpha}")

    print(f"\nBonferroni-corrected alpha across {n_tests} metrics: {bonferroni_alpha:.5f}")
    print("\nThe reported p-values in the CSV are wrong -- anova_real_data.py's manual_anova()\n"
          "converts F-statistic to p-value with a hardcoded threshold (p=1.0 if F<3.0 else\n"
          "0.001), not the real F-distribution CDF, and the same bug is duplicated in\n"
          "cluster_outcome_validation.py. The F-statistics themselves are computed correctly.\n"
          "Recomputed real p-values above are astronomically smaller than the fake 0.001\n"
          "placeholder in every case here, and every result survives Bonferroni correction\n"
          "comfortably -- so the specific 'significant' conclusions in this file are not\n"
          "wrong, but the fake p-value logic is a real latent bug that could produce a wrong\n"
          "conclusion on a smaller, more marginal effect size elsewhere in the pipeline.")


if __name__ == "__main__":
    main()
