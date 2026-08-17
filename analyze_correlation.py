import pandas as pd
import numpy as np
from scipy import stats

# Load the xERA results
xera_results = pd.read_csv('stuff_plus_weight_tuning_xera_2025.csv')
valid_data = xera_results.dropna(subset=['weighted_stuff_z', 'xERA'])

r = valid_data['weighted_stuff_z'].corr(valid_data['xERA'])
n = len(valid_data)

print('=== Statistical Analysis of xERA Correlation ===')
print(f'Correlation coefficient (r): {r:.4f}')
print(f'R-squared (r²): {r**2:.6f} ({r**2*100:.3f}%)')
print(f'Sample size (n): {n}')
print()

# Test statistical significance
t_stat = r * np.sqrt((n-2) / (1 - r**2))
p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n-2))

print(f't-statistic: {t_stat:.4f}')
print(f'p-value: {p_value:.6f}')
print(f'Statistically significant (p < 0.05): {p_value < 0.05}')
print()

# Confidence interval for correlation
z = np.arctanh(r)
se = 1 / np.sqrt(n - 3)
ci_lower = np.tanh(z - 1.96 * se)
ci_upper = np.tanh(z + 1.96 * se)

print(f'95% Confidence interval for r: [{ci_lower:.4f}, {ci_upper:.4f}]')
print()

# Effect size interpretation
if abs(r) < 0.1:
    effect_size = 'Very weak'
elif abs(r) < 0.3:
    effect_size = 'Weak'
elif abs(r) < 0.5:
    effect_size = 'Moderate'
else:
    effect_size = 'Strong'

print(f'Effect size: {effect_size}')
print()

print('=== My Thoughts on This Correlation ===')
print('From a quantitative perspective, this correlation is indeed very weak.')
print('However, there are several important considerations:')
print()
print('1. STATISTICAL SIGNIFICANCE: With n=409, even small correlations')
print('   can be statistically significant (which this is).')
print()
print('2. PRACTICAL SIGNIFICANCE: The question is whether this relationship')
print('   is meaningful in the real world, not just statistically significant.')
print()
print('3. CONTEXT: xERA is a very specific metric focused on contact quality.')
print('   Stuff+ also measures pitch quality, so we might expect higher correlation.')
print()
print('4. POSSIBLE REASONS FOR WEAK CORRELATION:')
print('   - xERA includes factors beyond pure stuff (defense, luck, etc.)')
print('   - Our 4-day sample might not be representative')
print('   - xERA might be more influenced by pitch mix than raw stuff')
print('   - Measurement error in either metric')
print()
print('5. VALUE PROPOSITION: Even weak correlations can be valuable if:')
print('   - They are consistent across different samples')
print('   - The metric is easy to compute and interpret')
print('   - They provide unique insights not captured by other metrics')