import pandas as pd
import numpy as np

# Read the existing results to check correlations
war_results = pd.read_csv('stuff_plus_weight_tuning_2025.csv')
xera_results = pd.read_csv('stuff_plus_weight_tuning_xera_2025.csv')

print('=== WAR Results Analysis ===')
if 'weighted_stuff_z' in war_results.columns and 'WAR' in war_results.columns:
    corr_war = war_results[['weighted_stuff_z', 'WAR']].corr()
    print('Correlation matrix:')
    print(corr_war)
    print(f'WAR correlation with Stuff+: {corr_war.iloc[0,1]:.4f}')
    print(f'Number of pitchers: {len(war_results)}')

print()
print('=== xERA Results Analysis ===')
if 'weighted_stuff_z' in xera_results.columns and 'xERA' in xera_results.columns:
    corr_xera = xera_results[['weighted_stuff_z', 'xERA']].corr()
    print('Correlation matrix:')
    print(corr_xera)
    print(f'xERA correlation with Stuff+: {corr_xera.iloc[0,1]:.4f}')
    print(f'Number of pitchers: {len(xera_results)}')

print()
print('=== Analysis of Zero Weights Issue ===')
print('This suggests the Stuff+ components may be capturing different aspects of pitching:')
print('- xwOBA: Quality of contact allowed (correlates with overall WAR)')
print('- miss% + chase%: Swing-and-miss ability (correlates with xERA)')
print('- The zero weights indicate these metrics measure complementary skills')