import json
import numpy as np
import matplotlib.pyplot as plt

# Load distances
with open('graph_distances.json', 'r') as f:
    distances = json.load(f)

# Get all unique distances
all_dists = set()
for source, targets in distances.items():
    for target, dist in targets.items():
        all_dists.add(dist)

all_dists = sorted(list(all_dists))
print(f"Unique distances: {all_dists}")
print(f"Max distance: {max(all_dists)}")

# Define penalty functions
def current_penalty(d):
    """Current: 1/(d+1)"""
    return 1 / (d + 1)

def linear_penalty(d, max_d=9):
    """Linear: 1 - d/max_d"""
    return max(0, 1 - d / max_d)

def exponential_penalty(d, decay=0.5):
    """Exponential: exp(-decay*d)"""
    return np.exp(-decay * d)

def gaussian_penalty(d, sigma=2):
    """Gaussian: exp(-d^2/(2*sigma^2))"""
    return np.exp(-d**2 / (2 * sigma**2))

def power_penalty(d, alpha=2):
    """Power law: 1/(d+1)^alpha"""
    return 1 / (d + 1)**alpha

def log_penalty(d):
    """Logarithmic: 1/log(d+2)"""
    return 1 / np.log(d + 2)

def step_penalty(d, threshold=2):
    """Step: 1 if d <= threshold, else 0.1"""
    return 1.0 if d <= threshold else 0.1

# Test functions
distances_test = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

functions = [
    ("Current (1/(d+1))", current_penalty),
    ("Linear (1-d/9)", lambda d: linear_penalty(d, 9)),
    ("Exponential (exp(-0.5*d))", lambda d: exponential_penalty(d, 0.5)),
    ("Gaussian (sigma=2)", lambda d: gaussian_penalty(d, 2)),
    ("Power (alpha=2)", lambda d: power_penalty(d, 2)),
    ("Log (1/log(d+2))", log_penalty),
    ("Step (thresh=2)", lambda d: step_penalty(d, 2)),
]

print("\nPenalty function comparison:")
header = "Distance | " + " | ".join(f"{name[:15]:<15}" for name, _ in functions)
print(header)
print("-" * len(header))

for d in distances_test:
    values = [func(d) for _, func in functions]
    row = f"{d:8d} | " + " | ".join(f"{v:6.3f}" for v in values)
    print(row)

# Analyze what this means for consistency
print("\nInterpretation:")
print("- Higher values = more consistent (closer to center)")
print("- Current method gives high weight even to distance 9 (0.100)")
print("- Linear gives 0 at max distance")
print("- Exponential decays quickly")
print("- Gaussian is smooth but penalizes moderately")
print("- Power law decays faster than current")
print("- Log is slow decay")
print("- Step is binary")

# Suggest based on domain knowledge
print("\nRecommendations:")
print("1. Linear or Gaussian: Smooth penalty, good for continuous pitch variation")
print("2. Exponential: Sharp penalty for 'outlier' pitches")
print("3. Power law: Compromise between current and exponential")
print("4. Consider domain: How 'different' should a pitch at distance 5 be vs distance 1?")

# Visualize
plt.figure(figsize=(12, 8))
for name, func in functions:
    y = [func(d) for d in distances_test]
    plt.plot(distances_test, y, label=name, marker='o')

plt.xlabel('Graph Distance')
plt.ylabel('Consistency Weight')
plt.title('Penalty Function Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('penalty_functions_comparison.png', dpi=150, bbox_inches='tight')
print("Saved plot to penalty_functions_comparison.png")