"""
Data Exploration Script - Windows Compatible
--------------------------------------------
Understanding ECG signals before processing!

WHY THIS MATTERS:
Before applying DSP or ML, you MUST understand your data:
- What does a normal heartbeat look like?
- How do abnormal beats differ?
- What noise patterns exist?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================
# IMPORTANT: UPDATE THIS PATH!
# ============================================
# Paste the path from download_data.py output here
# Example: r"C:\Users\YourName\.cache\kagglehub\datasets\shayanfazeli\heartbeat\versions\2"
DATA_PATH = r"C:\Users\ishan\.cache\kagglehub\datasets\shayanfazeli\heartbeat\versions\1"  # The 'r' makes it a raw string (handles backslashes)

# Verify path exists
if not os.path.exists(DATA_PATH):
    print(f"❌ ERROR: Path not found: {DATA_PATH}")
    print(f"Please update DATA_PATH with the output from download_data.py")
    exit(1)

# Load training data
train_file = os.path.join(DATA_PATH, "mitbih_train.csv")
print(f"Loading: {train_file}")
train_df = pd.read_csv(train_file, header=None)

print("=" * 60)
print("DATASET STATISTICS")
print("=" * 60)
print(f"Total samples: {len(train_df)}")
print(f"Columns: {train_df.shape[1]}")
print(f"  - Signal points: {train_df.shape[1] - 1}")
print(f"  - Label column: 1")

# Last column is the label
labels = train_df.iloc[:, -1]
print(f"\nClass distribution:")
for class_id in range(5):
    count = (labels == class_id).sum()
    percentage = 100 * count / len(labels)
    print(f"  Class {class_id}: {count:6d} samples ({percentage:5.2f}%)")

# Extract signals and labels
signals = train_df.iloc[:, :-1].values  # All columns except last
labels = train_df.iloc[:, -1].values    # Last column

print(f"\nSignal shape: {signals.shape}")
print(f"Labels shape: {labels.shape}")

# ============================================
# CRITICAL OBSERVATION TASK FOR YOU!
# ============================================
# Before running the visualization, answer this:
# 
# Looking at the class distribution above:
# Q1: Is this a balanced dataset? 
# Q2: Which class has the most samples?
# Q3: Why might imbalanced classes be a problem for ML?
#
# Think about it, then continue...
# ============================================

input("\nPress Enter to continue to visualization...")

# ============================================
# VISUALIZATION: Normal vs Abnormal Heartbeats
# ============================================

fig, axes = plt.subplots(5, 3, figsize=(15, 12))
fig.suptitle('ECG Heartbeat Patterns by Class', fontsize=16, fontweight='bold')

class_names = [
    'Normal',
    'Supraventricular',
    'Premature Ventricular',
    'Fusion',
    'Unclassifiable'
]

for class_id in range(5):
    # Get 3 random samples from this class
    class_indices = np.where(labels == class_id)[0]
    random_samples = np.random.choice(class_indices, 3, replace=False)
    
    for i, sample_idx in enumerate(random_samples):
        ax = axes[class_id, i]
        signal = signals[sample_idx]
        
        # Time axis (assuming 125 Hz sampling rate)
        # WHY 125 Hz? This is the sampling rate of the MIT-BIH database
        # Nyquist theorem: Must sample at 2x highest frequency of interest
        # Heart signals go up to ~40 Hz, so 125 Hz is sufficient
        sampling_rate = 125  # Hz
        time = np.arange(len(signal)) / sampling_rate  # Convert to seconds
        
        ax.plot(time, signal, linewidth=1.5, color='steelblue')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time (seconds)', fontsize=9)
        
        if i == 0:
            ax.set_ylabel(f'Class {class_id}\n{class_names[class_id]}\nAmplitude', 
                         fontweight='bold', fontsize=9)
        
        if class_id == 0:
            ax.set_title(f'Sample {i+1}', fontsize=10)

plt.tight_layout()
output_file = 'ecg_heartbeat_patterns.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✓ Visualization saved as '{output_file}'")
print(f"✓ Opening image...")
plt.show()

# ============================================
# YOUR OBSERVATION TASKS:
# ============================================
print("\n" + "=" * 60)
print("OBSERVATION TASKS - Write down your answers!")
print("=" * 60)
print("""
Look at the plot and answer:

1. PATTERN RECOGNITION:
   - What does a "normal" heartbeat (Class 0) look like?
   - Describe the shape in words (peaks, valleys, duration)

2. ABNORMALITY DETECTION:
   - How does Class 2 (Premature Ventricular) differ from Class 0?
   - Which feature is most different? (amplitude? width? shape?)

3. NOISE OBSERVATION:
   - Do you see any high-frequency jitter/wiggle in the signals?
   - Are all samples perfectly smooth?

4. VARIABILITY:
   - Within the same class, do all 3 samples look identical?
   - What varies? (amplitude? baseline? exact shape?)

5. CHALLENGE AHEAD:
   - Based on what you see, what will make ML classification difficult?
   - Which classes look most similar to each other?
""")