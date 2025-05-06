from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter
import random
import shutil
import os

# Define the root directory
root_dir = Path(r'/mnt/eeg/N2')

# Collect all .mat files
mat_files = [mat_file for mat_file in root_dir.glob('*.mat')]

# Extract participant IDs from filenames (e.g., N2 from bdc14_N2_0106.mat)
# Alternative: use f.name[6] for letter (e.g., N, J) if 18th char meant letter
participant_ids = [str(f)[18] for f in mat_files]

# Group files by participant
files_by_participant = {}
for f, pid in zip(mat_files, participant_ids):
    if pid not in files_by_participant:
        files_by_participant[pid] = []
    files_by_participant[pid].append(f)

# Count the number of files per participant
participant_counts = Counter(participant_ids)

# Print original counts
print("Original number of epochs per participant:")
for participant, count in sorted(participant_counts.items()):
    print(f"{participant}: {count}")

# Find the minimum number of epochs
min_epochs = 30
print(f"Minimum number of epochs: {min_epochs}")

# Create N2_small: select min_epochs files per participant
N2_small_files = []
for participant in files_by_participant:
    # Randomly select min_epochs files for this participant
    selected_files = random.sample(files_by_participant[participant], min_epochs)
    N2_small_files.extend(selected_files)

# Count epochs in N2_small for verification
N2_small_counts = Counter([str(f)[18] for f in N2_small_files])

# Print N2_small counts
print("\nNumber of epochs per participant in N2_small:")
for participant, count in sorted(N2_small_counts.items()):
    print(f"{participant}: {count}")

# Save N2_small files to a new directory
N2_small_dir = Path('/mnt/eeg/N2_small')
N2_small_dir.mkdir(exist_ok=True)
for f in N2_small_files:
    shutil.copy(f, N2_small_dir / f.name)

print(f"\nN2_small files saved to: {N2_small_dir}")
print(f"Total files in N2_small: {len(N2_small_files)}")

# Plot original distribution
plt.figure(figsize=(12, 6))
participants = sorted(participant_counts.keys())
epoch_counts = [participant_counts[p] for p in participants]
plt.bar(participants, epoch_counts, color='skyblue', alpha=0.5, label='Original')
plt.xlabel('Participant')
plt.ylabel('Number of Epochs')
plt.title('Original Epochs per Participant')
plt.xticks(rotation=45)
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('parps_original.png')
plt.show()

# Plot N2_small distribution
plt.figure(figsize=(12, 6))
N2_small_participants = sorted(N2_small_counts.keys())
N2_small_epoch_counts = [N2_small_counts[p] for p in N2_small_participants]
plt.bar(N2_small_participants, N2_small_epoch_counts, color='lightgreen', label='N2_small')
plt.xlabel('Participant')
plt.ylabel('Number of Epochs')
plt.title('N2_small Epochs per Participant')
plt.xticks(rotation=45)
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('parps_N2_small.png')
plt.show()