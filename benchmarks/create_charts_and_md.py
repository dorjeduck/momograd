import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the combined benchmarks data
df = pd.read_csv('results/benchmarks_comparison.csv')

# Set the plot style
plt.style.use('seaborn-darkgrid' if 'seaborn-darkgrid' in plt.style.available else 'classic')

# Create a new figure and set its size
fig, ax = plt.subplots(figsize=(12, 7))

# Plotting
n_groups = len(df)
index = np.arange(n_groups)
bar_width = 0.35

# Plotting Py bars and text annotations
bars_micrograd = ax.bar(index - bar_width/2, df['time_micrograd'], bar_width, label='micrograd', alpha=0.8)
# Plotting Mojo bars and text annotations
bars_momograd = ax.bar(index + bar_width/2, df['time_momograd'], bar_width, label='momograd', alpha=0.8, color='orange')

# Function to add text annotations above bars with increased distance
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        # Increase the vertical distance; adjust the multiplier as needed
        vertical_distance = 3  # Increase or adjust as needed
        ax.text(bar.get_x() + bar.get_width() / 2., height + vertical_distance,
                 '{:.2f}'.format(height), ha='center', va='bottom', rotation=90)


# Adding labels and title
ax.set_xlabel('samples')
ax.set_ylabel('Time (s)')
ax.set_title('Time Comparison micrograd / momograd')

# Set the x-ticks to be the indexes with n_samples as labels
ax.set_xticks(index)
ax.set_xticklabels(df['n_samples'].astype(str))

# Adjust the x-axis limits to reduce the gap between the y-axis and the first bar
ax.set_xlim(-0.5, n_groups - 0.5)

# Add value labels
add_value_labels(bars_micrograd)
add_value_labels(bars_momograd)

# Place the legend in the upper left corner of the plot
ax.legend(loc='upper left')

# Adjust y-axis limit to accommodate text annotations
ymax = df[['time_momograd', 'time_micrograd']].max().max()  # Find the maximum value
y_limit_buffer = 30  
ax.set_ylim(0, ymax + y_limit_buffer)

plt.tight_layout()

# Save the plot to a file
plt.savefig('charts/chart_time_comparison.png', dpi=300, bbox_inches="tight")
print("Time comparison chart has been saved to 'charts/chart_time_comparison.png'")



# Calculate the speedup factor for each row
df['speedup'] = df['time_micrograd'] / df['time_momograd']

# Create a new figure for the speedup chart
plt.figure(figsize=(10, 6))

# Plotting the speedup factor as a bar chart
plt.bar(df['n_samples'].astype(str), df['speedup'], color='orange', label='Speedup Factor (Py/Mojo)')

# Adding labels and title
plt.xlabel('samples')
plt.ylabel('Speedup Factor (micrograd/momograd)')
plt.title('Speedup of momograd over micrograd by samples')

# Adding x-ticks for clarity
plt.xticks(rotation=45)

plt.tight_layout()

# Save the plot to a file
plt.savefig('charts/chart_speedup_comparison.png', dpi=300)
print("Speedup comparison chart has been saved to 'charts/chart_speedup_comparison.png'")

## markdown table


# Calculate the speedup factor for each row
df['speedup_micro_momo'] = df['time_micrograd'] / df['time_momograd']
df['speedup_micro_momox'] = df['time_micrograd'] / df['time_momogradx']
df['speedup_momo_momox'] = df['time_momograd'] / df['time_momogradx']


# Open a Markdown file to write the table
with open('results/benchmark_results.md', 'w') as md_file:
    # Write the table header
    md_file.write('# Benchmark Results\n\n')
    md_file.write('| samples | micrograd (sec) | momograd (sec) | momograd.x (sec) | speedup micro/momo | speedup micro/momo.x | speedup momo/momo.x |\n')
    md_file.write('| --- | --- |---| --- | --- | ---| --- |\n')
    
    # Iterate over each row in the DataFrame and write the table row
    for index, row in df.iterrows():
        md_file.write(f"| {int(row['n_samples'])} | {row['time_micrograd']:.2f} | {row['time_momograd']:.2f} | {row['time_momogradx']:.2f} | {row['speedup_micro_momo']:.1f}x | {row['speedup_micro_momox']:.1f}x | {row['speedup_momo_momox']:.1f}x |\n")

print("Markdown table has been written to 'results/benchmark_results.md'")

