import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# Font size configuration for better readability
FONT_SIZES = {
    'title': 24,
    'axes': 20,
    'ticks': 18,
    'legend': 16,
    'annotation': 18,  # For correlation values
    'colorbar': 16     # For colorbar ticks
}

# Load the CSV file
file_path = "./data/GM_NASA_TLX.csv"
df = pd.read_csv(file_path)

# Rename columns for clarity
df.rename(columns={
    "Q2": "Task",
    "Q3_1": "Mental Demand",
    "Q3_2": "Physical Demand",
    "Q3_3": "Temporal Demand",
    "Q3_4": "Own Performance",
    "Q3_5": "Effort",
    "Q3_6": "Frustration Level"
}, inplace=True)

# Define question columns and their shortened labels
question_cols = ["Mental Demand", "Physical Demand", "Temporal Demand", "Own Performance", "Effort", "Frustration Level"]
short_labels = ["Mental", "Physical", "Temporal", "Performance", "Effort", "Frustration"]

# Create a mapping dictionary for the shortened labels
label_map = dict(zip(question_cols, short_labels))

# Calculate cognitive load score (sum of all dimensions except "Own Performance")
cognitive_load_cols = [col for col in question_cols if col != "Own Performance"]
df["Cognitive Load Score"] = df[cognitive_load_cols].sum(axis=1)

# Add the new column to our lists
question_cols.append("Cognitive Load Score")
short_labels.append("Cog Load")
label_map["Cognitive Load Score"] = "Cog Load"

# Calculate mean and std of the cognitive load score
cog_load_mean = df["Cognitive Load Score"].mean()
cog_load_std = df["Cognitive Load Score"].std()
print(f"Cognitive Load Score - Mean: {cog_load_mean:.2f}, Std: {cog_load_std:.2f}")

# Set up the figure size and style for histograms
plt.figure(figsize=(12, 8))
plt.suptitle("Score Distributions for NASA TLX Questions", fontsize=FONT_SIZES['title'])

# Plot histograms for each question
df[question_cols].hist(bins=20, figsize=(12, 8), layout=(3, 3))
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Show histograms
plt.savefig("./temp_assets/tlx/tlx_histograms.png")
plt.show()

# Boxplots of TLX scores grouped by task
plt.figure(figsize=(12, 8))
df_melted = df.melt(id_vars=["Task"], value_vars=question_cols, var_name="Question", value_name="Score")
# Using a custom color palette
custom_palette = sns.color_palette("viridis", len(df["Task"].unique()))
sns.boxplot(x="Question", y="Score", hue="Task", data=df_melted, palette=custom_palette)
plt.xticks(rotation=45, fontsize=FONT_SIZES['ticks'])
plt.yticks(fontsize=FONT_SIZES['ticks'])
plt.title("NASA TLX Scores by Task", fontsize=FONT_SIZES['title'])
plt.legend(title="Task", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=FONT_SIZES['legend'])
plt.tight_layout()
plt.savefig("./temp_assets/tlx/tlx_boxplots.png", dpi=300)
plt.show()

# Mean and standard deviation of scores for each task
task_stats = df.groupby("Task")[question_cols].agg(["mean", "std"])

# Display the results in a structured table
print(task_stats.round(2).to_string())

# Correlation heatmap with improved font sizes and shortened labels
plt.figure(figsize=(12, 10))  # Made taller for better label visibility

# Create a copy of the dataframe with renamed columns for the heatmap
df_short = df[question_cols].copy()
df_short.columns = [label_map[col] for col in df_short.columns]

# Create the heatmap with larger annotation font size
heatmap = sns.heatmap(df_short.corr(), 
                     annot=True, 
                     cmap="coolwarm", 
                     fmt=".2f",
                     annot_kws={"size": FONT_SIZES['annotation']})

# Get the colorbar and modify its font size
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=FONT_SIZES['colorbar'])

# Set title with larger font size
plt.title("Task Correlation of NASA-TLX Question Responses", fontsize=FONT_SIZES['title'], pad=20)

# Adjust tick label font sizes
plt.xticks(fontsize=FONT_SIZES['ticks'])
plt.yticks(fontsize=FONT_SIZES['ticks'])

# Add more padding around the figure for the labels
plt.tight_layout()
plt.savefig("./assets/tlx_correlation_heatmap.png", bbox_inches="tight", dpi=300)
plt.show()