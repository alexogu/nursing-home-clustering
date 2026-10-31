
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Settings
k = 4
random_state = 42

os.makedirs("figures", exist_ok=True)
os.makedirs("output", exist_ok=True)

# Load the data
df = pd.read_csv("data/NH_ProviderInfo.csv", encoding="latin-1", low_memory=False)

print("Facilities loaded:", len(df))

# Rename columns
df = df.rename(columns={
    "Provider Name": "name",
    "City/Town": "city",
    "State": "state",
    "Urban": "urban",
    "Ownership Type": "ownership",
    "Number of Certified Beds": "beds",
    "Average Number of Residents per Day": "residents",
    "Continuing Care Retirement Community": "ccrc",
    "Chain Name": "chain",
    "Overall Rating": "overall",
    "Health Inspection Rating": "health",
    "QM Rating": "qm",
    "Reported Total Nurse Staffing Hours per Resident per Day": "nurse_hours",
    "Reported RN Staffing Hours per Resident per Day": "rn_hours",
    "Total nursing staff turnover": "turnover",
    "Total Weighted Health Survey Score": "health_score",
    "Total Amount of Fines in Dollars": "fines"
})

# Clean the data
df = df.dropna(subset=["beds", "residents"])
df = df[df["beds"] >= 10]

df["occupancy"] = df["residents"] / df["beds"]
df = df[df["occupancy"] <= 1.10]
df["occupancy"] = df["occupancy"].clip(upper=1.0)

print("Facilities after cleaning:", len(df))

# Create new columns
df["log_beds"] = np.log(df["beds"])
df["log_residents"] = np.log(df["residents"].clip(lower=1))
df["log_health"] = np.log1p(df["health_score"])
df["log_fines"] = np.log1p(df["fines"])

# Choose the features for clustering
features = [
    "occupancy", "log_beds", "log_residents",
    "nurse_hours", "rn_hours", "turnover",
    "overall", "health", "qm", "log_health", "log_fines"
]

df = df.dropna(subset=features).reset_index(drop=True)

# Scale the data
scaler = StandardScaler()
X = scaler.fit_transform(df[features])

# Test different numbers of clusters
inertia = []
silhouette = []
k_values = range(2, 11)

sample = np.random.RandomState(random_state).choice(
    len(X), min(5000, len(X)), replace=False
)

for number in k_values:
    model = KMeans(n_clusters=number, n_init=10, random_state=random_state)
    model.fit(X)

    inertia.append(model.inertia_)
    silhouette.append(silhouette_score(X[sample], model.labels_[sample]))

    print("Clusters:", number)
    print("Inertia:", round(model.inertia_, 2))
    print("Silhouette:", round(silhouette[-1], 3))

# Save cluster test results
results = pd.DataFrame({
    "k": list(k_values),
    "inertia": inertia,
    "silhouette": silhouette
})

results.to_csv("output/k_selection.csv", index=False)

# Plot elbow and silhouette graphs
plt.figure(figsize=(11, 4))

plt.subplot(1, 2, 1)
plt.plot(k_values, inertia, marker="o")
plt.title("Elbow Plot")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(k_values, silhouette, marker="o")
plt.title("Silhouette Score")
plt.xlabel("Number of clusters")
plt.ylabel("Score")
plt.grid()

plt.tight_layout()
plt.savefig("figures/fig1_k_selection.png")
plt.close()

# Create the final clusters
model = KMeans(n_clusters=k, n_init=20, random_state=random_state)
df["cluster"] = model.fit_predict(X)

# Put clusters in order based on occupancy
order = df.groupby("cluster")["occupancy"].mean().sort_values().index
cluster_numbers = {old: new for new, old in enumerate(order)}

df["cluster"] = df["cluster"].map(cluster_numbers)

# Get average information for each cluster
profile = df.groupby("cluster")[[
    "occupancy", "beds", "residents", "nurse_hours",
    "rn_hours", "turnover", "overall", "health",
    "qm", "health_score", "fines"
]].mean().round(2)

profile["number_of_facilities"] = df.groupby("cluster").size()

profile["for_profit"] = df.groupby("cluster")["ownership"].apply(
    lambda x: (x.str.startswith("For profit")).mean() * 100
).round(1)

profile["nonprofit"] = df.groupby("cluster")["ownership"].apply(
    lambda x: (x.str.startswith("Non profit")).mean() * 100
).round(1)

profile["urban"] = df.groupby("cluster")["urban"].apply(
    lambda x: (x == "Y").mean() * 100
).round(1)

profile["ccrc"] = df.groupby("cluster")["ccrc"].apply(
    lambda x: (x == "Y").mean() * 100
).round(1)

profile["chain"] = df.groupby("cluster")["chain"].apply(
    lambda x: x.notna().mean() * 100
).round(1)

profile["empty_beds"] = (df["beds"] - df["residents"]).groupby(df["cluster"]).sum()

profile.to_csv("output/cluster_profiles.csv")

print("\nCluster Profiles:")
print(profile)

# Compare clusters by state
state_mix = pd.crosstab(
    df["state"], df["cluster"], normalize="index"
)

state_mix["n"] = df["state"].value_counts()
state_mix.to_csv("output/state_cluster_mix.csv")

# Find three example facilities in each cluster
examples = []

for cluster in range(k):

    center = X[df["cluster"] == cluster].mean(axis=0)
    distances = np.linalg.norm(X - center, axis=1)

    distances[df["cluster"] != cluster] = np.inf

    closest = np.argsort(distances)[:3]

    for i in closest:
        row = df.loc[i]

        examples.append({
            "cluster": cluster,
            "name": row["name"],
            "city": row["city"],
            "state": row["state"],
            "ownership": row["ownership"],
            "beds": row["beds"],
            "residents": row["residents"],
            "occupancy": round(row["occupancy"], 2),
            "overall": row["overall"],
            "nurse_hours": row["nurse_hours"],
            "turnover": row["turnover"]
        })

examples = pd.DataFrame(examples)
examples.to_csv("output/cluster_examples.csv", index=False)

print("\nExample Facilities:")
print(examples)

# Save the full dataset
df.to_csv("output/facilities_with_clusters.csv", index=False)

# Create cluster heatmap
cluster_data = pd.DataFrame(X, columns=features)
cluster_data["cluster"] = df["cluster"]

average = cluster_data.groupby("cluster").mean()

plt.figure(figsize=(11, 5))
plt.imshow(average, cmap="RdBu_r", aspect="auto", vmin=-1.2, vmax=1.2)

plt.xticks(range(len(features)), features, rotation=45, ha="right")
plt.yticks(range(k), ["Cluster 0", "Cluster 1", "Cluster 2", "Cluster 3"])

plt.colorbar(label="Difference from average")
plt.title("Cluster Differences")

plt.tight_layout()
plt.savefig("figures/fig2_cluster_heatmap.png")
plt.close()

# Compare occupancy between clusters
plt.figure(figsize=(7, 5))

data = [
    df[df["cluster"] == c]["occupancy"] * 100
    for c in range(k)
]

plt.boxplot(data, showfliers=False)
plt.xticks(range(1, k + 1), ["Cluster 0", "Cluster 1", "Cluster 2", "Cluster 3"])
plt.ylabel("Occupancy (%)")
plt.title("Occupancy by Cluster")
plt.grid(axis="y")

plt.tight_layout()
plt.savefig("figures/fig3_occupancy.png")
plt.close()

# Compare ownership between clusters
ownership = pd.crosstab(
    df["cluster"],
    df["ownership"].str.split(" - ").str[0],
    normalize="index"
) * 100

ownership.plot(kind="bar", stacked=True, figsize=(8, 5))

plt.title("Ownership by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Percentage of Facilities")
plt.tight_layout()
plt.savefig("figures/fig3_ownership.png")
plt.close()

# Compare states with the fullest facilities
fullest = state_mix[state_mix["n"] >= 40][k - 1].sort_values()

states = pd.concat([fullest.head(8), fullest.tail(8)])

plt.figure(figsize=(10, 5))
plt.bar(states.index, states.values * 100)

plt.title("States with the Fewest and Most Full Facilities")
plt.xlabel("State")
plt.ylabel("Facilities in Cluster 3 (%)")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("figures/fig4_states.png")
plt.close()

print("\nDone!")
print("Charts saved in figures/")
print("Tables saved in output/")
