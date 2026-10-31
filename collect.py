
import os
import requests
import pandas as pd

# CMS dataset link
url = "https://data.cms.gov/provider-data/api/1/metastore/schemas/dataset/items/4pq5-n9py"

response = requests.get(url)
data = response.json()

download_url = ""

for file in data["distribution"]:
    if file["downloadURL"].endswith(".csv"):
        download_url = file["downloadURL"]
        break

os.makedirs("data", exist_ok=True)
response = requests.get(download_url)

with open("data/NH_ProviderInfo.csv", "wb") as file:
    file.write(response.content)

print("Dataset downloaded successfully!")
df = pd.read_csv("data/NH_ProviderInfo.csv", encoding="latin-1", low_memory=False)
print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))
print("\nFirst six columns:")
print(df.columns[:6].tolist())
