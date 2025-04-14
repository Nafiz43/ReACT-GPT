"""
Developed at DECAL Lab in CS Department @ UC Davis by Nafiz Imtiaz Khan (nikhan@ucdavis.edu)
Copyright © 2025 The Regents of the University of California, Davis campus. All Rights Reserved. Used with permission.
"""
import pandas as pd
import json 
import numpy as np
import os


df = pd.read_csv('data/react_set.csv')
df['importance'] = df['features'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)

df

# Sample DataFrame
# df = pd.read_csv('your_file.csv')  # Uncomment and modify this as needed

# Define the mapping
def row_to_custom_json(row):
    return {
        "article": {
            "title": row["article_title"],
            "link": row["article_link"],
            "venue": row["article_venue"]
        },
        "recommendation": row["recommendation"],
        "positive_impact": row["positive_impact"],
        "evidence": row["evidence"],
        "confidence_score": row["confidence_score"],
        "category": row["category"],
        "Features": row["features"],
        "importance": row["importance"]
    }

# Apply transformation
custom_json_list = df.apply(row_to_custom_json, axis=1).tolist()

# If you want to save it
import json
with open("data/updated_react_set.json", "w") as f:
    json.dump(custom_json_list, f, indent=4)
custom_json_list