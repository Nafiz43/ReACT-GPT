"""
Developed at DECAL Lab in CS Department @ UC Davis by Nafiz Imtiaz Khan (nikhan@ucdavis.edu)
Copyright © 2025 The Regents of the University of California, Davis campus. All Rights Reserved. Used with permission.
"""
import json
import pandas as pd

json_data = []

# Function to filter JSON data based on the feature list
def filter_json_by_features(data, features):
    # print("data printing: ", data)
    filtered_data = []
    for entry in data:
        entry_features =  set(map(str.strip, str(entry.get("Features", "") if not pd.isna(entry.get("Features", "")) else "").split(",")))
        if any(feature in entry_features for feature in features):
            filtered_data.append(entry)
    return filtered_data

def calculate_feature_differences(df, month_n, feature_list):
    avg_feature_values = {}
    for feature in feature_list:
        avg_feature_values[feature] = df[feature].mean()
        # print("Average", avg_feature_values)

    # Calculate the monthly feature values for the month n-1 to n+1
    monthly_feature_values = {}
    for feature in feature_list:
        # Filter the dataframe for the relevant months (n-1 to n+1)
        monthly_values = df[(df['month'] >= month_n-2) & (df['month'] <= month_n)][feature]
        monthly_feature_values[feature] = monthly_values.sum()  # Summing the monthly values
        # print("Monthly", monthly_feature_values)

    # Calculate the difference between monthly_feature_values and average_feature_values
    differences = {}
    for feature in feature_list:
        differences[feature] = monthly_feature_values[feature] - avg_feature_values[feature]

    # Get the features with differences <= 0
    negative_or_zero_features = [feature for feature, diff in differences.items() if diff <= 0]

    return negative_or_zero_features




def ReACT_Extractor(project_name, original_data, feature_data: pd.DataFrame, month_n: int) -> dict:
    feature_list = ['s_avg_clustering_coef', 't_num_dev_nodes', 't_num_dev_per_file', 't_graph_density', 'st_num_dev', 't_net_overlap'] #features impacting positive outcomes
    processed_features = calculate_feature_differences(feature_data, month_n, feature_list) 
    # print(processed_features)
    filtered_json = filter_json_by_features(original_data, processed_features)
    sorted_json = sorted(filtered_json, key=lambda x: x["importance"], reverse=True)
    # print(sorted_json)

    nested_output = {
        project_name: {
            str(month_n): sorted_json
        }
    }


    return nested_output

reacts_to_recommend = {}
with open("data/updated_react_set.json", 'r') as json_file:
    original_data = json.load(json_file)

feature_data = pd.read_csv('data/clean-apache-network-data-1-1.csv')
x = feature_data.groupby('proj_name')

for proj_name, group_df in x:
    # print(f"Project: {proj_name}")
    # print(len(group_df))
    
    for i in range(len(group_df)):
        # print(f"Month: {i}")
        react_output = ReACT_Extractor(proj_name, original_data, group_df, i)
        # print(react_output)
        
        # Safe nested merge
        for project, months in react_output.items():
            if project not in reacts_to_recommend:
                reacts_to_recommend[project] = {}

            for month, entries in months.items():
                if month not in reacts_to_recommend[project]:
                    reacts_to_recommend[project][month] = []

                reacts_to_recommend[project][month].extend(entries)  # append entries

with open("data/extracted_react_apache.json", 'w') as json_file:
    json.dump(reacts_to_recommend, json_file, indent=4)

print("All ReACTs saved across projects and months!")

