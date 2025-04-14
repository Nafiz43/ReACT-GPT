"""
Developed at DECAL Lab in CS Department @ UC Davis by Nafiz Imtiaz Khan (nikhan@ucdavis.edu)
Copyright © 2025 The Regents of the University of California, Davis campus. All Rights Reserved. Used with permission.
"""
import os
import logging
import click
import pandas as pd
import csv
from datetime import datetime
from langchain_ollama import OllamaLLM as Ollama
import json
import re
from _constant_func import *

# import boto3


# data = pd.read_csv('data/Labeled/labels_v2.csv')
# data = pd.read_csv('data/ground_truth_data.csv')
data = pd.read_csv('local_history/CategoryAssignment-CoT0llama3.3:70b-12025-04-10 02:46.csv')
total_report_count = len(data)
# data = data[83:]


prompt = """
You are given an actionable recommendation related to OSS project sustainability.

Your task is to identify which features from the list below would likely be impacted by this actionable.

Each feature captures a specific aspect of socio-technical activity in open-source software projects:

- "s_avg_clustering_coef": Measures how interconnected a developer's immediate social contacts are within the social network. Higher values indicate tighter community cohesion.
- "s_net_overlap": Number of developers who remain consistently active in the social network over time. Reflects long-term social engagement.
- "t_num_dev_nodes": Number of unique developers participating in technical contributions during a given period. Indicates the scale of technical involvement.
- "t_num_dev_per_file": Average number of developers contributing to each file. Reflects collaborative intensity and shared ownership of code.
- "t_graph_density": Density of the technical network. Higher values suggest more collaboration via shared files between developers.
- "t_net_overlap": Number of developers who remain consistently active in technical work across time. Reflects technical contributor stability.
- "st_num_dev": Number of developers who are active in both social and technical networks. Indicates integration of communication and contribution—a known driver of sustainability.

Now, based on the actionable provided below, identify which features it is most likely to influence. List the feature names only, separated by commas.

ACTIONABLE:
"""


cleaning_prompt = """
You will be given a response that may contain noisy or unrelated text.

Your task is to extract and return only the names of the relevant features mentioned in the response. Use the exact feature names listed below and ignore any other text or variations:

Valid feature names:
- 's_avg_clustering_coef'
- 't_num_dev_nodes'
- 't_num_dev_per_file'
- 't_graph_density'
- 'st_num_dev'
- 't_net_overlap'

Return your answer as a comma-separated list of the exact feature names found in the response. Do not include any explanations or extra text.
"""




@click.command()
@click.option(
    "--model_name",
    default="llama3.1:latest",
    type=click.Choice(allowable_models),
    help="model type, llama, mistral or non_llama",
)

@click.option(
    "--temp",
    default=0,
    type=int,
    help="Randomness of the model",
)

@click.option(
    "--prompting_method",
    default="IP",
    type=click.Choice(allowable_prompting_methods),
    help="here goes the prompting methods",
)


@click.option(
    "--reports_to_process", 
    default=-1,  # Default value
    type=int, 
    help="An extra integer to be passed via command line"
)

def main(model_name, prompting_method, reports_to_process, temp):
    print(f"Received model_name: {model_name}")
    print(f"Received value for actionable_to_process: {reports_to_process}")
    print(f"Received value for prompting method: {prompting_method}")
    print(f"Received value for Temperature: {temp}")


    # if(prompting_method =="IP"):
    #     question = IP_template
    # elif(prompting_method=="CoT"):
    #     question = CoT_template
    # elif(prompting_method=="RA"):
    #     question = RA_template



    global data 
    global total_report_count 

    if(reports_to_process > 0):
        data = data.head(reports_to_process)
        total_report_count = reports_to_process
        print(f"Processing only {reports_to_process} actionables")
    else:
        print(f"Processing {total_report_count} actionables")


    # Your existing logic to handle logging
    log_dir, log_file = "local_history", f'{"Feature-Assignment-"+prompting_method+str(temp)+model_name+str(reports_to_process)+datetime.now().strftime("%Y-%m-%d %H:%M")}.csv'
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, log_file)

    if not os.path.isfile(log_path):
        with open(log_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["feature-assignment-timestamp", "article_title", "article_link", "article_venue", "recommendation","positive_impact", "evidence", "confidence_score", "category", "features"])

    cnt = 1
    # print(questions)

    for index, row in data.iterrows():
        print("Actionable Number:", cnt, end="\r")

        query = prompt+row['recommendation']
        ollama = Ollama(model=model_name, temperature=temp)
        logging.getLogger().setLevel(logging.ERROR)  
        response = ollama.invoke(query)

        response = ollama.invoke(cleaning_prompt+response)
         
        # row['category'] = response 
        # response = dummy_response
        # response = clean_response(response)
        
        with open(log_path, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow([timestamp, row['article_title'], row['article_link'], row['article_venue'], row['recommendation'], row['positive_impact'], row['evidence'], row['confidence_score'], row['category'], response])

                print("Actionables Processing Completed", cnt, end="\r")

        cnt = cnt+1

    print("\nTotal Actionables Processed", len(data))


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
