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
data = pd.read_csv('data/article_data.csv')
total_report_count = len(data)
# data = data[83:]



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
    print(f"Received value for reports_to_process: {reports_to_process}")
    print(f"Received value for prompting method: {prompting_method}")
    print(f"Received value for Temperature: {temp}")


    if(prompting_method =="IP"):
        question = IP_template
    elif(prompting_method=="CoT"):
        question = CoT_template
    elif(prompting_method=="RA"):
        question = RA_template



    global data 
    global total_report_count 

    if(reports_to_process > 0):
        data = data.head(reports_to_process)
        total_report_count = reports_to_process
        print(f"Processing only {reports_to_process} reports")
    else:
        print(f"Processing {total_report_count} reports")


    # Your existing logic to handle logging
    log_dir, log_file = "local_history", f"{prompting_method+str(temp)+model_name+str(reports_to_process)+datetime.now().strftime('%Y-%m-%d %H:%M')}.csv"
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, log_file)

    if not os.path.isfile(log_path):
        with open(log_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "venue", "article_title", "article_link", "answer","model_name"])

    cnt = 1
    # print(questions)

    for index, row in data.iterrows():
        print("Article Number:", cnt, end="\r")

        query = prompt_template+row['Text']+question
        ollama = Ollama(model=model_name, temperature=temp)
        logging.getLogger().setLevel(logging.ERROR)  
        response = ollama.invoke(query)
        # response = dummy_response
        response = clean_response(response)
        
        with open(log_path, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow([timestamp, row['File-path'], extract_article_title(row['Text']), row['Link'], response, model_name])
                print("Article Processing Completed", cnt, end="\r")

        cnt = cnt+1

    print("\nTotal Reports Processed", len(data))


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
