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
            writer.writerow(["timestamp", "article_title", "answer","model_name"])

    cnt = 1
    # print(questions)

    for index, row in data.iterrows():
        print("Article Number:", cnt, end="\r")

        query = prompt_template+row['Text']+question
        ollama = Ollama(model=model_name, temperature=temp)
        logging.getLogger().setLevel(logging.ERROR)  
        # response = ollama.invoke(query)
        response = dummy_response
        response = clean_response(response)
        
        with open(log_path, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow([timestamp, extract_article_title(row['Text']), response, model_name])
                print("Article Processing Completed", cnt, end="\r")

        cnt = cnt+1

        # cleaned_report = clean_radiology_report(row['Report Text'])
        # print(cleaned_report)
        # for i in range(0, 39):
        #     if(prompting_method=="IP"):
        #         query = process_question_for_IP_prompting(cleaned_report, str(questions.iloc[i]['Questions']), i)
        #     elif(prompting_method=="CoT"):
        #         query = process_question_for_CoT_prompting(cleaned_report, str(questions.iloc[i]['Questions']), i)

        #     if(model_name=="anthropic.claude-3-5-haiku-20241022-v1:0"):
        #         import boto3
        #         bedrock_client = boto3.client(service_name='bedrock-runtime', region_name="us-west-2")

        #         ################### BEDROCK CONVERSE###################################
        #         system_prompts = [{"text": "System Prompt"}]
        #         messages = [{
        #                 "role": "user",
        #                 "content": [{"text": ""+query+""}]
        #             }]

        #         inference_config = {"temperature": temp}

        #         res = bedrock_client.converse(
        #                 modelId=model_name,
        #                 messages=messages,
        #                 system=system_prompts,
        #                 inferenceConfig=inference_config
        #                 # additionalModelRequestFields=additional_model_fields
        #             )
        #         response = res['output']['message']['content'][0]['text']
        #         # print(res)

        #         ################### BEDROCK CONVERSE###################################
        #     else:
                

        #     print("Question Number", (i+1), "; MODEL RESPONSE:", response+"\n \n \n")


        #     json_match = re.search(r"\{.*\}", response.strip(), re.DOTALL)
        #     if json_match in [None, ""]:
        #         json_match = {"reason_for_the_label": response, "label": extract_first_binary(response)}
        #     else:
        #         json_match = fix_json(json_match.group(0), response)

        #     if json_match:
        #         json_text = json.dumps(json_match)  # Convert dictionary to JSON string
        #         try:
        #             classification = ClassificationResponse.model_validate_json(json_text)
        #         except Exception as e: 
        #             print(f"An error occurred: {type(e).__name__}: {e}")
        #             print("JSON text:", json_text)
        #             classification = ClassificationResponse(reason_for_the_label=response, label=extract_first_binary(response))
        #     else:
        #         print("No valid JSON found in the response:", response)
        #         classification = ClassificationResponse(reason_for_the_label=response, label=extract_first_binary(response))

        #     # remove_part = 'REPORT: ' + report_to_pass
        #     # query = query.replace(remove_part, "", 1)

            
        
        # cnt = cnt+1
        # progress_percentage = ((index+1) / len(data)) * 100
        # print(f"Processed {index+1}/{len(data)} reports ({progress_percentage:.2f}% complete)")


    # print("\n")
    print("\nTotal Reports Processed", len(data))


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
