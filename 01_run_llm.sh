# Developed at DECAL Lab in CS Department @ UC Davis by Nafiz Imtiaz Khan (nikhan@ucdavis.edu)
# Copyright © 2025 The Regents of the University of California, Davis campus. All Rights Reserved. Used with permission.
# This file contains some example run commands
# --reports_to_process=-1 means it will process all the reports in the dataset; 
# provide a valid number to process that many reports

# Models that are finally being considered:
# claude-3.5-haiku (bedrock)
# mixtral:8x7b-instruct-v0.1-q4_K_M 
# llama3.3:70b
# medllama2:latest
# qwen2.5:72b
# thewindmom/llama3-med42-8b:latest

# Prompt Techniqyes:
# Instruction-Prompting
# CoT
# CoT (with Self-Consistency)

# python3 01_run_llm.py --model_name=llama3.2:latest --prompting_method=IP --reports_to_process=-1 

# python3 01_run_llm.py --model_name=medllama2:latest --prompting_method=IP --reports_to_process=-1 >> log-files/medllama2_IP_temp0.txt

# python3 01_run_llm.py --model_name=mixtral:8x7b-instruct-v0.1-q4_K_M --prompting_method=IP --reports_to_process=-1  >> log-files/mixtral_IP_temp0.txt

# python3 01_run_llm.py --model_name=llama3.3:70b --prompting_method=IP --reports_to_process=-1 >> log-files/llama3.3_IP_temp0.txt

# python3 01_run_llm.py --model_name=qwen2.5:72b --prompting_method=IP --reports_to_process=-1 >> log-files/qwen2.5_IP_temp0.txt

# python3 01_run_llm.py --model_name=thewindmom/llama3-med42-8b:latest --prompting_method=IP --reports_to_process=-1 log-files/llama3med42_IP_temp0.txt

# python3 01_run_llm.py --model_name=anthropic.claude-3-5-haiku-20241022-v1:0 --prompting_method=IP --reports_to_process=-1 >> log-files/claude_haiku_IP_temp0.txt



# python3 01_run_llm.py --model_name=medllama2:latest --prompting_method=CoT --reports_to_process=-1 >> log-files/medllama2_CoT_temp0.txt

# python3 01_run_llm.py --model_name=mixtral:8x7b-instruct-v0.1-q4_K_M --prompting_method=CoT --reports_to_process=-1  >> log-files/mixtral_CoT_temp0.txt

# python3 01_run_llm.py --model_name=llama3.3:70b --prompting_method=CoT --reports_to_process=-1  >> log-files/llama3.3_CoT_temp0.txt

# python3 01_run_llm.py --model_name=qwen2.5:72b --prompting_method=CoT --reports_to_process=-1 >> log-files/qwen2.5_CoT_temp0.txt

# python3 01_run_llm.py --model_name=thewindmom/llama3-med42-8b:latest --prompting_method=CoT --reports_to_process=-1 >> log-files/llama3med42_CoT_temp0.txt

# python3 01_run_llm.py --model_name=anthropic.claude-3-5-haiku-20241022-v1:0 --prompting_method=CoT --reports_to_process=-1 >> log-files/claude_haiku_CoT_temp0.txt



# python3 01_run_llm.py --model_name=medllama2:latest --prompting_method=CoT --temp=1 --reports_to_process=-1  >> log-files/medllama2_CoT_temp1.txt

# python3 01_run_llm.py --model_name=mixtral:8x7b-instruct-v0.1-q4_K_M --prompting_method=CoT --temp=1 --reports_to_process=-1  >> log-files/mixtral_CoT_temp1.txt

# python3 01_run_llm.py --model_name=llama3.3:70b --prompting_method=CoT --temp=1 --reports_to_process=-1 >> log-files/llama3.3_CoT_temp1.txt

# python3 01_run_llm.py --model_name=qwen2.5:72b --prompting_method=CoT --temp=1 --reports_to_process=-1 >> log-files/qwen2.5_CoT_temp1.txt

# python3 01_run_llm.py --model_name=thewindmom/llama3-med42-8b:latest --prompting_method=CoT --temp=1 --reports_to_process=-1 >>log-files/llama3med42_CoT_temp1.txt

# python3 01_run_llm.py --model_name=anthropic.claude-3-5-haiku-20241022-v1:0 --prompting_method=CoT --temp=1 --reports_to_process=-1  >> log-files/claude_haiku_CoT_temp1.txt

# Some older trials
# python3 01_run_llm.py --model_name=mixtral:latest --reports_to_process=-1

# python3 01_run_llm.py --model_name=llama3.1:latest --reports_to_process=-1

# python3 01_run_llm.py --model_name=llama3.2:latest --reports_to_process=-1

# python3 01_run_llm.py --model_name=medllama2:latest --reports_to_process=-1

# python3 01_run_llm.py --model_name=meditron:70b --reports_to_process=-1

# python3 01_run_llm.py --model_name=tinyllama --reports_to_process=-1

# python3 01_run_llm.py --model_name=llama3.3:70b --reports_to_process=-1

# python3 01_run_llm.py --model_name=qordmlwls/llama3.1-medical:latest --reports_to_process=-1

# python3 01_run_llm.py --model_name=deepseek-r1:1.5b --reports_to_process=-1

# python3 01_run_llm.py --model_name=deepseek-r1:7b --reports_to_process=-1

# python3 01_run_llm.py --model_name=deepseek-r1:70b --reports_to_process=-1

# python3 01_run_llm.py --model_name=medllama2:latest --reports_to_process=-1

# python3 01_run_llm.py --model_name=mistral-nemo:latest --reports_to_process=-1

# python3 01_run_llm.py --model_name=qordmlwls/llama3.1-medical:latest --reports_to_process=-1



# python3 01_run_llm.py --model_name=llama3.2:latest --reports_to_process=-1

# python3 01_run_llm.py --model_name=meditron:70b --reports_to_process=-1

# python3 01_run_llm.py --model_name=medllama2:latest --reports_to_process=-1

# python3 01_run_llm.py --model_name=tinyllama --reports_to_process=1

# python3 01_run_llm.py --model_name=deepseek-r1:1.5b --reports_to_process=1

# python3 01_run_llm.py --model_name=deepseek-r1:7b --reports_to_process=1

# python3 01_run_llm.py --model_name=thewindmom/llama3-med42-8b:latest --reports_to_process=1 --



# python3 01_run_llm_CoT.py --model_name=llama3.2:latest --reports_to_process=-1





python3 01_run_llm.py --model_name=llama3.3:70b --prompting_method=CoT --reports_to_process=-1  >> log-files/llama3.3_CoT_temp0.txt
