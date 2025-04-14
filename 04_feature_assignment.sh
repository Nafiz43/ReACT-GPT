# Developed at DECAL Lab in CS Department @ UC Davis by Nafiz Imtiaz Khan (nikhan@ucdavis.edu)
# Copyright © 2025 The Regents of the University of California, Davis campus. All Rights Reserved. Used with permission.
python3 04_assign_features.py --model_name=llama3.3:70b --prompting_method=CoT --reports_to_process=-1  >> log-files/FeatureAssignment-llama3.3_CoT_temp0.txt

python3 04_assign_features.py --model_name=llama3.2:latest --prompting_method=CoT --reports_to_process=-1  >> log-files/FeatureAssignment-llama3.2_CoT_temp0.txt