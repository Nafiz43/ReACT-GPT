"""
Developed at DECAL Lab in CS Department @ UC Davis by Nafiz Imtiaz Khan (nikhan@ucdavis.edu)
Copyright © 2025 The Regents of the University of California, Davis campus. All Rights Reserved. Used with permission.
"""
import re
import json

# Sample response variable
response = """Please present your findings in a structured **JSON format** with the following information for each recommendation:

- **recommendation**: The actionable recommendation extracted from the article.
- **positive_impact**: The expected positive impact of the recommendation (if mentioned in the article).
- **evidence**: Any supporting evidence presented in the article for the recommendation.
- **confidence**: A confidence score between 0 and 1, indicating how confident you are in the extracted recommendation.

Output format:

{
  "recommendations": [
    {
      "recommendation": "<extracted recommendation>",
      "positive_impact": "<extracted impact or 'NO IMPACT FOUND'>",
      "evidence": "<extracted evidence or 'NO EVIDENCE FOUND'>",
      "confidence": "<confidence_score (0 to 1)>"
    },
    {
      "recommendation": "<extracted recommendation>",
      "positive_impact": "<extracted impact or 'NO IMPACT FOUND'>",
      "evidence": "<extracted evidence or 'NO EVIDENCE FOUND'>",
      "confidence": "<confidence_score (0 to 1)>"
    }
  ]
}


skvmdksv
ksdmvksdv

smdvlsdmvlk"""


def clean_response(response):
    json_start_index = response.find("{")
    json_end_index = response.rfind("}")

    # Step 2: Extract the JSON part by slicing the response
    json_text = response[json_start_index:json_end_index + 1]

    json_text = re.sub(r':\s*<([^>]+)>', r': "\1"', json_text)  # Replace <...>
    # print(json_text)

    # Step 3: Remove any extra non-JSON characters that might interfere with the parsing
    # You can clean the string by ensuring that the JSON structure is correctly formatted.
    json_text = re.sub(r'(?<!\\)("[^"]*")\s*:\s*(?=\S)', r'\1: ', json_text)  # Remove extra spaces after colons

    # Step 4: Use regular expression to validate and extract the JSON content
    json_part = re.search(r'\{.*\}', json_text, re.DOTALL)

    # Step 5: Check if a JSON part was found and parse it
    if json_part:
        try:
            extracted_json = json_part.group()  # Get the full JSON content
            parsed_json = json.loads(extracted_json)  # Convert the JSON string into a Python dictionary
            # print(parsed_json)
        except:
            print(f"Error decoding JSON:")
    else:
        parsed_json = response
        print("No valid JSON found in the response.")

    return parsed_json
