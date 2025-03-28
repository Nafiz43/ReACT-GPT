import pandas as pd
import ast

# Load the CSV file
df = pd.read_csv('local_history/IP0llama3.3:70b-12025-03-28 04:34.csv')

# Prepare a list to store structured data
structured_data = []

# Iterate over rows in the DataFrame
for _, row in df.iterrows():
    article_title = row['article_title']
    
    # Safely parse the answer column (which contains a dictionary)
    try:
        answer_dict = ast.literal_eval(row['answer'])  # Convert string to dictionary
        recommendations = answer_dict.get('recommendations', [])
        
        # Extract relevant fields from each recommendation
        for rec in recommendations:
            structured_data.append({
                'article_title': article_title,
                'recommendation': rec.get('recommendation', ''),
                'positive_impact': rec.get('positive_impact', 'NO IMPACT FOUND'),
                'evidence': rec.get('evidence', 'NO EVIDENCE FOUND'),
                'confidence_score': rec.get('confidence', '')
            })
    except Exception as e:
        print(f"Error parsing row: {e}")

# Create a new DataFrame
output_df = pd.DataFrame(structured_data)

# Save the structured data to a new CSV file
output_df.to_csv('data/structured_output.csv', index=False)

print("Data transformation complete. Output saved to 'structured_output.csv'.")
