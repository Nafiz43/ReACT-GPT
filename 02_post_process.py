import pandas as pd
import ast

# Load the CSV file 
df = pd.read_csv('local_history/CoT0llama3.3:70b-12025-04-03 21:04.csv') # Adjust the file path as needed

# Prepare a list to store structured data
structured_data = []

# Iterate over rows in the DataFrame
for _, row in df.iterrows():
    article_title = row['article_title']
    article_venue = row['venue']
    article_link = row['article_link'] if 'article_link' in row else ''  # Handle missing column gracefully
    
    # Safely parse the answer column (which contains a dictionary)
    try:
        answer_dict = ast.literal_eval(row['answer'])  # Convert string to dictionary
        recommendations = answer_dict.get('recommendations', [])
        
        # Extract relevant fields from each recommendation
        for rec in recommendations:
            structured_data.append({
                'article_title': article_title,
                'article_link': article_link, 
                'article_venue': article_venue,
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
