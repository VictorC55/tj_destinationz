import sys
import os
from bs4 import BeautifulSoup
from openai import OpenAI

# RUN IN CMD/POWERSHELL

# Capture command line arguments
args = sys.argv[1:]

# Make sure to rotate this key if it gets public exposure!
os.environ["OPENROUTER_API_KEY"] = "put key here"

# 1. Configure the OpenRouter Client
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("API key not found. Please set OPENROUTER_API_KEY.")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# Using gpt-4o-mini (supports up to 128k context tokens, perfect for combining multiple files)
MODEL_NAME = "openai/gpt-4o-mini"

# 2. Parse the HTML file
def extract_text_from_html(file_path):
    """Reads an HTML file and extracts the visible text."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'html.parser')
            
            # Remove script and style elements so they don't clutter the text
            for script_or_style in soup(['script', 'style']):
                script_or_style.decompose()
                
            # Extract plain text, separating blocks with a newline
            text = soup.get_text(separator='\n', strip=True)
            return text
    except FileNotFoundError:
        return f"Error: File {file_path} not found."

# ==========================================
# Processing Logic
# ==========================================
def process_all_together():
    # Validation checks for arguments
    if len(args) < 2:
        print("Usage: python script.py <folder_of_html_files> <instruction_text_file>")
        return

    folder_path = args[0]
    instruction_file_path = args[1]

    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a valid directory.")
        return

    if not os.path.isfile(instruction_file_path):
        print(f"Error: Instruction file '{instruction_file_path}' not found.")
        return

    # 1. Read the text instruction from the text file
    print(f"Reading instruction from {instruction_file_path}...")
    with open(instruction_file_path, 'r', encoding='utf-8') as f:
        user_instruction = f.read().strip()

    # 2. Get all HTML files in the folder
    html_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.html')]

    if not html_files:
        print(f"No .html files found in folder '{folder_path}'.")
        return

    print(f"Found {len(html_files)} HTML file(s). Gathering contents...")

    # 3. Print raw contents and combine the text into a single variable
    combined_documents_text = ""
    
    for file_name in html_files:
        full_html_path = os.path.join(folder_path, file_name)
        
        # Clean and append text
        clean_text = extract_text_from_html(full_html_path)
        
        # We wrap each document section with clear headers so the LLM knows where files split
        combined_documents_text += f"\n\nDOCUMENT SOURCE: {file_name}\n"
        combined_documents_text += f"=========================================\n"
        combined_documents_text += f"{clean_text}\n"
        combined_documents_text += f"=========================================\n"

    # 4. Construct the Single Combined Prompt
    print(f"\nSending all document data together to {MODEL_NAME} via OpenRouter...")
    
    messages = [
        {
            "role": "system", 
            "content": "You are a high school college admissions advisor. You will be provided with text extracted from an HTML webpage. Students will ask you for advice regarding school and colleges; use the text from the .html file to guide your answers. Provide reasoning for all your answers. Make sure to use quotes from the provided html files. "
        },
        {
            "role": "user", 
            "content": f"Here is the collected context data from multiple webpages:\n{combined_documents_text}\nBased on all the information listed above, please fulfill this specific request:\n{user_instruction}"
        }
    ]
    
    # Make the single API call
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        extra_headers={
            "HTTP-Referer": "http://localhost:3000", 
            "X-Title": "HTML Batch Analyzer",             
        }
    )
    
    # 5. Output the result
    print("\n--- OpenRouter Combined Response ---")
    print(response.choices[0].message.content)
    print("=" * 50)

def main():
    process_all_together()

if __name__ == "__main__": 
    main()