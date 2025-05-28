"""
Reformats xlam-function-calling-60k dataset to be compatible with the MoE-PEFT format.
"""
from datasets import load_dataset
import json

ds = load_dataset("Salesforce/xlam-function-calling-60k", split="train")

list_data = []
count = 0
for i in ds:
    # Limit to roughly half dataset
    if count >= 5000: 
        break
    count += 1
    tools = i["tools"]
    prompt = f"""You are a helpful assistant with access to the following functions to help
you answer queries from the user. Based on the user's query, use the functions if required.

Some rules you always follow:
1. You always respond with valid code: fn_name(arg=value)
2. If the function is not relevant to answer the query, notify the user why.
3. If no function is provided, say you don't have access to any functions.

functions:
{tools}
"""    
    list_data.append({"instruction": prompt, "input": i["query"], "output": i["answers"]})


with open("datasets/recast/xlam.json", "w") as f:
    json.dump(list_data, f)
