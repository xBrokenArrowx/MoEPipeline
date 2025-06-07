"""
Reformats xlam-function-calling-60k dataset to be compatible with the MoE-PEFT format.
"""
from datasets import load_dataset
import json
import argparse
import os

DEFAULT_SIZE = 1000
DEFAULT_PROMPT_LOCATION = './prompts/default.txt'

def main():
    parser = argparse.ArgumentParser('Dataset subseting')
    #Dataset Size
    parser.add_argument(
        '-c',
        '--count',
        type=int,
        help=f'Size of dataset to create (DEFAULT = {DEFAULT_SIZE})',
        default=DEFAULT_SIZE
    )
    #Prompt
    parser.add_argument(
        '-p',
        '--prompt',
        type=str,
        help=f'The prompt file (should be a .txt and has a tools varaible for formatting) [DEFAULT= {DEFAULT_PROMPT_LOCATION}]',
        default=DEFAULT_PROMPT_LOCATION
    )

    #outfile
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        help='Output file name',
        default=f'./datasets/xlam-{DEFAULT_SIZE}.json'
    )

    args = parser.parse_args()
    
    size = args.count
    prompt_file = args.prompt

    #try to open the prompt file
    try:
        with open(prompt_file, 'r') as f:
            prompt = f.read()
    except FileNotFoundError as e:
        print(f'Prompt  File: {prompt_file} unable to be located')
        raise e
    
    print(f'Creating subset with args:\n\tSize: {args.count}\n\tPrompt Location: {args.prompt}\n\tOutput: {args.output}\n\nFull Prompt\n{"-"*20}\n{prompt}')

    
    
    ds = load_dataset("Salesforce/xlam-function-calling-60k", split="train")

    list_data = []
    count = 0
    for i in ds:
        # Limit to roughly half dataset
        if count >= size: 
            break
        count += 1
        tools = i["tools"]
        formatted = prompt.format(tools=tools)
        list_data.append({"instruction": formatted, "input": i["query"], "output": i["answers"]})


    with open(args.output, "w") as f:
        json.dump(list_data, f, indent=4)


if __name__ == "__main__":
    main()