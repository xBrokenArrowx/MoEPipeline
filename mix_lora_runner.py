"""Run a MixLora model on BFCL data"""
import argparse
import os
import json

import torch
import pandas as pd
from mixlora import MixLoraModelForCausalLM
from transformers import AutoTokenizer
from tqdm import tqdm

DEFAULT_BATCH_SIZE = 32
DEFAULT_OUTPUT = 'results/'
DEFAULT_PROMPT = """You are a helpful assistant with access to the following functions to help
you answer queries from the user. Based on the user's query, use the functions if required.

Some rules you always follow:
1. You always respond with valid code: fn_name(arg=value)
2. If the function is not relevant to answer the query, notify the user why.
3. If no function is provided, say you don't have access to any functions.

functions:
{{tools}}
"""
def fix_model_name(model_path:str)->str:
    """Replace \ with _ to generate a name for the model (should be like models_your_model_name)"""
    model_name = model_path.replace('/', '_')
    return model_name

def load_tests_paths(test_file:str) -> list:
    """load the list of tests we want to run"""
    with open(rel_path(test_file), 'r', encoding='utf8') as f:
        j = json.loads(f.read())
    return j

def rel_path(file:str) -> str:
    """return a relative path to this file"""
    return os.path.join(os.path.dirname(__file__), file)

def add_message(question:list, tools:dict):
    """Add our prompt to the user query and incorperate the tools"""
    
    prompt = DEFAULT_PROMPT.format(tools)
    query = question[0][0]['content']

    return f'{prompt}\n\n{query}\n\nAssistant:'

def generate_results(model:MixLoraModelForCausalLM, tokenizer:AutoTokenizer, test:str, batch_size)->pd.DataFrame:
    """Run batch inference on input test"""
    df = pd.read_json(test, lines=True, orient='records')
    df['question_formatted'] = df.apply(lambda row: add_message(row['question'], row.get('function', {})), axis=1)

    questions = df['question_formatted'].to_list()

    results = []

    for i in tqdm(range(0, len(questions), batch_size), desc=f'Running inference on {test}'):
        batch = questions[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        # Move tensors to model device if needed
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(**inputs)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        results.extend(decoded)

    df['result'] = results

    return df

def save_results(output_dir:str, test_name:str, df:pd.DataFrame)->None:
    """Save the result df to a json"""
    path = rel_path(os.path.join(output_dir, test_name))
    df['result'] = df['result'].apply(lambda x: x.split('Assistant:')[-1].strip())
    df[['id', 'result']].to_json(path, lines=True, orient='records', index=False)


def main():
    """Main"""
    parser = argparse.ArgumentParser(
        "MixLora Runner",
        description="Run a trained MixLora model on BFCL data"
    )

    parser.add_argument('-a', '--adapter', type=str, required=True, help='Adapter model for the eval')
    parser.add_argument('-o', '--output', type=str, default=DEFAULT_OUTPUT, help='Output directory (defaults to "results/" in the main dir of the repo)')
    parser.add_argument('-t', '--test', default='single_turn.json', help='File containing a list of paths to evaluate from (default runs all the single turn tests)')
    parser.add_argument('-b', '--batch_size', default=DEFAULT_BATCH_SIZE, type=int, help='Batch size for inference')

    args = parser.parse_args()
    assert args.adapter and os.path.isdir(args.adapter), "Did not pass an adapter folder, or was not a folder"

    print("\n", '-'*20, '\n')
    print(f'Running Eval with args:\n\tAdapter: {args.adapter}\n\tOutput Dir: {args.output}\n\tTest File: {args.test}')
    
    print("Loading Model")
    model, config = MixLoraModelForCausalLM.from_pretrained(args.adapter)
    print("Loading Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    
    tests = load_tests_paths(args.test) #load test files

    #Set up output directories
    model_name = fix_model_name(args.adapter)
    output_dir = f'{args.output}/{model_name}'
    os.makedirs(output_dir, exist_ok=True)

    for test in tests:
        test_name = os.path.basename(test)
        df = generate_results(model, tokenizer, test, args.batch_size)

        save_results(output_dir, test_name, df)
    


    





if __name__ == "__main__":
    main()

