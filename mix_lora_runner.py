"""Run a MixLora model on BFCL data"""
import argparse
import os
import json
import gc

import torch
import pandas as pd
from transformers import AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
from mixlora_mods import MixLoraModelForCausalLM  # Rewrote MixLora to be used on GPU

DEFAULT_BATCH_SIZE = 32  # Might run out of memory with 64 on GPUs
DEFAULT_OUTPUT = 'results/'
DEFAULT_PROMPT_PATH = 'prompts/default.txt'

def get_best_device():
    """Just return the most performant device we can use in order: Cuda, MPS, CPU"""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

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

def add_message(prompt:str, question:list, tools:dict):
    """Add our prompt to the user query and incorperate the tools"""
    
    formatted = prompt.format(tools=tools)
    query = question[0][0]['content']

    return f'System:\n{formatted}\n\nUser\n\n{query}\n\nAssistant:'

def generate_results(model:MixLoraModelForCausalLM, tokenizer:AutoTokenizer, test:str, batch_size:int, prompt:str)->pd.DataFrame:
    """Run batch inference on input test"""
    df = pd.read_json(test, lines=True, orient='records')
    df['question_formatted'] = df.apply(lambda row: add_message(prompt, row['question'], row.get('function', [])), axis=1)

    questions = df['question_formatted'].to_list()
    results = []
    for i in tqdm(range(0, len(questions), batch_size), desc=f'Running inference on {test}'):
        batch = questions[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        decoded = [x[x.find('Assistant:')+len('Assistant:'):] for x in decoded]
        
        
        results.extend(decoded)

    df['result'] = results

    try:
        torch.cuda.empty_cache()
        gc.collect()
    except:
        print('unable to clean memory')
        
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
    parser.add_argument('-p', '--prompt', type=str, default=DEFAULT_PROMPT_PATH, help="File path to prompt")

    args = parser.parse_args()
    assert args.adapter and os.path.isdir(args.adapter), "Did not pass an adapter folder, or was not a folder"

    try:
        with open(args.prompt, 'r') as f:
            prompt = f.read()
            
    except FileNotFoundError as e:
        print('Unable to find prompt file')
        raise e

    print("\n", '-'*20, '\n')
    print(f'Running Eval with args:\n\tAdapter: {args.adapter}\n\tOutput Dir: {args.output}\n\tTest File: {args.test}\n\tPrompt File: {args.prompt}')

    print("Loading Model")
    device_name = get_best_device()
    print(f'Most Suitable Device: {device_name}')
    device = torch.device(device_name)
    bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
    model, config = MixLoraModelForCausalLM.from_pretrained(args.adapter, config=bnb_config, device_map="auto")
    # model = model.to(device)
    model.gradient_checkpointing_enable()
    print(f"Running on device: {torch.cuda.current_device()}")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True  

    print("Loading Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    tokenizer.padding_side = 'left' # HuggingFace will cry if you don't do this
    
    tests = load_tests_paths(args.test) # load test files

    #Set up output directories
    model_name = fix_model_name(args.adapter)
    output_dir = f'{args.output}/{model_name}'
    os.makedirs(output_dir, exist_ok=True)

    for test in tests:
        test_name = os.path.basename(test)
        if os.path.exists(os.path.join(output_dir,test_name)):
            continue
        try:
            df = generate_results(model, tokenizer, test, args.batch_size, prompt)
            save_results(output_dir, test_name, df)
        except Exception as e:
            print(e)
            print(f'Unable to generate results for test: {test_name}')
    
if __name__ == "__main__":
    main()