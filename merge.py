"""Merge an adapter into a model so that it can be used in BFCL evaluation / VLLM serving"""
import argparse
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig


def main():
    parser = argparse.ArgumentParser(description="Merge PEFT adapter into base model and save.")
    parser.add_argument(
        '-a',
        "--adapter-dir",
        type=str,
        required=True,
        help="Directory containing the adapter to merge."
    )
    parser.add_argument(
        '-o',
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the merged model."
    )

    args = parser.parse_args()

    assert os.path.isdir(args.adapter_dir), "Specified directory for the adapter does not exist"

    

    peft_config = PeftConfig.from_pretrained(args.adapter_dir)
    base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path)

    model = PeftModel.from_pretrained(base_model, args.adapter_dir)

    model = model.merge_and_unload()

    model.save_pretrained(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
    tokenizer.save_pretrained(args.output_dir)

    print("Finished merging models")


if __name__ == "__main__":
    main()