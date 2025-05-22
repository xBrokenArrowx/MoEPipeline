# Pipeline for MoE-ifying a model and running BFCL

Howdy NLPee(ne)rs

I have packaged up the entire situationship I have been using to MoE-PEFT a model, and then evaluate it on BFCL

## Set Up
Create a virtual environment in the root directory (make sure it is either called .venv or venv so that it is ignored by github pweety pwease don't push your venv)

`uv venv` or the python one should be fine

Make sure to activate the venv with `source <path_to_venv>` before continuing on to next steps

### BFCL
To set up BFCL first enter the directory
`cd bfcl`

Then download the requirements from their directory via 
`pip install -e .[oss_eval_vllm]`

This will install VLLM for you

### MoE-PEFT
To setup MoE-PEFT enter the `MoE-PEFT` directory via `cd MoE-PEFT` from the root dir

then call `pip install -r requirements.txt`

After doing this and the BFCL set up you should have all the needed requirements to get started

### Scripts
Make sure you give the script files executable permissions
`chmod +x run_bfcl.sh` and `chmod +x run_moe.sh`


## Running Pipeline

### Creating an MoE model
To create an adapter MoE we want to use `./run_moe.sh` this takes a couple of command line arguments and also has some predefined variables you can change in the file.

`./run_moe.sh <base_model_name> <dataset> <name_for_new_adapter>`

- base_model_name: should be the model from huggingface i.e `Qwen/Qwen2.5-1.5B-Instruct` (make sure thi model is supported by BFCL) https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/SUPPORTED_MODELS.md 
> BFCL adds on an additional word like -FC to certain models, do not include that in this script but you will need to include that in the bfcl script for this to work 

- dataset: name of the dataset from huggingface, or the path to the dataset relative to the root directory

- name_for_new_adapter: What you want your new adapter to be called `my_coool_adapter`

### Example of `./run_moe.sh`

`./run_moe.sh Qwen/Qwen2.5-1.5B-Instruct recast_fixed_fr.json coolest_adapter`

This will perform the MoE adapter training on Qwen2.5-1.5B-Instruct using the `recast_fixed_fr.json` dataset. Naming the model `coolest_adapter` which will appear in the `models` directory as a usable model for BFCL/VLLM as `coolest_adapter_merged`. Ensure you used the merged version for later steps it is the only way to be usable for BFCL

### Running `./run_bfcl.sh`
To generate and evaluate the new model on BFCL we can call this script.

`./run_bfcl.sh <model> <local_model_path>`

- model: The name of the supported model from this list https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/SUPPORTED_MODELS.md

- local_model_path: the path to your new merged adapter model

Running this script will create a results folder inside the `bfcl/result` directory which is then evaluated creating a folder in `bfcl/score`

### Example using with our previously generated model
`./run_bfcl.sh Qwen/Qwen2.5-1.5B-Instruct-FC models/coolest_adapter_merged`

The `-FC` part is just because BFCL has some weird distinction between certain models I don't fully understand it but you need to add it for some of the models. See the above link.

## Further Customization

There are a lot of arguments that are not being used in the scripts. Feel free to checkout the github for both BFCL, and MoE-PEFT to discover more of the options.

I think `python MoE-PEFT/launch.py help` will display a list of options that can be passed.
The `gen` argument also creates a editable config file called `MoE-PEFT/moe-peft.json` with more arguments.

Feel free to use the scripts for an understanding of the process flow rather than needing to be used all the time.
