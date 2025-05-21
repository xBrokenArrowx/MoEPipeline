#Script for training MoE
# ./run_moe.sh <BASE_MODEL_NAME> <DATASET_NAME_OR_PATH> <NAME_FOR_ADAPTER>
# BASE_MODEL_NAME: should be the name of the model from huggingface (this will download it from hf)
# DATASET_NAME_OR_PATH: the name of the dataset on huggingface, or the local path to it (needs to be [ {instruction:..., output:...}]) 
# NAME_FOR_ADAPTER: what you want the adapter to be called in the models/ folder
# for example of the dataset see MoE-PEFT/tests/dummy_data.json
#
# example of calling this script: ./run_moe.sh Qwen/Qwen2.5-1.5B-Instruct yahma/alpaca-cleaned my_cool_adapter
batch=16
epochs=2

echo "Generating Config"
echo "Making new model in models/$3"

python ./MoE-PEFT/launch.py gen \
    --template lora \
    --tasks $2 \
    --adapter_name "models/$3" \
    --batch_size $batch \
    --num_epochs $epochs


python ./MoE-PEFT/moe_peft.py \
    --base_model $1 \
    --config ./MoE-PEFT/moe_peft.json \
    --bf16

python merge.py \
    -a "models/${3}_0" \
    -o "models/${3}_merged"

echo "Model should be available in models/${3}_merged"

