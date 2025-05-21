
bfcl generate \
    --model $1 \
    --test-category single_turn \
    --backend vllm \
    --local-model-path $2

bfcl evaluate \
    --model $1 \
    --test-category single_turn 



