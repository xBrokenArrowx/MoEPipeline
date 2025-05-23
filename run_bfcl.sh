
bfcl generate \
    --model $1 \
    --test-category single_turn \
    --backend vllm \
    --local-model-path $2 \
    --result-dir ../results/


bfcl evaluate \
    --model $1 \
    --test-category single_turn \
    --result-dir ../results/ \
    --score-dir ../scores/



