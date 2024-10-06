for i in 0 1 2 3 4 5 6 7;do
    CUDA_VISIBLE_DEVICES=$i python -m vllm.entrypoints.api_server \
        --model model_path \
        --served-model-name=model_name \
        --gpu-memory-utilization=0.95 \
        --max-model-len=70000 \
        --tensor-parallel-size 1 \
        --host 127.0.0.1 \
        --port $((4100+i)) \
        --trust-remote-code \
        --swap-space 0 &
done

### inference
# config: Specify the config file of your generation
# model: The path of your model
# exp: Define the experiment class, choosing between 'main' (main experiment, L-CiteEval) and 'l-citeeval-length'
# num_port: Number of gpus in use
python script/inference_1shot_vllm.py --config config/main/narrativeqa.yaml --model model_path --exp main --num_port 8