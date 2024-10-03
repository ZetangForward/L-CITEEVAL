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

    
python script/inference_1shot_vllm.py --config config/narrativeqa.yaml --bsz 1 --model model_path --exp citation --num_port 8