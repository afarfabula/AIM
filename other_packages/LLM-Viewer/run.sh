# Script to calculate the prefill for the final llava1.5 anchor16 schedule.
python3 analyze_flex_prefill_only.py lmsys/vicuna-7b-v1.5 nvidia_A100 --config_file configs/Llama.py --preset llava15_anchor16 --skip-mlp-bias

# Legacy examples:
# python3 analyze_flex_prefill_only.py Qwen/Qwen2-7B nvidia_A100 --config_file configs/Llama.py --preset qwen2_default_aim --skip-mlp-bias
# python3 analyze_flex_prefill_only.py lmsys/vicuna-7b-v1.5 nvidia_A100 --config_file configs/Llama.py --preset vicuna_default_aim


# Scipt to calculate the prefill with same token numbers at different layers (LLaVA-PruMerge)
# python3 analyze_flex_prefill_only.py Qwen/Qwen2-7B nvidia_A100 --config_file configs/Llama.py  --promptlen 18532 --skip-mlp-bias

# python3 analyze_flex_prefill_only.py lmsys/vicuna-7b-v1.5 nvidia_A100 --config_file configs/Llama.py  --promptlen 328

# python3 analyze_flex_prefill_only.py Qwen/Qwen-VL-Chat nvidia_A100 --config_file configs/Llama.py  --promptlen 296  # Qwen-VL-Chat with 256 learnable queries
