try:
    from models.llava_next.model.language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig
    from models.llava_next.model.language_model.llava_mpt import LlavaMptForCausalLM, LlavaMptConfig
    from models.llava_next.model.language_model.llava_mistral import LlavaMistralForCausalLM, LlavaMistralConfig
except Exception as e:
    print(e)
    pass


AVAILABLE_MODELS = {
    "llava_llama": "LlavaLlamaForCausalLM, LlavaConfig",
    "llava_qwen": "LlavaQwenForCausalLM, LlavaQwenConfig",
    "llava_mistral": "LlavaMistralForCausalLM, LlavaMistralConfig",
    "llava_mixtral": "LlavaMixtralForCausalLM, LlavaMixtralConfig",
    # "llava_qwen_moe": "LlavaQwenMoeForCausalLM, LlavaQwenMoeConfig",    
    # Add other models as needed
}

for model_name, model_classes in AVAILABLE_MODELS.items():
    try:
        exec(f"from .language_model.{model_name} import {model_classes}")
    except Exception as e:
        print(f"Failed to import {model_name} from models.llava_next.language_model.{model_name}. Error: {e}")
