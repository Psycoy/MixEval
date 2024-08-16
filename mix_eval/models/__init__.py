AVAILABLE_MODELS = {
    # open models
    "llama_2_7b": "LLAMA_2_7B",
    "llama_2_70b": "LLAMA_2_70B",
    "llama_2_7b_chat": "LLAMA_2_7B_Chat",
    "llama_2_70b_chat": "LLAMA_2_70B_Chat",
    
    "llama_3_8b": "Llama_3_8B",
    "llama_3_8b_instruct": "Llama_3_8B_Instruct",
    "llama_3_70b": "Llama_3_70B",
    "llama_3_70b_instruct": "Llama_3_70B_Instruct",
    
    "qwen_15_4b": "Qwen_15_4B",
    "qwen_15_7b": "Qwen_15_7B",
    "qwen_15_32b": "Qwen_15_32B",
    "qwen_15_72b": "Qwen_15_72B",
    "qwen_15_110b": "Qwen_15_110B",
    "qwen_15_moe_a27b": "Qwen_15_MoE_A27B",
    "qwen_15_4b_chat": "Qwen_15_4B_Chat",
    "qwen_15_7b_chat": "Qwen_15_7B_Chat",
    "qwen_15_32b_chat": "Qwen_15_32B_Chat",
    "qwen_15_72b_chat": "Qwen_15_72B_Chat",
    "qwen_15_110b_chat": "Qwen_15_110B_Chat",
    "qwen_15_moe_a27b_chat": "Qwen_15_MoE_A27B_Chat",
    "qwen_max_0428": "Qwen_Max_0428",
    "qwen_2_7b_instruct": "Qwen_2_7B_Instruct",
    "qwen_2_72b_instruct": "Qwen_2_72B_Instruct",
    
    "yi_6b": "Yi_6B",
    "yi_34b": "Yi_34B",
    "yi_6b_chat": "Yi_6B_Chat",
    "yi_34b_chat": "Yi_34B_Chat",
    "yi_large": "YI_Large",
    "yi_15_9b_chat": "Yi_15_9B_Chat",
    "yi_15_34b_chat": "Yi_15_34B_Chat",
    
    "gemma_2b": "Gemma_2B",
    "gemma_7b": "Gemma_7B",
    "gemma_11_2b_instruct": "Gemma_11_2B_Instruct",
    "gemma_11_7b_instruct": "Gemma_11_7B_Instruct",
    "gemma_2_9b_instruct": "Gemma_2_9B_Instruct",
    "gemma_2_27b_instruct": "Gemma_2_27B_Instruct",
    
    "mistral_7b": "Mistral_7B",
    "mixtral_8_7b": "Mixtral_8_7B",
    "mixtral_8_22b": "Mixtral_8_22B",
    "mistral_7b_instruct_v02": "Mistral_7B_Instruct_V02",
    "mistral_8_7b_instruct_v01": "Mistral_8_7B_Instruct_V01",
    "mistral_8_22b_instruct_v01": "Mistral_8_22B_Instruct_V01",
    
    "mammooth2_8_7b_plus": "MAmmooTH2_8_7B_Plus",
    
    "phi_2": "Phi_2",
    
    "deepseek_7b": "Deepseek_7B",
    "deepseek_67b": "Deepseek_67B",
    "deepseek_moe_16b": "Deepseek_MoE_16B",
    "deepseek_7b_chat": "Deepseek_7B_Chat",
    "deepseek_67b_chat": "Deepseek_67B_Chat",
    "deepseek_moe_16b_chat": "Deepseek_MoE_16B_Chat",
    "deepseek_v2": "Deepseek_v2",
    
    "dbrx_base": "DBRX_Base",
    "dbrx_instruct": "DBRX_Instruct",
    
    "olmo_7b": "OLMo_7B",
    "olmo_7b_instruct": "OLMo_7B_Instruct",
    
    # "jet_moe": "JetMoE",
    # "jet_moe_chat": "JetMoE_Chat",
    
    "mpt_7b": "MPT_7B",
    "mpt_30b": "MPT_30B",
    "mpt_7b_chat": "MPT_7B_Chat",
    "mpt_30b_chat": "MPT_30B_Chat",
    
    "vicuna_7b_v15": "Vicuna_7B_V15",
    "vicuna_33b_v13": "Vicuna_33B_V13",
    
    "command_r": "Command_R",
    "command_r_plus": "Command_R_Plus",
    
    "tulu_v2_dpo_7b": "Tulu_V2_DPO_7B",
    "tulu_v2_dpo_70b": "Tulu_V2_DPO_70B",
    
    "starling_lm_7b_beta": "Starling_LM_7B_Beta",
    
    "zephyr_7b_beta": "Zephyr_7B_Beta",
    
    "solar_107b_instruct_v1": "Solar_107B_Instruct_V1",
    
    
    # api models
    "gpt_35_turbo_0125": "GPT_35_Turbo_0125",
    "gpt_35_turbo_1106": "GPT_35_Turbo_1106",
    "gpt_4_turbo_2024_04_09": "GPT_4_Turbo_2024_04_09",
    "gpt_4_0613": "GPT_4_0613",
    "gpt_4_0314": "GPT_4_0314",
    "gpt_4_0125_preview": "GPT_4_0125_Preview",
    "gpt_4_1106_preview": "GPT_4_1106_Preview",
    "gpt_4o": "GPT_4o",
    "gpt_4o_mini": "GPT_4o_Mini",
    
    "claude_3_haiku": "Claude_3_Haiku",
    "claude_3_sonnet": "Claude_3_Sonnet",
    "claude_3_opus": "Claude_3_Opus",
    "claude_3_5_sonnet": "Claude_3_5_Sonnet",
    
    "mistral_small": "Mistral_Small",
    "mistral_medium": "Mistral_Medium",
    "mistral_large": "Mistral_Large",
    "mistral_large_2": "Mistral_Large_2",
    "mistral_nemo": "Mistral_Nemo",
    
    
    "gemini_10_pro": "Gemini_10_Pro",
    "gemini_15_pro": "Gemini_15_Pro",
    "gemini_10_ultra": "Gemini_10_Ultra",
    
    "reka_edge": "Reka_Edge",
    "reka_flash": "Reka_Flash",
    "reka_core": "Reka_Core",
    
    
    # additional
    "internlm_chat_7b": "InternLM_Chat_7B",
    "internlm2_chat_7b": "InternLM2_Chat_7B",
    
    "xverse_7b_chat": "XVerse_7B_Chat",
    "xverse_13b_chat": "XVerse_13B_Chat",
    
    "yulan_chat_2_13b": "Yulan_Chat_2_13B",
    
    "qwen_7b_chat": "Qwen_7B_Chat",
    "qwen_15_18b_chat": "Qwen_15_18B_Chat",
    
    "notus_7b_v1": "Notus_7B_V1",
    
    "baichuan_13b_chat": "Baichuan_13B_Chat",
    "baichuan2_7b_chat": "Baichuan2_7B_Chat",
    "baichuan2_13b_chat": "Baichuan2_13B_Chat",
    
    "vicuna_7b_v13": "Vicuna_7B_V13",
    "vicuna_7b_v15_16k": "Vicuna_7B_V15_16K",
    "vicuna_13b_v13": "Vicuna_13B_V13",
    "vicuna_13b_v15_16k": "Vicuna_13B_V15_16K",
    
    "tigerbot_7b_sft_v1": "TigerBot_7B_SFT_V1",
    "tigerbot_7b_sft_v2": "TigerBot_7B_SFT_V2",
    "tigerbot_13b_chat_v1": "TigerBot_13B_Chat_V1",
    "tigerbot_13b_chat_v2": "TigerBot_13B_Chat_V2",
    "tigerbot_13b_chat_v3": "TigerBot_13B_Chat_V3",
    
    "moss_moon_003_sft": "Moss_Moon_003_SFT",
    
    "mpt_7b_instruct": "MPT_7B_Instruct",
    
    "xwin_lm_7b_v01": "XWin_LM_7B_V01",
    
    "local_chat": "LocalChatModel",
    "local_base": "LocalBaseModel",
    
}

for model_name, model_class in AVAILABLE_MODELS.items():
    try:
        exec(f"from .{model_name} import {model_class}")
    except ImportError as e:
        print(e)
        pass
    
    
if __name__ == "__main__":
    print(AVAILABLE_MODELS.keys())
