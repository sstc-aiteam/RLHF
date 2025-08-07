import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, set_seed
from peft import PeftModel
import argparse
import random
import numpy as np
import os
import logging
import json
import time

# 設置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def set_random_seed(seed=43):
    """
    設置所有隨機種子以確保測試環境與訓練環境相同
    
    Args:
        seed (int): 隨機種子值，默認為43（與訓練腳本一致）
    """
    logger.info(f"🎲 設置隨機種子: {seed}")
    
    # Python內置random模塊
    random.seed(seed)
    
    # NumPy隨機種子
    np.random.seed(seed)
    
    # PyTorch隨機種子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    
    # Transformers庫的隨機種子
    set_seed(seed)
    
    # 確保CUDA操作的確定性（與訓練時保持一致）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 設置環境變量
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info("✅ 隨機種子設置完成，測試環境已與訓練環境同步")

def load_training_seed_if_available(model_dir):
    """
    嘗試從訓練好的模型目錄中加載訓練時使用的種子
    
    Args:
        model_dir (str): 模型目錄路徑
        
    Returns:
        int: 訓練時使用的種子，如果找不到則返回默認值43
    """
    # 嘗試從training_config.json加載種子
    config_path = os.path.join(model_dir, "training_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                training_seed = config.get('final_seed_used', config.get('seed', 43))
                logger.info(f"📁 從訓練配置({model_dir})中找到種子: {training_seed}")
                return training_seed
        except Exception as e:
            logger.warning(f"讀取訓練配置失敗({model_dir}): {e}")
    
    # 嘗試從random_states.pkl加載種子（如果存在）
    seed_state_path = os.path.join(model_dir, "random_states.pkl")
    if os.path.exists(seed_state_path):
        try:
            import pickle
            with open(seed_state_path, 'rb') as f:
                random_states = pickle.load(f)
                training_seed = random_states.get('seed_used', 43)
                logger.info(f"📁 從隨機狀態文件({model_dir})中找到種子: {training_seed}")
                return training_seed
        except Exception as e:
            logger.warning(f"讀取隨機狀態文件失敗({model_dir}): {e}")
    
    logger.info(f"📁 未找到訓練時的種子配置({model_dir})，使用默認種子: 43")
    return 43

def load_training_config(model_dir):
    """
    加載訓練配置信息
    
    Args:
        model_dir (str): 模型目錄路徑
        
    Returns:
        dict: 訓練配置信息
    """
    config_path = os.path.join(model_dir, "training_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config
        except Exception as e:
            logger.warning(f"讀取訓練配置失敗: {e}")
    return {}

def parse_args():
    parser = argparse.ArgumentParser(description='比較基礎模型和訓練後模型在單張圖片上的表現 (支持LoRA和完整模型)')
    parser.add_argument('--base_model_path', type=str, default="/home/itrib30156/llm_vision/qwen3b",
                        help='基礎模型路徑')
    parser.add_argument('--lora_model_path', type=str, default="/home/itrib30156/llm_vision/outputs/counting-model_11/checkpoint-370",
                        help='訓練後模型路徑（支持LoRA適配器或完整微調模型）。如果不指定，只測試基礎模型')
    parser.add_argument('--image_path', type=str, default="/home/itrib30156/llm_vision/self_grpo/images/R.jpg",
                        help='測試圖片路徑')
    parser.add_argument('--question', type=str, default="圖中共有多少瓶裝咖啡豆的瓶子?,",
                        help='測試問題')
    
    # 隨機種子相關參數
    parser.add_argument('--seed', type=int, default=None,
                        help='隨機種子值。如果不指定，會嘗試從訓練模型目錄讀取訓練時的種子')
    parser.add_argument('--deterministic', action='store_true', default=True,
                        help='啟用確定性模式，與訓練環境保持一致')
    parser.add_argument('--auto_load_seed', action='store_true', default=True,
                        help='自動從訓練模型目錄加載訓練時使用的種子')
    
    # 生成參數
    parser.add_argument('--max_new_tokens', type=int, default=128,
                        help='最大生成token數')
    parser.add_argument('--temperature', type=float, default=0.6,
                        help='生成溫度')
    parser.add_argument('--do_sample', action='store_true', default=True,
                        help='是否使用采樣生成')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='nucleus sampling的p值')
    parser.add_argument('--top_k', type=int, default=50,
                        help='top-k sampling的k值')
    
    # 測試控制參數
    parser.add_argument('--compare_mode', action='store_true', default=False,
                        help='比較模式：同時測試基礎模型和訓練後模型')
    parser.add_argument('--only_lora', action='store_true', default=False,
                        help='只測試訓練後模型（LoRA或完整模型）')
    parser.add_argument('--only_base', action='store_true', default=False,
                        help='只測試基礎模型')
    
    # 其他參數
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='顯示詳細信息')
    parser.add_argument('--save_result', type=str, default=None,
                        help='保存測試結果到指定文件')
    parser.add_argument('--use_answer_guided_prompt', action='store_true', default=False,
                        help='使用答案引導的系統提示詞（與訓練時一致）')
    
    return parser.parse_args()

def get_system_prompt(use_answer_guided=False):
    """獲取系統提示詞"""
    if use_answer_guided:
        # 答案引導模式的系統提示詞（與訓練時一致）
        return (
            "你是一個專業的視覺分析助手，特別擅長精確統計圖片中的物體數量。"
            "請遵循以下步驟回答：\n"
            "1. 在 <think> </think> 標籤中進行詳細的觀察和推理：\n"
            "   - 仔細描述你在圖片中看到的內容\n"
            "   - 說明每個物體的位置和特徵\n"
            "   - 面對計數問題必須逐步進行計數，可以用編號方式（如：第1個...第2個...）\n"
            "   - 解釋你的計數邏輯或判斷方法\n"
            "2. 在 <answer> </answer> 標籤中給出最終的數量\n\n"
            "請用繁體中文回答，數字請使用阿拉伯數字。"
        )
    else:
        # 基礎系統提示詞
        return "你是一個專業的視覺分析助手，特別擅長精確統計圖片中的物體數量。請用繁體中文回答，數字請使用阿拉伯數字。"

def load_models(args):
    """
    加載基礎模型和訓練後的模型（支持LoRA和完整模型）
    
    Returns:
        tuple: (processor, base_model, trained_model, training_config, model_type)
    """
    print("🔧 正在加載處理器和模型...")
    
    # 加載處理器
    processor = AutoProcessor.from_pretrained(args.base_model_path, trust_remote_code=True)
    
    # 加載基礎模型
    print(f"📦 加載基礎模型: {args.base_model_path}")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).eval()
    
    trained_model = None
    training_config = {}
    model_type = "none"
    
    # 加載訓練後的模型（如果指定）
    if args.lora_model_path and not args.only_base:
        print(f"🎯 檢查訓練模型類型: {args.lora_model_path}")
        
        # 首先檢查是否存在必要文件
        if not os.path.exists(args.lora_model_path):
            logger.error(f"訓練模型路徑不存在: {args.lora_model_path}")
            return processor, base_model, None, {}, "none"
        
        # 檢查模型類型
        adapter_config_path = os.path.join(args.lora_model_path, "adapter_config.json")
        full_model_config_path = os.path.join(args.lora_model_path, "config.json")
        
        # 加載訓練配置信息
        training_config = load_training_config(args.lora_model_path)
        if training_config:
            print(f"📋 發現訓練配置:")
            print(f"   - 訓練模式: {training_config.get('training_mode', '未知')}")
            print(f"   - LoRA策略: {training_config.get('lora_strategy', '未知')}")
            print(f"   - 學習率: {training_config.get('learning_rate', '未知')}")
            print(f"   - 訓練輪數: {training_config.get('num_epochs', '未知')}")
            if 'guidance_enabled' in training_config:
                print(f"   - 答案引導: {'是' if training_config['guidance_enabled'] else '否'}")
        
        # 方法1: 嘗試作為GRPO模型加載
        if os.path.exists(adapter_config_path):
            print("🎯 檢測到GRPO模型文件，嘗試作為LoRA適配器加載...")
            try:
                from peft import PeftModel
                trained_model = PeftModel.from_pretrained(
                    base_model,
                    args.lora_model_path,
                    torch_dtype=torch.bfloat16,
                ).eval()
                
                model_type = "lora"
                print(f"✅ GRPO模型加載成功")
                
                # 顯示GRPO模型信息
                if hasattr(trained_model, 'peft_config') and trained_model.peft_config:
                    peft_config = trained_model.peft_config[list(trained_model.peft_config.keys())[0]]
                    print(f"🎯 LoRA配置信息:")
                    print(f"   - LoRA秩 (r): {peft_config.r}")
                    print(f"   - LoRA alpha: {peft_config.lora_alpha}")
                    print(f"   - LoRA dropout: {peft_config.lora_dropout}")
                    print(f"   - 目標模組: {len(peft_config.target_modules)} 個")
                    if args.verbose:
                        print(f"   - 目標模組列表: {peft_config.target_modules}")
                
            except Exception as e:
                print(f"❌ 作為GRPO模型加載失敗: {e}")
                print("🔄 嘗試作為完整模型加載...")
                trained_model = None
        
        # 方法2: 作為完整模型加載
        if trained_model is None and os.path.exists(full_model_config_path):
            print("📦 檢測到完整模型文件，作為完整模型加載...")
            try:
                trained_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    args.lora_model_path,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True
                ).eval()
                
                model_type = "full_model"
                print(f"✅ 完整訓練模型加載成功")
                
                # 比較參數量
                base_params = sum(p.numel() for p in base_model.parameters())
                trained_params = sum(p.numel() for p in trained_model.parameters())
                print(f"📊 模型參數對比:")
                print(f"   - 基礎模型參數: {base_params:,}")
                print(f"   - 訓練模型參數: {trained_params:,}")
                
                if abs(base_params - trained_params) < 1000:  # 參數量基本相同
                    print(f"   - ✅ 參數量一致，確認為微調後的完整模型")
                else:
                    print(f"   - ⚠️ 參數量差異: {abs(base_params - trained_params):,}")
                
            except Exception as e:
                print(f"❌ 作為完整模型加載失敗: {e}")
                trained_model = None
        
        # 如果兩種方法都失敗
        if trained_model is None:
            print("❌ 無法識別訓練模型格式，將只測試基礎模型")
            model_type = "failed"
        
    return processor, base_model, trained_model, training_config, model_type

def generate_response(model, processor, prompt, image, args, model_name="Model"):
    """
    生成模型回答
    
    Args:
        model: 要測試的模型
        processor: 處理器
        prompt: 提示詞
        image: 圖片
        args: 命令行參數
        model_name: 模型名稱（用於日志顯示）
    
    Returns:
        tuple: (response_text, generation_time)
    """
    print(f"🧠 {model_name} 正在生成回答...")
    
    # 處理提示
    text = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    )
    
    # 將輸入移至適當的設備
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 生成參數設置
    generation_kwargs = {
        'max_new_tokens': args.max_new_tokens,
        'do_sample': args.do_sample,
    }
    
    # 只有在採樣模式下才添加相關參數
    if args.do_sample:
        generation_kwargs.update({
            'temperature': args.temperature,
            'top_p': args.top_p,
            'top_k': args.top_k,
        })
    else:
        # 非採樣模式下可以使用的參數
        generation_kwargs.update({
            'num_beams': 1,  # 確保是貪婪解碼
        })
    
    # 記錄生成時間
    start_time = time.time()
    
    # 生成回答
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_kwargs
        )
    
    generation_time = time.time() - start_time
    
    # 解碼回答
    response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    # 只獲取模型生成的部分
    response = response.split("assistant:")[-1].strip()
    
    return response, generation_time

def analyze_response_structure(response):
    """
    分析回答結構
    
    Args:
        response (str): 模型回答
        
    Returns:
        dict: 分析結果
    """
    import re
    
    analysis = {
        'has_think_tag': False,
        'has_answer_tag': False,
        'think_content': '',
        'answer_content': '',
        'think_length': 0,
        'total_length': len(response),
        'has_enumeration': False,
        'number_count': 0
    }
    
    # 檢查think標籤
    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    if think_match:
        analysis['has_think_tag'] = True
        analysis['think_content'] = think_match.group(1).strip()
        analysis['think_length'] = len(analysis['think_content'])
        
        # 檢查是否有枚舉
        analysis['has_enumeration'] = bool(re.search(r'[1-9]\s*[.、)）]', analysis['think_content']))
    
    # 檢查answer標籤
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if answer_match:
        analysis['has_answer_tag'] = True
        analysis['answer_content'] = answer_match.group(1).strip()
    
    # 統計數字出現次數
    numbers = re.findall(r'\d+', response)
    analysis['number_count'] = len(numbers)
    
    return analysis

def print_comparison_results(base_response, trained_response, base_analysis, trained_analysis, 
                           base_time, trained_time, model_type):
    """打印比較結果（支持不同模型類型）"""
    print("\n" + "="*80)
    print("📊 模型比較結果")
    print("="*80)
    
    # 基礎模型結果
    print("🔵 基礎模型回答:")
    print("-"*60)
    print(base_response)
    print(f"\n⏱️  生成時間: {base_time:.2f}秒")
    
    # 訓練後模型結果
    model_name = "GRPO模型" if model_type == "lora" else "GRPO模型"
    model_emoji = "🎯" if model_type == "lora" else "🚀"
    
    print(f"\n{model_emoji} {model_name}回答:")
    print("-"*60)
    print(trained_response)
    print(f"\n⏱️  生成時間: {trained_time:.2f}秒")
    
    # 結構化分析比較
    print(f"\n📋 結構化分析比較:")
    print("-"*60)
    print(f"{'項目':<20} {'基礎模型':<15} {f'{model_name}':<15} {'改進':<10}")
    print("-"*60)
    
    base_think = '是' if base_analysis['has_think_tag'] else '否'
    trained_think = '是' if trained_analysis['has_think_tag'] else '否'
    think_improve = '✅' if trained_analysis['has_think_tag'] and not base_analysis['has_think_tag'] else '➖'
    print(f"{'包含思考過程':<20} {base_think:<15} {trained_think:<15} {think_improve:<10}")
    
    base_answer = '是' if base_analysis['has_answer_tag'] else '否'
    trained_answer = '是' if trained_analysis['has_answer_tag'] else '否'
    answer_improve = '✅' if trained_analysis['has_answer_tag'] and not base_analysis['has_answer_tag'] else '➖'
    print(f"{'包含答案標簽':<20} {base_answer:<15} {trained_answer:<15} {answer_improve:<10}")
    
    length_improve = '✅' if trained_analysis['think_length'] > base_analysis['think_length'] else '➖'
    print(f"{'思考內容長度':<20} {base_analysis['think_length']:<15} {trained_analysis['think_length']:<15} {length_improve:<10}")
    
    base_enum = '是' if base_analysis['has_enumeration'] else '否'
    trained_enum = '是' if trained_analysis['has_enumeration'] else '否'
    enum_improve = '✅' if trained_analysis['has_enumeration'] and not base_analysis['has_enumeration'] else '➖'
    print(f"{'包含枚舉計數':<20} {base_enum:<15} {trained_enum:<15} {enum_improve:<10}")
    
    total_improve = '✅' if trained_analysis['total_length'] > base_analysis['total_length'] else '➖'
    print(f"{'總回答長度':<20} {base_analysis['total_length']:<15} {trained_analysis['total_length']:<15} {total_improve:<10}")
    
    time_improve = '✅' if trained_time < base_time else '➖'
    print(f"{'生成時間(秒)':<20} {base_time:.2f}<{'':<14} {trained_time:.2f}<{'':<14} {time_improve:<10}")
    
    # 最終答案提取
    if base_analysis['has_answer_tag'] and trained_analysis['has_answer_tag']:
        print(f"\n🎯 最終答案比較:")
        print(f"  基礎模型: {base_analysis['answer_content']}")
        print(f"  {model_name}: {trained_analysis['answer_content']}")
        if base_analysis['answer_content'] != trained_analysis['answer_content']:
            print("  ⚠️  答案不同！")
        else:
            print("  ✅ 答案一致")
    
    # 模型類型說明
    print(f"\n📝 模型類型說明:")
    if model_type == "lora":
        print("  🎯 LoRA適配器: 在基礎模型上附加的輕量級適配器")
        print("  📊 優勢: 參數效率高，訓練快速，存儲占用小")
    elif model_type == "full_model":
        print("  🚀 完整微調模型: 基礎模型的所有參數都經過訓練更新")
        print("  📊 優勢: 可能有更強的表現，但需要更多存儲空間")
        print("  ⚠️ 注意: 完整模型可能過擬合，需要仔細評估泛化能力")

def main():
    args = parse_args()
    
    print("🚀 啟動 Qwen VL 模型比較測試（支持完整模型）")
    print("=" * 80)
    
    # [種子設置部分保持不變...]
    if args.seed is not None:
        final_seed = args.seed
        logger.info(f"使用命令行指定的種子: {final_seed}")
    elif args.lora_model_path and args.auto_load_seed:
        final_seed = load_training_seed_if_available(args.lora_model_path)
        logger.info(f"檢測到訓練模型，使用訓練時的種子以確保一致性: {final_seed}")
    elif args.auto_load_seed:
        final_seed = load_training_seed_if_available(args.base_model_path)
    else:
        final_seed = 43
        logger.info(f"使用默認種子: {final_seed}")
    
    set_random_seed(final_seed)
    
    print(f"🎲 測試環境種子: {final_seed}")
    print(f"🔧 確定性模式: {'啟用' if args.deterministic else '禁用'}")
    
    # 顯示測試模式
    if args.only_base:
        print("🔵 測試模式: 僅基礎模型")
    elif args.only_lora:
        print("🎯 測試模式: 僅訓練後模型")
    else:
        print("⚖️  測試模式: 比較基礎模型與訓練後模型")
    
    print("-" * 80)
    
    # 加載模型（新的方法）
    processor, base_model, trained_model, training_config, model_type = load_models(args)
    
    # 根據加載結果顯示信息
    if model_type == "lora":
        print("✅ 成功加載LoRA適配器模型")
    elif model_type == "full_model":
        print("✅ 成功加載完整訓練模型")
        print("ℹ️ 注意: 這是一個完整的微調模型，而不是LoRA適配器")
    elif model_type == "failed":
        print("❌ 訓練模型加載失敗，將只測試基礎模型")
    
    # [圖片加載和測試部分保持不變...]
    print(f"🖼️  加載圖片: {args.image_path}")
    try:
        image = Image.open(args.image_path).convert("RGB")
        print(f"📐 圖片尺寸: {image.size}")
    except Exception as e:
        print(f"❌ 加載圖片失敗: {e}")
        return
    
    question = args.question
    print(f"❓ 測試問題: {question}")
    
    # 根據訓練配置或命令行參數決定是否使用答案引導提示
    use_guided_prompt = args.use_answer_guided_prompt
    if not use_guided_prompt and training_config.get('guidance_enabled', False):
        use_guided_prompt = True
        print(f"🎯 檢測到答案引導訓練，自動啟用相應的系統提示")
    
    system_prompt = get_system_prompt(use_guided_prompt)
    
    if args.verbose:
        print(f"\n📝 系統提示詞:")
        print("-" * 40)
        print(system_prompt)
        print("-" * 40)
    
    # 構建提示
    prompt = [
        {'role': 'system', 'content': [{"type": "text", "text": system_prompt}]},
        {'role': 'user', 'content': [
            {"type": "image"},
            {"type": "text", "text": question}
        ]}
    ]
    
    # 顯示生成參數
    print(f"\n⚙️  生成參數:")
    print(f"  - max_new_tokens: {args.max_new_tokens}")
    print(f"  - do_sample: {args.do_sample}")
    if args.do_sample:
        print(f"  - temperature: {args.temperature}")
        print(f"  - top_p: {args.top_p}")
        print(f"  - top_k: {args.top_k}")
    else:
        print(f"  - 模式: 貪婪解碼 (greedy decoding)")
    
    # 測試結果存儲
    results = {
        'seed_used': final_seed,
        'base_model_path': args.base_model_path,
        'trained_model_path': args.lora_model_path,
        'model_type': model_type,
        'image_path': args.image_path,
        'question': question,
        'use_guided_prompt': use_guided_prompt,
        'training_config': training_config,
        'generation_params': {
            'max_new_tokens': args.max_new_tokens,
            'do_sample': args.do_sample,
            'temperature': args.temperature if args.do_sample else None,
            'top_p': args.top_p if args.do_sample else None,
            'top_k': args.top_k if args.do_sample else None,
        }
    }
    
    # 測試基礎模型
    base_response = None
    base_analysis = None
    base_time = None
    
    if not args.only_lora:
        base_response, base_time = generate_response(
            base_model, processor, prompt, image, args, "基礎模型"
        )
        base_analysis = analyze_response_structure(base_response)
        results['base_model'] = {
            'response': base_response,
            'generation_time': base_time,
            'analysis': base_analysis
        }
    
    # 測試訓練後模型
    trained_response = None
    trained_analysis = None
    trained_time = None
    
    if trained_model is not None and not args.only_base:
        model_name = "GRPO模型" if model_type == "lora" else "訓練後模型"
        trained_response, trained_time = generate_response(
            trained_model, processor, prompt, image, args, model_name
        )
        trained_analysis = analyze_response_structure(trained_response)
        results['trained_model'] = {
            'response': trained_response,
            'generation_time': trained_time,
            'analysis': trained_analysis,
            'model_type': model_type
        }
    
    # 顯示結果
    if args.compare_mode and base_response and trained_response:
        # 比較模式
        print_comparison_results(base_response, trained_response, base_analysis, 
                               trained_analysis, base_time, trained_time, model_type)
    else:
        # 單模型模式
        if base_response and not args.only_lora:
            print("\n" + "="*60)
            print("🔵 基礎模型回答:")
            print("-"*60)
            print(base_response)
            print(f"\n⏱️  生成時間: {base_time:.2f}秒")
            
            if base_analysis['has_think_tag'] or base_analysis['has_answer_tag']:
                print(f"\n📊 結構分析:")
                print(f"  ✅ 思考過程: {'是' if base_analysis['has_think_tag'] else '否'}")
                print(f"  ✅ 結構化回答: {'是' if base_analysis['has_answer_tag'] else '否'}")
                if base_analysis['has_answer_tag']:
                    print(f"  🎯 最終答案: {base_analysis['answer_content']}")
        
        if trained_response and not args.only_base:
            model_name = "GRPO模型" if model_type == "lora" else "GRPO模型"
            model_emoji = "🎯" if model_type == "lora" else "🚀"
            
            print(f"\n" + "="*60)
            print(f"{model_emoji} {model_name}回答:")
            print("-"*60)
            print(trained_response)
            print(f"\n⏱️  生成時間: {trained_time:.2f}秒")
            
            if trained_analysis['has_think_tag'] or trained_analysis['has_answer_tag']:
                print(f"\n📊 結構分析:")
                print(f"  ✅ 思考過程: {'是' if trained_analysis['has_think_tag'] else '否'}")
                print(f"  ✅ 結構化回答: {'是' if trained_analysis['has_answer_tag'] else '否'}")
                if trained_analysis['has_answer_tag']:
                    print(f"  🎯 最終答案: {trained_analysis['answer_content']}")
    
    # 保存結果（如果指定）
    if args.save_result:
        with open(args.save_result, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 測試結果已保存到: {args.save_result}")
    
    print(f"\n🎯 測試完成！")
    
    # 根據模型類型給出不同的總結
    if trained_model and base_response and trained_response:
        print(f"\n📈 訓練效果評估:")
        print(f"   🎲 測試一致性: 基礎模型和訓練模型使用相同種子({final_seed})確保公平比較")
        
        if model_type == "lora":
            print(f"   📊 LoRA適配器效果:")
        else:
            print(f"   📊 完整微調效果:")
        
        if trained_analysis['has_think_tag'] and not base_analysis['has_think_tag']:
            print("   ✅ 成功學會結構化思考")
        if trained_analysis['think_length'] > base_analysis['think_length']:
            print("   ✅ 推理過程更加詳細")
        if trained_analysis['has_enumeration'] and not base_analysis['has_enumeration']:
            print("   ✅ 學會了逐步計數")
        if trained_time < base_time:
            print("   ✅ 生成速度有所提升")
        
        if model_type == "full_model":
            print("   ⚠️ 注意: 完整模型占用更多存儲空間，請評估是否值得")

if __name__ == "__main__":
    main()


# ============================================================================
# GRPO模型比較測試使用範例（帶隨機種子固定）
# ============================================================================

"""
🎯 GRPO模型比較測試範例（優化種子一致性）：

1. 完整比較測試（自動使用LoRA訓練種子確保一致性）:
python test_with_lora_comparison.py \
    --base_model_path "/home/itrib30156/llm_vision/new_qwen3b" \
    --lora_model_path "/home/itrib30156/llm_vision/outputs/Qwen25VL-Counting-GRPO-Guided_3" \
    --image_path "/home/itrib30156/llm_vision/self_grpo/images/R.jpg" \
    --question "圖中有多少櫃子放了咖啡豆?" \
    --verbose
    # 注意：基礎模型和GRPO模型將自動使用相同的訓練種子進行公平比較

2. 指定種子的比較測試（手動確保一致性）:
python test_with_lora_comparison.py \
    --seed 43 \
    --base_model_path "/path/to/base/model" \
    --lora_model_path "/path/to/lora/model" \
    --image_path "/path/to/test/image.jpg" \
    --use_answer_guided_prompt \
    --save_result "comparison_result.json"
    # 注意：手動指定種子會覆蓋自動讀取的LoRA訓練種子

3. 僅測試GRPO模型（使用LoRA訓練種子）:
python test_with_lora_comparison.py \
    --only_lora \
    --lora_model_path "/path/to/lora/model" \
    --image_path "/path/to/image.jpg" \
    --max_new_tokens 512
    # 注意：即使只測試GRPO模型，也會使用LoRA訓練時的種子

4. 僅測試基礎模型（如果有LoRA路徑，仍使用LoRA種子確保可比較性）:
python test_with_lora_comparison.py \
    --only_base \
    --lora_model_path "/path/to/lora/model" \
    --image_path "/path/to/image.jpg"
    # 注意：即使只測試基礎模型，如果指定了lora_model_path，
    # 基礎模型仍會使用LoRA訓練時的種子，以便後續比較

5. 批量測試（確保種子一致性）:
# 第一次測試
python test_with_lora_comparison.py \
    --base_model_path "/path/to/base/model" \
    --lora_model_path "/path/to/lora/model" \
    --image_path "/path/to/image1.jpg" \
    --question "圖中有多少個蘋果？" \
    --save_result "test1.json"

# 第二次測試（自動使用相同種子）
python test_with_lora_comparison.py \
    --base_model_path "/path/to/base/model" \
    --lora_model_path "/path/to/lora/model" \
    --image_path "/path/to/image2.jpg" \
    --question "圖中有多少個橘子？" \
    --save_result "test2.json"
    # 注意：兩次測試會自動使用相同的LoRA訓練種子，確保結果可比較

🔧 種子一致性邏輯（重要更新）:

**新的種子選擇優先級**:
1. **命令行指定種子**: 如果使用 --seed 參數，直接使用該值
2. **LoRA訓練種子優先**: 如果指定了 lora_model_path，優先使用LoRA訓練時的種子
3. **基礎模型種子**: 只有在沒有GRPO模型時，才從基礎模型目錄讀取種子  
4. **默認種子**: 以上都沒有則使用43

**關鍵改進**:
- ✅ **確保一致性**: 基礎模型和GRPO模型始終使用相同種子
- ✅ **公平比較**: 消除隨機性差異對比較結果的影響
- ✅ **可重現性**: 多次運行產生相同結果
- ✅ **智能選擇**: 自動選擇最合適的種子來源

**使用場景說明**:

1. **完整比較模式** (推薦):
   ```bash
   python script.py --lora_model_path "/path/to/lora"
   ```
   - 基礎模型和GRPO模型使用相同的LoRA訓練種子
   - 確保公平比較，消除隨機性影響

2. **僅基礎模型但考慮後續比較**:
   ```bash
   python script.py --only_base --lora_model_path "/path/to/lora"
   ```
   - 基礎模型使用LoRA訓練種子
   - 為後續與GRPO模型比較做準備

3. **純基礎模型測試**:
   ```bash
   python script.py --only_base
   ```
   - 不指定lora_model_path
   - 使用基礎模型自己的種子或默認種子

4. **強制指定種子**:
   ```bash
   python script.py --seed 42 --lora_model_path "/path/to/lora"
   ```
   - 覆蓋自動種子選擇
   - 基礎模型和GRPO模型都使用指定種子

📊 測試結果可信度提升:

由於確保了種子一致性，測試結果更加可靠：
- ✅ **消除隨機性**: 生成差異純粹來自模型能力差異
- ✅ **可重現結果**: 相同輸入產生相同輸出
- ✅ **公平評估**: 基礎模型和GRPO模型處於相同隨機環境
- ✅ **準確分析**: 結構化分析結果更加準確

⚠️ 重要注意事項:

1. **種子優先級**: GRPO模型存在時優先使用其訓練種子
2. **一致性保證**: 基礎模型會自動同步GRPO模型的種子
3. **覆蓋機制**: --seed 參數可以覆蓋任何自動選擇
4. **日志提示**: 腳本會清楚顯示使用的種子來源和邏輯

💡 最佳實踐:

1. **比較測試**: 讓腳本自動選擇LoRA訓練種子
2. **重現實驗**: 記錄使用的種子值以便後續重現
3. **批量測試**: 使用相同的模型路徑確保種子一致
4. **結果分析**: 重點關注結構化改進而非隨機差異
"""