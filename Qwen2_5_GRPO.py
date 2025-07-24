import os
from collections import defaultdict
from typing import Callable, Optional, Union
from accelerate.utils.other import is_compiled_module
from accelerate.utils import broadcast_object_list
import torch
import torch.nn as nn
from unittest.mock import patch
from datasets import Dataset, IterableDataset
from transformers import (
    AriaForConditionalGeneration,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.import_utils import is_vllm_available
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig

if is_peft_available():
    from peft import PeftConfig

if is_vllm_available():
    from vllm import LLM, SamplingParams

if is_wandb_available():
    import wandb

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class Qwen2VLGRPOTrainer(Trainer):
    """
    GRPO Trainer for Qwen2.5-VL with LoRA checkpoint support.
    Optimized for visual-language counting tasks with answer guidance.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
        resume_from_checkpoint: Optional[str] = None,
    ):
        self._tensor_parallel_size = 1
        
        # Initialize args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Load model
        model = self._load_base_model(model, args, attn_implementation)
        model_id = model if isinstance(model, str) else model.config._name_or_path
        
        # Handle LoRA configuration and checkpoint loading
        model = self._setup_lora_model(model, peft_config, resume_from_checkpoint)
        
        # Setup reference model
        self.ref_model = self._setup_reference_model(model, model_id, peft_config, args)
        self.resume_from_checkpoint = resume_from_checkpoint

        # Setup processing class
        processing_class = self._setup_processing_class(processing_class, model_id)
        self._pad_token_id = processing_class.pad_token_id if hasattr(processing_class, 'pad_token_id') else processing_class.tokenizer.pad_token_id
        self._eos_token_id = processing_class.eos_token_id if hasattr(processing_class, 'eos_token_id') else processing_class.tokenizer.eos_token_id

        # Setup reward functions
        self.reward_funcs, self.reward_processing_classes = self._setup_reward_functions(
            reward_funcs, reward_processing_classes, args.model_init_kwargs or {}
        )

        # Training configuration
        self._setup_training_config(args)
        
        # Initialize metrics and vLLM
        model.warnings_issued["estimate_tokens"] = True
        self._metrics = defaultdict(list)
        self.use_vllm = getattr(args, "use_vllm", False)

        # Initialize parent trainer
        super().__init__(
            model=model,
            args=args,
            data_collator=lambda features: features,  # No data collation needed
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        self.model_accepts_loss_kwargs = False
        
        # Setup vLLM if enabled
        if self.use_vllm:
            self._setup_vllm(args, model_id, max_pixels, min_pixels)
        
        # Prepare models for distributed training
        self._prepare_models_for_training()

    def _load_base_model(self, model, args, attn_implementation):
        """Load the base model based on model type"""
        if isinstance(model, str):
            model_init_kwargs = args.model_init_kwargs or {}
            model_init_kwargs["attn_implementation"] = attn_implementation
            
            # Handle torch_dtype
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, str) and torch_dtype != "auto":
                model_init_kwargs["torch_dtype"] = getattr(torch, torch_dtype)
            
            model_init_kwargs["use_cache"] = False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            
            # Load specific model type
            if "Qwen2-VL" in model:
                return Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Qwen2.5-VL" in model:
                return Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Aria" in model:
                model_init_kwargs.pop("use_cache", None)
                return AriaForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                return AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            if args.model_init_kwargs is not None:
                raise ValueError("model_init_kwargs can only be used when model is a string.")
            return model

    def _setup_lora_model(self, model, peft_config, resume_from_checkpoint):
        """Setup LoRA model with proper error handling"""
        
        # 如果有 checkpoint 需要加載
        if resume_from_checkpoint is not None:
            adapter_config_path = os.path.join(resume_from_checkpoint, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                print(f"🔄 Loading LoRA from checkpoint: {resume_from_checkpoint}")
                try:
                    if is_peft_available():
                        from peft import PeftModel
                        # 直接從 checkpoint 加載 LoRA 適配器
                        model = PeftModel.from_pretrained(model, resume_from_checkpoint)
                        print("✅ LoRA loaded successfully from checkpoint")
                        
                        # 確保模型處於訓練模式
                        model.train()
                        
                        # 驗證 LoRA 是否正確加載
                        if hasattr(model, 'peft_config') and model.peft_config:
                            print(f"✅ PEFT config loaded: {list(model.peft_config.keys())}")
                        
                        # 檢查可訓練參數
                        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                        total_params = sum(p.numel() for p in model.parameters())
                        print(f"📊 Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.3f}%)")
                        
                        return model
                    else:
                        print("❌ PEFT not available, cannot load LoRA checkpoint")
                        
                except Exception as e:
                    print(f"❌ Failed to load LoRA from checkpoint: {e}")
                    print(f"🔄 Will create new LoRA configuration instead")
        
        # 創建新的 LoRA 配置
        if peft_config is not None:
            print(f"🆕 Creating new LoRA model with config")
            if is_peft_available():
                from peft import get_peft_model
                
                try:
                    model = get_peft_model(model, peft_config)
                    print("✅ LoRA model created successfully")
                    
                    # 打印 LoRA 配置詳情
                    print(f"🎯 LoRA Configuration:")
                    print(f"  - Rank (r): {peft_config.r}")
                    print(f"  - Alpha: {peft_config.lora_alpha}")
                    print(f"  - Dropout: {peft_config.lora_dropout}")
                    print(f"  - Target modules: {peft_config.target_modules}")
                    
                    # 檢查可訓練參數
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    total_params = sum(p.numel() for p in model.parameters())
                    print(f"📊 Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.3f}%)")
                    
                    if trainable_params == 0:
                        print("❌ Warning: No trainable parameters found!")
                        # 嘗試手動啟用 LoRA 層
                        for name, param in model.named_parameters():
                            if any(keyword in name.lower() for keyword in ['lora_a', 'lora_b', 'adapter']):
                                param.requires_grad = True
                                print(f"🔧 Manually enabled gradient for: {name}")
                    
                except Exception as e:
                    print(f"❌ Failed to create LoRA model: {e}")
                    print("🔄 Using base model without LoRA")
            else:
                print("❌ PEFT not available, using base model")
        else:
            print("ℹ️ No PEFT config provided, using base model")
        
        return model

    def _verify_lora_setup(self, model):
        """驗證 LoRA 設置是否正確"""
        is_peft_model = False
        lora_layers = []
        
        if is_peft_available():
            from peft import PeftModel
            is_peft_model = isinstance(model, PeftModel)
        
        if is_peft_model:
            print("✅ Model is a PEFT model")
            
            # 統計 LoRA 參數
            lora_params = 0
            for name, param in model.named_parameters():
                if any(keyword in name.lower() for keyword in ['lora_a', 'lora_b', 'adapter']):
                    lora_params += param.numel()
                    lora_layers.append(f"{name}: {param.numel():,} params ({'trainable' if param.requires_grad else 'frozen'})")
            
            print(f"🎯 LoRA Statistics:")
            print(f"  - LoRA parameters: {lora_params:,}")
            print(f"  - LoRA layers: {len(lora_layers)}")
            
            if len(lora_layers) > 0 and any('trainable' in layer for layer in lora_layers):
                print("✅ LoRA setup verified successfully")
            else:
                print("⚠️ Warning: LoRA layers found but may not be trainable")
                
        else:
            print("⚠️ Model is NOT recognized as PEFT model")
            print("🔍 Checking for LoRA-like parameters...")
            
            # 檢查是否有類似 LoRA 的參數
            for name, param in model.named_parameters():
                if any(keyword in name.lower() for keyword in ['lora', 'adapter', 'merger']):
                    lora_layers.append(f"{name}: {param.numel():,} params")
            
            if lora_layers:
                print(f"🔍 Found {len(lora_layers)} LoRA-like parameters")
            else:
                print("❌ No LoRA parameters found - this will save the full model!")
        
        return is_peft_model

    def _setup_reference_model(self, model, model_id, peft_config, args):
        """Setup reference model for GRPO"""
        if peft_config is None:
            if is_deepspeed_zero3_enabled():
                model_init_kwargs = args.model_init_kwargs or {}
                if "Qwen2-VL" in model_id:
                    return Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
                elif "Qwen2.5-VL" in model_id:
                    return Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
                elif "Aria" in model_id:
                    return AriaForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
                else:
                    return AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
            else:
                return create_reference_model(model)
        else:
            return None  # LoRA uses adapter disable for reference

    def _setup_processing_class(self, processing_class, model_id):
        """Setup processing class for tokenization and image processing"""
        if processing_class is None:
            if any(model_name in model_id for model_name in ["Qwen2-VL", "Qwen2.5-VL", "Aria"]):
                processing_class = AutoProcessor.from_pretrained(model_id)
                # Add token IDs at processor level for consistency
                if not hasattr(processing_class, 'pad_token_id'):
                    processing_class.pad_token_id = processing_class.tokenizer.pad_token_id
                if not hasattr(processing_class, 'eos_token_id'):
                    processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
            else:
                processing_class = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        else:
            # Ensure token IDs are accessible
            if hasattr(processing_class, 'tokenizer'):
                if not hasattr(processing_class, 'pad_token_id'):
                    processing_class.pad_token_id = processing_class.tokenizer.pad_token_id
                if not hasattr(processing_class, 'eos_token_id'):
                    processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
        
        return processing_class

    def _setup_reward_functions(self, reward_funcs, reward_processing_classes, model_init_kwargs):
        """Setup reward functions and their processing classes"""
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        
        # Load reward models if they are strings
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )

        # Setup reward processing classes
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("Number of reward processing classes must match number of reward functions.")

        # Configure each reward processing class
        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class

        return reward_funcs, reward_processing_classes

    def _setup_training_config(self, args):
        """Setup training configuration"""
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=1.,
            num_return_sequences=self.num_generations,
            pad_token_id=self._pad_token_id,
        )
        self.beta = args.beta

    def _setup_vllm(self, args, model_id, max_pixels, min_pixels):
        """Setup vLLM for faster generation"""
        if not is_vllm_available():
            raise ImportError("vLLM is not available. Install with `pip install vllm`")

        if self.accelerator.is_main_process:
            vllm_device = getattr(args, "vllm_device", "auto")
            self._tensor_parallel_size = getattr(args, "tensor_parallel_size", 1)
            
            if vllm_device == "auto":
                vllm_device = "cuda"
            
            # Handle multi-GPU specification
            if vllm_device.startswith("cuda:") and "," in vllm_device:
                try:
                    gpu_ids = vllm_device.split(":")[1].split(",")
                    self._tensor_parallel_size = len(gpu_ids)
                    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
                    print(f"Using {self._tensor_parallel_size} GPUs for vLLM: {gpu_ids}")
                except Exception as e:
                    print(f"Error parsing vllm_device: {e}")
                    self._tensor_parallel_size = 1
                vllm_device = "cuda"
            
            # Setup vLLM with patches
            world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
            profiling_patch = patch(
                "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
                return_value=None,
            )
            
            with world_size_patch, profiling_patch:
                gpu_memory_utilization = getattr(args, "vllm_gpu_memory_utilization", 0.8)
                self.llm = LLM(
                    model=model_id,
                    device=vllm_device,
                    tensor_parallel_size=self._tensor_parallel_size,
                    gpu_memory_utilization=gpu_memory_utilization,
                    dtype=torch.bfloat16,
                    enable_prefix_caching=True,
                    enforce_eager=True,
                    mm_processor_kwargs={
                        "max_pixels": max_pixels,
                        "min_pixels": min_pixels,
                    },
                    max_model_len=args.max_prompt_length + args.max_completion_length,
                )
            
            self.sampling_params = SamplingParams(
                temperature=getattr(args, "temperature", 1.0),
                max_tokens=self.max_completion_length,
            )

        self._last_loaded_step = 0
        self.accelerator.wait_for_everyone()

    def _prepare_models_for_training(self):
        """Prepare reference model and reward functions for distributed training"""
        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        """Set signature columns for GRPO data processing"""
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def train(self, resume_from_checkpoint=None, **kwargs):
        """Enhanced LoRA checkpoint handling with full training state recovery"""
        if resume_from_checkpoint is None:
            resume_from_checkpoint = self.resume_from_checkpoint
        
        if resume_from_checkpoint is not None:
            # Check if it's a LoRA checkpoint
            adapter_config_path = os.path.join(resume_from_checkpoint, "adapter_config.json")
            trainer_state_path = os.path.join(resume_from_checkpoint, "trainer_state.json")
            optimizer_path = os.path.join(resume_from_checkpoint, "optimizer.pt")
            scheduler_path = os.path.join(resume_from_checkpoint, "scheduler.pt")
            
            if os.path.exists(adapter_config_path):
                print(f"🎯 LoRA checkpoint detected: {resume_from_checkpoint}")
                
                # Check what can be restored
                has_training_state = os.path.exists(trainer_state_path)
                has_optimizer = os.path.exists(optimizer_path)
                has_scheduler = os.path.exists(scheduler_path)
                
                print(f"📊 Checkpoint contents:")
                print(f"   ✅ LoRA weights: Available")
                print(f"   {'✅' if has_training_state else '❌'} Training state: {'Available' if has_training_state else 'Missing'}")
                print(f"   {'✅' if has_optimizer else '❌'} Optimizer state: {'Available' if has_optimizer else 'Missing'}")
                print(f"   {'✅' if has_scheduler else '❌'} Scheduler state: {'Available' if has_scheduler else 'Missing'}")
                
                # Decision making based on available files
                if has_training_state and has_optimizer and has_scheduler:
                    print("🔄 Full checkpoint recovery: LoRA weights + training state + optimizer + scheduler")
                    # Let parent class handle the full recovery
                    return super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)
                
                elif has_training_state:
                    print("🔄 Partial checkpoint recovery: LoRA weights + training state (fresh optimizer/scheduler)")
                    
                    # Load training state manually to get step/epoch info
                    try:
                        import json
                        with open(trainer_state_path, 'r') as f:
                            trainer_state = json.load(f)
                        
                        global_step = trainer_state.get('global_step', 0)
                        epoch = trainer_state.get('epoch', 0)
                        
                        print(f"📈 Resuming from step {global_step}, epoch {epoch:.2f}")
                        
                        # Try to resume with training state but expect optimizer issues
                        try:
                            return super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)
                        except Exception as e:
                            print(f"⚠️ Full recovery failed ({e}), trying LoRA-only recovery")
                            print("🔄 Falling back to LoRA weights only with fresh training state")
                            resume_from_checkpoint = None
                            
                    except Exception as e:
                        print(f"❌ Failed to read training state: {e}")
                        print("🔄 Using LoRA weights only")
                        resume_from_checkpoint = None
                
                else:
                    print("🔄 LoRA-only recovery: Using LoRA weights with fresh optimizer/scheduler/state")
                    resume_from_checkpoint = None
            
            else:
                print(f"📁 Standard checkpoint (non-LoRA): {resume_from_checkpoint}")
                # Let parent handle standard checkpoints
                return super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)
        
        # Call parent with potentially modified resume_from_checkpoint
        return super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)

    def save_model(self, output_dir=None, _internal_call=False):
        """Save LoRA adapter and processing class - Fixed version"""
        if output_dir is None:
            output_dir = self.args.output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        if hasattr(self.model, 'save_pretrained'):
            # 更準確的 PEFT 模型檢測
            is_peft_model = False
            
            # 方法1: 檢查是否是 PeftModel 實例
            if is_peft_available():
                from peft import PeftModel
                is_peft_model = isinstance(self.model, PeftModel)
            
            # 方法2: 檢查是否有 peft_config 屬性且不為空
            if not is_peft_model and hasattr(self.model, 'peft_config'):
                is_peft_model = self.model.peft_config is not None and len(self.model.peft_config) > 0
            
            # 方法3: 檢查是否有 base_model 屬性 (PEFT 模型特有)
            if not is_peft_model and hasattr(self.model, 'base_model'):
                is_peft_model = True
            
            # 方法4: 檢查模型類名
            if not is_peft_model:
                model_class_name = self.model.__class__.__name__
                is_peft_model = 'Peft' in model_class_name or 'LoRA' in model_class_name
            
            if is_peft_model:
                print(f"✅ Detected PEFT/LoRA model, saving adapter to {output_dir}")
                self.model.save_pretrained(output_dir)
                
                # 驗證保存結果
                adapter_config_path = os.path.join(output_dir, "adapter_config.json")
                if os.path.exists(adapter_config_path):
                    print(f"✅ LoRA adapter successfully saved")
                    # 打印保存的文件列表
                    saved_files = [f for f in os.listdir(output_dir) if f.endswith(('.json', '.bin', '.safetensors'))]
                    print(f"📁 Saved files: {saved_files}")
                else:
                    print(f"⚠️ Warning: adapter_config.json not found, might have saved full model")
            else:
                print(f"⚠️ Full model detected, saving complete model to {output_dir}")
                self.model.save_pretrained(output_dir)
        
        # 保存處理器
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
            print(f"✅ Saved processing class to {output_dir}")
        
        # 保存訓練參數
        if hasattr(self.args, 'save_to_json'):
            self.args.save_to_json(os.path.join(output_dir, "training_args.json"))
            print(f"✅ Saved training arguments")
        
        # 額外的驗證步驟
        self._validate_saved_model(output_dir)


    def _validate_saved_model(self, output_dir):
        """驗證保存的模型類型"""
        adapter_config_path = os.path.join(output_dir, "adapter_config.json")
        pytorch_model_path = os.path.join(output_dir, "pytorch_model.bin")
        safetensors_model_path = os.path.join(output_dir, "model.safetensors")
        adapter_model_path = os.path.join(output_dir, "adapter_model.safetensors")
        
        if os.path.exists(adapter_config_path):
            print(f"🎯 Confirmed: LoRA adapter saved successfully")
            
            # 計算保存的文件大小
            total_size = 0
            for file_name in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file_name)
                if os.path.isfile(file_path):
                    total_size += os.path.getsize(file_path)
            
            print(f"📊 Total saved size: {total_size / (1024**2):.2f} MB")
            
            # LoRA 適配器通常只有幾十MB，如果超過1GB可能是完整模型
            if total_size > 1024**3:  # 1GB
                print(f"⚠️ Warning: Saved size is {total_size / (1024**3):.2f} GB, might be full model instead of LoRA")
            else:
                print(f"✅ Size looks correct for LoRA adapter")
                
        elif os.path.exists(pytorch_model_path) or os.path.exists(safetensors_model_path):
            print(f"⚠️ Warning: Full model saved instead of LoRA adapter")
            print(f"📁 This will consume much more disk space")
        else:
            print(f"❌ Error: No model files found in {output_dir}")

    def _load_training_state_from_checkpoint(self, checkpoint_path):
        """Load training state from checkpoint, skipping optimizer to avoid conflicts"""
        if not os.path.exists(checkpoint_path):
            return
        
        trainer_state_file = os.path.join(checkpoint_path, "trainer_state.json")
        
        try:
            if os.path.exists(trainer_state_file):
                import json
                with open(trainer_state_file, 'r') as f:
                    trainer_state = json.load(f)
                print(f"Resuming from step: {trainer_state.get('global_step', 0)}")
            
            # Skip optimizer and scheduler loading to avoid parameter mismatch
            # optimizer_file = os.path.join(checkpoint_path, "optimizer.pt")
            # if os.path.exists(optimizer_file):
            #     print(optimizer_file)
            #     print("⚠️ Skipping optimizer state loading to avoid parameter mismatch")
            
            # scheduler_file = os.path.join(checkpoint_path, "scheduler.pt")
            # if os.path.exists(scheduler_file):
            #     print(scheduler_file)
            #     print("⚠️ Skipping scheduler state loading")
                        
        except Exception as e:
            print(f"Error loading training state: {e}")

    def _get_per_token_logps(self, model, input_ids, attention_mask, pixel_values, image_grid_thw, logits_to_keep=None):
        """Get per-token log probabilities from model output"""
        device = model.device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        pixel_values = pixel_values.to(device)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(device)
        
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }
        
        if image_grid_thw is not None:
            model_inputs["image_grid_thw"] = image_grid_thw
        
        try:
            logits = model(**model_inputs).logits
        except Exception as e:
            print(f"Model forward failed: {e}")
            print(f"Input shapes: input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}, "
                  f"pixel_values: {pixel_values.shape}, image_grid_thw: {image_grid_thw.shape if image_grid_thw is not None else None}")
            raise
            
        logits = logits[:, :-1, :]  # Remove last logit
        
        if logits_to_keep is not None:
            if not isinstance(logits_to_keep, int):
                try:
                    logits_to_keep = int(logits_to_keep)
                except:
                    logits_to_keep = None
        
        if logits_to_keep is not None:
            input_ids = input_ids[:, -logits_to_keep:]
            logits = logits[:, -logits_to_keep:]
        else:
            input_ids = input_ids[:, 1:]
        
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        
        return torch.stack(per_token_logps)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Main GRPO loss computation"""
        if return_outputs:
            raise ValueError("GRPOTrainer does not support returning outputs")
    
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        images = [x["image"] for x in inputs]
        
        if all(i is None for i in images):
            images = None
            
        prompt_inputs = self.processing_class(
            text=prompts_text,
            images=images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        pixel_values = prompt_inputs.get("pixel_values")
        image_grid_thw = prompt_inputs.get("image_grid_thw")
        
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]

        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)
            
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)

        # Create completion mask
        is_eos = completion_ids == self.processing_class.tokenizer.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Prepare inputs for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        if pixel_values is not None:
            pixel_values = pixel_values.repeat_interleave(self.num_generations, dim=0).view(-1, pixel_values.shape[-1])
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.repeat_interleave(self.num_generations, dim=0)

        # Compute logits
        per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw)
        per_token_logps = per_token_logps[:, prompt_length - 1:]

        # Compute reference logits
        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(self.ref_model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw)
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw)
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]

        # Compute KL divergence
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # Decode completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        # Compute rewards
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        
        for i, (reward_func, reward_processing_class) in enumerate(zip(self.reward_funcs, self.reward_processing_classes)):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                # Handle callable reward functions
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        reward_kwargs[key].extend([example[key]] * self.num_generations)
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        rewards = rewards_per_func.sum(dim=1)

        # Compute advantages
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Compute loss
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Log metrics
        self._log_metrics(completion_mask, rewards_per_func, rewards, std_grouped_rewards, per_token_kl)
        
        return loss

    def _log_metrics(self, completion_mask, rewards_per_func, rewards, std_grouped_rewards, per_token_kl):
        """Log training metrics"""
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())
        
        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())