"""
Qwen2.5-VL 3B 物體計數專用 GRPO 訓練 FastAPI 服務 (簡化版)
專注於提升物體計數的準確性，特別針對遮擋情況的處理
支持上傳包含 main.csv 和 images 資料夾的 ZIP 文件
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import asyncio
import os
# 關閉 wandb 和 swanlab 追蹤
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["SWANLAB_MODE"] = "disabled"
import json
import uuid
import logging
from datetime import datetime
from pathlib import Path
import shutil
import pandas as pd
from enum import Enum
import zipfile
import tempfile
import threading

# 導入訓練相關模組
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import GRPOConfig
from transformers import (
    Qwen2_5_VLForConditionalGeneration, 
    AutoProcessor,
    BitsAndBytesConfig,
    set_seed,
    TrainerCallback
)
import torch
from PIL import Image
import re
import random
import numpy as np
from Qwen2_5_GRPO import Qwen2VLGRPOTrainer

# 設置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 應用
app = FastAPI(
    title="Qwen2.5-VL GRPO訓練API (簡化版)",
    description="專門用於物體計數任務的視覺語言模型GRPO訓練服務",
    version="1.0.0"
)

# 目錄設置
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "training_outputs" 
DEFAULT_MODEL_PATH = "/home/itrib30156/llm_vision/qwen3b"

# 創建必要目錄
for dir_path in [UPLOAD_DIR, OUTPUT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# 訓練狀態
class TrainingStatus(str, Enum):
    PENDING = "pending"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

# 配置模型
class TrainingConfig(BaseModel):
    num_epochs: int = Field(default=50, ge=1, le=1000)
    learning_rate: float = Field(default=5e-5, gt=0, le=1)
    batch_size: int = Field(default=1, ge=1, le=32)
    gradient_accumulation_steps: int = Field(default=8, ge=1, le=128)
    num_generations: int = Field(default=8, ge=1, le=20)
    temperature: float = Field(default=0.2, gt=0, le=2.0)
    use_4bit: bool = Field(default=True)
    max_image_size: int = Field(default=512, ge=128, le=2048)
    seed: int = Field(default=43, ge=0)

class LoRAConfig(BaseModel):
    r: int = Field(default=64, ge=1, le=512)
    lora_alpha: int = Field(default=128, ge=1, le=512)
    lora_dropout: float = Field(default=0.05, ge=0, le=0.5)

class TrainingRequest(BaseModel):
    job_name: str
    base_model_path: str = Field(default=DEFAULT_MODEL_PATH)
    data_file_id: str
    training_config: TrainingConfig = Field(default_factory=TrainingConfig)
    lora_config: LoRAConfig = Field(default_factory=LoRAConfig)

class JobStatus(BaseModel):
    job_id: str
    job_name: str
    status: TrainingStatus
    progress: float = 0.0
    current_step: Optional[int] = None
    total_steps: Optional[int] = None
    loss: Optional[float] = None
    created_at: datetime
    error_message: Optional[str] = None
    output_dir: Optional[str] = None

# 全局變量
training_jobs: Dict[str, JobStatus] = {}
uploaded_files: Dict[str, Dict[str, Any]] = {}
training_processes: Dict[str, bool] = {}

# 核心函數
def set_random_seed(seed=43):
    """設置隨機種子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)

def extract_number_from_answer(text):
    """從答案中提取數字"""
    numbers = re.findall(r'\b(\d+)\b', text.strip())
    return int(numbers[0]) if numbers else None

def progressive_counting_reward_func(completions, solution, **kwargs):
    """計數獎勵函數"""
    rewards = []
    for completion, sol in zip(completions, solution):
        try:
            predicted = extract_number_from_answer(completion[0]['content'])
            true_count = extract_number_from_answer(sol)
            
            if predicted is None or true_count is None:
                rewards.append(0.0)
                continue
            
            diff = abs(predicted - true_count)
            if diff == 0:
                reward = 1.0
            elif diff == 1:
                reward = 0.85
            elif diff == 2:
                reward = 0.7
            elif diff <= 5:
                reward = max(0.0, 0.5 - (diff - 2) * 0.1)
            else:
                reward = 0.0
            
            rewards.append(reward)
        except:
            rewards.append(0.0)
    
    return rewards

def resize_image(img_pil, max_size=512):
    """調整圖片大小"""
    width, height = img_pil.size
    if width > max_size or height > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        return img_pil.resize((new_width, new_height), Image.LANCZOS)
    return img_pil

def get_prompt_rft(example, max_image_size=512):
    """處理訓練樣本"""
    results = []
    try:
        img_path = example['images_path']
        question = example['question']
        answer = example['answer']
        
        if not os.path.exists(img_path):
            return results
        
        img_pil = Image.open(img_path).convert('RGB')
        img_pil = resize_image(img_pil, max_image_size)
        
        system_prompt = (
            "You are an expert at counting objects in images. "
            "Count ALL visible objects carefully, including those that might be partially hidden. "
            "Provide your answer in the format: 'A photo of [NUMBER] [OBJECT_TYPE]'"
        )
        
        results.append({
            'prompt': [
                {'role': 'system', 'content': [{"type": "text", "text": system_prompt}]},
                {'role': 'user', 'content': [
                    {"type": "image"},  
                    {"type": "text", "text": question}
                ]}
            ],
            'image': img_pil,
            'solution': answer,
        })
    except Exception as e:
        logger.warning(f"處理樣本失敗: {e}")
    
    return results

def extract_and_validate_zip(zip_content: bytes, extract_to: str) -> dict:
    """解壓並驗證ZIP文件"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_zip:
            temp_zip.write(zip_content)
            temp_zip_path = temp_zip.name
        
        try:
            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                
                # 檢查必要文件
                has_csv = any('main.csv' in f for f in file_list)
                has_images = any('images/' in f for f in file_list)
                
                if not has_csv or not has_images:
                    return {"success": False, "error": "ZIP必須包含main.csv和images目錄"}
                
                # 解壓文件
                zip_ref.extractall(extract_to)
                
                # 重新組織文件結構
                csv_path = None
                images_dir = None
                
                for root, dirs, files in os.walk(extract_to):
                    if 'main.csv' in files:
                        csv_path = os.path.join(root, 'main.csv')
                    if 'images' in dirs:
                        images_dir = os.path.join(root, 'images')
                
                # 移動到根目錄
                target_csv = os.path.join(extract_to, 'main.csv')
                target_images = os.path.join(extract_to, 'images')
                
                if csv_path != target_csv:
                    shutil.move(csv_path, target_csv)
                if images_dir != target_images:
                    if os.path.exists(target_images):
                        shutil.rmtree(target_images)
                    shutil.move(images_dir, target_images)
                
                # 驗證CSV
                df = pd.read_csv(target_csv)
                required_cols = ['images_path', 'question', 'answer']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    return {"success": False, "error": f"CSV缺少列: {missing_cols}"}
                
                # 更新圖片路徑
                for idx, row in df.iterrows():
                    img_filename = os.path.basename(row['images_path'])
                    new_path = os.path.join(target_images, img_filename)
                    df.at[idx, 'images_path'] = new_path
                
                df.to_csv(target_csv, index=False)
                
                # 統計有效圖片
                valid_images = sum(1 for _, row in df.iterrows() 
                                 if os.path.exists(row['images_path']))
                
                return {
                    "success": True,
                    "csv_path": target_csv,
                    "images_dir": target_images,
                    "rows": len(df),
                    "valid_images": valid_images
                }
                
        finally:
            os.unlink(temp_zip_path)
            
    except Exception as e:
        return {"success": False, "error": f"處理失敗: {str(e)}"}

class TrainingProgressCallback(TrainerCallback):
    """訓練進度回調"""
    def __init__(self, job_id):
        self.job_id = job_id
        
    def on_train_begin(self, args, state, control, **kwargs):
        # 檢查是否在訓練開始時就被取消
        if self.job_id in training_processes and not training_processes[self.job_id]:
            control.should_training_stop = True
            if self.job_id in training_jobs:
                training_jobs[self.job_id].status = TrainingStatus.CANCELLED
            return control
            
        if self.job_id in training_jobs:
            training_jobs[self.job_id].status = TrainingStatus.TRAINING
            if state.max_steps:
                training_jobs[self.job_id].total_steps = state.max_steps
        
    def on_step_end(self, args, state, control, **kwargs):
        # 每步檢查是否被取消
        if self.job_id in training_processes and not training_processes[self.job_id]:
            control.should_training_stop = True
            if self.job_id in training_jobs:
                training_jobs[self.job_id].status = TrainingStatus.CANCELLED
            return control
            
        if self.job_id in training_jobs:
            job = training_jobs[self.job_id]
            if state.global_step:
                job.current_step = state.global_step
            if state.max_steps:
                job.total_steps = state.max_steps
                job.progress = (state.global_step / state.max_steps) * 100
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        # 日誌時也檢查取消狀態
        if self.job_id in training_processes and not training_processes[self.job_id]:
            control.should_training_stop = True
            if self.job_id in training_jobs:
                training_jobs[self.job_id].status = TrainingStatus.CANCELLED
            return control
            
        if self.job_id in training_jobs and logs:
            job = training_jobs[self.job_id]
            if 'train_loss' in logs:
                job.loss = logs['train_loss']

def train_model_background(job_id: str, training_request: TrainingRequest, csv_path: str):
    """後台訓練函數"""
    def run_training():
        try:
            # 檢查是否已被取消
            if job_id in training_processes and not training_processes[job_id]:
                if job_id in training_jobs:
                    training_jobs[job_id].status = TrainingStatus.CANCELLED
                return
            
            # 更新狀態
            if job_id in training_jobs:
                training_jobs[job_id].status = TrainingStatus.TRAINING
            
            # 創建輸出目錄
            output_dir = os.path.join(OUTPUT_DIR, job_id)
            os.makedirs(output_dir, exist_ok=True)
            training_jobs[job_id].output_dir = output_dir
            
            # 設置隨機種子
            set_random_seed(training_request.training_config.seed)
            
            # 處理數據
            ds = load_dataset("csv", data_files=csv_path, split="train")
            processed_data = []
            
            for i, example in enumerate(ds):
                # 每處理一些樣本就檢查是否被取消
                if i % 10 == 0 and job_id in training_processes and not training_processes[job_id]:
                    if job_id in training_jobs:
                        training_jobs[job_id].status = TrainingStatus.CANCELLED
                    return
                    
                if os.path.exists(example['images_path']):
                    sample_data = get_prompt_rft(example, training_request.training_config.max_image_size)
                    processed_data.extend(sample_data)
            
            if not processed_data:
                raise ValueError("沒有有效的訓練樣本")
            
            # 再次檢查是否被取消
            if job_id in training_processes and not training_processes[job_id]:
                if job_id in training_jobs:
                    training_jobs[job_id].status = TrainingStatus.CANCELLED
                return
            
            dataset_train = Dataset.from_list(processed_data)
            
            # 載入模型前檢查取消狀態
            if job_id in training_processes and not training_processes[job_id]:
                if job_id in training_jobs:
                    training_jobs[job_id].status = TrainingStatus.CANCELLED
                return
            
            # 載入模型
            processor = AutoProcessor.from_pretrained(
                training_request.base_model_path, 
                trust_remote_code=True
            )
            
            model_kwargs = {
                'device_map': "auto",
                'trust_remote_code': True,
                'torch_dtype': torch.bfloat16
            }
            
            if training_request.training_config.use_4bit:
                model_kwargs['quantization_config'] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                training_request.base_model_path, 
                **model_kwargs
            )
            
            model.config.use_cache = False
            model.gradient_checkpointing_enable()
            
            if training_request.training_config.use_4bit:
                model = prepare_model_for_kbit_training(model)
            
            # 應用LoRA
            peft_config = LoraConfig(
                r=training_request.lora_config.r,
                lora_alpha=training_request.lora_config.lora_alpha,
                target_modules=["q_proj", "v_proj", "o_proj"],
                bias="none",
                lora_dropout=training_request.lora_config.lora_dropout,
                task_type="CAUSAL_LM",
            )
            
            model = get_peft_model(model, peft_config)
            
            # 訓練前最後檢查
            if job_id in training_processes and not training_processes[job_id]:
                if job_id in training_jobs:
                    training_jobs[job_id].status = TrainingStatus.CANCELLED
                return
            
            # 訓練配置
            training_args = GRPOConfig(
                learning_rate=training_request.training_config.learning_rate,
                per_device_train_batch_size=training_request.training_config.batch_size,
                gradient_accumulation_steps=training_request.training_config.gradient_accumulation_steps,
                num_generations=training_request.training_config.num_generations,
                num_train_epochs=training_request.training_config.num_epochs,
                output_dir=output_dir,
                logging_steps=10,
                save_steps=50,
                bf16=True,
                temperature=training_request.training_config.temperature,
                remove_unused_columns=False,
                gradient_checkpointing=True,
                report_to=None,  # 關閉所有實驗追蹤
                logging_dir=None,  # 關閉日誌目錄
            )
            
            # 創建訓練器
            trainer = Qwen2VLGRPOTrainer(
                model=model,
                processing_class=processor,
                reward_funcs=[progressive_counting_reward_func],
                args=training_args,
                train_dataset=dataset_train,
            )
            
            # 添加回調
            trainer.add_callback(TrainingProgressCallback(job_id))
            
            # 開始訓練
            try:
                trainer.train()
            except KeyboardInterrupt:
                # 處理中斷
                if job_id in training_jobs:
                    training_jobs[job_id].status = TrainingStatus.CANCELLED
                return
            
            # 訓練後檢查是否被取消（以防在訓練最後階段被取消）
            if job_id in training_processes and not training_processes[job_id]:
                if job_id in training_jobs:
                    training_jobs[job_id].status = TrainingStatus.CANCELLED
                return
            
            # 保存模型
            trainer.save_model(output_dir)
            processor.save_pretrained(output_dir)
            
            # 完成
            if job_id in training_jobs:
                training_jobs[job_id].status = TrainingStatus.COMPLETED
                training_jobs[job_id].progress = 100.0
            
        except Exception as e:
            if job_id in training_jobs:
                training_jobs[job_id].status = TrainingStatus.FAILED
                training_jobs[job_id].error_message = str(e)
            logger.error(f"訓練失敗: {e}")
    
    # 在新線程中運行
    thread = threading.Thread(target=run_training, daemon=True)
    thread.start()

# API 端點
@app.get("/data-format")
async def get_data_format_guide():
    """獲取訓練數據格式說明"""
    return {
        "zip_structure": {
            "required_files": {
                "main.csv": "主要數據文件，包含 images_path, question, answer 三個欄位",
                "images/": "圖片目錄，包含所有訓練圖片"
            },
            "example_structure": {
                "training_data.zip": {
                    "main.csv": "CSV數據文件",
                    "images/": {
                        "apple1.jpg": "圖片文件",
                        "apple2.png": "圖片文件",
                        "orange1.jpg": "圖片文件"
                    }
                }
            }
        },
        "csv_format": {
            "required_columns": [
                {
                    "name": "images_path",
                    "description": "圖片檔案名稱",
                    "example": "apple1.jpg"
                },
                {
                    "name": "question", 
                    "description": "對圖片的計數問題",
                    "example": "How many apples are in this image?"
                },
                {
                    "name": "answer",
                    "description": "標準答案",
                    "example": "A photo of 3 apples"
                }
            ],
            "csv_example": {
                "headers": ["images_path", "question", "answer"],
                "sample_rows": [
                    ["apple1.jpg", "How many apples are in this image?", "A photo of 3 apples"],
                    ["apple2.png", "Count the apples in the picture", "A photo of 5 apples"],
                    ["orange1.jpg", "How many oranges do you see?", "A photo of 2 oranges"]
                ]
            }
        },
        "image_requirements": {
            "supported_formats": [".jpg", ".jpeg", ".png", ".bmp", ".gif"],
            "recommended_size": "224x224 到 1024x1024 像素",
            "max_file_size": "建議每張圖片小於 5MB",
            "naming": "使用英文字母和數字，避免特殊字符"
        },
        "answer_format": {
            "recommended_pattern": "A photo of [數量] [物體名稱]",
            "correct_examples": [
                "A photo of 3 apples",
                "A photo of 5 cars", 
                "A photo of 1 cat"
            ],
            "avoid": [
                "There are 3 apples",
                "3",
                "Three apples"
            ]
        },
        "tips": [
            "確保 CSV 中的 images_path 與 images 目錄中的檔案名對應",
            "建議包含不同數量的樣本以平衡數據",
            "可以包含部分遮擋的物體，系統會學習處理",
            "使用 /upload-data/validate 端點可以在上傳前驗證格式"
        ]
    }

@app.get("/")
async def root():
    return {
        "message": "Qwen2.5-VL GRPO訓練API (簡化版)",
        "version": "1.0.0", 
        "gpu_available": torch.cuda.is_available(),
        "endpoints": {
            "data_format": "/data-format - 獲取數據格式說明",
            "upload": "/upload-data - 上傳ZIP訓練數據包",
            "files": "/files - 查看已上傳文件",
            "train": "/train - 開始訓練, 使用Qwen2.5VL-3B 下載網址 https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/tree/main",
            "jobs": "/jobs - 查看訓練任務"
        }
    }

@app.post("/upload-data")
async def upload_training_data(file: UploadFile = File(...)):
    """上傳ZIP訓練數據包"""
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="只支持ZIP格式")
    
    file_id = str(uuid.uuid4())
    extract_dir = os.path.join(UPLOAD_DIR, file_id)
    
    try:
        zip_content = await file.read()
        os.makedirs(extract_dir, exist_ok=True)
        
        validation_result = extract_and_validate_zip(zip_content, extract_dir)
        
        if not validation_result["success"]:
            shutil.rmtree(extract_dir)
            raise HTTPException(status_code=400, detail=validation_result["error"])
        
        uploaded_files[file_id] = {
            "filename": file.filename,
            "csv_path": validation_result["csv_path"],
            "images_dir": validation_result["images_dir"],
            "upload_time": datetime.now(),
            "rows": validation_result["rows"],
            "valid_images": validation_result["valid_images"]
        }
        
        return {
            "file_id": file_id,
            "filename": file.filename,
            "upload_time": datetime.now(),
            "csv_rows": validation_result["rows"],
            "valid_images": validation_result["valid_images"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
        raise HTTPException(status_code=500, detail=f"處理失敗: {str(e)}")

@app.get("/files")
async def list_uploaded_files():
    """獲取已上傳文件列表"""
    return {"files": [
        {
            "file_id": file_id,
            "filename": info["filename"],
            "upload_time": info["upload_time"],
            "csv_rows": info["rows"],
            "valid_images": info["valid_images"]
        }
        for file_id, info in uploaded_files.items()
    ]}

@app.post("/train")
async def start_training(training_request: TrainingRequest, background_tasks: BackgroundTasks):
    """開始訓練\n
    base_model_path:使用Qwen2.5VL-3B\n 
    下載網址 https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/tree/main
    """
    if training_request.data_file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="數據文件不存在")
    
    if not torch.cuda.is_available():
        raise HTTPException(status_code=400, detail="GPU不可用")
    
    file_info = uploaded_files[training_request.data_file_id]
    csv_path = file_info["csv_path"]
    
    job_id = str(uuid.uuid4())
    training_jobs[job_id] = JobStatus(
        job_id=job_id,
        job_name=training_request.job_name,
        status=TrainingStatus.PENDING,
        created_at=datetime.now()
    )
    
    training_processes[job_id] = True
    background_tasks.add_task(train_model_background, job_id, training_request, csv_path)
    
    return {
        "job_id": job_id,
        "message": "訓練已啟動",
        "status": "started"
    }

@app.get("/jobs")
async def list_jobs():
    """獲取所有訓練任務"""
    return {"jobs": list(training_jobs.values())}

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """獲取訓練狀態"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="任務不存在")
    return training_jobs[job_id]


@app.get("/jobs/{job_id}/download")
async def download_model(job_id: str):
    """下載訓練完成的模型"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="任務不存在")
    
    job = training_jobs[job_id]
    if job.status != TrainingStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="訓練未完成")
    
    output_dir = job.output_dir
    if not output_dir or not os.path.exists(output_dir):
        raise HTTPException(status_code=404, detail="模型文件不存在")
    
    # 創建zip文件
    zip_path = os.path.join(OUTPUT_DIR, f"{job_id}_model.zip")
    shutil.make_archive(zip_path[:-4], 'zip', output_dir)
    
    return FileResponse(
        zip_path,
        media_type='application/zip',
        filename=f"{job.job_name}_model.zip"
    )

@app.delete("/jobs/{job_id}", deprecated=True)
async def cancel_job(job_id: str):
    """取消訓練任務\n可能會導致GPU錯誤，暫停使用"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="任務不存在")
    
    job = training_jobs[job_id]
    
    # 檢查任務狀態
    if job.status in [TrainingStatus.COMPLETED, TrainingStatus.FAILED, TrainingStatus.CANCELLED]:
        raise HTTPException(status_code=400, detail="任務已結束，無法取消")
    
    # 設置取消標記
    training_processes[job_id] = False
    
    # 立即更新狀態
    training_jobs[job_id].status = TrainingStatus.CANCELLED
    
    return {"message": "任務取消指令已發送，訓練將在下一個檢查點停止"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)