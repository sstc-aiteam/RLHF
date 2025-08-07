# Qwen2.5-VL GRPO 物體計數訓練系統

這是一個專門用於 Qwen2.5-VL 視覺語言模型的 GRPO (Generative Reward Preference Optimization) 訓練系統，特別針對物體計數任務進行優化。

## 系統概述

### 主要組件

1. **Qwen2_5_GRPO.py** - 核心 GRPO 訓練器實現
2. **Grpo_fastapi_3.py** - FastAPI 網頁服務接口
3. **test_with_lora_comparison.py** - 測試訓練好的GRPO模型
4. **training_data.zip** - 訓練/測試資料
5. **引用相關套件**
    - TRL https://github.com/huggingface/trl
    - Verl https://github.com/volcengine/verl
    - EasyR1 https://github.com/hiyouga/EasyR1

### 特色功能

- 🎯 專門針對物體計數任務優化
- 🔧 支持 LoRA (Low-Rank Adaptation) 微調
- 📊 漸進式計數獎勵機制
- 🚀 FastAPI 網頁界面，易於使用
- 💾 完整的訓練狀態管理和恢復
- 🔄 支持訓練中斷和繼續
- 📦 自動模型打包和下載

---

## 文件說明

### 1. Qwen2_5_GRPO.py

**功能：** GRPO 訓練器的核心實現

**主要特性：**
- 繼承自 Transformers Trainer
- 支持 Qwen2.5-VL、Qwen2-VL、Aria 等視覺語言模型
- 完整的 LoRA 檢查點加載和保存機制
- 自定義獎勵函數支持
- 分布式訓練支持
- vLLM 加速生成（可選）

**核心方法：**
- `_setup_lora_model()` - LoRA 模型配置和檢查點恢復
- `compute_loss()` - GRPO 損失計算
- `save_model()` - 智能模型保存（LoRA 適配器）
- `train()` - 增強的訓練流程，支持檢查點恢復

### 2. Grpo_fastapi_3.py

**功能：** 提供完整的網頁訓練服務接口

**主要特性：**
- RESTful API 接口
- 數據上傳和驗證
- 後台訓練任務管理
- 實時訓練進度監控
- 訓練任務取消功能
- 模型下載服務

---

POST
/upload-data
Upload Training Data

## 輸入格式

### 訓練數據結構

訓練數據需要以 **ZIP 文件** 格式上傳，包含以下結構：

```
training_data.zip
├── main.csv                    # 主要數據文件
└── images/                     # 圖片目錄
    ├── image1.jpg
    ├── image2.png
    └── ...
```

### CSV 格式要求

`main.csv` 必須包含以下三個欄位：

| 欄位名稱 | 描述 | 範例 |
|---------|------|------|
| `images_path` | 圖片檔案名稱 | `apple1.jpg` |
| `question` | 計數問題 | `How many apples are in this image?` |
| `answer` | 標準答案 | `A photo of 3 apples` |

### CSV 範例

```csv
images_path,question,answer
apple1.jpg,How many apples are in this image?,A photo of 3 apples
apple2.png,Count the apples in the picture,A photo of 5 apples
orange1.jpg,How many oranges do you see?,A photo of 2 oranges
car1.jpg,Count the number of cars,A photo of 4 cars
```

### 訓練配置

```json
{
  "training_config": {
    "num_epochs": 50,
    "learning_rate": 5e-5,
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "num_generations": 8,
    "temperature": 0.2,
    "use_4bit": true,
    "max_image_size": 512,
    "seed": 43
  },
  "lora_config": {
    "r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05
  }
}
```

---

## 輸出格式

### 訓練完成後的模型文件

```
output_directory/
├── adapter_config.json         # LoRA 配置
├── adapter_model.safetensors   # LoRA 權重
├── training_args.json          # 訓練參數
├── tokenizer.json             # 分詞器
├── tokenizer_config.json      # 分詞器配置
├── preprocessor_config.json   # 預處理器配置
└── trainer_state.json         # 訓練狀態
```

### API 響應格式

**訓練狀態響應：**
```json
{
  "job_id": "uuid-string",
  "job_name": "my_counting_model",
  "status": "training|completed|failed|cancelled",
  "progress": 75.5,
  "current_step": 150,
  "total_steps": 200,
  "loss": 0.125,
  "created_at": "2025-01-20T10:30:00",
  "error_message": null,
  "output_dir": "/path/to/output"
}
```

**文件上傳響應：**
```json
{
  "file_id": "uuid-string",
  "filename": "training_data.zip",
  "upload_time": "2025-01-20T10:30:00",
  "csv_rows": 500,
  "valid_images": 498
}
```

---

## 使用方法

### 1. 啟動服務

```bash
python Grpo_fastapi_3.py
```

服務將在 `http://localhost:8000` 啟動

### 2. API 端點

#### 基本信息
- `GET /` - 服務基本信息
- `GET /data-format` - 獲取詳細的數據格式說明

#### 數據管理
- `POST /upload-data` - 上傳 ZIP 訓練數據
- `GET /files` - 查看已上傳的文件

#### 訓練管理
- `POST /train` - 開始訓練任務
- `GET /jobs` - 查看所有訓練任務
- `GET /jobs/{job_id}` - 獲取特定任務狀態
- `DELETE /jobs/{job_id}` - 取消訓練任務

#### 模型下載
- `GET /jobs/{job_id}/download` - 下載訓練完成的模型

### 3. 使用流程

1. **準備數據**：按照格式要求準備 ZIP 文件
2. **上傳數據**：使用 `/upload-data` 端點上傳
3. **配置訓練**：設置訓練參數
4. **開始訓練**：調用 `/train` 端點
5. **監控進度**：定期查詢 `/jobs/{job_id}` 獲取狀態
6. **下載模型**：訓練完成後下載結果

### 4. 使用範例

```python
import requests
import json

# 上傳數據
with open('training_data.zip', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/upload-data',
        files={'file': f}
    )
file_id = response.json()['file_id']

# 開始訓練
training_config = {
    "job_name": "apple_counting_model",
    "data_file_id": file_id,
    "training_config": {
        "num_epochs": 30,
        "learning_rate": 5e-5,
        "batch_size": 1
    }
}

response = requests.post(
    'http://localhost:8000/train',
    json=training_config
)
job_id = response.json()['job_id']

# 查詢狀態
status_response = requests.get(f'http://localhost:8000/jobs/{job_id}')
print(status_response.json())
```

---

## 獎勵機制

### 漸進式計數獎勵函數

系統使用智能的獎勵機制來評估計數準確性：

- **完全正確 (差異=0)**：獎勵 = 1.0
- **差異1個**：獎勵 = 0.85
- **差異2個**：獎勵 = 0.7
- **差異3-5個**：獎勵 = 0.5 - 0.1*(差異-2)
- **差異>5個**：獎勵 = 0.0

這種設計鼓勵模型追求精確計數，同時對小幅偏差保持寬容。

---

### 硬體需求

- **GPU**：至少 8GB VRAM (推薦 16GB+)
- **RAM**：16GB+ 系統記憶體
- **儲存**：足夠空間存放模型和數據

### 支持的模型

- Qwen2.5-VL (推薦)
- Qwen2-VL
- Aria

---

## 注意事項

1. **答案格式**：訓練答案建議使用 "A photo of [數量] [物體]" 格式
2. **圖片品質**：建議使用清晰、對比度好的圖片
3. **數據平衡**：盡量包含不同數量的樣本
4. **檢查點**：系統會自動保存訓練進度，支持中斷恢復
5. **取消訓練**：可以隨時取消，已保存的檢查點不會丟失

---

## 故障排除

### 常見問題

1. **GPU 記憶體不足**：
   - 減少 `batch_size`
   - 增加 `gradient_accumulation_steps`
   - 啟用 `use_4bit` 量化

2. **模型加載失敗**：
   - 檢查模型路徑是否正確
   - 確認具有足夠的磁盤空間

3. **數據格式錯誤**：
   - 使用 `/data-format` 端點查看詳細要求
   - 確保 CSV 和圖片路徑對應

4. **訓練中斷**：
   - 系統會自動保存檢查點
   - 可以使用相同配置重新開始訓練

---
