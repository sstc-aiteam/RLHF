# GRPO模型測試工具使用說明

### 基本語法
```bash
python test_with_lora_comparison.py [參數]
```

### 最簡單的使用方式
```bash
python test_with_lora_comparison.py \
    --base_model_path "/path/to/your/base/model" \
    --lora_model_path "/path/to/your/grpo/model" \
    --image_path "/path/to/your/test/image.jpg" \
    --question "你的問題"
```

## 📝 常用使用場景

### 1. 完整比較測試（最常用）
**目的**：同時測試基礎模型和GRPO模型，並比較結果

```bash
python test_with_lora_comparison.py \
    --base_model_path "/home/user/models/base_model" \
    --lora_model_path "/home/user/models/grpo_model" \
    --image_path "/home/user/images/test.jpg" \
    --question "圖中有多少個蘋果？" \
    --verbose
```

**預期結果**：
- 顯示兩個模型的回答
- 比較分析結果
- 顯示哪個模型表現更好

### 2. 只測試GRPO模型
**目的**：快速測試訓練後的模型效果

```bash
python test_with_lora_comparison.py \
    --only_lora \
    --lora_model_path "/home/user/models/grpo_model" \
    --image_path "/home/user/images/test.jpg" \
    --question "圖中有多少瓶咖啡？"
```

### 3. 只測試基礎模型
**目的**：測試原始模型的基準表現

```bash
python test_with_lora_comparison.py \
    --only_base \
    --base_model_path "/home/user/models/base_model" \
    --image_path "/home/user/images/test.jpg" \
    --question "圖中有多少個物品？"
```

### 4. 批量測試多張圖片
**目的**：測試模型在不同圖片上的表現

```bash
# 測試圖片1
python test_with_lora_comparison.py \
    --base_model_path "/home/user/models/base_model" \
    --lora_model_path "/home/user/models/grpo_model" \
    --image_path "/home/user/images/image1.jpg" \
    --question "圖中有多少個蘋果？" \
    --save_result "result1.json"

# 測試圖片2
python test_with_lora_comparison.py \
    --base_model_path "/home/user/models/base_model" \
    --lora_model_path "/home/user/models/grpo_model" \
    --image_path "/home/user/images/image2.jpg" \
    --question "圖中有多少個橘子？" \
    --save_result "result2.json"
```

## ⚙️ 重要參數說明

### 必需參數
| 參數 | 說明 | 示例 |
|-----|------|------|
| `--base_model_path` | 基礎模型路徑 | `/home/user/models/base_model` |
| `--image_path` | 測試圖片路徑 | `/home/user/images/test.jpg` |
| `--question` | 要問AI的問題 | `"圖中有多少個蘋果？"` |

### 可選參數
| 參數 | 說明 | 建議值 |
|-----|------|--------|
| `--lora_model_path` | GRPO模型路徑 | 如要比較則必填 |
| `--verbose` | 顯示詳細資訊 | 建議加上 |
| `--save_result` | 保存結果到文件 | `result.json` |
| `--max_new_tokens` | 回答最大長度 | `128`（短回答）或`512`（詳細回答） |

### 測試模式參數
| 參數 | 說明 | 使用時機 |
|-----|------|---------|
| `--compare_mode` | 比較模式 | 想看詳細比較分析時 |
| `--only_base` | 只測基礎模型 | 只想測試基礎模型 |
| `--only_lora` | 只測GRPO模型 | 只想測試訓練後模型 |

## 📊 結果解讀

### 成功運行的標誌
看到以下信息表示程序正常運行：
```
🚀 啟動 Qwen VL 模型比較測試
✅ 成功加載LoRA適配器模型
🖼️ 加載圖片: /path/to/image.jpg
📐 圖片尺寸: (width, height)
```

### 結果分析重點
1. **思考過程**：看AI是否有邏輯推理
2. **結構化回答**：看是否按格式回答
3. **最終答案**：重點關注數字是否準確
4. **生成時間**：了解模型效率
