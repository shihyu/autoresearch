# autoresearch 核心程式說明

一句話總結：這個 repo 是給 AI agent 自動修改 `train.py`、反覆跑 5 分鐘 LLM 預訓練實驗、再用 `val_bpb` 判斷改動有沒有變好的研究框架，不是交易策略系統，也不是直接拿來分析 C/C++ profiler 的工具。

## 1. 這個專案到底在做什麼

這個專案把「研究流程」拆成兩層：

- 固定層：`prepare.py`
  - 準備資料
  - 訓練 tokenizer
  - 提供 dataloader
  - 提供固定評估指標 `evaluate_bpb`
- 可實驗層：`train.py`
  - 定義 GPT 模型
  - 定義 optimizer
  - 定義訓練迴圈
  - 讓 agent 只改這一個檔案做研究

因此它的核心不是「把某個模型訓練到最強」，而是「讓 agent 有一個可重複、可比較、可自動迭代的研究環境」。

## 2. 核心檔案分工

| 檔案 | 角色 | 可否修改 | 重點 |
| --- | --- | --- | --- |
| `prepare.py` | 固定資料與評估層 | 不建議改 | 決定資料來源、tokenizer、dataloader、BPB 評估方式 |
| `train.py` | 研究主戰場 | 可以改 | 模型架構、參數、optimizer、訓練 loop 都在這裡 |
| `program.md` | 給 agent 的操作手冊 | 可以改 | 告訴 agent 怎麼開實驗、怎麼記錄結果、怎麼持續迭代 |

## 3. 核心資料流

```text
prepare.py
  ├─ download_data()
  │    下載 parquet shards 到 ~/.cache/autoresearch/data
  ├─ train_tokenizer()
  │    用訓練資料建立 BPE tokenizer
  ├─ make_dataloader()
  │    把文件打包成 BOS 對齊、無 padding 的 token batch
  └─ evaluate_bpb()
       用固定驗證 shard 算 val_bpb

train.py
  ├─ 讀 tokenizer / 常數
  ├─ build_model_config()
  ├─ GPT(...)
  ├─ setup_optimizer()
  ├─ while True:
  │    forward -> backward -> optimizer.step()
  └─ evaluate_bpb()
       輸出 val_bpb / VRAM / tokens / steps
```

## 4. `prepare.py` 在做什麼

### 4.1 固定訓練與評估邊界

`prepare.py` 先把幾個關鍵常數固定住，讓不同實驗可以公平比較：

- `MAX_SEQ_LEN = 2048`：上下文長度固定。位置：[prepare.py](/home/shihyu/github/autoresearch/prepare.py#L30)
- `TIME_BUDGET = 300`：每次訓練只算 300 秒。位置：[prepare.py](/home/shihyu/github/autoresearch/prepare.py#L31)
- `EVAL_TOKENS = 40 * 524288`：驗證 token 數量固定。位置：[prepare.py](/home/shihyu/github/autoresearch/prepare.py#L32)

這代表 agent 可以亂改模型，但比較標準不能亂掉。

### 4.2 `download_data()`：把資料抓到本地快取

位置：[prepare.py](/home/shihyu/github/autoresearch/prepare.py#L91)

- 輸入：要下載多少個 shard
- 判斷條件：
  - 如果檔案已存在就跳過
  - 驗證 shard 一定會加進去
- 輸出：`~/.cache/autoresearch/data/` 下的 parquet 檔
- 影響：沒有資料就不能訓練，也不能做 tokenizer

### 4.3 `train_tokenizer()`：把文字資料轉成 token 系統

位置：[prepare.py](/home/shihyu/github/autoresearch/prepare.py#L141)

- 輸入：本地 parquet 訓練資料
- 判斷條件：
  - 若 tokenizer 已存在就直接跳過
  - 若 shard 不足 2 個就退出
- 輸出：
  - `tokenizer.pkl`
  - `token_bytes.pt`
- 影響：
  - `tokenizer.pkl` 給訓練時編碼文字
  - `token_bytes.pt` 給 BPB 評估時計算每個 token 的 byte 長度

### 4.4 `make_dataloader()`：把文件塞滿每一列，避免 padding 浪費

位置：[prepare.py](/home/shihyu/github/autoresearch/prepare.py#L275)

這是很重要的 runtime 核心。

- 輸入：
  - `tokenizer`
  - batch size `B`
  - 序列長度 `T`
  - split (`train` / `val`)
- 判斷條件：
  - 每列都以 `BOS` 開頭
  - 優先找「剛好放得下的最大文件」
  - 如果沒有文件放得下，就裁最短文件補滿剩餘空間
- 輸出：
  - `inputs`
  - `targets`
  - `epoch`
- 影響：
  - 幾乎把每個 batch 塞滿
  - 降低 padding 浪費
  - 讓固定 5 分鐘內能訓練更多有效 token

可把它理解成「為了 GPU 吞吐量做的 packing dataloader」。

### 4.5 `evaluate_bpb()`：固定評估標準

位置：[prepare.py](/home/shihyu/github/autoresearch/prepare.py#L343)

- 輸入：`model`、`tokenizer`、驗證 batch size
- 判斷條件：
  - 只用固定 validation shard
  - 特殊 token 不計入 byte 數
  - 交叉熵先累加 nats，再轉 bits/byte
- 輸出：`val_bpb`
- 影響：
  - 這是整個 repo 最重要的勝負指標
  - vocab 大小不同時仍可比較

## 5. `train.py` 在做什麼

### 5.1 `GPTConfig`：集中定義模型結構

位置：[train.py](/home/shihyu/github/autoresearch/train.py#L31)

它描述模型的主要維度：

- `n_layer`
- `n_head`
- `n_embd`
- `window_pattern`

agent 改這裡或其上游組裝邏輯，就等於在改研究假設。

### 5.2 `CausalSelfAttention`：注意力主體

位置：[train.py](/home/shihyu/github/autoresearch/train.py#L60)

這段除了標準 Q/K/V 外，還做了兩件實驗性設計：

1. `Value Embedding`
   - 位置：[train.py](/home/shihyu/github/autoresearch/train.py#L82)
   - 作用：把 token 對應的 value embedding 混進 `v`
   - 特點：不是每層都有，而是交錯啟用，最後一層一定有

2. `window_size` + Flash Attention 3
   - 位置：[train.py](/home/shihyu/github/autoresearch/train.py#L92)
   - 作用：讓某些層只看半個 context，某些層看完整 context
   - 目的：降低部分層的注意力成本，換更高吞吐量

簡化後的關鍵概念如下：

```python
q = self.c_q(x)
k = self.c_k(x)
v = self.c_v(x)

# 用 gate 決定要不要把 value embedding 混進去
if ve is not None:
    v = v + gate * ve

# 套 rotary position embedding
q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)

# 用 flash attention 做因果注意力
y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
```

### 5.3 `MLP`：前饋層

位置：[train.py](/home/shihyu/github/autoresearch/train.py#L98)

- 輸入：每個 token 的 hidden state
- 判斷條件：使用 `relu(x).square()`，不是 GeLU / SwiGLU
- 輸出：投影回原本 embedding 維度
- 影響：這是目前作者選的簡化版激活設計，agent 也可以改這裡測試別的 MLP 形式

### 5.4 `GPT.forward()`：整個模型前向流程

位置：[train.py](/home/shihyu/github/autoresearch/train.py#L267)

它做的事可以讀成：

1. token id 進 embedding
2. 做 RMSNorm
3. 保留一份 `x0` 當跨層殘差來源
4. 每層都先做
   - `resid_lambdas[i] * x`
   - `x0_lambdas[i] * x0`
5. 進 block 做 attention + MLP
6. 最後經過 `lm_head` 產生 logits
7. 若有 target，就算 cross entropy

這裡的重點不是純標準 GPT，而是多了「每層都可學的殘差比例」。

### 5.5 `MuonAdamW`：把不同參數分不同 optimizer

位置：[train.py](/home/shihyu/github/autoresearch/train.py#L355)

這個 optimizer 是 repo 很關鍵的一段。

- matrix 類參數：走 `Muon`
- embedding / lm_head / scalar 類參數：走 `AdamW`

而且 `setup_optimizer()` 會把參數切成多組，給不同 learning rate。位置：[train.py](/home/shihyu/github/autoresearch/train.py#L235)

這代表此 repo 不只是「調模型」，也是在研究：

- 哪類參數應該用哪種 optimizer
- 不同參數群要不要用不同 LR
- Muon 在這個小型預訓練問題上是否比純 AdamW 更好

### 5.6 超參數區：agent 主要調這裡

位置：[train.py](/home/shihyu/github/autoresearch/train.py#L427)

最常被改動的地方通常是：

| 參數 | 作用 |
| --- | --- |
| `ASPECT_RATIO` | 決定 model_dim 與 depth 的比例 |
| `HEAD_DIM` | 決定每個 attention head 的寬度 |
| `WINDOW_PATTERN` | 決定哪些層看長窗、哪些層看短窗 |
| `TOTAL_BATCH_SIZE` | 每次 optimizer step 的總 token 數 |
| `EMBEDDING_LR` / `UNEMBEDDING_LR` / `MATRIX_LR` | 各類參數的學習率 |
| `WEIGHT_DECAY` | Muon 的 cautious weight decay 強度 |
| `DEPTH` | transformer 層數 |
| `DEVICE_BATCH_SIZE` | 單卡 micro-batch，大了可能 OOM |

### 5.7 訓練迴圈：固定 5 分鐘內盡量學到最多

位置：[train.py](/home/shihyu/github/autoresearch/train.py#L542)

核心工作：

1. 做 gradient accumulation
2. 根據目前進度調整 LR / Muon momentum / weight decay
3. 更新參數
4. 追蹤吞吐量與 MFU
5. 超過時間預算就停
6. 最後再做一次固定驗證

可直接把這段理解成：

```python
while True:
    for micro_step in range(grad_accum_steps):
        loss = model(x, y)
        loss.backward()
        x, y = next(train_loader)

    # 依照已經花掉的訓練時間調整 schedule
    update_lr_and_muon_state()
    optimizer.step()
    model.zero_grad(set_to_none=True)

    if total_training_time >= TIME_BUDGET:
        break
```

這裡不是追求「訓練到收斂」，而是追求「在固定 5 分鐘內得到最低的 `val_bpb`」。

## 6. `program.md` 範例是用來做什麼

`program.md` 不是模型設定檔，而是「給 agent 的研究 SOP」。

位置：[program.md](/home/shihyu/github/autoresearch/program.md)

它要求 agent：

1. 建新 branch
2. 讀 `README.md`、`prepare.py`、`train.py`
3. 檢查資料是否已下載
4. 建 `results.tsv`
5. 每次只改 `train.py`
6. 跑 `uv run train.py > run.log 2>&1`
7. 從 log 擷取 `val_bpb` / `peak_vram_mb`
8. 改進就保留 commit，沒改進就回退
9. 無限迴圈做實驗

所以這份範例真正的用途是：

- 把 agent 從「一般聊天助理」變成「會持續做研究迭代的自動化研究員」
- 規範它的邊界
- 規範它如何記錄結果
- 規範它如何判斷實驗成功或失敗

簡單講，`program.md` 是研究流程的 prompt 程式，不是模型程式。

## 7. 這份程式能拿來做什麼

### 可以直接拿來做的事

- 自動化 LLM 預訓練研究
- 比較不同模型架構
- 比較不同 optimizer / LR schedule
- 比較不同 batch size / depth / attention window 設計
- 讓 agent 在固定時間預算內自己做 ablation

### 不能直接拿來做的事

- 直接優化 C/C++ 程式效能
- 直接研究交易策略
- 直接做回測撮合、風控、下單
- 直接分析 CPU cache miss、branch prediction、SIMD 熱點

## 8. 那它能不能拿去優化 C/C++ 效能？

結論：不能直接拿來做，但可以改造成「自動實驗框架」。

原因很直接：

- 它目前的輸入是文字資料，不是 C/C++ 程式碼與 profiler 結果
- 它目前的目標函數是 `val_bpb`，不是執行時間、IPC、cache miss、binary size
- 它目前的執行流程是訓練神經網路，不是編譯 C/C++、跑 benchmark、比較 perf 結果

如果要改成 C/C++ 效能優化平台，至少要把下面幾塊換掉：

| 現在 | 要改成什麼 |
| --- | --- |
| `prepare.py` 的資料集 | C/C++ 專案、benchmark case、編譯腳本 |
| `train.py` 的目標 | 編譯後跑 benchmark / `perf` / `hyperfine` |
| `evaluate_bpb()` | 執行時間、throughput、cache miss、記憶體占用 |
| `program.md` | 改成 agent 自動改 C/C++ 或 build flags，再驗證效能是否提升 |

所以它比較像「可借鏡的自治實驗框架」，不是現成的 C/C++ optimizer。

## 9. 那它能不能拿去研究交易策略？

結論：也不能直接拿來做，但可以借概念改造成策略研究 agent。

原因：

- 沒有行情資料 ingest
- 沒有訊號產生邏輯
- 沒有部位管理
- 沒有回測撮合
- 沒有風控
- 沒有 PnL / Sharpe / drawdown 指標

如果要改成交易策略研究框架，需要把核心指標換成：

- 年化報酬
- Sharpe / Sortino
- 最大回撤
- turnover
- slippage 後績效

而且 `train.py` 也要從「神經網路預訓練」改成「策略生成 / 特徵組合 / 參數搜尋 / 回測」。

## 10. 最精準的定位

如果只用一句話定義這個 repo：

> 這是一個讓 AI agent 自動做 LLM 訓練實驗與模型/優化器探索的最小研究平台。

不是：

- C/C++ profiler
- 自動 benchmark 調校器
- 交易策略回測框架
- 量化交易 bot

## 11. 這程式厲害的地方

這個 repo 真正強的不是「能跑一個 GPT」，而是把「自動研究流程」壓到很小、很乾淨、很容易反覆實驗。

### 11.1 公平比較機制做得很完整

- 固定 `MAX_SEQ_LEN`、`TIME_BUDGET`、`EVAL_TOKENS`。位置：[prepare.py](/home/shihyu/github/autoresearch/prepare.py#L30)
- 固定用 `evaluate_bpb()` 當最終指標。位置：[prepare.py](/home/shihyu/github/autoresearch/prepare.py#L343)
- `val_bpb` 是 vocab-size-independent metric，不容易被 tokenizer 大小誤導

這很重要，因為很多研究腳本可以跑，但不容易公平比較；這個 repo 一開始就把比較標準鎖住了。

### 11.2 只改一個檔案，agent 很容易自動化

- 真正研究主戰場幾乎都在 [train.py](/home/shihyu/github/autoresearch/train.py)
- `prepare.py` 保持固定
- `program.md` 把 branch、log、保留/回退規則都定義好

這種切法很厲害，因為它刻意降低 agent 亂改整個 repo 的風險，讓每次實驗變得可控。

### 11.3 dataloader 很重視吞吐量，不是隨便湊 batch

`make_dataloader()` 會：

- 每列都從 `BOS` 開始
- 優先找放得下的最大文件
- 放不下就裁最短文件補滿
- 盡量維持 100% token 利用率

位置：[prepare.py](/home/shihyu/github/autoresearch/prepare.py#L275)

這種 best-fit packing 的實作很務實，因為在固定 5 分鐘預算下，能多餵一點有效 token，通常就更有研究價值。

### 11.4 模型不是純 baseline，裡面有真的研究點

模型裡面不是只有標準 GPT，還混了幾個可探索設計：

- `Value Embedding` 混進 attention value。位置：[train.py](/home/shihyu/github/autoresearch/train.py#L82)
- 長窗/短窗交錯 attention pattern。位置：[train.py](/home/shihyu/github/autoresearch/train.py#L194)
- 每層可學的殘差比例 `resid_lambdas` / `x0_lambdas`。位置：[train.py](/home/shihyu/github/autoresearch/train.py#L133)

這表示它不是單純調參平台，也是在鼓勵架構實驗。

### 11.5 optimizer 設計有工程深度

這個 repo 不是整包都用 AdamW，而是把參數分群：

- 矩陣類參數走 Muon
- embedding / lm_head / scalar 走 AdamW

位置：

- [train.py](/home/shihyu/github/autoresearch/train.py#L235)
- [train.py](/home/shihyu/github/autoresearch/train.py#L355)

再加上 `torch.compile` 和 fused step kernel，這代表作者連 optimizer update path 都在追效能，不只是追模型正確性。

### 11.6 它真正厲害的是「研究流程產品化」

`program.md` 的設計讓 agent 可以：

1. 改 `train.py`
2. commit
3. 跑實驗
4. 擷取 `val_bpb`
5. 改進就保留，退步就回退
6. 持續循環

位置：[program.md](/home/shihyu/github/autoresearch/program.md)

這很像把研究員的日常流程，包裝成一個可自動執行的最小 operating system。

### 11.7 它的厲害是「小而完整」

很多研究專案的問題是：

- 太大，agent 看不完
- 太鬆，評估標準常變
- 太雜，改一個點會連帶破很多地方

這個 repo 反過來：

- 檔案少
- 邊界清楚
- 指標固定
- 迭代路徑明確

所以非常適合做 autonomous research 的起點。

## 12. 你如果想往別的方向延伸

### 想做 C/C++ 效能優化

保留的概念：

- agent 自動改檔
- 固定時間或固定 benchmark 預算
- 單一指標比較好壞
- 改進就保留、退步就回退

要替換的核心：

- `train.py` 改成 build-and-benchmark loop
- `evaluate_bpb()` 改成 benchmark evaluator
- `prepare.py` 改成 benchmark dataset / build environment setup

### 想做交易策略研究

保留的概念：

- agent 自動產生實驗
- 固定回測資料區間
- 固定績效指標
- 改進就保留、退步就回退

要替換的核心：

- `train.py` 改成回測與策略搜尋 loop
- `prepare.py` 改成市場資料整理與特徵快取
- `evaluate_bpb()` 改成策略績效評估器

## 13. 總結

最重要的三句話：

1. 這個 repo 的主題是「AI agent 自動做 LLM 預訓練研究」。
2. `program.md` 範例是讓 agent 依照固定 SOP 持續做實驗，不是一般說明文件。
3. 它不能直接拿來優化 C/C++ 效能或研究交易策略，但它的「自治實驗框架」設計很值得拿去改造成別的研究系統。
