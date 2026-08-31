# word2vec（Skip-gram + Negative Sampling）

對應程式：[`./word2vec.py`](./word2vec.py)

## TL;DR
用「一個詞的意思由它周遭的詞決定」（distributional hypothesis）這個假設，訓練出每個詞的低維度向量（embedding），讓語意相近的詞在向量空間中距離也相近。

## 為什麼需要它
- One-hot / BoW 表示法維度等於字典大小，而且任兩個詞的向量都正交 —— 完全看不出「cat」跟「dog」比「cat」跟「car」更相關。
- word2vec 把每個詞壓縮成一個幾十到幾百維的稠密向量，向量之間的距離/方向本身就帶有語意（例如著名的 `king - man + woman ≈ queen`）。
- 是後續 GloVe、FastText，乃至 Transformer 的 token embedding 的概念前身，理解它有助於理解「embedding」這個概念從哪來。

## 核心原理
Skip-gram 的訓練目標：給定中心詞（center word），預測它視窗（window）內的每個上下文詞（context word）。

- 兩組向量：`W`（中心詞用）與 `W_prime`（上下文詞用），訓練完通常只取 `W` 當作最終 embedding。
- 直接對整個字典做 softmax 太貴（字典可能有幾萬到幾十萬字），所以用 **Negative Sampling** 近似：
  - 對每個正樣本 (center, true context)，額外抽 k 個「負樣本」（隨機詞，假設它們大機率不是真正的上下文）。
  - 訓練目標變成一個二元分類：真正的 pair 要讓 `sigmoid(v_c · u_context)` 趨近 1，負樣本的 pair 要讓 `sigmoid(v_c · u_negative)` 趨近 0。
  - 這樣每次更新只碰 `1 + k` 個詞向量，而不是整個字典。

## 程式碼導覽
| 函式 | 對應到理論的哪個部分 |
|---|---|
| `build_vocab` | 把 token 轉成整數索引，之後所有向量運算都靠這個索引查表 |
| `skipgram_pairs` | 依 `window` 大小產生 (center, context) 正樣本 pair |
| `init_embeddings` | 建立 `W`（中心詞向量）與 `W_prime`（上下文詞向量）兩個矩陣 |
| `sigmoid` | 把內積分數轉成機率，`clip` 是為了避免 `exp` 溢位 |
| `train_pair` | 核心：對單一 pair 做一次 negative-sampling 梯度更新（見下方「實作細節」）|
| `train` | 外層迴圈：每個 epoch 洗牌所有 pair，逐一呼叫 `train_pair` |
| `nearest` | 用 cosine similarity 找向量空間中最近的詞，訓練完拿來驗證結果用 |
| `analogy` | 用向量加減法做類比推理（`b - a + c`），本質上就是呼叫 `nearest` |
| `gensim_test` | 用成熟套件 `gensim` 跑同樣的 skip-gram，當作 from-scratch 版本的對照/驗證 |

**實作細節 / 容易看漏的地方：**
- `train_pair` 裡 `grad_center` 是把「正樣本方向」跟「所有負樣本方向」的梯度加總，再一次更新 `W[center_idx]` —— 這對應到 loss 對 `v_c` 的偏微分是所有相關項的和。
- `W[context_idx] = W[context_idx]` 那行其實是 no-op（沒有作用），真正更新 context 向量的是下一行 `W_prime[context_idx] -= ...`；讀程式碼時不要被誤導成 `W` 也有更新 context。
- 負樣本用均勻隨機抽（`rng.integers`），且用 list comprehension 把跟正樣本/中心詞重複的索引濾掉；正式實作（如原論文）通常會用「詞頻的 3/4 次方」做加權抽樣，讓高頻詞被抽到的機率被壓低一些 —— 這裡是簡化版。

## 使用時機 / 優缺點
- ✅ 需要一個輕量、可解釋、不依賴大型預訓練模型的詞向量時（教學、資源有限、baseline 比較）。
- ✅ 想快速做同義詞/類比查詢的小型應用。
- ❌ 每個詞只有「一個」固定向量，無法處理多義詞（例如 "bank" 河岸 vs 銀行意思不會分開）—— 這是後來 contextual embedding（ELMo、BERT）要解決的問題。
- ❌ 對罕見詞、拼字變化（typo、詞形變化）處理不好，這點 FastText（用 subword）有改善，可對照 [`GloVe_FastText_BPE.py`](../GloVe_FastText_BPE/GloVe_FastText_BPE.py) 的筆記。

## 常見誤區
1. **word2vec 不是深度學習模型**：本質上只有一層線性投影 + softmax（或 negative sampling 近似），沒有非線性隱藏層，不要跟後面的 RNN/Transformer 語言模型搞混。
2. **Skip-gram vs CBOW**：Skip-gram 是「中心詞 → 預測上下文」，CBOW 是「上下文 → 預測中心詞」，方向相反。這份程式實作的是 Skip-gram。
3. **負採樣數量 k 不是越多越好**：k 太大等於逼近原本昂貴的 full softmax，失去負採樣省算力的意義；論文建議小資料集 k=5~20，大資料集 k=2~5。

## 複習自問
- 為什麼不直接對整個字典做 softmax？負採樣近似的是哪一步？
- `W` 跟 `W_prime` 分別代表什麼？最後拿來用的是哪一個？
- 如果把 `window` 從 2 改成 10，訓練出來的向量語意會偏向「語法相似」還是「主題相似」？（提示：窗口越大越偏主題/領域相關，窗口小偏語法角色相近）
- `king - man + woman ≈ queen` 這個類比在程式裡對應到哪個函式？
