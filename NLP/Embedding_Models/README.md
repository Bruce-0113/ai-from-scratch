# Embedding Models（Bi-Encoder 檢索 + Matryoshka 截斷 + BGE-M3 混合多向量 + MTEB 評估）

對應程式：[`./embedding_models.py`](./embedding_models.py)

## TL;DR
把文字轉成向量之後要怎麼比對、怎麼壓縮、怎麼組合多種訊號來檢索，這份程式用四個小區塊示範：用bi-encoder算cosine similarity做語意檢索、用Matryoshka截斷向量維度省儲存空間、用BGE-M3同時吐出dense/sparse/multi-vector三種訊號做混合檢索、以及用MTEB標準benchmark驗證模型好壞，而不是只看幾筆玩具語料的排序結果。

## 為什麼需要它
- 語意搜尋、RAG的第一步永遠是「把query跟document都變成向量，再找最近的」——embedding模型怎麼選、向量怎麼壓縮、多種訊號怎麼組合，直接決定retrieval的準確率、儲存成本與延遲。
- 單一dense向量對「意思相近」很敏感，但對罕見詞、專有名詞、型號等字面比對常常抓不住，這是bi-encoder的已知弱點，也是BGE-M3要額外保留sparse訊號的原因。
- 向量維度越高，儲存與計算成本越高（1024維向量存上百萬篇文件，空間相當可觀），Matryoshka讓你不用重新訓練模型，就能在「準確率」與「儲存成本」之間依需求取捨。
- MTEB榜單分數高不代表在自己的領域資料一樣準（醫療、法律、多語言等場景常常跟通用benchmark結果不一致），需要養成用自己資料驗證、而不是只看排行榜選型號的習慣。

## 核心原理

### 1. Bi-encoder dense retrieval
Sentence-BERT系列模型把query跟document各自「獨立」編碼成一個固定長度的向量（不像cross-encoder要把兩段文字一起丟進模型才能算相關性），檢索時只需要對預先算好的文件向量做一次dot product排序即可，這也是bi-encoder能撐起大規模、低延遲檢索的原因——cross-encoder雖然通常更準，但每個(query, doc)配對都要重新跑一次模型，語料一大就跑不動。`normalize_embeddings=True`會讓每個向量長度都變成1，這時dot product在數學上就等於cosine similarity，所以`emb @ q_emb`可以直接當相似度分數使用。

### 2. Matryoshka Representation Learning（MRL）截斷
一般embedding模型如果把向量直接砍短（例如384維只取前128維），語意品質通常會斷崖式下降，因為每個維度重要性相近，砍哪一段都會丟資訊。MRL在訓練時刻意讓「向量的前N維本身就是一個夠用的低維embedding」，因此同一顆模型不用重新訓練，就能依下游需求決定要不要截斷、截多短。常見的官方案例是把1536維向量砍到256維，準確率大約只掉1%，卻換來6倍的儲存空間節省。程式裡的`truncate`就是最小實作：切下前`dim`維後再重新做一次L2 normalize——因為截斷後向量長度不再是1，若忘記重新normalize，後續拿它做cosine similarity會算錯。

### 3. BGE-M3 混合多向量檢索
BGE-M3是單一模型，一次針對同一段文字吐出三種互補的表示法，各自捕捉不同性質的相關性：
- **Dense**（`dense_vecs`，1024維）：跟bi-encoder一樣的整體語意向量，抓的是「意思相不相近」。
- **Sparse / lexical**（`lexical_weights`，格式是`{token_id: weight}`）：模型自己學出來的詞權重，概念上像是可學習版的BM25，抓的是「關鍵字/字面有沒有對上」。
- **ColBERT / multi-vector**（`colbert_vecs`，每個token各一個向量）：用MaxSim計算相似度——對query的每個token，在document所有token向量裡找出最相似的那一個並加總，抓的是細粒度的token級對應，比單一dense向量精細，又比cross-encoder便宜。

三種訊號各有各的弱點（dense抓不到罕見詞/專有名詞，sparse抓不到同義詞改寫，colbert儲存成本最高），所以BGE-M3官方建議把三者依權重加權平均成一個最終分數，程式裡`0.4*dense + 0.2*sparse + 0.4*colbert`就是這個組合公式的示範（實際權重需要依領域資料調整，不是固定不變的常數）。

### 4. MTEB 評估
MTEB（Massive Text Embedding Benchmark）在2022年推出時涵蓋56個任務、8種任務類型，之後的MTEB v2擴充到100+個任務，是目前業界最常用來比較embedding模型優劣的統一榜單。程式最後一段用`MTEB(tasks=[...]).run(encoder, ...)`，把選好的bi-encoder丟進標準檢索任務（ArguAna、SciFact、NFCorpus都是retrieval類任務）驗證，而不是只憑開頭那3筆玩具語料的排序結果就下結論。

## 程式碼導覽
| 區塊 / 函式 | 對應到理論的哪個部分 |
|---|---|
| `SentenceTransformer("BAAI/bge-small-en-v1.5")` + `encoder.encode(...)` | Bi-encoder：把corpus跟query各自獨立編碼成dense向量 |
| `normalize_embeddings=True` | 讓向量變成unit vector，之後`emb @ q_emb`的dot product才等於cosine similarity |
| `scores = emb @ q_emb` | 檢索評分：對每篇文件計算跟query的相似度，`sorted(...)`依分數由高到低排序 |
| `truncate(vectors, dim)` | Matryoshka截斷：切前`dim`維後重新normalize |
| `emb_256` / `emb_128` | 示範同一顆embedding可以截到不同長度，權衡準確率與儲存成本 |
| `BGEM3FlagModel("BAAI/bge-m3", ...)` + `model.encode(..., return_dense=True, return_sparse=True, return_colbert_vecs=True)` | 一次取得BGE-M3的三種訊號：`dense_vecs`、`lexical_weights`、`colbert_vecs` |
| `model.compute_lexical_matching_score(...)` | Sparse訊號的分數計算（對應BM25-like的關鍵字比對） |
| `model.colbert_score(...)` | Multi-vector訊號的MaxSim計算 |
| `final = 0.4*dense + 0.2*sparse + 0.4*colbert` | 把三種訊號加權組合成最終混合檢索分數 |
| `MTEB(tasks=[...]).run(encoder, ...)` | 用標準benchmark（而非玩具語料）評估bi-encoder的檢索能力 |

**實作細節 / 容易看漏的地方：**
- BGE-M3那段（`dense_score` / `sparse_score` / `colbert_score` / `final`）是示範組合公式用的假程式碼：`q_lex`、`d_lex`、`q_col`、`d_col`從沒被賦值，`dense_score = 0.5`也只是佔位數字——重點是理解公式怎麼組合，不是直接執行這段程式。
- `truncate`裡一定要重新做`np.linalg.norm(...)`normalize，因為只取前`dim`維之後向量長度已經不再是1；若忘記重新normalize，後面拿截斷後的向量當cosine similarity用會算錯。
- `normalize_embeddings=True`是retrieval正確性的關鍵開關：沒設的話`emb @ q_emb`算出來的只是dot product，不等於cosine similarity，排序結果會被向量長度而非純語意相似度影響。
- MTEB的`tasks`清單（ArguAna、SciFact、NFCorpus）都屬於retrieval任務，MTEB其實還涵蓋classification、clustering、reranking等8大類任務，這裡只挑了跟這份程式主題（檢索）相關的子集。

## 使用時機 / 優缺點
- ✅ 需要語意搜尋/RAG檢索的第一層召回（recall）：bi-encoder速度快、文件向量可預先算好，擴充到百萬級文件仍然撐得住。
- ✅ 儲存或頻寬吃緊（向量資料庫成本高、embedding傳輸量大）：可用Matryoshka截斷在不重新訓練模型的前提下換取空間，通常截到256維左右仍能保留大部分準確率。
- ✅ 專有名詞、型號、代碼等字面比對很重要的領域（法律條文、產品型號檢索）：BGE-M3的sparse訊號能補上dense訊號抓不到的字面match。
- ❌ ColBERT multi-vector儲存成本最高（每個token都要存一個向量），大規模語料上索引/儲存成本會比純dense高出許多，需要先評估是否真的需要這麼精細的訊號。
- ❌ Bi-encoder獨立編碼本質上犧牲了query與document之間的交互資訊，準確率通常比不上cross-encoder；正式系統常見做法是「bi-encoder做初篩、cross-encoder做重排序（re-rank）」的兩階段架構。
- ❌ MTEB分數高不保證在你的領域資料一樣好，排行榜只能當初篩門檻，正式上線前務必用自己的資料集驗證。

## 常見誤區
1. **任何embedding模型都能安全截斷**：只有明確用Matryoshka目標訓練過的模型，前N維才會是「夠用的低維embedding」；沒訓練過MRL的模型直接砍維度，語意品質可能斷崖式下降，不能一概而論。
2. **cosine similarity跟dot product永遠是同一件事**：只有在向量都被normalize成unit length的前提下，dot product才等於cosine similarity；忘記`normalize_embeddings=True`，或截斷後忘記重新normalize，兩者結果就會不一致。
3. **BGE-M3的三個分數只是同一件事算三次**：dense、sparse、colbert分別捕捉語意相似、字面關鍵字比對、token級細粒度對應，是三種不同性質的訊號，混合權重（`0.4/0.2/0.4`）不是隨便設的比例，而是需要依實際任務調整的超參數。
4. **MTEB排名等於模型在自己資料上表現最好**：MTEB是通用benchmark的平均表現，不是特定領域資料的表現保證，選型號時MTEB只能當篩選門檻，不能取代自己動手做的評估。

## 複習自問
- 為什麼設定`normalize_embeddings=True`之後，`emb @ q_emb`可以直接當成cosine similarity使用？如果拿掉這個設定會發生什麼事？
- `truncate`函式裡如果拿掉重新normalize那一步，後續用截斷後的向量做cosine similarity檢索，會出現什麼問題？
- BGE-M3的dense、sparse、colbert三種訊號分別擅長抓住哪一種相關性？如果檢索場景高度依賴罕見的產品型號或代碼字串，你會傾向調高哪個訊號的權重？
- 為什麼bi-encoder適合拿來做大規模的第一階段召回，而不是直接拿cross-encoder跑過全部文件？
- 為什麼在MTEB榜單上分數最高的模型，不一定是你系統裡最終該選用的模型？
