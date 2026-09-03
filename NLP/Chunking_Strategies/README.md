# Chunking Strategies for RAG（Fixed / Recursive / Semantic / Parent-Document / Contextual Retrieval）

對應程式：[`./chunking_strategies.py`](./chunking_strategies.py)

## TL;DR
RAG系統裡，把文件切成多大、怎麼切、切完要不要加上下文，對檢索準確率的影響不亞於選哪個embedding模型。這份程式用五種互補策略示範：最簡單的定長切割（fixed）、依段落/換行/句子逐層退讓的遞迴切割（recursive）、依句子間語意相似度斷句的語意切割（semantic）、切小塊檢索但回傳大塊上下文的parent-document、以及在索引前替每個chunk加上LLM生成之上下文摘要的contextual retrieval，最後用`recall_at_k`把「哪種策略比較好」變成可以實測、而不是憑印象猜的問題。

## 為什麼需要它
- 把一份50頁合約丟進RAG，問「終止條款是什麼」，檢索卻回傳封面頁——問題通常不是embedding模型不夠好，而是chunking切錯了：條款跨頁被攔腰切斷，附近沒有能跟查詢對上的關鍵字。
- 「語意切割 + 20%重疊 + 1000 tokens」聽起來像是最穩妥的預設值，但2026年的多份benchmark顯示現實常常相反：Vectara的研究裡，單純的512-token遞迴切割（69%）反而贏過語意切割（54%）；SPLADE + Mistral-8B在Natural Questions上發現重疊幾乎沒有帶來可量測的效益。這代表chunking策略必須針對自己的資料與查詢類型實測，不能照抄部落格的「最佳實踐」。
- 不同查詢類型需要不同的chunk大小：找一個事實（factoid）用小chunk（256-512 tokens）雜訊少；跨段落的分析型查詢（analytical / multi-hop）需要512-1024 tokens才裝得下足夠的上下文；理解整個章節則需要1024-2048 tokens。chunk太小裝不下答案，太大則稀釋掉檢索分數、引入不相關內容。
- Context cliff（上下文懸崖）:研究觀察到當放入LLM的上下文超過約2500 tokens時，回答品質會明顯下降——這代表就算檢索到了對的內容，chunk切法（以及retrieve回來的parent chunk大小）如果讓上下文塞得太多，一樣會拖累最終答案品質。

## 核心原理

### 1. Fixed chunking（`chunk_fixed`）
最陽春的做法：每`size`個字元切一刀，重疊`overlap`個字元再往下切。完全不管句子或段落邊界，切點常常落在句子中間，但實作最簡單、壓縮率最好，是所有策略的效能與品質基準線（baseline）。

### 2. Recursive chunking（`chunk_recursive`）
對應LangChain `RecursiveCharacterTextSplitter`的做法，也是目前最常見的正式環境預設值。核心想法是「依優先順序嘗試分隔符」：先試段落分隔（`\n\n`），找不到才試換行（`\n`），再來是句子（`. `），最後是空白。切開之後貪婪地把片段黏回去，盡量湊到接近但不超過`size`；如果某個片段切完還是超過`size`（例如一段沒有任何換行的超長文字），就把剩下更細的分隔符（`seps[1:]`）遞迴丟進去繼續切，直到全部分隔符都試過為止才退回逐字元的`chunk_fixed`。相較fixed chunking，好處是切點會盡量落在自然邊界上，壞處是實作邏輯比定長切割複雜、需要遞迴。

### 3. Semantic chunking（`chunk_semantic`）
先把文字切成一句一句（外部提供的`split_sentences`），對每句做embedding，再依序比較相鄰兩句的cosine similarity（因為向量都是L2-normalized，dot product直接等於cosine similarity）。當相似度低於`threshold`、且目前累積的chunk長度已經超過`min_chars`時才切一刀開新chunk——`min_chars`這個門檻是關鍵防呆機制，沒有它的話，只要中間有一句話語意稍微跳躍，就會切出40 tokens左右的破碎小chunk，反而傷害檢索品質。切出來若超過`max_chars`,則呼叫`chunk_recursive`進一步拆分,不會回傳過長的chunk。語意切割理論上能讓每個chunk的主題更集中,但計算成本比recursive高很多,而且不見得永遠比recursive準——這正是核心原理最前面提到的Vectara benchmark結果。

### 4. Parent-document chunking（`chunk_parent_child` + `retrieve_parent`）
把「拿什麼去檢索」跟「回傳什麼當上下文」拆成兩層:`chunk_parent_child`先用`chunk_recursive`切出大的parent chunk（預設2048字元），每個parent再切成小的child chunk（預設256字元），並記錄每個child屬於哪個parent。檢索時（`retrieve_parent`）只拿小巧、主題聚焦的child chunk去算相似度分數，因為小chunk的embedding訊噪比通常更好；但真正回傳給LLM的是child所屬的parent chunk，藉此拿回child切割時流失的前後文。這個設計即使child chunk切得不理想,通常也能還原出合理的parent，屬於「優雅降級」的架構。

### 5. Contextual retrieval（`contextualize_chunks`）
Anthropic在2024年提出的做法:在把每個chunk送進索引之前,先讓LLM讀「整份文件 + 這個chunk」,生成一段50-100字、描述這個chunk在文件中位置與脈絡的摘要,再把摘要接在chunk前面一起索引。這解決了chunk被獨立抽出來後語意不完整的問題（例如一段只寫「該條款於次年生效」,不知道在講哪個條款）。代價是索引階段每個chunk都要多跑一次LLM呼叫,成本明顯高於前四種策略,官方benchmark顯示能帶來35-50%的檢索改善。

### 6. 用`recall_at_k`實測，而非直接採用預設值
`recall_at_k`對每筆(query, gold_idxs)算出查詢的embedding,對整個語料庫的chunk依相似度排序,檢查前k名裡有沒有命中標註為正確答案的chunk索引,最後回傳命中率。這是本檔案五種策略共用的評估工具——選哪種策略、多大的chunk size、要不要重疊,都應該先用一組有標註答案的查詢集在自己的資料上跑過recall@k,而不是直接套用「recursive 512 tokens是2026年預設值」這類通則。

## 程式碼導覽
| 函式 | 對應到理論的哪個部分 |
|---|---|
| `chunk_fixed(text, size, overlap)` | 定長切割 baseline：每`size`字元切一刀，`step = size - overlap`控制重疊量 |
| `chunk_recursive(text, size, seps)` | 遞迴切割：依`seps`優先順序找第一個存在於文字中的分隔符，貪婪合併片段到接近`size`，超長片段用`seps[1:]`遞迴細切 |
| `chunk_semantic(text, encoder, threshold, min_chars, max_chars)` | 語意切割：`embs[i] @ embs[i-1]`算相鄰句子cosine similarity，低於`threshold`且長度超過`min_chars`才切新chunk |
| `chunk_parent_child(text, parent_size, child_size)` | Parent-document索引建置：先切parent，再對每個parent切child，記錄`parent_idx`對應關係 |
| `retrieve_parent(child_query, mapping, encoder, top_k)` | Parent-document檢索：對child chunk排序取`top_k`，依命中順序回傳去重後的parent |
| `contextualize_chunks(document, chunks, llm)` | Contextual retrieval：對每個chunk生成情境摘要（`llm.batch`），摘要接在chunk前面一起回傳 |
| `recall_at_k(queries, corpus_chunks, encoder, k)` | 評估工具：對每筆查詢排序語料庫chunk，統計前`k`名命中`gold_idxs`的比例 |

**實作細節 / 容易看漏的地方：**
- `chunk_recursive`只嘗試`seps`裡「第一個出現在文字中」的分隔符就直接處理並回傳,不會逐一嘗試每種分隔符;如果`seps`裡沒有任何一種出現在文字裡,才會落到最後一行的`chunk_fixed`當作退路。
- `chunk_semantic`依賴的`split_sentences`與`contextualize_chunks`依賴的`llm.batch`都不在這支檔案裡定義,屬於呼叫端要提供的外部依賴（前者是斷句工具如nltk，後者是LLM client的批次呼叫介面）。
- `retrieve_parent`用`seen`集合對`parent_idx`去重——多個child chunk很可能對應到同一個parent，若不去重，`top_k`筆結果裡可能會塞進重複的parent，浪費上下文額度。
- 所有`encoder.encode(..., normalize_embeddings=True)`都假設回傳的向量是L2-normalized，因此程式裡用`@`（dot product）直接當cosine similarity使用；如果自己替換成不會normalize的encoder，相似度分數與排序都會跑掉。
- `recall_at_k`的`gold_idxs`是需要人工標註的正確答案索引集，這代表要用這支函式評估任何策略之前，必須先準備一組有標註的查詢集，不能省略這一步直接比較策略優劣。

## 使用時機 / 優缺點
依查詢類型挑chunk大小（一般性的起始建議，仍需用`recall_at_k`在自己資料上驗證）：

| 查詢類型 | 建議chunk大小 |
|---|---|
| Factoid（「執行長叫什麼名字？」） | 256-512 tokens |
| Analytical / multi-hop | 512-1024 tokens |
| 整個章節的理解型查詢 | 1024-2048 tokens |

- ✅ 第一次建置、還不了解語料特性：`chunk_recursive`搭配512 tokens、不加重疊，是最省事也最少踩雷的起手式。
- ✅ 需要小chunk精準檢索、但又不想犧牲上下文：`chunk_parent_child` + `retrieve_parent`的兩層設計，兼顧檢索精度與回傳內容的完整性。
- ✅ 合約、論文等大量跨頁/跨段引用的文件：`contextualize_chunks`能顯著改善檢索，但要能負擔索引時的LLM呼叫成本。
- ❌ `chunk_semantic`在同質性高、語意連貫的文字上不見得比`chunk_recursive`準，卻要多付出embedding每一句話的計算成本，需要先用`recall_at_k`驗證是否真的值得。
- ❌ `contextualize_chunks`每個chunk都要呼叫一次LLM，索引成本隨文件數量線性增加，不適合語料量巨大或需要頻繁重新索引的場景。
- ❌ 本檔案沒有實作「句子級切割」與「late chunking（先embedding整份文件、再pool成chunk向量）」這兩種策略——需要更細粒度切割或想保留跨chunk上下文時，得另外實作或改用支援長上下文的embedding模型（如BGE-M3、Jina v3）。

## 常見誤區
1. **重疊（overlap）是必需的安全邊際**：多份2026年的研究發現overlap經常對recall沒有可量測的幫助，卻讓索引成本翻倍（`chunk_fixed`的`overlap`參數要實測後再決定是否使用，不是預設就該調高）。
2. **語意切割一定比recursive準**：Vectara的benchmark顯示recursive 512-token切割（69%）在他們的評測中反而贏過語意切割（54%），策略優劣高度依賴語料與查詢分佈，不能假設「更聰明的方法」永遠更準。
3. **`chunk_semantic`不設`min_chars`也沒關係**：拿掉`min_chars`門檻後，任何一句話語意稍微跳躍就會切出極短的片段，這種零碎chunk通常會拖累檢索品質，`min_chars`不是可有可無的參數。
4. **只用factoid查詢評估chunking策略就夠了**：factoid與multi-hop查詢對chunk大小的最佳選擇明顯不同，只用單一查詢類型的`recall_at_k`結果去決定全站的chunking設定，容易得出偏頗的結論。
5. **chunk可以跨文件切**：`chunk_recursive`與`chunk_semantic`都只接受單一`text`參數，正確用法是先依文件切分、對每份文件各自呼叫chunking函式，絕不能把多份文件串在一起再整批切，否則chunk會混入不相關文件的內容。

## 複習自問
- `chunk_recursive`裡`seps=seps[1:] or (" ",)`這行在做什麼？如果`seps`只剩一個元素時繼續遞迴，退路是什麼？
- `chunk_semantic`為什麼需要`min_chars`這個門檻？拿掉它之後，在一段語意跳躍頻繁的文字上會發生什麼事？
- `retrieve_parent`如果拿掉`seen`去重邏輯，`top_k=3`卻只回傳1種parent內容的情況會如何發生？
- 為什麼`contextualize_chunks`要把整份`document`也丟進prompt，而不是只把`chunk`本身丟給LLM生成摘要？
- 如果要用`recall_at_k`比較`chunk_recursive`與`chunk_semantic`在自己語料上的優劣，除了程式本身，還需要額外準備什麼資料？為什麼不能只看兩三筆查詢的結果就下結論？
