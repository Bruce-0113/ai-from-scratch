# Long-Context Evaluation（NIAH / Multi-Needle / RULER / LongBench v2）

對應程式：[`./long_context_evaluation.py`](./long_context_evaluation.py)

## TL;DR
模型廠商說支援1M甚至10M tokens的context window，不代表這1M tokens都「真的能用」。把一份200頁合約整份丟進去問「終止條款是什麼」，模型答得出來，但答案來自封面頁——因為真正的條款藏在12萬token深的地方，落在模型實際會「注意到」的範圍之外。這正是2026年的context容量落差：規格表寫1M，實際可用（依任務而定）往往只有60-70%。這份程式示範四種互補的量測方式：單一事實的needle-in-a-haystack（NIAH）、需要同時抓住多個事實的multi-needle、需要串連多步推理的RULER風格變數追蹤（multi-hop tracing）、以及在真實世界文件上跑的LongBench v2——分別對應「能不能找到」跟「找到之後能不能推理」這兩個不同難度的問題。

## 為什麼需要它
- Context window的規格數字只講「模型技術上能吃下多少token」，不講「吃下去之後還記不記得住、用不用得到」——這兩件事在2026年的模型上落差可以很大：Gemini 3 Pro宣稱1M tokens，但在1M深度的8-needle MRCR測試中，準確率掉到只剩26.3%。
- 三種任務對長context的敏感度完全不同：單一事實檢索（retrieval）在frontier模型上幾乎能做到接近滿分、撐到規格表宣稱的上限；多跳/彙總（multi-hop / aggregation）過了約128k就開始明顯degrade；分散在整份文件各處、需要串連推理的任務則是最先垮掉的一種。只看retrieval分數會嚴重高估模型的實際長文本能力。
- 「Lost in the middle」（Liu et al., 2024）：模型對長輸入「中段」內容的注意力系統性地比開頭跟結尾弱，如果NIAH測試只固定測`depth_ratio=0.5`，量到的其實是最差情況附近的分數，若只測開頭或結尾則會顯著低估風險——必須把多個深度都掃過一輪才看得出真正的形狀。
- Needle如果跟filler內容有字面上的關鍵字重疊，檢索會變得太簡單、量不出模型真正的長文本理解能力；真正有鑑別度的測試（如NoLiMa）needle跟question之間刻意不共用任何字面詞彙，逼模型做一步語意推理才能找到答案。
- 每次模型升級（換家供應商、換版本）都可能改變其真實可用的context長度——沒有一套固定的回歸測試組（regression harness），這種「這次升級到底是變好還是變差」的問題只能靠上線後使用者回報才會發現。

## 核心原理

### 1. Needle-in-a-Haystack (NIAH)：長文本評估的起點（`build_haystack` / `score_niah`）
最初由Greg Kamradt在2023年提出，做法是把一句與filler內容無關的事實（needle，例如「the magic word is pineapple」）插進一段長度可控的filler文字（haystack）裡的某個深度，再問模型「這個事實是什麼」。掃過`depth_ratio`（插入深度，0=最前面、1=最後面）× `total_tokens`（haystack總長度）這兩個維度，畫成一張熱力圖，就是NIAH的標準呈現方式。這是長文本評估最早、也最基礎的一種測試——目前的frontier模型大多已經能在這個測試上跑到接近滿分，代表它是必要但不充分的baseline：能通過NIAH只證明模型「能從長輸入裡撈出一句沒有推理需求的話」，不代表它能做多跳推理或彙總。

### 2. Multi-needle：單一needle測不出的「同時記住多件事」（`build_multi_needle`）
在haystack裡的多個深度（本程式用10%、40%、70%）各埋一個needle，問題要求把全部needle都答出來（例如「這三個magic word分別是什麼？」）。單一needle測試只驗證「模型能不能從長輸入抓出一個事實」，不代表模型能同時在注意力裡「hold住」多個分散的事實——實務上單一needle測試接近滿分的模型，multi-needle測試的分數經常明顯下降，這是MRCR（Multi-Round Coreference Resolution）等benchmark特別設計8-needle、24-needle、甚至100-needle版本的原因：needle數量一多，就能看出模型的注意力機制在哪個規模開始飽和。

### 3. RULER風格的多跳變數追蹤：從「找得到」到「推得出來」
NVIDIA在2024年提出的RULER，把長文本評估擴充到13種任務、4大類別：retrieval（單一/多key/多value）、multi-hop tracing（變數追蹤）、aggregation（詞頻統計）、以及QA。本程式裡的module-level範例`haystack`/`question`（`X1 = 42 ... X2 = X1 + 10 ... X3 = X2 * 2`，問「X3是多少？」）就是最簡化版本的multi-hop tracing：答案需要依序完成三次賦值運算，而不是單純把某一行文字複製貼上。RULER在2024年釋出時發現，17個宣稱支援32k以上context的模型裡，只有一半在32k時還能維持品質——代表「能通過NIAH」跟「能通過RULER」是兩件不同的事，NIAH飽和的模型在multi-hop任務上依然可能大幅退步（frontier模型在128k常常掉到只剩50-70%準確率）。

### 4. LongBench v2：在真實世界文件上量測，而非合成的needle（`eval_model_on_longbench`）
NIAH與RULER都是「人工插入的合成事實」，跟真實使用情境（真的很長的合約、程式碼庫、多輪對話）之間仍有落差。LongBench v2（2024）提供503題選擇題，context長度從8k到2M字不等，涵蓋六大類真實任務：單文件QA、多文件QA、長文本in-context learning、長對話、程式碼庫理解、長結構化資料。本程式透過Hugging Face的`datasets.load_dataset("THUDM/LongBench-v2")`載入資料集，並用`eval_model_on_longbench`過濾出指定`subset`（如`single-doc-qa`）逐題送進模型評分，是目前業界公認最貼近正式環境長文本表現的benchmark。

### 5. 其他值得知道的benchmark：NoLiMa / HELMET / BABILong
- **NoLiMa（"Non-lexical needle"）**：needle與question刻意不共用任何字面詞彙，逼模型做至少一步語意推理才能連結兩者，比傳統NIAH更難、也更能反映真實檢索場景（真實查詢很少跟答案逐字重複）。
- **HELMET**：把多份文件串接起來，問題可能只跟其中一份文件有關，測試模型能不能做「選擇性注意力」（selective attention），忽略掉不相關的文件。
- **BABILong**：把bAbI推理鏈（一系列需要邏輯推理的小故事題）埋進大量不相關的filler裡，測的是「在haystack裡做推理」而不只是「在haystack裡做檢索」，是目前這幾個benchmark裡對推理能力要求最高的一種。

### 6. 該回報的兩個數字：有效檢索長度 vs. 有效推理長度
規格表上的「advertised context window」只是廠商的宣稱值，正式評估報告應該額外算出兩個更誠實的數字：**有效檢索長度**（effective retrieval length，NIAH在某個門檻，例如90%通過率下能撐到的最大長度）與**有效推理長度**（effective reasoning length，multi-hop或aggregation任務在較低門檻，例如70%通過率下能撐到的最大長度）。經驗法則是，有效推理長度通常只有宣稱context window的25%-50%——這才是規劃RAG chunk大小、決定要不要做多輪摘要、評估「這個模型能不能撐住我的使用情境」時真正該參考的數字。

## 程式碼導覽
| 函式 / 區塊 | 對應到理論的哪個部分 |
|---|---|
| `build_haystack(filler_text, needle, depth_ratio, total_tokens)` | NIAH haystack產生器：把filler tokens重複到填滿`total_tokens - len(needle)`，再依`depth_ratio`把needle插入指定深度 |
| `score_niah(model, haystack, question, expected)` | NIAH評分：對模型輸出做寬鬆的、不分大小寫的substring比對，而非精確匹配 |
| `build_multi_needle(filler, needles, total_tokens)` | Multi-needle haystack產生器：在10%/40%/70%三個固定深度插入needle，`zip`只會用到前3個needle |
| 模組層級的`haystack` / `question`（`X1 = 42 ... X3 = X2 * 2`） | RULER風格的multi-hop變數追蹤範例：答案需要串連三次賦值，而非單一事實檢索 |
| `longbench = load_dataset("THUDM/LongBench-v2")` | 載入真實世界長文本QA benchmark；模組層級程式碼，`import`這個檔案時就會立刻觸發資料集下載 |
| `eval_model_on_longbench(model, subset)` | 用指定`subset`（如`single-doc-qa`）過濾LongBench v2的`test` split，逐題送進模型並用`normalize`後的exact match計算accuracy |

**實作細節 / 容易看漏的地方：**
- `build_haystack`用的`tokenize`跟`eval_model_on_longbench`用的`normalize`在檔案裡都沒有定義，是需要呼叫端自行提供的佔位符（對應原始教材的示範程式碼），直接執行會丟出`NameError`——`tokenize`通常對應某個真實tokenizer的`encode`/`decode`，`normalize`通常是「轉小寫 + 去標點」這類正規化函式。
- `score_niah`跟`eval_model_on_longbench`用的`model`參數，介面上只要求有`.complete(prompt, max_tokens) -> str`，本身沒有預設實作，呼叫端需要自行包一層符合這個介面的wrapper。
- `score_niah`用的是「`expected`是否為`answer`的不分大小寫substring」，比對邏輯比LongBench v2用的`normalize(answer) == normalize(x["answer"])`精確匹配寬鬆很多——這代表NIAH分數比較容易因為「答案剛好包含關鍵字，但整體答非所問」而被高估，正式使用時應視情況換成更嚴謹的比對邏輯。
- `build_multi_needle`直接對`filler`做位置切片（`filler[:int(total_tokens * 0.1)]`），而不是像`build_haystack`一樣先`tokenize`再切——如果`filler`是字串，切片單位就是字元而非token，呼叫端需要自行確保`total_tokens`與`filler`的單位一致。
- `depths = [0.1, 0.4, 0.7]`跟三個切片區段的比例（每段0.3）都寫死在函式內部，`needles`若少於3個，`zip`會提早停止；若多於3個，多出來的needle會被silently忽略，不會報錯提示。
- `longbench = load_dataset(...)`是模組層級程式碼，`import`這個檔案的當下就會立刻連網下載`THUDM/LongBench-v2`資料集，不是包在函式或`if __name__ == "__main__":`裡面——正式使用時應該把這行搬進函式，避免單純`import`就觸發一次資料集下載。
- `eval_model_on_longbench`如果`tasks`過濾後是空list（`subset`打錯或該類別在資料集裡不存在），`return correct / len(tasks)`會丟出`ZeroDivisionError`，沒有額外的防呆訊息。

## 使用時機 / 優缺點
2026年常見的用法對照表（依評估目的挑benchmark，正式環境通常會同時用到多種）：

| 用途 | 建議做法 |
|---|---|
| 快速健檢（sanity check） | 自訂NIAH，3個深度 × 3個長度 |
| 正式環境的模型選型 | RULER（13種任務），跑在目標context長度上 |
| 真實世界QA品質 | LongBench v2的`single-doc-qa`子集 |
| 多跳推理能力 | BABILong或自訂的變數追蹤任務 |
| 對話／多輪coreference | MRCR的8-needle版本，跑在目標context長度上 |
| 模型升級的回歸測試 | 固定的NIAH + RULER測試組，每次換模型都重跑一次，比對前後差異 |

- ✅ NIAH的heatmap直觀、實作成本低，是最快能發現「這個模型在特定深度區間退步」的方式，尤其能揪出lost-in-the-middle效應。
- ✅ Multi-needle與RULER的multi-hop tracing能揭露NIAH測不出的弱點——很多模型NIAH接近滿分，但multi-hop一測就掉到五六成，兩者放在一起看才不會被單一指標誤導。
- ✅ LongBench v2用真實世界文件與人工標註問題，比合成needle更貼近正式環境的實際表現，能抓到「合成測試表現很好、真實資料表現很差」的落差。
- ✅ 用固定的評估組（fixed harness）在每次模型升級後重跑，能把「這次升級到底變好還是變差」變成可以量化比較的問題，而不是憑感覺猜測。
- ❌ 只測NIAH會嚴重高估模型的長文本能力：在1M tokens通過NIAH，不代表模型能在1M tokens裡做multi-hop推理，兩者必須分開測。
- ❌ 如果needle跟filler內容有字面關鍵字重疊，或所有測試都只測`depth_ratio=0.5`，量出來的分數會系統性偏高，掩蓋掉真正的弱點——深度要多點掃過，needle最好做到字面不重疊（NoLiMa風格）。
- ❌ 廠商自行公布的長文本benchmark分數（OpenAI、Google、Anthropic皆然）僅供參考，正式決策前應該在自己的使用情境與資料上獨立重跑一次，不能直接採信官方數字。
- ❌ 超長prompt（如1M tokens）的prefill時間可能長達30-120秒，只看準確率會忽略延遲成本——正式評估報告應該同時記錄time-to-first-token，而不只是分數。
- ❌ 合成benchmark（NIAH、RULER、BABILong）再怎麼精心設計，終究是人工構造的測試情境，跟自己領域的真實文件、真實問題分佈仍有落差，不能完全取代針對自己使用情境的客製化測試。

## 常見誤區
1. **模型卡上寫的context window就是「可用」的長度**：規格表數字只代表技術上能吃下的token數，不代表在那個長度下模型還記得住、答得準——正式評估前不能只信任model card，一定要自己跑一輪NIAH加上至少一項推理型任務。
2. **NIAH滿分代表這個模型的長文本能力沒問題**：NIAH只驗證單一事實retrieval，完全不驗證multi-hop推理或跨段落彙總能力——這正是RULER被設計出來、且會刻意涵蓋multi-hop與aggregation任務的原因。
3. **只測`depth_ratio=0.5`就夠代表整體表現**：很多評估實作偷懶只測中間深度，但lost-in-the-middle效應代表中段本身就是相對弱的位置，沒有掃過0、0.25、0.5、0.75、1.0這幾個深度，看不出真正的深度敏感度曲線。
4. **needle跟filler用同一批文字、關鍵字重疊也沒關係**：字面重疊會讓模型可以靠簡單的關鍵字匹配「作弊」通過測試，量到的其實是字串比對能力而非長文本理解能力，該用NoLiMa風格的非字面重疊needle才有鑑別度。
5. **廠商自己公布的long-context benchmark分數可以直接拿來做選型依據**：供應商公布的分數存在選擇性報告與測試條件不透明的風險，正式選型決策前應該在自己的資料與使用情境上獨立重新跑一次。
6. **只看準確率就能判斷這個context長度能不能用在正式環境**：超長prompt的prefill延遲（100萬token常見要30-120秒）在互動式應用中可能才是真正的瓶頸，評估報告漏掉time-to-first-token，會讓「能用」的判斷失真。
7. **模型升級後，之前測過的context能力理所當然還在**：新版本模型即使整體更強，長文本能力也可能在特定深度或特定任務類型上退步——每次升級都應該用固定的回歸測試組重新驗證，而不是假設「新的一定比舊的好」。

## 複習自問
- 為什麼「模型能通過1M tokens的NIAH測試」不足以說明這個模型可以拿來處理100萬token的長文件分析任務？RULER想額外驗證的是什麼？
- Lost-in-the-middle效應具體指的是什麼？如果評估只測`depth_ratio=0.5`，會對評估結果帶來什麼偏誤？
- Multi-needle測試和single-needle NIAH測試分別驗證了模型的哪一種能力？一個模型single-needle接近滿分、multi-needle明顯下降，可能代表什麼問題？
- NoLiMa的needle設計（跟query不共用字面詞彙）想避免什麼樣的「假陽性」？如果needle跟filler字面重疊，NIAH分數可能被什麼因素灌水？
- 「advertised context window」、「effective retrieval length」、「effective reasoning length」這三個數字分別代表什麼？為什麼正式評估報告應該同時列出後兩者，而不是只看廠商宣稱的數字？
- 如果一個模型升級之後，NIAH分數維持不變，但RULER的multi-hop tracing分數明顯下降，這代表這次升級對什麼類型的應用場景風險最高？
- 為什麼超長prompt的評估報告不能只看準確率，還需要額外記錄time-to-first-token？這在什麼樣的應用場景下特別關鍵？
- 如果要幫自己的RAG系統選一個長文本模型，除了跑NIAH之外，至少還應該再補上哪一種測試，才能對「這個模型撐不撐得住我的使用情境」有基本的信心？
