# LLM Evaluation（LLM-as-judge：Faithfulness + Answer Relevance + G-Eval + CI Gate）

對應程式：[`./llm_evaluation.py`](./llm_evaluation.py)

## TL;DR
Exact Match和F1這類傳統指標量不到「語意相不相同」——標準答案是「June 29, 2007.」，模型答「June 29th, 2007.」，Exact Match打0分、F1打約75分，但人類一看就知道是100分的正確答案。當測試案例從1筆變成上萬筆，還要在每次改retriever、chunking、prompt、模型之後重新跑一次，就不可能靠人工逐筆看。這份程式示範業界目前的解法——LLM-as-judge：用LLM（或NLI模型）本身去評分LLM的輸出，包含RAGAS式的faithfulness（答案有沒有幻覺）、answer relevance（答案有沒有離題）、DeepEval的G-Eval（自訂規則的LLM裁判）、以及能接進CI/CD的回歸測試關卡。

## 為什麼需要它
- 傳統字面比對指標（Exact Match、F1、BLEU、ROUGE）只看字面重疊，量不到「意思是否正確」——同一個事實可以用無數種說法表達，字面比對會系統性低估用不同措辭寫出的正確答案。
- RAG系統上線後最大的風險是生成答案「編了檢索內容裡沒有的東西」（幻覺），這種錯誤如果沒有自動化評估，只能等使用者回報或人工抽查才會發現，規模一大就顧不過來。
- 每次調整retriever、chunking策略、prompt或換模型，都可能讓某個之前正常的case壞掉——沒有自動化的回歸測試，這種「改A壞B」的問題很容易在上線後才被發現。
- 人工評分能給出最準確的判斷，但無法規模化到每天數千筆的CI/CD流程；LLM-as-judge用一顆LLM取代人工評分，成本可以壓到每筆幾毫美分，讓大規模自動化評估變得可行。
- 但LLM-as-judge本身也是一個需要被驗證的系統：裁判LLM有自己的偏好（偏好長答案、偏好與自己同家族模型的輸出、偏好符合prompt語氣的答案），沒有校準過的裁判分數不能直接信任——這正是本程式與README要強調的重點之一。

## 核心原理

### 1. LLM-as-judge：用LLM取代靜態指標
核心想法很簡單：給定`(query, context, answer)`，直接prompt一個裁判LLM「照這個規則打0-1分」，回傳分數。它之所以有效，是因為LLM在語意理解上已經逼近人類判斷，但成本只是人工評分的一小部分——用GPT-4o-mini這類便宜模型，每筆評分成本可以壓到約0.003美元，1000筆的回歸測試整套跑下來不到5美元。但它也有三個常見的「悄悄失效」模式：
- **裁判偏誤（judge bias）**：裁判LLM會偏好比較長的答案、偏好跟自己同家族模型生成的答案、偏好語氣風格符合prompt的答案，這些偏好跟「答案品質」不必然相關。
- **JSON解析失敗**：裁判回傳的JSON格式不對時，很多框架會直接把該筆分數記成NaN並悄悄排除在彙總結果外——RAGAS使用者對這個痛點很熟悉。務必用try/except明確處理解析失敗，而不是讓它悄悄消失在平均值裡。
- **裁判版本漂移**：升級裁判模型版本，等於同時改變了所有歷史分數的基準——所有指標都會跟著變動。正式環境應該把裁判模型與版本號一起釘死（pin），版本要換也要走明確的重新校準流程。

### 2. Faithfulness：拆解claims + NLI逐一驗證（`atomic_claims` / `faithfulness`）
`faithfulness`是RAGAS的核心指標之一，回答「答案裡的每個說法，檢索到的context有沒有支持？」。做法分兩步：`atomic_claims`先請`llm`把答案拆成一行一個的原子事實陳述（例如把「他在2007年推出iPhone，並在同年成為Time年度風雲人物」拆成兩條獨立可查證的claim）；`faithfulness`再對每一條claim，把`context`當premise、該claim當hypothesis，丟進NLI模型（`nli`，這裡用`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`）做entailment判斷，entailment分數超過0.5才算「被支持」，最後回傳「被支持的claim數 / 總claim數」。用專門的NLI模型而不是直接問裁判LLM「這句話對不對」，除了成本更低、結果更穩定可重現之外，也避免了裁判LLM「自己拆的claim自己說了算」的循環驗證問題。

### 3. Answer Relevance：反向生成問題再比對embedding（`answer_relevance`）
`answer_relevance`回答的是另一個問題：「答案本身有沒有離題？」它不直接比對答案與問題的字面或語意相似度，而是反過來操作——請`llm`根據答案「反推」出`n`個「這個答案可能是在回答什麼問題」，再用`encoder`把這些生成的問題與原始`question`都做normalize過的embedding，用內積算cosine similarity，取平均。如果答案確實針對原問題作答，LLM反推出的問題應該會很接近原問題；如果答案掉進了離題的細節裡，反推出的問題會偏離原問題，即使答案裡每一條claim個別來看都對context忠實（`faithfulness`高）。這正是faithfulness量不到的維度：一個完全忠於context、卻不是在回答使用者實際問題的答案。

### 4. G-Eval：用rubric（evaluation_steps）取代模糊的「請打0-1分」
`GEval`（DeepEval提供的類別）是G-Eval方法的實作：與其直接問裁判LLM「這個答案好不好，請給0-1分」，G-Eval會把評分邏輯拆解成一串明確的`evaluation_steps`（例如「讀expected output」→「讀actual output」→「列出actual output裡的事實陳述」→「逐一標記是否被expected output支持」→「分數=支持比例的claim」），讓裁判模型照著這串chain-of-thought步驟走，而不是一次性給出一個難以復現的整體印象分數。這份程式定義的`Correctness`指標需要`expected_output`（標準答案）搭配`INPUT`/`ACTUAL_OUTPUT`/`EXPECTED_OUTPUT`三個欄位，屬於「有參考答案」的離線評估，跟`faithfulness`/`answer_relevance`這種不需要標準答案的reference-free指標定位不同——G-Eval適合評估RAGAS涵蓋不到的自訂/領域專屬品質維度。

### 5. RAG評估的完整四指標（RAGAS）
除了程式裡實作的`faithfulness`與`answer_relevance`，業界通用的RAGAS框架完整定義了四個指標，分別對應RAG流程的不同環節：

| 指標 | 回答的問題 | 常見作法 |
|---|---|---|
| Faithfulness | 答案裡的每個claim，檢索到的context有沒有支持？ | NLI-based entailment（本程式的做法） |
| Answer Relevance | 答案有沒有針對使用者的問題作答？ | 從答案反推假設性問題，比對與真實問題的相似度（本程式的做法） |
| Context Precision | 檢索回來的chunk裡，有多少比例是真的相關的？ | LLM-judge |
| Context Recall | 檢索有沒有把回答問題所需的內容都撈回來？ | 對照標準答案的LLM-judge |

前兩者是reference-free（不需要標準答案），後兩者通常需要LLM裁判、Context Recall還需要標準答案。四個指標分別檢查生成端（faithfulness、relevance）與檢索端（precision、recall）的品質，缺一個都可能讓整體RAG系統的問題被漏掉。

### 6. 校準（Calibration）：裁判分數在被驗證之前不能信任
裁判LLM給出的原始分數，在跟人類標註做過相關性驗證之前，都只是「這顆模型覺得」的分數，不代表真的可信。標準做法是：找100筆左右人工標註過的範例，把裁判分數與人類分數畫成散佈圖，計算Spearman rho（等級相關係數）；如果rho小於0.7，代表裁判的rubric或prompt設計需要調整，不能直接拿去上線當作品質門檻。這一步在程式碼裡沒有實作，但是任何LLM-as-judge指標正式使用前不能省略的驗證流程。

### 7. CI Gate：把評估寫成pytest測試，接進CI/CD（`test_rag_system`）
`test_rag_system`示範如何把評估「常態化」——不是跑一次性的分析報告，而是寫成pytest可以直接發現、每次PR都會執行的測試函式。它用DeepEval的`FaithfulnessMetric(threshold=0.85)`與`ContextualRelevancyMetric(threshold=0.7)`對固定的回歸測試集（`load_regression_cases()`）逐筆評分，任何一筆低於門檻就`assert`失敗、擋下合併。這是把「評估」從人工的、事後的分析行為，轉變成自動化、常態化的品質關卡的關鍵一步。

## 程式碼導覽
| 函式 / 區塊 | 對應到理論的哪個部分 |
|---|---|
| `nli = pipeline("text-classification", model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", top_k=None)` | Faithfulness檢查用的NLI entailment分類器，`top_k=None`回傳全部三個標籤的分數 |
| `LLM = Callable[[str], str]` | 型別別名：任何「輸入prompt字串、回傳生成字串」的可呼叫物件都符合，讓`llm`參數可以接任意LLM client的wrapper |
| `atomic_claims(answer, llm)` | 用`llm`把答案拆解成一行一個的原子事實陳述，供`faithfulness`逐一驗證 |
| `faithfulness(answer, context, llm)` | RAGAS faithfulness：對每條claim跑NLI entailment，回傳「被context支持的claim比例」 |
| `answer_relevance(question, answer, encoder, llm, n)` | RAGAS answer relevance：從答案反推`n`個假設性問題，取與原問題的平均cosine similarity |
| `GEval(name="Correctness", evaluation_steps=[...], ...)` | G-Eval自訂指標：用明確的`evaluation_steps`（rubric）取代模糊的「請打分」prompt |
| `LLMTestCase(input=..., actual_output=..., expected_output=...)` + `metric.measure(test)` | 對單一範例套用G-Eval指標並印出分數與理由（`metric.reason`） |
| `test_rag_system()` | CI回歸測試：用`FaithfulnessMetric`/`ContextualRelevancyMetric`對固定回歸集逐筆檢查，低於threshold就`assert`失敗 |

**實作細節 / 容易看漏的地方：**
- `faithfulness`裡的NLI模型（`MoritzLaurer/...`）跟`atomic_claims`用的`llm`是兩個獨立的模型——用專門的小模型做entailment檢查，而不是讓同一顆裁判LLM「自己拆claim、自己判斷claim對不對」，可以避免評分邏輯自我循環驗證、也比每次都呼叫LLM便宜。
- `faithfulness`的entailment門檻寫死在函式內部（`entail["score"] > 0.5`），不像`is_faithful`（見[[NLI]]筆記）把threshold開放成參數；如果要調整靈敏度，需要直接改程式碼而非傳參數。
- `answer_relevance`的`encoder`與`llm`都是外部依賴，程式本身沒有預設值——注解裡建議的`SentenceTransformer("BAAI/bge-small-en-v1.5")`只是範例，實際使用時`encoder`與`llm`都需要呼叫端自行提供並保證介面相符（`.encode(texts, normalize_embeddings=True)`、`str -> str`）。
- `answer_relevance`回傳的是cosine similarity的平均值，理論範圍是`[-1, 1]`，但在SentenceTransformer這類語意相近的embedding空間裡，實務上分數通常落在`[0, 1]`附近。
- `GEval`的`metric`與`test`是模組層級（module-level）程式碼，`import`這個檔案時就會立刻執行`metric.measure(test)`並印出結果，不是包在函式或`if __name__ == "__main__":`裡面——正式使用時應該把這段搬進函式或測試檔案，避免import就觸發一次真實的LLM呼叫。
- `test_rag_system`呼叫的`load_regression_cases()`在檔案裡沒有定義，是需要呼叫端自行提供的佔位符（對應原始教材的示範程式碼），直接執行這個函式會丟出`NameError`。
- `test_rag_system`裡的兩個threshold（faithfulness 0.85、relevancy 0.7）是DeepEval官方文件範例的示範值，不是通用的業界標準——正式使用前應該用自己的回歸資料集校準出合理門檻。

## 使用時機 / 優缺點
2026年常見的組合拳（依用途挑框架，可同時使用多種）：

| 用途 | 建議框架 |
|---|---|
| RAG品質日常監控 | RAGAS（faithfulness / answer relevance / context precision / context recall 四指標） |
| CI/CD回歸測試關卡 | DeepEval + pytest（本程式`test_rag_system`的模式） |
| 自訂／領域專屬品質維度 | DeepEval裡的G-Eval |
| 線上即時流量監控 | RAGAS的reference-free模式（不需要標準答案） |
| 人工抽查／標註介面 | LangSmith 或 Phoenix 這類帶標註UI的工具 |
| 紅隊測試／安全性評估 | Promptfoo + DeepEval |

- ✅ 比字面比對指標（Exact Match、F1、ROUGE）更能反映語意正確性，同一個事實用不同措辭表達也能拿到合理分數。
- ✅ 成本遠低於人工評分，可以規模化跑在每一次PR、每一次模型/prompt/retriever變更上，是唯一能撐起「大量case + 頻繁迭代」場景的評估方式。
- ✅ Faithfulness用NLI模型而非裁判LLM本身，兼具速度快、成本低、結果可重現的優點，適合當高頻執行的檢查關卡。
- ✅ G-Eval的明確`evaluation_steps`設計，比起模糊的「請打0-1分」prompt更穩定，也更容易讓不同的人看懂「這個分數是怎麼算出來的」。
- ❌ 沒有校準過的裁判分數不可信：裁判LLM有長度偏好、同源模型偏好等系統性偏誤，正式上線前一定要跑過人工標註對照（Spearman rho門檻建議0.7以上）。
- ❌ 用同一顆（或同家族）LLM同時生成答案與評分答案，分數容易被系統性灌水10-20%，裁判模型應該選跟受測系統不同家族的模型。
- ❌ 平均分數會掩蓋少數嚴重失敗：整體平均0.85看起來很健康，底層可能藏著5%的災難性失敗案例，只看平均值容易漏掉這些長尾問題，務必額外檢查分數最低的分位數。
- ❌ 回歸測試集本身如果沒有版本控制，會隨時間漂移，導致「上個月85分、這個月85分」其實是在兩份不同的測試集上跑出來的，無法做時間序列比較。
- ❌ 大規模跑LLM裁判的成本會隨case數量線性成長，正式環境應該選「剛好通過校準門檻」的最便宜模型，而不是無腦用最強模型當裁判。

## 常見誤區
1. **只要換成LLM-as-judge，分數就一定可信**：裁判LLM本身也可能判斷錯誤，尤其是claim很長、包含多個子陳述時；沒有跟人類標註做過相關性校準的裁判分數，只是「這顆模型的意見」，不是可以直接拿來當品質門檻的真值。
2. **用同一顆模型生成答案又評分答案沒問題**：這是self-evaluation偏誤，同源模型互評會系統性高估分數（常見幅度10-20%），裁判模型應該來自不同的模型家族。
3. **平均分數高就代表整體品質沒問題**：平均值會掩蓋分佈——0.85的平均分數底下，可能有5%的case是完全不合格的災難性失敗，只看平均容易忽略需要優先修的長尾問題，應該額外檢查最低分位數。
4. **faithfulness分數高，答案就一定切題**：faithfulness只驗證「答案裡的說法有沒有被context支持」，不驗證「答案有沒有回答到使用者的問題」——一個完全忠於context、卻答非所問的回答，faithfulness可以很高，這正是需要另外算`answer_relevance`的原因。
5. **裁判模型的版本可以隨時升級，不影響歷史分數的可比性**：升級裁判LLM等於改變了評分標準本身，所有歷史分數的基準都會跟著漂移；正式環境應該把裁判模型與版本號一起釘死，要換版本得先跑一次平行的baseline比較。
6. **JSON解析失敗的case可以安靜地被排除在平均值之外**：很多框架遇到裁判回傳格式錯誤時會把該筆記成NaN並悄悄從彙總排除，這樣算出來的平均分數其實是「有效回應的平均」，不是「全部case的平均」，務必用明確的try/except攔截並列出失敗筆數。
7. **回歸測試的threshold（如本程式的0.85 / 0.7）放諸四海皆準**：這些數字是特定框架文件的示範值，不是業界標準，不同模型、不同任務、不同資料分布的合理門檻都不一樣，正式使用前要用自己的資料校準。
8. **pairwise比較（哪個答案比較好）不需要考慮呈現順序**：裁判LLM在做A/B兩兩比較時普遍偏好先出現的選項（positional bias），嚴謹的pairwise評估需要把兩種呈現順序都跑一次再取結果一致的部分，本程式雖然沒有實作pairwise比較，但這是延伸到該場景時必須注意的偏誤。

## 複習自問
- `faithfulness`為什麼要用一顆獨立的NLI模型做entailment判斷，而不是直接問生成答案用的那顆LLM「這句話對不對」？這樣做解決了什麼問題？
- `answer_relevance`為什麼要「從答案反推問題」再比對相似度，而不是直接算`question`與`answer`兩段文字的embedding相似度？這樣的設計能抓到什麼`faithfulness`抓不到的問題？
- 一個答案的`faithfulness`很高、但`answer_relevance`很低，可能代表這個答案出了什麼問題？反過來`faithfulness`低、`answer_relevance`高呢？
- G-Eval的`evaluation_steps`（明確的chain-of-thought評分步驟）相較於「請直接給這個答案打0-1分」的prompt，為什麼比較穩定？
- 如果裁判LLM跟被評估系統用的是同一顆模型，可能會發生什麼問題？業界的建議做法是什麼？
- 為什麼「平均faithfulness分數0.85」不足以說明一個RAG系統已經可以上線？還需要額外檢查什麼？
- 如果要驗證一個新導入的LLM裁判是否可信，具體應該怎麼做？Spearman rho在這個流程裡扮演什麼角色？
- `test_rag_system`裡的threshold（0.85 / 0.7）如果不做校準直接套用在自己的專案上，可能會有什麼風險？
