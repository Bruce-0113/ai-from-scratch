# Context Engineering(上下文工程)

對應程式:[`./context_engineering.py`](./context_engineering.py)

## TL;DR
把「context window有多大」跟「該往裡面塞什麼、塞多少、用什麼順序塞」這兩件事分開處理。`ContextBudget`用固定的token預算,依優先順序把system prompt、工具定義、檢索文件、對話歷史、目前問題切好空間;`ConversationManager`在歷史超過門檻時把最舊的對話壓縮成一行摘要;`select_tools`依查詢意圖動態挑出相關工具,不是每次都把全部工具定義塞進去;`reorder_lost_in_middle`把最相關的內容擺在最前面跟最後面,呼應「模型對開頭結尾的注意力最高、對中段注意力最低」的實證結果。核心問題從來不是「窗口夠不夠大」,而是「窗口裡的每一個token有沒有在做有用的事」——一段被塞進去的無關工具定義、一輪過期的對話、一段答非所問的檢索文字,都會讓模型在這個任務上表現得稍微差一點。

## 為什麼需要它
- 現在主流模型的context window都很大(百K到百萬token等級),但「window很大」不代表「塞滿它是有效的」。一個典型coding assistant的token拆解:system prompt 500、50個工具定義8,000、檢索到的文件4,000、10輪對話歷史6,000、目前問題200、生成預留4,000——加總22,700,只佔128K窗口的18%,而其中有相當比例其實是雜訊。
- Attention不是均勻分布在整個context上。窗口越長,不代表模型越能「均勻地記住」裡面每一段內容;相反地,放在中段的資訊經常被模型忽略掉。
- 每一個放進窗口的token都會排擠掉另一個可能更相關的token。多餘的工具定義、過期的對話輪次、答非所問的檢索片段,都會讓模型在這個任務上的表現變差,而不是「反正模型會自己過濾掉」。

## 核心原理
- **Context window是稀缺資源,不是硬碟**:可以想像成RAM而不是disk——快、但有限,必須做取捨,而不是能塞多少就塞多少。
- **Lost-in-the-middle**:Liu et al. (2023)把一份相關文件混在20份不相關文件裡,分別放在不同位置測準確率。放在最前或最後,準確率85-90%;放在中間(第10/20的位置),準確率掉到60-70%。實務上的意涵——重要指令放最前面(system prompt)、目前問題跟最相關內容放最後面(利用recency bias)、把窗口中段當成最低優先權區域、真的必須把重要資訊放中間時,在結尾再重複一次重點。
- **Context組成會互相競爭同一份預算**:system prompt(人設、規則,固定不變)、工具定義(每個約50-200 token)、檢索文件(品質決定回應品質,retrieval做壞了比不做還糟)、對話歷史(隨對話輪數線性成長,大部分跟目前問題無關)、few-shot範例(兩三個好範例常常比大量文字指令更有效)、生成預留(至少留2,000-4,000 token給模型作答,不然窗口塞滿模型會無法生成)。
- **三種時間尺度的memory**:short-term(目前對話,直接存在context window裡,靠壓縮/截斷管理)、long-term(跨對話保存的事實與偏好,例如「使用者偏好TypeScript」,存在DB裡,session開始時撈出來,對應Claude Code的`CLAUDE.md`)、episodic(特定的過去互動,存成embedding,依相似度檢索)。**這支腳本只實作了short-term memory**(`ConversationManager`),long-term跟episodic是概念延伸,尚未在程式碼裡出現。
- **Dynamic context assembly**:好的系統不是「固定system prompt + 固定工具 + 固定歷史」,而是依每次查詢動態組裝——分類意圖→只選相關工具→只檢索相關文件→只納入相關歷史輪次→依重要性排序(關鍵內容放頭尾、次要內容放中間)。

## 程式碼導覽
| 函式/類別 | 對應到理論的哪個部分 |
|---|---|
| `count_tokens` / `count_tokens_json` | Token計數器:用「字數 × 1.3」估算token數,取代真正的tokenizer,足夠拿來做相對預算分配,但不是精確值 |
| `ContextBudget` | 預算管理核心:`__init__`先扣掉`generation_reserve`算出`available`;`allocate`對每個component做兩層截斷(先套per-component的`max_tokens`上限,再套目前剩餘的全域預算);`report`把各component的佔比印成長條圖 |
| `reorder_lost_in_middle` / `score_relevance` | Lost-in-the-middle的具體實作:先依分數由高到低排序,再用`[::2]`/`[1::2]`切成兩半、後半反轉重組,讓高分項目落在序列的頭尾、低分項目落在中段;`score_relevance`用query/document的詞彙交集比例(bag-of-words overlap)當相關性分數 |
| `ConversationManager` | 對話歷史壓縮:近期輪次逐字保留,一旦總token數超過`max_history_tokens`,就把最舊的兩輪`_summarize_turns`成一行摘要並丟棄原文,直到低於門檻或只剩4輪為止 |
| `TOOL_REGISTRY` / `classify_intent` / `select_tools` | 動態工具選擇:`classify_intent`用關鍵字比對算出每個意圖類別的命中分數,取得分≥最高分一半的所有類別;`select_tools`依`TOOL_REGISTRY`的字典順序,把類別命中的工具貪婪塞入,直到碰到`token_budget`上限為止 |
| `ContextEngine.assemble` / `.chat` | Step 6整合pipeline:依優先順序呼叫`budget.allocate`——system prompt→工具→(用`score_relevance`篩選、`reorder_lost_in_middle`重排後的)檢索文件→對話歷史→目前問題,每個component都有各自的per-component上限;`chat`把一問一答記錄進`ConversationManager`,再呼叫`assemble`組裝這一輪的context |
| `run_demo` | 依序示範:對話輪數增加時budget report如何變化、四種查詢的意圖分類與工具選擇結果、以及`reorder_lost_in_middle`把5份不同分數的文件重排的實際效果 |

**實作細節 / 容易看漏的地方:**
- `ContextBudget.allocate`的截斷是兩階段的:先用傳入的`max_tokens`(per-component上限)截一次,再用「目前剩餘的全域預算」截第二次——代表同一段內容有可能被連續截斷兩次,長度比單看`max_tokens`還短。
- `ContextEngine.assemble`每次呼叫都會用`ContextBudget(self.budget.max_tokens, self.budget.generation_reserve)`整個重建budget,等於每次查詢的預算配置都是無狀態、從零開始的;只有`conversation`(對話歷史)跟`knowledge_base`(檢索來源)會跨查詢保留下來。
- `select_tools`是依`TOOL_REGISTRY`的**字典插入順序**貪婪塞入,不是依相關性分數排序——如果同一個意圖底下有多個工具,排在字典後面的工具即使一樣相關,也可能因為預算已滿而被跳過(例如`generate_chart`排在`TOOL_REGISTRY`最後,只要`data`類別的預算被`query_database`先佔滿,`generate_chart`就進不去)。
- `classify_intent`允許回傳多個意圖(只要分數≥最高分的一半),所以像「Search for best practices on error handling」這種查詢會同時命中`code`跟`research`兩個類別,選出的工具集也是兩邊的聯集。
- `ConversationManager._compress_if_needed`的while迴圈條件是`len(self.turns) > 4`,代表無論對話多長,永遠至少保留最後4輪(2組問答)逐字內容不壓縮。
- `count_tokens`用「字數 × 1.3」估算,對英文句子還算堪用,但對中文(沒有空白分詞,`split()`容易把一整句當成一個「字」)、程式碼、URL這類內容會嚴重低估——這是刻意簡化的教學版heuristic,不是可以直接拿來卡真實API token上限的實作。

## 使用時機 / 優缺點
- ✅ 想理解「token預算怎麼在多個component之間分配」「為什麼要把重要內容放頭尾」這些概念時,這支腳本是一個可以直接執行、看報表的具體示範。
- ✅ 需要依查詢意圖動態裁減工具定義(tool pruning)以節省token時,`classify_intent` + `select_tools`提供了一個最小可行的參考架構。
- ✅ 想設計自己的context assembly pipeline時,`ContextEngine.assemble`裡「依優先順序逐一`allocate`」的寫法是一個可以直接套用的骨架。
- ❌ `count_tokens`只是字數估計,離真實的provider tokenizer(如`tiktoken`)有落差,正式環境要先換成對應SDK提供的官方計數方式,才能真的守住硬性的context window上限。
- ❌ `score_relevance`是純字面詞彙重疊,沒有語意理解——同義詞、換句話說的查詢會被誤判成不相關,真正的RAG系統需要換成embedding-based的向量檢索。
- ❌ `_summarize_turns`只是「取前100字元+`...`」的截斷,不是語言模型產生的摘要,壓縮後容易遺失關鍵資訊;正式系統應該呼叫LLM對舊對話做真正的摘要。
- ❌ 只實作了short-term memory(當前對話),沒有long-term memory(跨對話持久保存的事實/偏好,例如`CLAUDE.md`)或episodic memory(用embedding檢索過去的相似對話)——這兩塊在概念上很重要,但這支腳本沒有涵蓋,需要額外接資料庫與向量檢索。

## 常見誤區
1. **以為窗口大就可以隨便塞**:200K/1M token的窗口不代表把200K token都塞滿是有效的——一段精心篩選過的1萬token context,經常比隨便塞的10萬token context表現更好。塞進去的每一段無關內容都在稀釋signal-to-noise ratio。
2. **把「有沒有放進context」當成唯一的判準,忽略「放在哪個位置」**:同一段關鍵資訊放在開頭/結尾跟放在中間,準確率可以差到10-20個百分點。設計context時要同時考慮「放不放」跟「放哪裡」。
3. **把`select_tools`的貪婪填裝誤認為是按相關性排序**:目前實作是照`TOOL_REGISTRY`的字典順序塞、碰到`token_budget`就停,不是先幫工具打分數再取前N名——如果需求是「預算有限時優先保留最相關的工具」,需要額外加上排序邏輯,不能直接假設現在的行為已經是按重要性排序。
4. **忽略generation reserve**:如果把整個context window都塞滿輸入,模型會沒有空間生成回覆。`ContextBudget.__init__`已經用`available = max_tokens - generation_reserve`把這塊預留出來,任何新增的component都應該對著`available`(而不是`max_tokens`)去要預算。
5. **把截斷式摘要當成語意摘要**:`_summarize_turns`只是字串截斷,不是LLM生成的摘要——如果拿它的輸出去回答「使用者之前提過什麼細節」,遺失資訊的機率很高,正式系統應該用LLM做摘要並保留關鍵事實。

## 延伸閱讀
- [Liu et al., 2023 — "Lost in the Middle: How Language Models Use Long Contexts"](https://arxiv.org/abs/2307.03172):`reorder_lost_in_middle`所依據的實證研究,系統性測量了相關文件位置與回答準確率的關係。
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval):說明如何在檢索階段就替每個chunk補上上下文,而不是只在prompt階段做重排,可以視為`score_relevance`/`reorder_lost_in_middle`的進階版做法。
- [Simon Willison — Context Engineering](https://simonwillison.net/2025/Jun/27/context-engineering/):把「context engineering」跟「prompt engineering」的差異講清楚的原始文章,可以當作這份README第一段TL;DR的背景脈絡。

## 複習自問
- 為什麼「200K token的窗口」不等於「200K token都該被塞滿」?把這個道理套用到`ContextBudget`的`generation_reserve`上,它具體在防止什麼問題?
- `reorder_lost_in_middle`為什麼要把分數排序後用`[::2]`、`[1::2]`切成兩半,再把後半反轉重新接起來?如果不做「反轉」這一步,最終順序會變成什麼樣子、少了什麼效果?
- `select_tools`目前是照`TOOL_REGISTRY`的字典順序貪婪塞入,如果同一個意圖底下有5個工具但預算只夠放3個,現在的實作會選中哪3個?這是不是你想要的行為?如果不是,要怎麼改?
- `ConversationManager`為什麼要保留「至少4輪」不壓縮,而不是把所有歷史都摘要掉?如果拿掉這個下限,可能會出現什麼問題?
- `count_tokens`用「字數 × 1.3」估算token數,這個估計在哪些文字型態(中文、程式碼、URL)上最容易失準?如果要換成真正的provider tokenizer,應該在哪個函式做替換,又會牽動哪些呼叫它的地方?
