# Anthropic Workflow Patterns — Prompt Chaining、Routing、Parallelization、Orchestrator-Workers、Evaluator-Optimizer 五種單Agent工作流

對應程式: [`./anthropic_workflow_patterns.py`](./anthropic_workflow_patterns.py)

參考: [ai-engineering-from-scratch – Anthropic's Workflow Patterns: Simple Over Complex 課程](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/14-agent-engineering/12-anthropic-workflow-patterns/docs/en.md)

前置概念: 課程原本列的前置課是Phase14·01(Agent Loop:ReAct風格的思考-行動-觀察控制迴圈)——這支程式假設你已經知道agent loop長什麼樣子,重點放在它的對照組:當步驟可以被預先列舉出來,根本不需要agent自己一步步決定要做什麼,寫死的程式路徑(workflow)就夠用、還更便宜更好稽核。跟本repo同屬Phase14的[`../Orchestration_Patterns`](../Orchestration_Patterns/README.md)是這一課的下一步——當單一agent配上這五個workflow pattern都還是不夠用時,才進一步討論要不要長成multi-agent、長成supervisor/swarm/hierarchical/debate哪一種拓撲;[`../Agent_Memory`](../Agent_Memory/README.md)(MemGPT兩層記憶)則是另一個維度的問題,處理的是「單一agent的context塞不下」,跟這裡「要不要用agent、還是workflow就夠」是兩件不同的事。

## TL;DR
團隊常常直接跳去接multi-agent框架,解決其實一個函式呼叫就能搞定的問題——教材引用Schluntz與Zhang(Anthropic, 2024年12月)在《Building Effective Agents》裡的區分:**workflow**是工程師預先寫死的程式路徑,LLM和工具照著這條路徑被呼叫,工程師擁有這張圖;**agent**是模型動態自己決定要呼叫哪個工具、走幾步,模型擁有這張圖。Workflow更便宜、更快、更容易稽核;agent能處理開放性問題,但失敗模式更難掌握。兩者都建立在同一個「augmented LLM」(接了search/tools/memory的模型)之上,教材給出五種workflow pattern:**prompt chaining**(上一步輸出接下一步輸入)、**routing**(先分類再分派給對應handler)、**parallelization**(同一個prompt平行跑N次,用sectioning或voting兩種方式聚合——本程式只實作voting)、**orchestrator-workers**(一次性決定要叫哪些worker、再把結果合成一個答案,不會像agent loop一樣反覆重新規劃)、**evaluator-optimizer**(一個LLM提案、另一個LLM評審,不過關就帶著回饋重新提案,是Self-Refine的一般化版本)。本程式用一個`ScriptedLLM`(dict腳本化的假LLM,不接任何真實API)把五個pattern各自寫成10-20行的小函式,全部跑完只呼叫20次「LLM」,直接呼應教材的結論:「五個pattern的程式碼量加起來大約是框架成本的千分之一」。

## 為什麼需要它
- **對抗「先上框架、再回頭想問題」的直覺**:教材點名的最常見失敗模式,是團隊看到「要串好幾個LLM呼叫」就直接去接LangGraph或CrewAI,而不是先問「這五步能不能被預先寫死」。本程式示範的是:能列舉步驟的任務,直接用函式+dict+list就能做完,不需要框架的圖結構、狀態機或actor model。
- **Workflow跟agent的取捨是有清楚判準的,不是喜好問題**:教材給出兩組判準——**workflow贏在**可預測的任務(步驟能列舉就該列舉)、成本有上限的任務(workflow的步驟數是固定的,agent可能失控地一直呼叫下去)、需要稽核的任務(稽核員想讀的是圖,不是從trajectory反推);**agent贏在**開放式研究(下一步取決於上一步回傳了什麼)、步驟數量不確定的任務(從幾分鐘到幾小時都有可能)、還沒摸清楚正確workflow的新領域(先探索、之後再固化成workflow)。搞懂這兩組判準,才不會預設什麼都要agent。
- **這五個pattern是2026年幾乎所有agent框架底層真正在做的事**:LangGraph的節點圖、CrewAI的Process、OpenAI Agents SDK的handoff,拆開來看都是這五個pattern的某種排列組合。先用stdlib把它們各自刻一次,才看得出框架幫你省了什麼(狀態持久化、並發原語、角色模板),又用什麼代價換來的(多一層抽象、除錯要多繞一層)。

## 核心原理
- **Augmented LLM是五個pattern共同的基礎單元**:教材說的「augmented LLM」是一個接了三種能力的模型呼叫——search(檢索)、tools(行動)、memory(持久化)。任何一次API呼叫都可以接上這三者;五個pattern的差別只在於*怎麼組織多次*這樣的呼叫,不在於單次呼叫本身。本程式的`ScriptedLLM.__call__`就是被組織的那個原子單元,五個pattern函式全部只是用不同方式呼叫它。

- **五個pattern對照表**:

  | 課程概念 | 本程式對應 | 使用時機 |
  |---|---|---|
  | Prompt chaining | `prompt_chain` | 任務有乾淨的線性拆解時(先summarize再title) |
  | Routing | `route` | 輸入類別本質不同、需要不同處理時(退款/bug/銷售) |
  | Parallelization(voting) | `parallel_vote` | 同一個問題想要多數決降低變異(教材另有sectioning shape,本程式未實作) |
  | Orchestrator-workers | `orchestrator_workers`/`Worker` | 一次性決定要叫哪些specialist、再合成結果,不需要像agent一樣反覆重新規劃 |
  | Evaluator-optimizer | `evaluator_optimizer` | 有明確評審標準、值得迭代打磨的產出(Self-Refine的一般化版本) |

- **`ScriptedLLM`的兩種腳本形式撐起了「同一個prompt字串,不同次呼叫要回傳不同答案」這件事**:一個prompt對到單一字串,永遠回傳同一個答案(路由分類用這種——同一句話永遠分到同一類);一個prompt對到字串list,每呼叫一次往前推進一格、到底之後停在最後一格不再前進——`parallel_vote`用這個機制讓同一個`"is this code safe to ship?"`prompt能在5次呼叫裡開出`yes/yes/no/yes/no`不同的票,`evaluator_optimizer`用這個機制讓同一個`evaluate:`prompt第一次回傳`FAIL`、改過之後的candidate再問一次回傳`PASS`。

- **`orchestrator_workers`不是「路由」也不是「agent loop」,是介於兩者之間的一次性動態dispatch**:跟`route`不同,它不是排他式地選一個handler,而是讓每個worker自己用`handles(task)`判斷「這是不是我的工作」,任何數量的worker都可以同時接手同一個任務(demo裡`security_reviewer`的`handles`永遠回傳`True`,所以每次都跟著跑)。跟真正的agent loop也不同,要跑哪些worker是一次性決定的,worker的輸出不會回頭觸發「要不要再多叫一個worker」的重新規劃——這正是教材說的「像agent loop,但不會無限迴圈」。

- **`evaluator_optimizer`是Self-Refine的一般化版本**:`proposer`拿到的第二個參數是前一輪evaluator給的回饋(第一輪是`None`),可以用它產生更好的candidate;迴圈在evaluator回傳`True`的那一輪立刻返回,不會多跑無謂的迭代。

## 程式碼導覽
| 函式/類別 | 對應到理論的哪個部分 |
|---|---|
| `ScriptedLLM` | Augmented LLM的假身:一次呼叫= 一次API call,字串腳本回傳固定答案,list腳本逐次前進、到底後停在最後一格 |
| `prompt_chain` | Pattern 1:依序執行`steps`,每一步的輸出接下一步的輸入 |
| `route` | Pattern 2:先分類、再分派給對應handler,查無handler時fallback到`"default"` |
| `parallel_vote` | Pattern 3(voting shape):同一個prompt呼叫`n`次,取多數決 |
| `Worker` | Orchestrator-workers的一個specialist:`handles`判斷要不要接手,`fn`是它自己的LLM呼叫 |
| `orchestrator_workers` | Pattern 4:跑過每個`handles(task)`為`True`的worker,再用`synth`合成一個結果 |
| `evaluator_optimizer` | Pattern 5:提案→評審→(不過關就)帶著回饋重新提案,直到`PASS`或`max_iter`用盡 |
| `demo_chain`/`demo_route`/`demo_parallel`/`demo_orchestrator`/`demo_evaluator_optimizer` | 各自對一個pattern餵一組固定輸入,把trace印出來 |
| `main` | 建好共用的`ScriptedLLM`腳本,跑完五個demo,印出總呼叫次數 |

**實作細節 / 容易看漏的地方:**
- **整理時移除的未使用import**:原始程式`from typing import Any, Callable`裡的`Any`整支程式沒有任何一處用到——五個pattern函式的型別標注全部用得到的只有`Callable`,整理時已移除,行為完全不變。
- **本次整理只補文件字串、清未用import、對齊續行縮排,沒有動任何邏輯**:五個pattern函式、`ScriptedLLM`、demo腳本的行為和原本逐行相同,`main()`跑完印出的`total llm calls`數字(20)沒有變。
- **`parallel_vote`只實作了voting,沒有sectioning**:教材的parallelization pattern有兩種形狀——sectioning(把輸入拆成獨立的幾塊,各自跑一次)跟voting(同一個輸入跑好幾次,多數決或綜合)。本程式的`parallel_vote`只對到voting這一半,想練sectioning需要自己另外寫一個函式。
- **`evaluator_optimizer`在`max_iter=0`時會丟`UnboundLocalError`**:迴圈一次都不執行,函式最後`return candidate, trace`裡的`candidate`從沒被賦值過。目前呼叫端(`demo_evaluator_optimizer`跟教材Exercise 3的bandit延伸)都用預設值`max_iter=5`,不會踩到這個邊界,但這是函式簽章沒有明講的隱含前提(`max_iter >= 1`),值得在改寫或加測試時留意。
- **`ScriptedLLM`的list腳本到底之後不會重來、只會停在最後一格**:`min(i + 1, len(value) - 1)`保證索引永遠不超界,但也代表呼叫次數一旦超過腳本長度,後面每次都拿到同一個(最後一個)答案,不是循環回第一個——`parallel_vote`的demo資料剛好呼叫5次、腳本也剛好5個元素,兩者對得上,不是巧合而是刻意配好的。

## 使用時機 / 優缺點
依教材的判準,workflow和agent不是二選一,是由簡入繁的順序:
1. 先看任務能不能列舉步驟——能列舉就用workflow,不需要agent。
2. 五個workflow pattern裡怎麼選:任務是線性的用prompt chaining;輸入類別本質不同用routing;想要多數決降低變異用parallelization;需要一次性動態分派給多個specialist用orchestrator-workers;有明確評審標準值得迭代用evaluator-optimizer。
3. 步驟數量不確定、下一步要看上一步的結果才能決定、或還沒摸清楚正確workflow的新領域,才輪到agent。

- ✅ 想快速搞懂五個pattern各自的形狀差異:五個函式加起來不到100行,每個demo印出的trace可以逐行對照「這一步在教材裡叫什麼」。
- ✅ 需要一個不接LLM、100%可重現的骨架,驗證pattern本身的控制流是否正確,或當成寫單元測試/教學示範的起點。
- ✅ 想直接把`ScriptedLLM.__call__`換成真正的`client.messages.create(...)`:五個pattern函式的介面完全不需要跟著改,因為它們拿到的一直只是一個`Callable[[str], str]`。
- ❌ `parallel_vote`只示範voting,沒有sectioning——想練「把輸入拆成獨立幾塊分別處理」的形狀,這支程式沒有現成範例。
- ❌ `orchestrator_workers`裡「要不要跑某個worker」是寫死的Python predicate(`lambda t: "python" in t.lower()`),不是真的讓一個LLM判斷要分派給誰——demo示範的是orchestrator-workers*之後*的靜態dispatch邏輯,不是orchestrator本身怎麼做決策。
- ❌ 沒有真正的augmented LLM:`ScriptedLLM`只回傳寫死的字串,沒有接search、沒有接真的tool、沒有跨呼叫的memory——五個pattern示範的是*控制流*,不是augmented LLM本身的三個能力。
- ❌ `evaluator_optimizer`對`max_iter=0`沒有防呆(見上方實作細節),把它當函式庫使用前記得留意這個邊界。

## 常見誤區
1. **以為`parallel_vote`涵蓋了parallelization的全部**:教材的parallelization有sectioning跟voting兩種形狀,本程式只做了voting;想拆分獨立輸入分別處理,得自己另外寫。
2. **以為orchestrator-workers就是agent loop**:兩者外觀相似(都是「動態決定要做什麼」),差別在於orchestrator-workers是一次性決定要跑哪些worker、跑完就合成結果,不會像agent loop一樣反覆根據新的observation重新規劃下一步。
3. **以為workflow比agent「差」、agent永遠是進階版**:教材的立場是兩者各有適用場景——可預測、成本有上限、需要稽核的任務workflow更好,不是agent做不到,是workflow更便宜、更容易debug。
4. **以為`main()`印出的`total llm calls: 20`是某種效能基準**:這個數字只是這份demo資料剛好呼叫了20次`ScriptedLLM`,換一組demo輸入或改變`max_iter`,數字就會不同——它示範的是「五個pattern的呼叫次數是可以直接數出來的」這件事本身,不是一個要拿去比較的benchmark。
5. **以為`ScriptedLLM`的list腳本會循環**:超過腳本長度後,`__call__`永遠回傳最後一個元素,不會回到第一個重新開始——這是刻意設計成「穩定在最後結果」,不是bug也不是循環佇列。

## 關鍵詞速查
| 詞彙 | 常聽到的說法 | 實際上是什麼 |
|---|---|---|
| Workflow | 「預先定義好的流程」 | 工程師寫死的LLM與工具呼叫路徑圖 |
| Agent | 「自主AI」 | 模型自己動態決定呼叫哪個工具、走幾步的圖 |
| Augmented LLM | 「加了工具的LLM」 | LLM + search + tools + memory,是五個pattern共同的原子單元 |
| Prompt chaining | 「一連串呼叫」 | 第N次呼叫的輸出是第N+1次呼叫的輸入 |
| Routing | 「分類器分派」 | 先決定用哪個chain/model處理輸入 |
| Parallelization | 「Fan out」 | N次並行呼叫,用sectioning(拆塊)或voting(多數決)聚合 |
| Orchestrator-workers | 「派工agent」 | Orchestrator一次性動態挑選要跑哪些specialist LLM |
| Evaluator-optimizer | 「提案+評審」 | 迭代到評審通過為止;Self-Refine的一般化版本 |

## 延伸閱讀
- [Anthropic — Building Effective Agents](https://www.anthropic.com/research/building-effective-agents):五種workflow pattern的原始出處,也是workflow vs agent這組區分的來源。
- [Anthropic — Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents):把context window當成預算而非容器的配套學科,決定何時該compact、何時該放手讓context長大。
- [LangGraph Overview](https://docs.langchain.com/oss/python/langgraph/overview):這五個pattern在真正框架裡怎麼被實作成節點圖。
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/):orchestrator-workers pattern的產品化版本。

## 複習自問
1. 教材Exercise 1要你替routing加上信心門檻,低於門檻就轉交真人。目前`route`的`classifier`只回傳一個label字串,沒有信心分數——要讓`route`支援這個功能,`classifier`的回傳型別要怎麼改?信心門檻的判斷該放在`route`函式內部,還是留給呼叫端在建構`handlers`時處理?
2. 教材Exercise 2要你替`parallel_vote`加上timeout,思考某次呼叫掛住不回應時該怎麼辦。目前`parallel_vote`是`[llm(prompt) for _ in range(n)]`,一次呼叫卡住整個list comprehension就卡住——如果要支援缺票情境下依然算出多數決,`votes`跟`counts`的資料結構要怎麼調整才能區分「投了no」跟「根本沒投到」?
3. 教材Exercise 3要你把`evaluator_optimizer`改成bandit:同時保留最好的兩個候選,避免晚出現的好結果被晚出現的壞結果覆蓋。目前的`trace`已經記錄了每一輪的`(candidate, verdict, feedback)`,但函式只回傳最後一個`candidate`——要選出「最好的兩個」,需要`evaluator`額外回傳分數還是只靠PASS/FAIL就夠?回傳值的型別要怎麼從單一`final`擴充成「保留兩個」?
4. 教材Exercise 4要你把prompt chaining跟routing結合:一個router從三條chain裡挑一條執行,並量測token成本對比單一大prompt的做法。如果要在這支程式上做,`route`的`handlers`字典要怎麼從「值是單一handler函式」改成「值是一整條`prompt_chain`」?
5. 教材Exercise 5要你挑一個自己專案裡的production功能,畫出它的workflow圖、數清楚步驟數,再判斷agent是否真的會表現更好。用本程式的判準(能否列舉步驟/是否成本有上限/是否需要稽核)套進去,你的功能落在workflow那一邊還是agent那一邊?如果答案不清楚,是缺了哪個資訊?
