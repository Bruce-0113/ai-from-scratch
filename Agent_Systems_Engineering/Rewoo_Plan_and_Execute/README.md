# ReWOO / Plan-and-Execute — 解耦式 Planner、Workers、Solver 三角色

對應程式: [`./rewoo_plan_and_execute.py`](./rewoo_plan_and_execute.py)

參考: [ai-engineering-from-scratch – ReWOO and Plan-and-Execute 課程](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/14-agent-engineering/02-rewoo-plan-and-execute/docs/en.md)

前置概念: 課程原本列的前置課是Phase14·01(Agent Loop,也就是ReAct風格的thought-action-observation交錯迴圈)——這支程式整段的論點都建立在「你已經知道ReAct怎麼運作」之上,重點放在ReWOO怎麼把這個交錯迴圈拆成Planner/Workers/Solver三個獨立角色。跟本repo同屬Phase14的[`../Orchestration_Patterns`](../Orchestration_Patterns/README.md)(supervisor/swarm/hierarchical/debate四種*多*agent拓撲)、[`../Stateful_Graph_Orchestration`](../Stateful_Graph_Orchestration/README.md)(把流程變成可存檔/可暫停的狀態機)處理的是不同維度的問題:那兩支討論的是「一個agent不夠用之後」或「流程中途失敗/需要人審之後」該怎麼辦,這支討論的是更前置的問題——**同一個agent、同一個任務**,要把「想」跟「做」黏在一起交錯進行(ReAct),還是先想好完整計畫再一次做完(ReWOO)。

## TL;DR
ReAct每一步工具呼叫都要把之前所有的thought/action/observation重新送一次給LLM,prompt長度(以及成本)隨步數增長,而且工具中途失敗時,模型得從錯誤的observation裡重新推導整個計畫。ReWOO(Xu et al., arXiv:2305.18323)打賭:規劃階段根本不需要看到observation——一次把整個計畫(一張帶依賴關係的DAG,用`#E1`、`#E2`這種字串引用彼此)生出來,worker平行(或依序)把每個節點的證據抓齊,solver最後只看「問題+計畫+全部證據」組出答案。論文在HotpotQA上量到約5倍token節省、準確率還多4個百分點,而且因為planner從不看observation,可以把它從175B teacher蒸餾成7B小模型,不用動到執行端。這支程式用純stdlib刻出這三個角色:`PlanStep`/`Plan`定義計畫DAG、`topological`只靠`#E<n>`字串引用(不是明確的邊列表)算出依賴順序、`run_workers`依序解引用並派工具呼叫、`ScriptedPlanner`/`ScriptedSolver`分別站在原本兩次LLM呼叫的位置上。`main()`用「法國首都的人口,四捨五入到百萬」這個兩跳問題跑一次ReWOO,再用`run_react_mock`重播同樣三次工具呼叫模擬ReAct風格的交錯歷史,兩者的字元數(token的替身指標)放在一起比,demo這次量到的比例是1.76倍——沒有論文的5倍,原因見下面「實作細節」。

## 為什麼需要它
- **對抗「每一步都要重送整個歷史」的隱性成本**:ReAct的prompt長度是隨步數線性(甚至因為每步都疊加前面所有內容而接近二次)成長的,`run_react_mock`裡`history_chars`每跑一步就往上疊加,直接把這個成長曲線變成可以印出來、算出來的數字,不用真的接LLM燒錢才發現某個40步的任務token爆了。
- **把「失敗定位」從整條trace精確到單一節點**:論文特別強調ReAct裡某一步工具失敗,模型得從那個錯誤observation倒推整個計畫該怎麼修;ReWOO裡`ToolRegistry.dispatch`把任何例外都包成一個`"error: ..."`字串存進對應的`evidence[step.id]`,只有solver最後看到這則錯誤,其他節點完全不受影響——失敗被鎖在它發生的那個節點裡,不會污染整條推理鏈。
- **拆開Planner與Worker,才有蒸餾的空間**:因為`ScriptedPlanner.plan_for`從頭到尾不看任何`evidence`,只看`question`就生出整張計畫(這支demo甚至連question都不看,直接回傳寫死的`Plan`),這個「planner不需要observation」的結構特性,正是論文能把planner蒸餾成7B小模型、同時保留175B大模型當executor的前提——這支程式沒有真的做蒸餾,但把這個結構性前提用程式碼的形狀直接體現出來。

## 核心原理
- **三個角色,一一對應到程式碼**:

  | 課程概念 | 本程式對應 |
  |---|---|
  | Planner | `ScriptedPlanner.plan_for`——站在「LLM讀question生出plan DAG」這次呼叫的位置上,這裡固定回傳建構時傳入的`Plan`,不看`question`也不看任何observation |
  | Plan DAG / 節點 | `PlanStep`(`id`/`tool`/`args`)與`Plan`(`steps`列表)——`args`裡的字串值可以包含`#E<n>`這種對更早節點結果的引用 |
  | Worker | `ToolRegistry.dispatch`+`run_workers`——依`topological`算出的順序,逐一解引用、呼叫工具、把結果記到`evidence[step.id]` |
  | Evidence reference(`#E1`,`#E2`,...) | `REFERENCE_RE`(正則)+`resolve_references`——把字串裡的`#E<n>`换成`evidence["E<n>"]`目前為止累積到的結果 |
  | Solver | `ScriptedSolver.solve`——站在「LLM讀question+plan+全部evidence組出最終答案」這次呼叫的位置上,這裡是格式化一個寫死的答案模板 |
  | Token效率的替身指標 | `ReWOORun.planner_chars`/`worker_chars`/`solver_chars`(ReWOO三段字元數) vs `run_react_mock`回傳值(ReAct累積字元數) |

- **`topological`只靠字串引用、沒有明確的邊列表就算出依賴順序**:每一輪掃過所有還沒排進去的`step`,用`REFERENCE_RE.findall(str(step.args))`從參數的字串形式裡挖出它引用的節點編號,如果這些編號全部都已經在`known`(已排好的節點id集合)裡,這個節點就可以排進去;一輪掃完如果完全没有新節點被排進去,代表有節點引用了不存在的id、或是兩個節點互相引用形成環,直接丟`RuntimeError`。這是一個O(n²)的fixed-point排序,對toy demo的節點數而言夠用,不是設計來處理大圖的。
- **`run_workers`是先算好完整執行順序,才開始真的呼叫工具**:`topological(plan)`回傳的是一份完整的、依賴關係已經排好的`PlanStep`列表,`run_workers`拿到這份列表後才逐一解引用、派工具呼叫——這代表依賴順序的計算(`topological`)跟真正的執行(`dispatch`)是兩個分開的階段,即使demo裡是循序執行,這個分階段的結構本身就是論文說的「worker之間可以平行跑」的必要前提(只要知道完整的依賴順序,獨立的節點理論上可以同時派發)。
- **ReWOO的成本是三段固定的呼叫,ReAct的成本是隨步數疊加的歷史**:`run_rewoo`只算三個數字——`planner_chars`(question+每個step的tool名稱與args)、`worker_chars`(每個step的args+它的結果)、`solver_chars`(question+全部worker輸出+最終答案),三者互不疊加、不會因為步數變多而讓前面算過的數字重複計入;`run_react_mock`則相反,`history_chars`每跑完一步工具呼叫就累加「這一步的工具名+參數+觀察結果+40字元的固定scaffolding開銷」,下一步的`total`會把這個累積中的`history_chars`整個重新算進去一次——這正是ReAct prompt長度隨步數增長的具體實作。

## 程式碼導覽
| 函式/類別 | 對應到理論的哪個部分 |
|---|---|
| `PlanStep`/`Plan` | 計畫DAG的資料結構:一個節點的`tool`+`args`+`id`,`args`裡可以嵌`#E<n>`引用 |
| `ToolRegistry` | Worker實際呼叫的工具表:`register`注册、`dispatch`呼叫並把任何例外包成`"error: ..."`字串 |
| `REFERENCE_RE`/`resolve_references` | 找出並替換字串裡的`#E<n>`引用,換成目前為止累積的`evidence` |
| `topological` | 只靠`#E<n>`引用(不靠明確邊列表)把`plan.steps`排成依賴順序,偵測環/未知引用 |
| `run_workers` | 依`topological`順序逐一解引用、派工具呼叫,回傳`id -> 結果`的`evidence` dict |
| `ScriptedPlanner`/`ScriptedSolver` | 分別站在「一次planner LLM呼叫」與「一次solver LLM呼叫」的位置上,這裡用寫死的`Plan`/答案模板取代 |
| `fake_search`/`rounded_million` | Demo用的兩個工具:關鍵字比對的假搜尋、從文字裡抓數字四捨五入成「N million」 |
| `ReWOORun`/`run_rewoo` | 跑一次Planner→Workers→Solver的完整流程,順便把三段字元數记录下来當token替身指標 |
| `run_react_mock` | 重播同一組工具呼叫,模擬ReAct每步都重送完整歷史的交錯迴圈,回傳總字元數 |
| `main` | 跑「法國首都人口」demo,印出plan/evidence/answer,再跑一次ReAct mock,印出兩者字元數與比例 |

**實作細節 / 容易看漏的地方:**
- **`resolve_references`裡「找不到對應evidence就原樣保留」這條路徑,在`run_workers`的正常流程裡永遠不會被走到**:`topological`只有在一個節點引用的所有id都已經排進`known`之後,才會讓這個節點排進最終順序;`run_workers`嚴格照這個順序執行,所以輪到某個節點被`resolve_references`處理時,它引用的每個`#E<n>`一定已經在`evidence`裡有值了。這條fallback不是死code(它讓`resolve_references`在被直接呼叫、繞過`topological`檢查時也不會拋例外),但在這支程式實際的呼叫路徑裡從未被觸發過——跟`Orchestration_Patterns`裡swarm的`hops < 3`上限是同一種「防呆但demo資料從不會踩到」的性質。
- **`topological`的環偵測是「一整輪掃描都沒有任何節點被排進去」的副作用,不是專門的環偵測演算法**:程式沒有另外維護一份圖結構去找環,`progress`這個flag只是「這輪有没有至少一個節點成功排入」的旗標——如果沒有,代表剩下的節點彼此循環引用或引用了不存在的id,兩種情況目前都丟同一個`RuntimeError("cyclic plan or unresolved reference")`,呼叫端從錯誤訊息本身無法分辨是哪一種。
- **`run_rewoo`裡`worker_chars`的計算,依賴`plan.steps`的原始順序剛好等於`topological`算出的執行順序**:`zip(plan.steps, evidence.values())`是照位置配對——`plan.steps`是`Plan`建構時給的原始列表順序,`evidence.values()`則是`run_workers`實際執行順序(由`topological`決定)插入`dict`的順序。demo裡的`Plan(steps=[E1, E2, E3])`本來就已經照依賴順序寫好,兩者剛好一致;但如果某個`Plan`的`steps`列表順序跟它真正的依賴順序不同(例如故意把`E2`寫在`E1`前面),`run_workers`透過`topological`仍然會算出正確的執行順序與正確的`evidence`(答案不會錯),但`run_rewoo`這裡的`zip`會把`plan.steps`裡的某個step的`args`錯位對到另一個step的`evidence`結果上,算出來的`worker_chars`(進而影響`solver_chars`)就會是錯的字元數——只影響token替身指標,不影響demo實際印出的答案是否正確。
- **這支demo量到的比例(1.76倍)遠比論文的5倍小,是trace長度太短的緣故**:`run_react_mock`的`history_chars`是隨步數疊加的,3步的demo疊加次數太少,還沒疊到能明顯拉開差距的程度;論文的5倍是在HotpotQA真正的多跳問答(通常牽涉更長的推理鏈)上量到的。把demo的步數拉長(例如把兩跳問題換成五跳、十跳),`react_chars`跟`rewoo_chars`的差距會用更明顯的比例拉開——這支程式的三步demo只夠展示「形狀」,不夠展示論文量到的具體倍率。

## 使用時機 / 優缺點
依課程給的決策表(依任務性質挑,不是越新的模式越好):
| 模式 | 適用情境 |
|---|---|
| ReAct | 短任務、環境未知、需要即時反應例外狀況 |
| ReWOO | 結構清楚、工具已知、對token敏感、證據可平行取得的任務 |
| Plan-and-Execute | 跟ReWOO類似,但執行後可以根據結果重新規劃(replan) |
| Plan-and-Act | 長流程(超過30步)的網頁/手機/電腦操作任務 |
| Tree of Thoughts | 值得為了搜尋品質多付成本的任務(見Lesson 04) |

Anthropic的建議是從最簡單的模式開始:一次工具呼叫加一句總結,不需要上ReWOO;但一個40步的研究任務,也不該只靠裸ReAct硬撐。

- ✅ 想搞懂ReWOO三角色怎麼具體對應到程式碼:`PlanStep`/`Plan`/`ToolRegistry`/`topological`/`run_workers`/`ScriptedPlanner`/`ScriptedSolver`加起來不到150行,`main()`印出的plan、evidence、answer可以逐行對照「這一步是規劃、執行還是收斂」。
- ✅ 需要一個不接LLM、100%可重現的骨架,拿來驗證`#E<n>`引用解析與依賴排序邏輯本身,或當成寫單元測試/教學示範的起點。
- ✅ 想直接看到「ReAct每步重送歷史」跟「ReWOO三段固定呼叫」這兩種成本曲線的具體算法差異:`run_react_mock`跟`run_rewoo`加起來不到40行,不用真的接LLM API就能比較兩者字元數。
- ❌ Planner是完全寫死的:`ScriptedPlanner.plan_for`連`question`都不看,永遠回傳同一份`Plan`——這支程式示範的是Workers/Solver怎麼消化一份計畫,不是「LLM怎麼根據問題生出計畫」,也没有Plan-and-Execute的replan機制(執行後發現結果不對,回頭修改計畫)。
- ❌ 沒有真正的平行執行:`run_workers`雖然靠`topological`算出了理論上可以平行跑的獨立節點,但實際呼叫`tools.dispatch`還是一個循序的Python迴圈,不是真的並發呼叫多個工具。
- ❌ 字元數只是token的粗略替身,demo比例(1.76x)沒有還原論文的5x:真正的token數要看tokenizer,而且論文的倍率是在更長的多跳推理鏈上量到的,3步demo的trace長度不夠把ReAct的歷史疊加效應體現出來。
- ❌ 沒有實作planner蒸餾:程式的結構(planner不看evidence)體現了蒸餾的前提,但沒有真的訓練或載入任何7B/175B模型——這部分留給讀者自己在Exercise裡延伸。

## 常見誤區
1. **以為`resolve_references`「找不到引用就原樣保留」這條分支是用不到的廢code**:它在`run_workers`的正常呼叫路徑裡確實從未被觸發(因為`topological`已經先過濾掉引用未就緒的節點),但它是`resolve_references`作為獨立函式被直接呼叫、或計畫結構之外的防呆機制,拿掉它會讓函式在異常輸入下改成拋例外,不是單純的無效程式碼。
2. **以為`topological`的`RuntimeError`一定代表「有環」**:訊息裡同時涵蓋兩種情況——節點引用了根本不存在的`#E<n>`,或是節點之間真的互相循環引用——目前的實作沒有區分兩者,看到這個錯誤不代表計畫裡一定有環,也可能只是引用了打錯的編號。
3. **以為`worker_chars`/`solver_chars`算出來的數字永遠正確**:這兩個數字依賴`run_rewoo`裡`zip(plan.steps, evidence.values())`這個位置對位,只有當`plan.steps`的原始順序恰好等於`topological`算出的依賴執行順序時才會對齊——`run_workers`本身透過`topological`永遠會得到正確的執行結果與正確答案,錯的只可能是這個额外统计出來的字元數。
4. **以為demo沒量到論文的5倍就代表ReWOO在小任務上沒有優勢,或是這支程式的比較方式有問題**:1.76倍是trace只有3步的緣故——`run_react_mock`的歷史疊加效應需要更多步數才能明顯拉開差距,論文的5倍是在HotpotQA真正的多跳問答(推理鏈通常更長)上量到的,不是這支demo的算法本身有誤差。
5. **以為ReWOO永遠比ReAct好**:課程的決策表很明確——短任務、環境未知、需要即時反應例外狀況的情境仍然適合ReAct,ReWOO的優勢建立在「工具已知、結構清楚、證據可以平行取得」這幾個前提成立的時候,Anthropic的建議也是先從最簡單的模式開始,不是預設一律上ReWOO。

## 關鍵詞速查
| 詞彙 | 常聽到的說法 | 實際上是什麼 |
|---|---|---|
| ReWOO | 「不看observation的推理」 | 先規劃、再平行取證據、最後才解答——規劃prompt裡完全不含observation |
| Plan-and-Execute | 「LangChain的plan-execute模式」 | ReWOO加上一個執行後可選的replanner節點 |
| Plan-and-Act | 「放大版plan-execute」 | 明確拆分planner/executor,並用合成的plan訓練資料處理長流程任務 |
| Evidence reference | 「`#E1`、`#E2`、...」 | 計畫節點裡的佔位符,派工具呼叫前才被替換成更早節點的實際輸出 |
| Planner distillation | 「小planner、大executor」 | 用大模型(teacher)產生的planner軌跡去微調一個小模型 |
| Token efficiency | 「更少的來回呼叫」 | 論文在HotpotQA上量到比ReAct少約5倍token |
| DAG executor | 「拓撲派工器」 | 依依賴順序執行計畫節點,理論上同一層級的節點可以平行 |

## 延伸閱讀
- [Xu et al., *ReWOO: Decoupling Reasoning from Observations* (arXiv:2305.18323)](https://arxiv.org/abs/2305.18323):ReWOO的原始論文,本程式Planner/Workers/Solver三角色與5倍token節省的說法都出自這裡。
- [Erdogan et al., *Plan-and-Act* (arXiv:2503.09572)](https://arxiv.org/abs/2503.09572):把plan-execute放大到長流程網頁/手機agent,用合成的plan訓練資料處理30~50步以上的任務。
- [LangGraph Plan-and-Execute tutorial](https://docs.langchain.com/oss/python/langgraph/overview):plan-execute在真正的框架裡怎麼實作。
- [Anthropic — Building Effective Agents](https://www.anthropic.com/research/building-effective-agents):「先選最簡單、夠用的模式」這個判斷順序的原始出處。

## 複習自問
1. 課程Exercise 1要你把獨立的計畫節點平行化執行。目前`run_workers`是照`topological`回傳的順序逐一`dispatch`,如果`plan`裡有兩個節點互不依賴(例如都只依賴`E1`、彼此之間沒有引用關係),`topological`回傳的列表要怎麼標示「這兩個節點其實可以同時跑」?`run_workers`要怎麼改,才能讓這種節點真的平行呼叫工具?
2. 課程Exercise 2要你加一個replanner節點,只要有任何worker回傳錯誤就觸發重新規劃——這是ReWOO變成Plan-and-Execute最小的一步改動。以現在的介面來看,`run_rewoo`要在哪個時間點檢查`evidence`裡有沒有`"error: "`開頭的結果?`ScriptedPlanner`要新增什麼方法,才能根據「原本的plan+目前的evidence」生出一份修正後的新plan?
3. 課程Exercise 3要你把`ScriptedPlanner`換成一個真正的小模型(7B等級),`ScriptedSolver`留在前沿大模型上,比較端到端品質。這樣拆分之後,如果小模型生出的`Plan`裡`#E<n>`引用寫錯(例如引用了不存在的節點),現在`topological`丟出的`RuntimeError`夠不夠讓上層知道「這是planner的錯,不是worker的錯」,需不需要更細的錯誤分類?
4. 課程Exercise 4要你讀論文Section 4的planner蒸餾方法,想清楚需要什麼訓練資料、怎麼評分plan品質。以這支程式的`Plan`/`PlanStep`資料結構來看,要收集「175B teacher生成的plan」當訓練資料,需要额外記錄哪些欄位(例如每個節點的信心分數、或者這個節點是否真的被後續證明是必要的)?
5. 課程Exercise 5要你把這支toy改寫成Plan-and-Act的形狀——plan是一個序列而不是DAG。如果拿掉`#E<n>`引用跟`topological`排序,單純把`Plan.steps`當成一個固定順序的序列執行,會失去現在DAG結構的哪個能力(提示:想想有兩個節點都依賴同一個更早節點、但彼此不依賴的情況)?這個取捨對什麼樣的任務是划算的?
