# Orchestration Patterns — Supervisor、Swarm、Hierarchical、Debate 四種多代理協作拓撲

對應程式: [`./orchestration_patterns.py`](./orchestration_patterns.py)

參考: [ai-engineering-from-scratch – Orchestration Patterns 課程](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/14-agent-engineering/28-orchestration-patterns/docs/en.md)

前置概念: 課程原本列的前置課是Phase14·12(Workflow Patterns:prompt chaining、routing、parallelization、orchestrator-workers、evaluator-optimizer五種單agent工作流)與Phase14·25(Multi-Agent Debate)——這支程式假設你已經知道「單一agent+這五種workflow pattern什麼時候就夠用了」,重點放在再往後一步:當一個agent真的不夠用,要往哪個方向長成多個agent、每種長法的代價是什麼。跟本repo同屬Phase14的[`../Agent_Memory`](../Agent_Memory/README.md)(MemGPT兩層記憶)、[`../Stateful_Graph_Orchestration`](../Stateful_Graph_Orchestration/README.md)(LangGraph狀態機)是同一個Phase處理不同維度的問題:Agent Memory處理「單一agent的context塞不下」,Stateful Graph處理「多步驟流程中途失敗、或需要人審該怎麼辦」,這支處理「一個agent的能力或context不夠時,要不要長出第二個agent、長成什麼形狀」。

## TL;DR
團隊常常「先決定要上multi-agent,再回頭找問題」——教材引用Anthropic的說法:「成功不在於做出最複雜的系統,而在於做出符合你需求的系統。」2026年的框架裡反覆出現的多代理拓撲其實只有四種:**supervisor-worker**(中央router分派給specialist,specialist之間互不對話)、**swarm/peer-to-peer**(沒有中央router,agent之間直接互相handoff)、**hierarchical**(supervisor管sub-supervisor管worker,巢狀結構)、**debate**(多個proposer平行提案,靠多數決收斂,本質上比較接近「驗證」而不是「路由」)。能叫得出這四個名字,才有辦法「選對拓撲」或「乾脆不用拓撲」。本程式把同一個三分類客服任務(退款/bug/銷售)餵給一個共用、**deterministic**的`classify`函式(用來代替一次「LLM決定路由到哪」的呼叫),分別跑過這四種拓撲,並用`ops`(每次模擬的agent turn計一次)當作真實成本指標(LLM呼叫次數)的替身——四條trace long相同、但op數不同:supervisor每個task 2 ops最省、swarm平均1~2 ops視是否需要handoff、hierarchical每個task固定3 ops(拓撲最深)、debate每個task 5 ops(拓撲最貴),對應`main()`結尾那句評語:「supervisor最乾淨、swarm最短、hierarchical最深、debate最貴,先想清楚問題,再選拓撲。」

## 為什麼需要它
- **對抗「topology-first」的直覺**:教材點名最常見的失敗模式就是「我們需要multi-agent」先於「multi-agent解決了什麼問題」。四個拓撲各自的op數在這支程式裡是可以直接算出來、印出來比較的,不用真的接上LLM、燒了真的錢跟時間才發現某個拓撲比別的貴兩三倍。
- **把「swarm bouncing」「假階層」這些反模式變成看得到的程式碼**:swarm的handoff上限(`hops < 3`)就是教材說的「加個hop counter防止A→B→A→B無限彈跳」的具體實作;hierarchical的兩層路由也是「三層是因為聽起來很enterprise,其實只有兩個真正的team」這個反模式的對照組——本程式刻意只做兩層乾淨的hierarchical,沒有無意義的第三層。
- **決策順序是有先後的,不是四選一的平行選單**:Anthropic的建議是由簡入繁——先單一agent+workflow pattern、不夠才上supervisor(2-4個specialist)、latency比推理清晰度重要才選swarm、supervisor的context塞不下所有specialist描述才需要hierarchical、accuracy比cost重要才用debate。搞懂這個順序,才不會預設一開始就選最貴的debate。

## 核心原理
- **四個拓撲共用同一個deterministic router,只是「誰在什麼時候問它」不同**:

  | 課程概念 | 本程式對應 |
  |---|---|
  | Router / 路由決策 | `classify(text)`——所有拓撲背後那次「LLM決定路由到哪」的呼叫,這裡用關鍵字比對取代,刻意做成deterministic,好讓四條trace的路由結果永遠一致,只比較拓撲的形狀與成本,不比較路由準不準 |
  | Specialist | `SPECIALISTS`——三個intent各自的終端處理函式,只回傳一句「已處理」字串,重點在拓撲不在specialist邏輯本身 |
  | ops | 每個拓撲function裡的`ops`計數器——每次+1代表一次模擬的agent turn,是「LLM呼叫次數/成本」的替身指標 |
  | Supervisor-worker | `supervisor_worker`——一次路由決策+一次specialist執行,2 ops/task,四者中最省 |
  | Swarm / peer-to-peer | `swarm`——沒有中央router,agent直接handoff給它覺得該接手的下一個agent,最多3次handoff防止彈跳 |
  | Hierarchical | `hierarchical`——先分department(`customer_ops`/`commercial`)、department內再分specialist,3 ops/task,拓撲最深 |
  | Debate | `debate`——三個proposer平行提案+一次多數決收斂,5 ops/task,四者中最貴 |

- **swarm的handoff機制**:每個task固定從`SPECIALISTS`裡第一個注册的agent(`"refund"`)開始問;輪到某個agent時,它自己跑一次`classify(task)`,如果分類結果跟自己的身分相同就直接處理掉,不同就把task交給它認為對的那個agent、換那個agent接手。因為`classify`是deterministic的,同一個task無論誰先問到,第一次handoff問到的對象一定是正確答案,所以demo資料最多只需要1次handoff就會收斂——`hops < 3`這個上限在這支demo裡永遠不會真的被觸發,它是「防止彈跳」的防呆機制,不是demo資料會走到的路徑。
- **hierarchical對同一個task呼叫了兩次`classify`**:一次決定`top_label`(粗分類成`customer_ops`或`commercial`),一次決定`sub_label`(具體要給哪個specialist)。這不是重複勞動,而是刻意模擬「兩層各自做一次獨立的路由決策」——只是這支demo兩層剛好都問同一個deterministic函式,現實中兩層通常是不同的agent、不同的prompt,才需要真的分成兩層。
- **debate的多數決在這支demo裡從沒真正被考驗過**:三個proposer(`alpha`/`beta`/`gamma`)各自呼叫同一個deterministic`classify`,所以提案永遠一致,`Counter(proposals).most_common(1)`每次都在一個「全票通過」的list裡找最大值,語法上是多數決,但實際上從沒解決過任何分歧。要讓這段邏輯真正發揮作用,三個proposer得換成三次獨立、有機率性的模型取樣。
- **op數是拓撲結構本身算出來的,不是寫死的常數**:supervisor固定2 ops/task,swarm依實際handoff次數浮動(這份demo資料落在1~2 ops/task),hierarchical固定3 ops/task,debate固定5 ops/task(3次提案+1次收斂決策+1次specialist執行)——四個數字排起來,剛好對應`main()`結尾「supervisor最乾淨、swarm最短、hierarchical最深、debate最貴」這句結論。

## 程式碼導覽
| 函式/物件 | 對應到理論的哪個部分 |
|---|---|
| `classify` | 共用router:用關鍵字把文字分成`refund`/`bug`/`sales`,代替一次LLM路由呼叫 |
| `SPECIALISTS` | 三個intent各自的終端handler,回傳一句代表「已處理」的字串 |
| `supervisor_worker` | Supervisor-worker拓撲:一次路由+一次specialist執行 |
| `swarm` | Swarm/peer-to-peer拓撲:agent之間直接handoff,`hops < 3`防彈跳 |
| `hierarchical` | Hierarchical拓撲:top-level部門路由→department內specialist路由→specialist執行 |
| `debate` | Debate拓撲:三個proposer平行提案→`Counter.most_common`多數決→specialist執行 |
| `main` | 用同一組三個task跑過四個拓撲,印出各自的trace跟`ops`小計 |

**實作細節 / 容易看漏的地方:**
- **整理時收斂掉的死分支**:`classify`原本多一段`if "pricing" in t or "quote" in t: return "sales"`,但`else`本來就會落到同一個`"sales"`結果——這段判斷式無論走不走,回傳值都一樣,是一個「看起來有特別處理、實際上什麼都没多做」的死分支,整理時已收斂成單一`return "sales"`,行為完全不變(pricing/quote相關的輸入一樣分類成`sales`)。
- **整理時移除的未使用import**:原始程式import了`dataclasses.dataclass`、`dataclasses.field`、`typing.Any`,但整支程式沒有任何一處用到——四個拓撲函式全部用純函式+`dict`+`list`實作,沒有定義任何dataclass,`Any`也沒被當成型別標注用過。這三個都是死code,整理時一併移除,只留下真正用到的`typing.Callable`。
- **`swarm`每個task都會把`current`重設回`list(SPECIALISTS)[0]`**:也就是永遠從`"refund"`這個agent開始問,不是從上一個task結束時的agent接著問——三個task之間彼此獨立,不會有「task 2從task 1收斂的agent開始」這種跨task的狀態殘留。
- **`debate`裡有兩個連續的`ops += 1`,不是重複計算的失誤**:一個算在收斂/計票決策本身(`Counter.most_common`那一步),一個算在specialist真正執行的那一步——對應到`supervisor_worker`裡「路由決策一次、specialist執行一次」的邏輯,只是debate多了「三個proposer各自提案」這個前置階段,所以總數是`3(提案)+1(收斂)+1(specialist)=5`,不是看起來的4。

## 使用時機 / 優缺點
依教材給的決策順序(由簡入繁,越後面代價越高):
1. 單一agent+workflow pattern(prompt chaining/routing/parallelization/orchestrator-workers/evaluator-optimizer)——多數情境先從這裡開始,本程式的四個拓撲都還沒用到。
2. Supervisor-worker——2-4個specialist、需要清楚的中央控制點時用。
3. Swarm——latency比「決策路徑好不好懂」更重要時用;沒有中央router,少一跳,但除錯時比較難追蹤「這個task到底怎麼被決定送到這裡的」。
4. Hierarchical——只有在單一supervisor的context塞不下所有specialist描述時才需要;純粹為了「看起來像大型系統」而加層,就是教材點名的假階層反模式。
5. Debate——accuracy比cost重要、值得為了正確率多付好幾倍op數時才用。

- ✅ 想快速搞懂四種拓撲的形狀差異、以及它們的op數(=成本)怎麼隨拓撲結構變化:四個函式加起來不到100行,`main()`印出的trace可以逐行對照「這一步是路由決策還是specialist執行」。
- ✅ 需要一個不接LLM、100%可重現的骨架,拿來驗證「拓撲成本差異」這個結論本身,或當成寫單元測試/教學示範的起點。
- ✅ 示範swarm的hop-limit防呆機制、hierarchical的兩層路由、debate的多數決收斂——三個常被提到但很少看到最小可執行實作的機制,這裡都各自不到10行程式碼。
- ❌ Router是deterministic的,四個拓撲永遠得到同一個路由結果:這支程式量的是「拓撲的成本與形狀」,不是「拓撲的路由品質」或「拓撲如何處理分歧」——尤其debate的多數決邏輯完全沒被考驗過,因為三個proposer從不會意見不合。
- ❌ 沒有真正的平行執行:swarm的handoff、debate的三個proposer在程式裡都是循序的Python迴圈,只是「概念上」代表平行的agent turn,不是真的並發呼叫。
- ❌ Hierarchical只有兩層、且兩層共用同一個分類函式:教材Exercise 3要求的12-specialist情境需要真正分開的sub-supervisor邏輯,本程式的`top_label`判斷式是寫死的兩個分支(`customer_ops`/`commercial`),不會自動long成更多層或更多部門。
- ❌ 沒有真的LLM呼叫,`ops`只是一個「一次turn記一次」的計數器:真實系統裡不同specialist、不同model size的呼叫成本天差地遠,`ops`目前是均一權重,拿來比拓撲*形狀*的相對貴賤沒問題,拿來估算真實美元成本就不夠精細。

## 常見誤區
1. **以為四個拓撲的op數是寫死的常數**:只有supervisor(2)、hierarchical(3)、debate(5)是固定的,swarm的op數其實是「實際需要幾次handoff才收斂」算出來的,demo資料剛好都落在1~2次handoff內,換一組task不一定得到一樣的數字。
2. **以為`hierarchical`裡呼叫兩次`classify`是重複勞動或bug**:兩次呼叫分別代表「top-level部門路由」跟「department內specialist路由」兩個獨立的決策層——只是這支demo省事共用同一個deterministic函式回答兩個問題,真實系統裡兩層通常是不同的agent。
3. **以為debate示範了「多數決真的解決了分歧」**:三個proposer的輸入完全相同(都呼叫同一個deterministic`classify`),`Counter.most_common`從沒真正需要在不一致的提案中選邊站——要看到這段邏輯發揮作用,得把三次提案換成三次獨立、有雜訊的LLM呼叫。
4. **以為選拓撲時「越花俏越好」**:Anthropic原文特別強調"building the right system for your needs",教材的決策順序是由簡入繁,每往後一步(supervisor→swarm→hierarchical→debate)都有明確的觸發條件(specialist數量、latency需求、context budget、accuracy需求),不是預設用最複雜的那個。
5. **以為swarm沒有中央router就等於沒有路由決策**:swarm一樣要跑`classify()`,差別只在「誰來問這個問題」——supervisor是固定的中央agent問一次,swarm是current agent自己問、覺得不是自己的工作就交棒——決策邏輯沒有消失,只是換了執行的位置,也因此天生比supervisor更難追蹤「這個task最後怎麼被決定的」。

## 關鍵詞速查
| 詞彙 | 常聽到的說法 | 實際上是什麼 |
|---|---|---|
| Supervisor-worker | 「router+specialist」 | 中央LLM分派給specialist;specialist彼此不對話,只對supervisor負責 |
| Swarm | 「peer-to-peer」 | 透過共用的tool介面直接handoff,沒有中央router |
| Hierarchical | 「supervisor的supervisor」 | 巢狀subgraph,用於specialist數量大到單一supervisor context塞不下的情境 |
| Debate | 「proposer+critique」 | 平行提案+交叉檢驗,收斂靠多數決,本質更接近驗證而非路由 |
| Tool-call-based supervision | 「不靠函式庫的supervisor」 | 2026年LangChain的建議:直接用tool call實作supervisor邏輯,換取更精細的context控制,而不是套用`create_supervisor`這類現成函式庫 |
| Crew | 「自主團隊」 | CrewAI裡角色制、自主協作的部署模式 |
| Flow | 「確定性工作流」 | CrewAI裡事件驅動、適合production的部署模式,教材建議當作起點 |

## 延伸閱讀
- [Anthropic — Building Effective Agents](https://www.anthropic.com/research/building-effective-agents):五種workflow pattern+agent與workflow的區別,是本課程「先workflow、不夠再上topology」這個判斷順序的原始出處。
- [LangGraph Overview](https://docs.langchain.com/oss/python/langgraph/overview):supervisor、swarm、hierarchical(巢狀subgraph)在真正的框架裡長什麼樣子。
- [CrewAI Docs](https://docs.crewai.com/en/introduction):Crew(自主角色協作)vs Flow(事件驅動確定性流程)兩種部署模式的官方說明。
- [Du et al., *Improving Factuality and Reasoning in Language Models through Multiagent Debate* (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325):debate拓撲的原始論文,本程式`debate()`函式簡化實作的理論依據。

## 複習自問
1. 教材Exercise 1要你把supervisor-worker拿掉router、改寫成swarm,看看什麼會壞掉、什麼會變好。如果照這個方向改`supervisor_worker`:讓`SPECIALISTS`裡每個handler自己判斷「這是不是我的工作」而不是靠中央`classify`一次決定,原本兩行的trace(`supervisor -> label`/`  label: ...`)會變成什麼形狀?需不需要像`swarm`一樣加一個hop counter?
2. 教材Exercise 2要你替swarm加上hop counter、超過3次handoff就refuse。`swarm`裡`hops < 3`這個上限其實已經在,但因為`classify`是deterministic的,demo資料從沒真的觸發過它——目前的程式碼在超過上限後,`while`迴圈會直接結束、什麼trace都不會多印一行,呼叫端拿到的是一個「這個task悄悄消失」的結果。如果要讓它真的觸發(例如刻意讓兩個agent對同一個task互相甩鍋),`swarm`要多印哪一行、回傳值要怎麼標示「這個task被拒絕了」?
3. 教材Exercise 3要你為12個specialist的情境設計兩層hierarchical,看看context budget在哪裡撐不住。目前`hierarchical`的`top_label`只有寫死的兩個分支(`"customer_ops"`/`"commercial"`),如果要一般化成12個specialist、可能好幾個部門的情境,這個判斷式要怎麼改,才不會變成一長串難以維護的if/elif?
4. 教材Exercise 4要你在真實production流量上量測四個拓撲,看哪個在哪個指標(latency/cost/accuracy/debuggability)勝出。目前四個函式都回傳`(trace, ops)`,`ops`是均一權重的「turn計數器」——如果要換成更貼近真實成本的量測(每次specialist呼叫的token數不同、debate的三個proposer可能用不同大小的model),這個回傳介面要怎麼擴充,才能同時回報好幾個維度而不只是一個數字?
5. 教材Exercise 5要你讀完Anthropic的Building Effective Agents之後,把自己專案裡的production流程對應到這四個拓撲之一。拿一個你自己專案裡「有多個步驟或多個角色」的流程,套用本程式`classify`+四個拓撲函式的寫法,能不能壓進supervisor-worker/swarm/hierarchical/debate四個裡的其中一個?如果四個都套不進去,問題出在哪裡——是需要混合拓撲,還是根本還沒到需要multi-agent的程度?
