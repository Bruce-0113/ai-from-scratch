# Stateful Graph Orchestration — 狀態機、Checkpoint 與 Human-in-the-Loop（LangGraph）

對應程式: [`./stateful_graph_orchestration.py`](./stateful_graph_orchestration.py)

參考: [ai-engineering-from-scratch – Stateful Graph Orchestration 課程](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/14-agent-engineering/13-langgraph-stateful-graphs/docs/en.md)

前置概念: Phase 14 的 Agent Loop(ReAct風格的agent控制迴圈)與Tool Use——這支程式假設你已經知道agent怎麼呼叫工具、怎麼跑一輪推理,重點放在「把整個agent流程本身變成一張可以中途存檔、中途暫停、中途被人類介入的圖」這件事。跟[`../Agent_Memory`](../Agent_Memory/README.md)(MemGPT兩層記憶)是同一個Phase處理不同問題的兩支程式:Agent Memory處理「context window裝不下」,這支處理「多步驟流程中途失敗、或需要人審該怎麼辦」。

## TL;DR
把agent寫成一長串Python function呼叫,中途在第38步crash就只能從第1步重跑;LangGraph的解法是把agent的控制流程本身變成一張圖——明確的state(一個typed dict)、多個node(純函式,讀state、回傳一個要合併進去的update dict)、多個edge(決定下一步去哪個node,包含依state內容分岔的conditional edge)。整張圖靠同一個規則運作:每個node跑完,runtime立刻把新state序列化存進checkpointer——所以恢復執行只是「load最後一次存的state,從那之後的node接著跑」,不用整個流程重來一次。這個「每個node都存檔」的機制同時解決兩件事:**durable execution**(第38步失敗,從第39步接著跑,不是從頭開始)和**human-in-the-loop**(state本身已經被序列化、有位址,可以直接被人看、被人改,改完state就能繼續跑)。本程式用純Python(`StateGraph`/`InMemoryCheckpointer`/`Runner`)刻出這個核心迴圈,並用一個「分類客服訊息→(退款/bug/銷售其中一條路徑)→人工審核閘門→送出」的demo,示範跑到`human_gate`暫停、檢查checkpoint歷史、模擬人工核准後從暫停處接續執行到底的完整流程。

## 為什麼需要它
- **多步驟流程失敗不該從頭重來**:40步的workflow在第38步失敗,重跑代價可能是真的錢和真的時間(重新呼叫外部API、重新寫資料庫……)。checkpoint機制讓恢復只需要讀回最後一次存的state,從上一個成功的node之後接著跑。教材點名Klarna、Uber、J.P. Morgan這幾個生產環境案例都靠這個模式撐住。
- **Human-in-the-loop需要state本身是可檢視、可修改的**:要讓人在流程中途看到state、改state、再把流程接著跑下去,前提是state已經被持久化、而且有明確的位址(`session_id`)——checkpointer的基礎建設剛好順便解決了這個前提,不需要另外設計一套「暫停中流程」的管理系統。
- **Conditional edge讓分岔邏輯變成圖的一部分,不是散落在程式碼裡的if/else**:把「這條訊息該去退款、bug還是銷售」這個決策,變成一個「從state讀路由結果、選擇下一個node」的edge,整張圖本身就是流程文件,不用另外去翻程式碼追蹤判斷式散落在哪裡。

## 核心原理
- **狀態機的基本元件,一一對應到程式碼**:

  | LangGraph概念 | 本程式對應 |
  |---|---|
  | Typed state | `State`(`dict[str, Any]`的type alias)——貫穿整個流程的單一資料結構 |
  | Node | `NodeFn`——讀`State`、回傳一個要合併進state的`Update` dict的純函式 |
  | Edge | `Edge` dataclass——`src`→`dst`,可選一個`router`決定這條邊是否要走 |
  | Reducer | `Runner.run`裡的`state = {**state, **update}`——最簡單的reducer:update裡的key直接覆蓋state裡對應的key |
  | Checkpointer | `InMemoryCheckpointer`——每個node跑完就`save`一次,`load_latest`/`history`負責讀回 |
  | Durable execution | `Runner.run`的`resume_from` + `state_override`——從指定node、指定state接著跑,不用回到entry重來 |

- **Conditional edge是「先注册所有可能的目的地,執行時只選一條」**:`add_conditional_edges(src, router, targets)`把`targets`這個`{值: node名稱}`的dict展開成好幾條`Edge`,每條都用`_make_router`包一層「`router(state)`是不是等於這個值」的判斷;`StateGraph._next`依注册順序掃過`src`的所有邊,回傳第一條「沒有router、或router(state)為真」的邊的目的地——這代表邊的注册順序有意義:如果多條邊的router同時為真,永遠是先注册的那條生效。

- **Checkpoint的時機是「node跑完、state更新之後,決定下一步之前」**:`Runner.run`每輪迴圈都是「跑node→合併update進state→存checkpoint→檢查要不要暫停→決定下一個node」,這個順序保證checkpoint裡存的state永遠是「某個node完整執行完畢」之後的快照,不會存到執行一半的中間狀態。

- **Human-in-the-loop靠一個「magic key」`_pause_reason`實作**:node只要在回傳的update裡塞`_pause_reason`這個key,`Runner.run`檢查到合併後的state裡有這個key就丟出`PausedAtNode`例外,把目前的node名稱、暫停原因、和state(已經checkpoint過)一起帶給呼叫端;人工審核完,呼叫端把改好的state存回checkpointer,再用`resume_from` + `state_override`把流程接著跑下去——state本身就是「暫停原因」和「恢復點」的唯一真相來源,不需要另外一套暫停狀態管理。

- **Resume會跳過已經跑完的node,不會重跑它**:demo裡`human_gate`暫停時,它自己這一步其實已經完整執行過(`step`已經+1、`_pause_reason`已經寫進state並被checkpoint),所以人工核准後resume的起點是`send`,`human_gate`本身不會被重跑一次;如果改成`resume_from="human_gate"`,因為此時`human_approval`已經是`True`,也會正確地直接放行、不再暫停,只是`step`會被多算一次。

## 程式碼導覽
| 函式/類別 | 對應到理論的哪個部分 |
|---|---|
| `State`/`Update`/`NodeFn`/`Router` | Type alias:state是一個dict、node是「state→update dict」的函式、router是「state→字串」的函式 |
| `START`/`END` | 圖的起點/終點哨兵值;`END`用來判斷`Runner.run`的while迴圈該不該停 |
| `Edge` | 一條`src`到`dst`的邊,`router`為`None`代表無條件邊 |
| `StateGraph` | 圖本身:`nodes`(名稱→函式)、`edges`(來源node→邊列表)、`entry`(起點node名稱) |
| `StateGraph.add_conditional_edges` | 把一個`{值: 目的node}`的dict展開成多條帶router的`Edge`,一次注册一整組分岔 |
| `StateGraph._next` | 依注册順序找第一條「無條件、或router(state)為真」的邊,回傳目的node名稱;找不到就回傳`None`(圖在此終止,不報錯) |
| `_make_router` | 把「router(state)」和「預期值」包成一個回傳bool的判斷式,供`add_conditional_edges`內部使用 |
| `InMemoryCheckpointer` | 用`dict[session_id, list[(node名稱, state快照)]]`存所有歷史checkpoint;`save`會對state做`deepcopy`,避免呼叫端後續原地修改state影響到已經存下的快照 |
| `PausedAtNode` | Human-in-the-loop用的例外:帶著暫停時的node名稱、state、和暫停原因(`reason`) |
| `Runner.run` | 核心迴圈:跑node→合併state→存checkpoint→檢查`_pause_reason`決定要不要暫停→用`StateGraph._next`決定下一個node,直到碰到`END`或無邊可走 |
| `_classify`/`_refund`/`_bug`/`_sales`/`_send` | Demo用的node:依關鍵字把輸入路由到退款/bug/銷售其中一條,各自產生一個ticket編號,最後`_send`組出輸出字串 |
| `_human_gate` | Demo用的human-in-the-loop node:`state["human_approval"]`為假就回傳帶`_pause_reason`的update觸發暫停,為真就正常放行 |
| `build_graph` | 組出完整demo圖:`classify`用conditional edge分岔到三條路徑,三條都匯流進`human_gate`再到`send`到`END` |
| `main` | Demo劇本:第一次執行在`human_gate`暫停→印出checkpoint歷史→模擬人工核准、把`human_approval`改成`True`存回checkpointer→用`resume_from="send"`接續執行到底 |

**實作細節 / 容易看漏的地方:**
- **暫停那一格的checkpoint,跟丟出來的例外裡的state,其實不是同一份東西**:`Runner.run`是先`self.checkpointer.save(session_id, current, state)`存檔,才`state.pop("_pause_reason")`,再丟出`PausedAtNode(current, state, reason)`。`save`裡對state做的是`deepcopy`,所以存進checkpointer的那份快照拍照時間點還帶著`_pause_reason`;而`pop`是原地修改同一個`state` dict,所以`PausedAtNode.state`(和之後`paused.state`)已經不含`_pause_reason`。兩者印出來很像,但仔細比對key會發現差一個欄位——這不是bug,是「先存檔、再清理暫停用的內部標記給呼叫端看」的刻意順序。
- **`Runner`只會寫checkpointer,從來不會讀它來resume**:`Runner.run`整個函式沒有任何一行呼叫`self.checkpointer.load_latest`或類似方法——`resume_from`和`state_override`都得由呼叫端自己組出來(`main()`裡是手動呼叫`ckpt.load_latest(session)`再組`approved_state`)。這代表把`checkpointer`傳給`Runner`,並不會讓resume「自動發生」,只是讓每一步的state有地方被記錄下來而已。
- **`classify`分岔後,三條路由值剛好跟目的node同名,但概念上是兩件事**:`add_conditional_edges`的`targets={"refund": "refund", ...}`,key是「router(state)預期回傳的值」,value是「要去的node名稱」——這裡两者字串剛好一樣容易讓人誤以為`add_conditional_edges`是直接拿router的回傳值當node名稱用,但其實兩者是分開查表比對的,只是這個demo刻意取了一樣的名字。
- **`_classify`原本有一段永遠不會分岔出不同結果的`elif`**(判斷`"pricing"`/`"quote"`關鍵字,結果也是設成`"sales"`),因為`else`分支本來就會落到同一個結果,已經在整理時收斂成單一`else`——語意完全不變(pricing/quote相關的輸入一樣會被分類成sales),只是少了一段讀起來像「有特別處理」、實際上什麼都沒多做的死分支。
- **`PausedAtNode`在整理時多帶了一個`reason`欄位**:原本`Runner.run`已經把`state.pop("_pause_reason")`的結果算進`reason`這個區域變數,卻只用來判斷要不要丟例外,從來沒有真的傳出去;現在`PausedAtNode(current, state, reason)`把這個值存成屬性,`main()`印暫停訊息時可以直接讀`paused.reason`,不用再回頭去翻state或checkpoint歷史找暫停原因。

## 使用時機 / 優缺點
- ✅ 想搞懂LangGraph的state graph怎麼具體對應到程式碼:`StateGraph`+`InMemoryCheckpointer`+`Runner`三個類別加起來不到150行,`main`印出的checkpoint歷史和暫停/恢復流程可以逐行對照「什麼時候state被存檔、暫停時state長什麼樣、恢復時從哪裡接著跑」。
- ✅ 需要一個不用接資料庫、可以100%重現的durable-execution骨架來驗證「resume是精確的」這件事:整支程式不呼叫任何外部服務或LLM,`checkpointer`是一個in-process dict,適合拿來寫unit test或當作接上真正資料庫checkpointer之前的原型。
- ✅ 示範conditional edge怎麼設計:`classify`是全圖唯一的分岔點,分岔完三條路徑立刻匯流回`human_gate`,對應教材點名的反面案例(edge幾乎都是conditional、圖難以理解)——這支程式刻意保持只有一處分岔,是「該怎麼做」的示範,不是「什麼都不做」的巧合。
- ❌ 不是production可用的checkpointer:`InMemoryCheckpointer`只是一個list,行程一結束歷史就沒了,也沒有任何壓縮/過期機制——checkpoint歷史會隨著node數量無限增長,真正的LangGraph checkpointer是SQLite/Postgres/Redis之類可持久化、可查詢、通常也有TTL或compaction機制的後端。
- ❌ 沒有parallel edge / 自訂reducer:`_next`一次只選一條邊往下走,`state = {**state, **update}`是最簡單的「新key覆蓋舊key」合併,沒有辦法表達「兩個node平行跑、結果要合併成一個list」這種需求(教材Exercise 3就是要練這個)——要支援平行分支,`Runner`跟reducer都得重新設計。
- ❌ 沒有實作subgraph/supervisor/swarm/hierarchical這些拓撲模式:教材另外討論了「supervisor路由到specialist subagent」「swarm去中心化交接」「hierarchical巢狀子圖」三種更複雜的多agent圖結構,本程式只示範最基本的單層線性圖加一處分岔,是理解狀態機核心迴圈最乾淨的起點,不是完整的LangGraph拓撲功能對照。
- ❌ 沒有streaming/partial output:`Runner.run`是整個node跑完才回傳一次update,沒辦法讓node中途吐出部分結果讓UI逐步更新(教材Exercise 5要求加這個)。
- ❌ Node假設是確定性的:demo裡的node都是純粹的字串處理,天生resume-safe;但如果node呼叫外部API、用亂數、或讀wall-clock時間,resume之後重跑「同一個」node時輸入不一定跟上次相同,checkpoint機制本身並不會幫你處理這種non-determinism,需要額外把這些來源存進state或做seed控制。

## 常見誤區
1. **以為把`checkpointer`傳給`Runner`,resume就會自動找到上次存的state**:`Runner.run`完全不會主動去讀`self.checkpointer`——它只在每個node跑完之後寫入。要resume,呼叫端必須自己呼叫`checkpointer.load_latest(session_id)`(或其他查詢方式)把state找回來,再透過`state_override`參數餵回去。`Runner`與`checkpointer`是「寫入端」跟「儲存端」的關係,不是「自動恢復」的關係。
2. **以為`paused.state`跟checkpoint歷史裡最後一筆的state是完全一樣的物件/內容**:兩者拍照時間點差了一個`state.pop("_pause_reason")`的原地修改——checkpoint裡那筆還帶著`_pause_reason`,`paused.state`已經清掉了。想確認「暫停原因」該讀`paused.reason`,不是去翻checkpoint歷史裡的dict。
3. **以為`add_conditional_edges`的`targets`裡,key是node名稱**:實際上key是「`router(state)`預期要回傳的值」,value才是node名稱——這支demo恰好把兩者取成一樣的字串(`"refund"`路由值對應`"refund"`這個node),容易造成「key就是目的地」的錯覺,換一組router值跟node名稱不同的例子(例如router回傳`"needs_refund"`、目的node叫`"refund"`)就能看出兩者是分開的。
4. **以為human_gate暫停時,這個node「還沒跑完」**:實際上`_human_gate`已經完整執行、回傳的update(含`_pause_reason`)已經合併進state並被checkpoint,只是`Runner.run`在合併之後才檢查要不要暫停——所以人工核准後應該接著跑`human_gate`*之後*的node,而不是重跑`human_gate`本身(雖然重跑`human_gate`因為`human_approval`已經是`True`也不會錯,只是`step`會被多加一次)。
5. **以為這支程式示範的是完整的LangGraph多agent拓撲**:supervisor、swarm、hierarchical subgraph都是教材裡提到、但本程式沒有實作的模式——這裡刻意只做最基礎的「單層圖+一處conditional分岔+checkpoint+暫停/恢復」,把durable execution和human-in-the-loop這兩個核心機制講乾淨,拓撲複雜度留給讀者自己延伸。

## 複習自問
1. 教材Exercise 1要你在`classify`node加一條conditional edge:當分類信心低於某個門檻時直接路由到`END`,並手動設定`route`後resume。如果要在這支程式上做,`_classify`要多回傳什麼欄位?`build_graph`裡的`add_conditional_edges`要怎麼多加一個`"END": END`的分支?
2. 教材Exercise 2要你把mock checkpointer換成真正的SQLite後端,並量測每一步序列化的成本。如果要在`InMemoryCheckpointer`的介面不變的前提下換掉底層儲存,`save`/`load_latest`/`history`三個方法的行為要怎麼對應到SQL的INSERT/SELECT?`deepcopy`這一步在SQLite版本裡還需要嗎?
3. 教材Exercise 3要你實作平行邊:兩個node同時跑,結果透過自訂reducer合併。目前`Runner.run`一次只往下走一個`current`,`_next`也只回傳單一目的地——要支援兩個node平行執行,`Runner.run`的while迴圈架構要怎麼改?`state = {**state, **update}`這個reducer在多個平行update同時要合併時會發生什麼問題(例如兩個node都回傳同一個key的不同值)?
4. 教材Exercise 4要你參考`langgraph-supervisor`,把這支程式的demo改寫成supervisor模式:一個中央router node決定要呼叫哪個specialist(而不是像現在這樣用固定的conditional edge分岔到refund/bug/sales)。這樣改之後,`classify`node跟現在的`add_conditional_edges`用法還適用嗎?router的決策現在要放進哪個node裡?
5. 教材Exercise 5要你加上streaming,讓每個node跑到一半就能吐出部分state更新給UI。以現在`NodeFn`的型別(`Callable[[State], Update]`,一次呼叫回傳一個完整的update dict)來看,要支援「node中途吐出好幾次部分update」,`NodeFn`的型別本身要怎麼改(例如改成generator/yield update)?`Runner.run`要在什麼時候把這些中途的片段也存進checkpoint?
