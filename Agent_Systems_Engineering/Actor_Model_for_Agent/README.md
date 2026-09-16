# Actor Model for Agents — 非同步訊息、Typed Runtime 與故障隔離（AutoGen v0.4）

對應程式: [`./actor_model_for_agent.py`](./actor_model_for_agent.py)

參考: [ai-engineering-from-scratch – The Actor Model for Agents: Async Messages and Typed Runtimes 課程](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/14-agent-engineering/14-autogen-actor-model/docs/en.md)

前置概念: 課程列的前置課是Phase14·01(Agent Loop,ReAct風格的思考-行動-觀察控制迴圈)和Phase14·12(Workflow Patterns)——但本repo目前沒有幫Agent Loop獨立開一課,其他好幾課([`../Agent_Memory`](../Agent_Memory/README.md)、[`../Stateful_Graph_Orchestration`](../Stateful_Graph_Orchestration/README.md)、[`../Tree_Of_Thoughts_Lats`](../Tree_Of_Thoughts_Lats/README.md)、[`../Anthropic_Workflow_Patterns`](../Anthropic_Workflow_Patterns/README.md))都把它列為「假設你已經懂」的背景知識,這裡也一樣,重點放在多個agent確定要存在之後,**它們之間該怎麼通訊**。跟[`../Anthropic_Workflow_Patterns`](../Anthropic_Workflow_Patterns/README.md)是不同維度的問題:那一課問的是「要不要用agent、還是workflow就夠」;這一課假設答案已經是「要有多個agent」,問的是「這些agent之間該用同步呼叫堆疊,還是非同步訊息」。跟[`../Orchestration_Patterns`](../Orchestration_Patterns/README.md)(Supervisor/Swarm/Hierarchical/Debate四種拓撲)也是不同維度:那一課關心的是「訊息該怎麼流動、誰指揮誰」的拓撲形狀,這一課關心的是拓撲底下**傳輸層**該怎麼設計(同步阻塞 vs. 解耦的send/receive),兩者可以互相搭配——任何一種拓撲都可以疊在actor model的傳輸層上面。

## TL;DR
大部分agent框架的預設寫法是同步呼叫堆疊:一個agent產生輸出、另一個agent消費,全部疊在同一個call stack上——一個環節壞掉整條堆疊就崩潰,並行處理要另外硬加,要分散到多台機器幾乎得重寫。教材引用的AutoGen v0.4(Microsoft Research, 2025年1月)用**actor model**回應這個問題:每個agent是一個actor,有自己的私有state、一個收訊息的inbox,只能靠傳訊息互動,不能直接共享記憶體;runtime把「送出訊息」跟「處理訊息」拆成兩個步驟——`send`只是把訊息放進佇列就立刻回傳,真正呼叫handler是runtime之後才做的事。這個拆分換來三個好處:**故障隔離**(一個actor的handler壞掉,runtime接住例外,不會拖垮其他actor)、**天然併發**(可以有多個訊息同時在飛)、**可分散**(inbox+transport是同一套抽象,不管actor是在同一個行程還是在另一台機器上)。本程式用Python stdlib刻一個最小actor runtime(`Message`/`Actor`/`Runtime`),再用兩個demo actor(`ReviewerAgent`審查程式碼、`ChecklistAgent`發起審查並收斂共識)示範:即使故意讓`ReviewerAgent`收到一則訊息就崩潰,另外三則審查訊息依然被正常處理完、`ChecklistAgent`依然能算出最終共識。

## 為什麼需要它
- **對抗「同步呼叫堆疊」這個預設寫法帶來的失敗模式**:多數團隊寫多agent互動時,直覺就是`agent_a.chat(agent_b)`這種同步阻塞呼叫——agent_a要等agent_b回傳才能繼續。這支程式示範的是最小可行的替代寫法:把「送出」跟「處理」拆成兩個函式(`Runtime.send`只入列,`Runtime.run_until_idle`才真正呼叫`receive`),看清楚這個拆分本身,才看得懂AutoGen v0.4從v0.2改版的核心動機。
- **故障隔離不是到處包try/except這麼簡單,是runtime本身要負責接住例外**:`run_until_idle`裡的`try/except`包住的是*每一則訊息的處理*,而不是呼叫端事先預判「這裡可能會爆」才特別小心——`ReviewerAgent`收到`crash_me`時完全不知道自己會被特殊對待,它就是單純`raise`,是runtime的迴圈負責接住、記錄成dead letter、然後處理下一則訊息。這是「一個actor失敗變成一筆有紀錄的訊息」而不是「整個程式當掉」的關鍵設計。
- **這是2026年多agent框架底層真正在做的事,而且正在換代**:AutoGen v0.4把這套actor model做成三層API(Core/AgentChat/Extensions),但教材也點名它已進入維護模式,微軟正式的生產後繼者是Microsoft Agent Framework(2025年10月公開預覽)。先用stdlib刻一次最小版本的actor+typed message+runtime,才看得出框架幫你做了什麼(真正的asyncio事件迴圈、跨行程transport、OpenTelemetry span)、又是用什麼代價換來的(多一層抽象、除錯要多繞一層)。

## 核心原理
- **Actor跟一般物件的差別,在於「只能透過訊息互動」這條界線**:一個actor有私有state(外部不能直接碰)、一個inbox、一個handler `receive(message, runtime) -> effects`,效果可以是「回覆」「送給另一個actor」「生出新actor」「更新state」「自我終止」。兩個actor之間**不共享記憶體**,`ReviewerAgent`和`ChecklistAgent`彼此完全不知道對方的內部屬性長怎樣,唯一的接觸點是`Message`。

- **解耦delivery跟handling,是fault isolation和天然併發的共同來源**:`Runtime.send`只做兩件事——把`Message`塞進共用的`deque`、把這次送出寫進`trace`——完全不會去呼叫`recipient`的`receive`。真正的投遞發生在`Runtime.run_until_idle`的while迴圈裡,一次從佇列取一則訊息、找到actor、呼叫`receive`、用`try/except`包住。這個時間差(送出的當下不代表立刻被處理)正是三個好處的根源。

- **本程式對應課程概念的落點**:

  | 課程概念 | 本程式對應 | 說明 |
  |---|---|---|
  | Actor | `Actor`抽象基底 | 只規定要有`receive(message, runtime)`,state放在子類別自己的屬性上 |
  | Message | `Message` dataclass | `sender`/`recipient`/`topic`/`body`/`mid`,actor之間唯一的互動方式 |
  | Inbox | `Runtime.queue` | 見下方「容易看漏的地方」——本程式簡化成全域共用一條佇列,不是per-actor各自一份 |
  | Runtime | `Runtime` | `register`登記actor、`send`入列、`run_until_idle`是真正的event loop |
  | Fault isolation | `run_until_idle`裡的`try/except` | handler壞掉被記進`dead_letters`,迴圈繼續處理下一則訊息 |
  | Topic | `Message.topic` | 見下方——本程式只拿來做`receive`裡的if/elif字串分派,不是真正的pub/sub broadcast |

- **`ReviewerAgent`跟`ChecklistAgent`示範的是「單向審查流程」,不是orchestrator-workers**:`ChecklistAgent`收到`"start"`就一次性對`partner`發出3則`"review"`訊息(fan-out),之後每收到一則`"review_result"`就往`self.results`塞一筆,直到湊滿3筆才算出最終`consensus`(fan-in)。這個fan-out/fan-in的形狀是actor之間協作的一種具體用法,跟[`../Anthropic_Workflow_Patterns`](../Anthropic_Workflow_Patterns/README.md)裡的`orchestrator_workers`(一次性動態決定要跑哪些worker、再合成)是不同機制——這裡沒有「動態決定要不要送」的邏輯,`ChecklistAgent`固定就是把收到的snippet全部轉送給同一個`partner`。

**實作細節 / 容易看漏的地方:**
- **Inbox其實是runtime全域共用的一條佇列,不是courseware講的「每個actor各自的inbox」**:`Runtime.queue`是單一`deque`,`run_until_idle`依序FIFO取出、逐一處理,不管訊息的`recipient`是誰。真正的per-actor inbox(每個actor有自己的佇列、可以各自用不同速度清空)在這支stdlib demo裡沒有實作,是刻意的簡化,不要跟正式的AutoGen Core搞混。
- **`Message.topic`不是真正的pub/sub broadcast channel**:這裡的`send`永遠指定單一`recipient`,`topic`只是給`receive`拿來做`if message.topic == "review"`這種字串分派用的標籤。AutoGen Core正式定義的topic是可以有多個訂閱者的廣播路由,一則訊息發到一個topic可能被好幾個actor同時收到——本程式沒有實作這個廣播語意。
- **`run_until_idle`是單執行緒依序處理,沒有真正的並行**:雖然「解耦delivery跟handling」在架構上*允許*多個訊息同時被不同actor處理,但這支demo的實作就是一個單純的`while`迴圈,一次處理一則訊息、處理完才輪到下一則。想看到真正的並行,需要換成asyncio事件迴圈(教材原文Core層用的就是這個),這支程式只示範了解耦本身,沒有兌現成真正的併發。
- **`ChecklistAgent.consensus`中途讀到的值不一定是最終值**:`receive`裡第一個`if all(r["ok"] for r in self.results)`只要目前收到的結果全部`ok`就會把`consensus`設成`True`,即使還有訊息沒回來;要等到`len(self.results) == 3`那個分支才會用完整的3筆結果覆寫成真正的最終值。因為`main()`裡是等`run_until_idle()`跑到idle才去讀`checklist.consensus`,這個「中途值」不會被外部觀察到,但如果你把讀取時機提前(例如在某個訊息處理完的當下就去讀),看到的可能是還沒收齊的暫時值。
- **`len(self.results) == 3`是寫死的demo批次大小,不是通用機制**:如果`main()`改成送4個snippet去審查,`ChecklistAgent`不會自動偵測「全部到齊」,`consensus`最終只會停在收到第3筆時被覆寫的值,第4筆回來後不會再更新。這不是bug,是這支demo刻意保持最小,換批次大小需要自己同步改這個數字或改成用預期總數的參數。
- **整理時移除的未使用import**:原始程式的`from typing import Any, Callable`裡`Callable`整支程式沒有任何一處用到,整理時已移除,行為完全不變(`Any`仍在用,保留)。

## 使用時機 / 優缺點
- ✅ 想直接看到「同步呼叫 vs. 解耦的send/receive」在程式碼層級的差異:`Runtime.send`只有4行、只做入列,對照`run_until_idle`才是真正呼叫`receive`的地方——兩者拆開讀,就能理解AutoGen從v0.2的`agent_a.chat(agent_b)`同步阻塞改成v0.4`send()`立刻回傳的核心動機。
- ✅ 想要一個不依賴asyncio、可以單步偵錯的故障隔離demo:`crash_me`故意讓`ReviewerAgent`在處理第2則訊息時整個handler壞掉,但另外3則`review`訊息依然全部被正常處理、`ChecklistAgent`依然算出共識——trace會逐行印出`[FAIL m002] reviewer raised RuntimeError: ... (others keep running)`,故障隔離不是文字說明,是看得到的執行結果。
- ✅ 想理解dead-letter queue的最小實作長怎樣:「送不到(收件人不存在)」和「handler拋例外」這兩種完全不同的失敗,在這支程式裡被同一個`dead_letters`列表接住,各自附帶一句原因說明。
- ❌ 沒有真正的並行:`run_until_idle`是單一while迴圈依序處理佇列,不是asyncio事件迴圈——這只示範了解耦delivery帶來的*架構可能性*,沒有真的把它兌現成多訊息同時處理。
- ❌ Inbox是全域共享的一條佇列,不是per-actor各自一份(見上方實作細節)——沒辦法示範「某個actor的inbox積壓、另一個actor的inbox很快清空」這種真實場景會出現的差異。
- ❌ Topic不是真正的pub/sub廣播——想練「一則訊息同時被多個訂閱者收到」的形狀,這支程式沒有現成範例。
- ❌ 沒有`RoundRobinGroupChat`/`SelectorGroupChat`這類多輪對話拓撲,也沒有跨行程transport、沒有接OpenTelemetry span——這些都留在教材的Use It清單和練習題裡,不在這支程式的範圍內。
- ❌ `ChecklistAgent`的共識收斂用寫死的`len == 3`,不是通用的「等到預期數量到齊」機制,拿來當函式庫用之前記得留意這個邊界。

## 常見誤區
1. **以為這篇是在講「要不要用agent」**:那是[`../Anthropic_Workflow_Patterns`](../Anthropic_Workflow_Patterns/README.md)的主題。這一課假設你已經確定要有多個agent,問題只在於「這些agent之間該怎麼通訊」,跟workflow vs. agent的取捨是完全不同層次的決定。
2. **以為`Runtime.send`會立刻執行對方的`receive`**:`send`只是把訊息塞進共用佇列就回傳,真正呼叫`receive`是在`run_until_idle`的迴圈裡才發生——兩者故意拆開,是fault isolation和天然併發潛力的來源,不是多餘的間接層。
3. **以為fault isolation代表訊息會自動重試**:`dead_letters`只是「記錄下來、不讓崩潰的效應蔓延到其他actor」,並沒有內建重送機制。真要重試,得自己在`run_until_idle`或呼叫端加邏輯去檢查`dead_letters`並重新`send`一次。
4. **以為這裡的inbox是每個actor各自一份**:本程式的inbox其實是`Runtime.queue`這一條全域共享佇列(見上方實作細節),不是courseware提到的per-actor mailbox,這是stdlib demo刻意的簡化。
5. **以為orchestrator-workers跟這裡的fan-out/fan-in是同一件事**:`ChecklistAgent`固定把每個snippet轉送給同一個`partner`,沒有「動態判斷要不要送給某個worker」的邏輯;[`../Anthropic_Workflow_Patterns`](../Anthropic_Workflow_Patterns/README.md)裡的`orchestrator_workers`才有這種一次性動態dispatch。

## 關鍵詞速查
| 詞彙 | 常聽到的說法 | 實際上是什麼 |
|---|---|---|
| Actor | 「Agent」 | 私有state + inbox + handler,不共享記憶體,只能靠訊息互動 |
| Message | 「Event」 | Typed payload,actor之間唯一的互動方式 |
| Inbox | 「Mailbox」 | 每個actor自己待處理訊息的佇列(本程式簡化成全域共用一條) |
| Runtime | 「Agent host」 | 負責路由訊息、隔離故障的event loop |
| Topic | 「Channel」 | 具名的pub/sub路由(本程式只拿來做字串分派,不是真正broadcast) |
| Fault isolation | 「Let it crash」 | 一個actor失敗不會拖垮其他actor |
| Dead-letter queue | 「錯誤紀錄」 | 送不到或handler壞掉的訊息集中存放,不會自動重試 |
| RoundRobinGroupChat | 「固定輪流的團隊」 | Agent依序輪流發言(本程式未實作) |
| SelectorGroupChat | 「依情境路由的團隊」 | 由selector agent依對話脈絡決定下一個發言者(本程式未實作) |

## 延伸閱讀
- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/):actor model重新設計的原始出處。
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/):AutoGen預設會發送的span遵循的規範,對應教材Exercise 4。
- [LangGraph Overview](https://docs.langchain.com/oss/python/langgraph/overview):用圖結構而非actor+訊息實作多agent協作的對照組。
- [`../Anthropic_Workflow_Patterns`](../Anthropic_Workflow_Patterns/README.md):workflow vs. agent的取捨,決定「要不要有多個agent」的上游問題。
- [`../Orchestration_Patterns`](../Orchestration_Patterns/README.md):多agent之間的協作拓撲(Supervisor/Swarm/Hierarchical/Debate),跟這一課的傳輸層問題是互相垂直的兩個維度。

## 複習自問
1. 教材Exercise 1要你加一個dead-letter重試機制。目前`dead_letters`只是一個`list[tuple[Message, str]]`,單純累積失敗紀錄——如果要支援「人工看過後決定要不要重送」,`Runtime`要新增什麼方法,才能把某筆dead letter重新塞回`queue`,而不用重新建構整個`Message`?
2. 教材Exercise 2要你實作`SelectorGroupChat`:一個selector actor依對話狀態決定下一個處理者。目前`ReviewerAgent`跟`ChecklistAgent`是互相寫死`partner`名稱——如果要讓「下一個該送給誰」變成動態決定,這個決策應該放在一個新的selector actor裡、還是該讓`Runtime.send`本身多一個「先問selector」的選項?兩種做法對「actor只能透過訊息互動」這條界線的破壞程度有什麼不同?
3. 教材Exercise 3要你把單一行程內的佇列換成跨行程的JSON-over-HTTP transport。`Runtime.send`目前直接對記憶體裡的`deque`做`append`——如果要換成網路傳輸,`send`和`run_until_idle`這兩個方法的介面要怎麼拆,才能讓`ReviewerAgent`、`ChecklistAgent`的程式碼完全不用改一行?
4. 教材Exercise 4要你替每則訊息接上OTel span,帶`gen_ai.agent.name`、`gen_ai.operation.name`屬性。目前`Runtime.send`和`run_until_idle`只把紀錄寫進`self.trace`這個字串list——如果要加span,是該包在這兩個方法內部,還是應該讓`trace`的每一筆紀錄都改成結構化物件、再由呼叫端決定要不要轉成span?
5. 教材Exercise 5要你讀AutoGen v0.4真正的`autogen_core` API、把玩具版本移植過去。這支程式跳過了哪些正式環境會咬人的細節?(提示:全域共用一條queue vs. per-actor mailbox、沒有asyncio事件迴圈、`topic`不是真正的broadcast channel)逐一對照,你覺得哪一個最先會在真實負載下出問題?
