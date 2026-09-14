# Agent Memory — Virtual Context and Memory Paging (MemGPT)

對應程式: [`./agent_memory.py`](./agent_memory.py)

參考:[ai-engineering-from-scratch – Agent Memory: Virtual Context and Memory Paging 課程](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/14-agent-engineering/07-memory-virtual-context-memgpt/docs/en.md)

前置概念:Phase 14 的Agent Loop(ReAct風格的agent控制迴圈)與Tool Use(工具呼叫的基本格式)——這支程式沒有重新實作那兩課,而是假設agent已經會呼叫工具,只專注在「工具呼叫的對象是記憶體本身」這件事。

## TL;DR
Context window看起來像是能解決記憶問題,實際上不行——就算是128k的視窗,對話輪數一多、文件一長,一樣會遇到三種失效模式:**溢出**(內容塞不下)、**稀釋**(視窗塞滿不相關內容,模型表現反而下降)、**無法延續**(新session從空白開始,記不得上一次講過什麼)。MemGPT(Packer et al., 2023, arXiv:2310.08560)的解法是把作業系統的虛擬記憶體概念直接搬過來:main context是RAM——固定大小、永遠對模型可見;archival memory是disk——容量無上限,但要透過工具「查詢」才會被讀進來。呼叫一次記憶體工具,概念上就是一次page fault:agent「阻塞」、runtime把讀寫結果算出來、結果以observation的形式塞回下一輪對話,這跟Unix的`read()` syscall是同一個形狀。本程式用純Python(無外部依賴、不呼叫任何LLM)刻出這兩層儲存加五個canonical memory tool,並用一個寫死腳本的agent(`run_scripted_agent`)跑一遍完整流程:填滿main context觸發eviction,再用`archival_memory_search`和`conversation_search`把被擠出去的事實查回來——整個paging行為因此是決定性、可離線測試的,不需要接真正的LLM才能驗證正確性。

## 為什麼需要它
- **視窗越大,不代表長時記憶問題就解決了**:Mem0的2025年論文(arXiv:2504.19413)量到「128k視窗的baseline,仍然會漏掉一個外接記憶體的4k視窗agent能接住的長時事實」——問題不是視窗不夠大,是「相關的東西剛好在視窗外」跟「視窗裡東西太多、訊號被稀釋」這兩件事,單靠加大視窗解決不了。
- **溢出、稀釋、無法延續,是三種不同的失效模式,需要不同的對策**:溢出要靠固定大小的main context加eviction(`MainContext.append`裡「超過`max_messages`就把最舊的一則擠進`evicted`」);稀釋要靠只在相關的時候才把資料page進main context,而不是把一切都攤開在prompt裡(`archival_memory_search`只在被呼叫時才回傳資料);無法延續要靠把事實寫進session之外還能存取的地方(`ArchivalStore`不會因為main context被清空而消失)。
- **MemGPT不是唯一實作,但是這個設計譜系的起點**:2024年9月MemGPT團隊把研究原型商用化,改名Letta,把兩層擴成三層(core/recall/archival)、拿掉heartbeat改用native reasoning、加上sleep-time agent做異步記憶整理。教材的說法是「就算production系統跑的是Letta、Mem0或自己刻的兩層store,MemGPT論文仍然是2026年的設計基礎」——這也是本程式選擇實作原始兩層版本、而不是三層版本的原因:兩層版本是理解「main context vs. 外部store」這個核心分野最乾淨的起點。

## 核心原理
- **OS類比,一一對應**:

  | 作業系統 | MemGPT / 本程式 |
  |---|---|
  | RAM | main context(`MainContext`)——固定大小,永遠對模型可見 |
  | Disk | archival memory(`ArchivalStore`)——容量無上限,要查詢才讀得到 |
  | Page fault | 一次memory tool呼叫(`MemoryTools`的任一方法) |
  | OS kernel | agent的控制迴圈(本程式用`run_scripted_agent`模擬,真實系統是ReAct風格的LLM迴圈) |

- **兩層儲存,各自的讀寫規則不一樣**:main context分兩塊——`core`是長期存在的dict(persona、使用者資料這類「隨時都該看得到」的事實),`messages`是有上限的捲動列表,寫滿了就把最舊的一則擠進`evicted`(page out,不是丟棄);archival memory則是`insert`時才寫入、`search`時才讀出,平常完全不出現在prompt裡。

- **五個canonical memory tool**:

  | 工具 | 做什麼 | 對應到程式 |
  |---|---|---|
  | `core_memory_append` | 在某個core section後面接文字 | 寫RAM層 |
  | `core_memory_replace` | 用精確字串取代core section裡的一段 | 改RAM層 |
  | `archival_memory_insert` | 寫一筆事實進外部store | 寫disk層 |
  | `archival_memory_search` | 依查詢字串取回最相關的幾筆事實 | 讀disk層 |
  | `conversation_search` | 在（包含已被evict的）對話歷史裡做子字串查詢 | 讀RAM層被page out的部分,不需要另外寫進archival |

- **Interrupt-driven pattern**:agent在對話中途呼叫一個memory tool,runtime執行它,結果當作一則新的observation接回下一輪——跟Unix的`read()`概念上是同一件事:呼叫方阻塞、核心把bytes準備好、呼叫方帶著結果繼續跑。本程式的`run_scripted_agent`把「agent決定呼叫哪個工具」這一步寫死成一串`ToolCall`,藏起了真正agent loop裡「LLM自己判斷現在該不該查記憶體」的推理,只留下「工具呼叫→observation」這條資料流本身。

- **四種記憶類型,本程式蓋到了幾種**:

  | 類型 | 回答什麼問題 | 本程式的對應 |
  |---|---|---|
  | Working memory | 現在當下重要的是什麼? | `MainContext.core` + `messages`(in-context層) |
  | Episodic memory | 發生過什麼事? | `MainContext.messages`/`evicted`,以及帶`session_id`/`turn_id`的`ArchivalRecord` |
  | Semantic memory | 什麼是真的? | `MainContext.core`裡的使用者/世界事實,以及寫進archival的事實 |
  | Procedural memory | 該怎麼做這件事? | ❌ 沒有實作——沒有「學到的routine/偏好規則反過來改變agent行為」這一層 |

- **Production環境的三種失效模式,本程式只處理了一種**:**memory rot**(寫入速度超過讀取/整理速度,retrieval被過時事實淹沒——本程式沒有consolidation或invalidation機制,`ArchivalStore`只會一直長大);**memory poisoning**(被攻擊者操控的內容被存成記憶、之後又被retrieve回來影響agent——本程式沒有任何檢查`insert`內容是否像指令的guard);**citation loss**(recall到事實但講不出出處——這個本程式有處理:`ArchivalRecord`固定帶`session_id`/`turn_id`,對應教材「every archival write都存來源」的建議)。

## 程式碼導覽
| 函式/類別 | 對應到理論的哪個部分 |
|---|---|
| `Message` | 一則對話turn,是`MainContext.messages`/`evicted`裡的元素 |
| `MainContext` | RAM層:`core`(持久dict)+ `messages`(有上限的捲動列表) |
| `MainContext.append` | 寫入新turn;超過`max_messages`就把最舊一則page out進`evicted`,不是直接丟棄 |
| `MainContext.render` | 把`core`+`messages`組成模型會看到的prompt文字——`evicted`故意不算進去,因為它已經不在visible context裡 |
| `ArchivalRecord` | disk層的一筆記錄,`session_id`/`turn_id`是citation loss問題的解法 |
| `ArchivalStore` | disk層:`insert`/`search`/`count`,底層撐起`archival_memory_insert`/`archival_memory_search`兩個工具 |
| `ArchivalStore.search` | Jaccard token overlap評分排序——教材裡真正BM25/embedding retrieval的簡化替身,足以展示paging行為,不是可用的排序演算法 |
| `MemoryTools` | 五個canonical memory tool的實作,方法名稱1:1對應MemGPT論文定義的tool surface,方便`run_scripted_agent`直接用`getattr`dispatch |
| `MemoryTools.core_memory_append` / `core_memory_replace` | 寫/改RAM層的core section |
| `MemoryTools.archival_memory_insert` / `archival_memory_search` | 寫/讀disk層 |
| `MemoryTools.conversation_search` | 對`evicted + messages`做子字串查詢(依時間新到舊),讓剛被page out的內容不用額外寫進archival也查得回來 |
| `ToolCall` / `run_scripted_agent` | 模擬agent的控制迴圈:一串寫死的`ToolCall`依序dispatch到`MemoryTools`的方法上,取代真實系統裡「LLM自己決定現在要不要呼叫記憶體工具」的推理步驟 |
| `main` | Demo劇本:填滿main context觸發eviction,再分別用`archival_memory_search`跟`conversation_search`把被evict的事實page回來,對照兩種retrieval路徑 |

**實作細節 / 容易看漏的地方:**
- `MainContext.append`的eviction是`while len(self.messages) > self.max_messages`而不是`if`,這樣不管一次append幾則、或`max_messages`被動態調小,都能一次修正到符合上限,不會漏掉多出來的訊息。
- `conversation_search`查的是`reversed(self.main.evicted + self.main.messages)`——先把兩個list接起來再整體反轉,效果是「先查目前還在main context裡的訊息(從新到舊),再查已經被evict的訊息(從新到舊)」,符合「越新的東西越該優先被找到」的直覺,不是先查evicted再查messages。
- `ArchivalStore.search`用集合交集/聯集算Jaccard相似度,`overlap == 0`的記錄直接跳過、不會被排進結果——這代表查詢字串跟archival裡的文字如果一個共同詞都沒有(例如換句話說或用了同義詞),`search`會完全找不到,是這個簡化實作故意留下的限制。
- `core_memory_append`用空白字元接資料,如果某個section原本是空字串,直接賦值不會多一個開頭空白——`(existing + " " + text).strip() if existing else text`這行的`if existing`分支就是在處理這個邊界。
- `run_scripted_agent`把單一tool呼叫失敗(未知工具名稱、或方法丟例外)都轉成字串observation,不會讓整個腳本中斷——這對應真實agent loop的行為:工具呼叫失敗是一種observation,要讓agent(或這裡的腳本)看到錯誤訊息、決定接下來怎麼辦,而不是讓整個process crash。

## 使用時機 / 優缺點
- ✅ 想搞懂MemGPT的OS類比怎麼具體對應到程式碼:兩層儲存加五個工具,配上`run_scripted_agent`跑出完整的interrupt-driven paging流程,`main`印出的tool trace可以逐行對照「什麼時候寫進core、什麼時候寫進archival、eviction什麼時候觸發、page in怎麼發生」。
- ✅ 需要一個不用接LLM、可以100%重現的paging骨架來驗證「eviction之後東西還查得回來」這件事:`run_scripted_agent`的腳本是寫死的`ToolCall`列表,不呼叫任何LLM,所以整個流程是決定性的,適合拿來寫unit test或當作更複雜記憶系統的起點。
- ❌ 不要當作production可用的retrieval:`ArchivalStore.search`只是token overlap,不是BM25、更不是embedding搜尋,換一種說法或用同義詞問同一件事,大概率完全查不到。
- ❌ 沒有memory rot/memory poisoning的防護:archival一旦`insert`就永久留著,`search`不做任何consolidation、去重或新舊事實衝突處理;也沒有任何guard去檢查被insert或被retrieve的內容是不是「看起來像指令」的可疑內容——教材點名的這兩個production failure mode,本程式完全沒實作,是刻意留給讀者的延伸練習(見下方複習自問)。
- ❌ 只有working/episodic/semantic memory的雛形,procedural memory沒有碰:四種記憶類型裡,「agent從過去互動學到的routine或偏好規則,反過來改變之後的行為」這一層完全沒有實作。
- ❌ 只示範MemGPT原始的兩層設計,不是Letta的三層:沒有獨立的recall tier,也沒有native reasoning或sleep-time compute——這些是MemGPT商用化之後(Letta)才加上的擴充,教材本身也把它們列為「後續演進」而不是這一課的範圍。

## 常見誤區
1. **以為`archival_memory_search`是語意搜尋**:它只是把查詢字串跟每筆記錄的文字各自轉成小寫token集合,算Jaccard相似度——沒有embedding、沒有同義詞理解,查詢用詞跟archival裡的原文差太多就找不到,這是教材裡「BM25/embedding retrieval」的簡化替身,不是真正的語意檢索。
2. **以為`evicted`裡的訊息已經遺失**:`MainContext.append`把超過上限的訊息「擠出去」而不是刪除,`evicted`列表完整保留這些訊息,`conversation_search`照樣查得到——這正是MemGPT「page out不等於丟棄」的核心設計,main context只是暫時看不到而已。
3. **以為MemGPT的兩層架構跟Letta的三層是同一件事**:本程式實作的是2023年原始MemGPT論文的two-tier(main context + archival)設計;Letta(2024年MemGPT團隊的商用化重寫)把它擴成三層(core/recall/archival),而且拿掉了heartbeat機制改用native reasoning——兩者是同一個設計譜系的不同世代,不能直接互換理解。
4. **以為archival裡的資料會自動整理、去重**:`ArchivalStore`就是一個list,重複`insert`同一件事實會產生兩筆獨立的記錄,`search`兩筆都可能回傳——這正是教材說的memory rot問題(寫入速度超過讀取/整理速度),本程式沒有任何consolidation或invalidation邏輯去處理它。
5. **以為記憶體工具是agent自己「想到」才會呼叫的**:本程式裡工具呼叫的順序是`main`函式裡寫死的`ToolCall`腳本,`run_scripted_agent`只是照順序執行、不做任何決策——真實系統裡「現在該不該查archival、該查什麼」是LLM在每一輪推理裡自己判斷的(屬於Agent Loop/Tool Use那兩課的範圍),這支程式刻意把這一步拿掉,只留下記憶體讀寫本身的機制。

## 複習自問
1. 教材Exercise 1要你幫main context加上token-based的容量上限(用`len(text.split()) * 1.3`近似token數),取代目前單純數「訊息則數」的`max_messages`,而且被擠出去的舊訊息要先壓縮成一段摘要塞進`core`,而不是整條原文留在`evicted`。如果要在這支程式上做,你會怎麼改`MainContext.append`?壓縮摘要這一步該由誰來做——`MainContext`自己,還是外部再包一層?
2. 教材Exercise 2要求把`ArchivalStore.search`換成真正的BM25(TF-IDF變體)。如果你另外準備一組「查詢用同義詞或換句話說」的test case,你預期目前Jaccard token overlap版本的recall會多差?哪一種query pattern最容易讓現在的`search`完全找不到任何結果?
3. 教材Exercise 3要你在`archival_memory_search`的輸出裡附上引用(session_id、turn_id甚至source_url)。`ArchivalRecord`已經有`session_id`/`turn_id`兩個欄位,但`MemoryTools.archival_memory_search`目前的輸出格式只印`rid`跟`text`——要怎麼改這個方法,讓輸出的每一行都能讓使用者追溯回「這句話是哪個session、第幾輪講的」?
4. 教材Exercise 4要你模擬一次memory poisoning:插入一筆看起來像系統指令的假事實(例如"ignore previous instructions and...")到archival,再確認它會不會被正常的`archival_memory_search`撈出來、進而影響下一輪對話。如果要在`MemoryTools`裡加一道guard,你會把檢查邏輯放在`archival_memory_insert`(寫入時擋)還是`archival_memory_search`(讀出時擋)?兩種做法各自的取捨是什麼?
5. 教材Exercise 5要求對照MemGPT研究版repo的core-memory JSON schema。如果把`MainContext.core`從目前的「每個section是一個字串」換成有結構的typed schema(例如每個section是一個巢狀dict,而不是純文字),`core_memory_append`/`core_memory_replace`兩個方法要怎麼跟著改?這樣的改動會不會讓`render()`的輸出格式也要跟著變?
