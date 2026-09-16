# Memory Blocks and Sleep-Time Compute — 具名記憶區塊與離峰時段整理

對應程式: [`./memory_blocks_sleep_time_compute.py`](./memory_blocks_sleep_time_compute.py)

參考: [ai-engineering-from-scratch – Memory Blocks and Sleep-Time Compute 課程](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/14-agent-engineering/08-memory-blocks-sleep-time-compute/docs/en.md)

前置概念: 課程列的前置課是Phase14·07(MemGPT,本repo對應[`../Agent_Memory`](../Agent_Memory/README.md))——main context是RAM、archival是disk、呼叫記憶體工具是一次page fault的那套OS類比。這支程式假設你已經懂兩層儲存本身,重點放在MemGPT論文之後、Letta商用化過程裡浮現的兩個production問題:**(1) main context是一整塊沒有結構的東西**——沒辦法表達「human block永遠在prompt裡、persona block永遠在prompt裡、task block換個session就整個換掉」這種差異,這支程式用具名的`Block`(human/persona/task)取代單一大塊context;**(2) 記憶整理本身佔用回應延遲**——如果摘要、去重、判斷矛盾都得在使用者等待的那一輪做完,tail latency會被拖垮,這支程式加入一個完全跑在使用者輪次之外的`SleepTimeAgent`。跟[`../Reflexion_Verbal_RL`](../Reflexion_Verbal_RL/README.md)提到的「把Self-Reflector搬到離峰時段執行」是同一個「非同步、離critical path」的設計動機,只是套用在不同的任務類型上——那支處理的是「反思一次失敗」,這支處理的是「整理一份持續累積的記憶」。

## TL;DR
MemGPT解決了「main context塞不下、要靠工具讀寫外部記憶」的控制流問題,但商用化過程中浮現三個新問題:**latency**(記憶整理佔用回應延遲)、**memory rot**(矛盾或過時的事實留在記憶裡,retrieval被淹沒)、**structure loss**(扁平的archival沒辦法表達「這塊永遠可見、那塊換session就變」)。Letta(letta.com,MemGPT團隊2024年商用化改名——2026年的Letta V1原生推理重寫是後續、獨立的一步)的解法是把main context拆成具名的**memory block**(`label`/`value`/`limit`/`description`四個欄位,透過`block_append`/`block_replace`/`block_summarize`工具編輯),外加一個**sleep-time agent**:跑在背景、不佔用主線回應延遲,理論上可以用比主線更強、更慢的模型,專門做摘要跟矛盾判定。本程式用純Python刻出`Block`+`BlockStore`(具名區塊)、`Archival`(扁平事實庫)、`PrimaryAgent`(處理輪次,只做原始寫入)、`SleepTimeAgent`(離峰時段整理),外加一個課程原文沒有寫死程式碼、但明確點名為production失效模式的機制:**版本追蹤(drift detection)**——`PrimaryAgent`記住自己上次看過每個block的版本號,下一輪開場先比對版本有沒有在它不注意的時候被別人(sleep-time agent)動過。`main()`跑一段四輪對話(用到`block_append`、`block_replace`兩種寫入,human/persona/task三個block都被寫到),接著跑一次sleep pass(task block被灌到超過80%上限,觸發摘要;一筆關於Berlin時區的archival舊事實因為使用者搬到Lisbon而被判定矛盾、標記失效),最後再讓primary agent多接一輪,印出`[drift]`訊息,證明它偵測到task block在它沒看著的時候被sleep-time agent動過。

## 為什麼需要它
- **記憶整理不能跟回應共用同一段延遲預算**:`SleepTimeAgent.run`跟`PrimaryAgent.turn`是兩個完全獨立的方法,`main()`裡所有`primary.turn(...)`呼叫都在`sleep.run(...)`之前執行完畢——這對應課程說的「every memory operation on the critical path blows up tail latency」:把摘要、矛盾判定挪到`SleepTimeAgent`,`PrimaryAgent`的四輪對話完全不需要等它。
- **Block bloat要有真正會被觸發的防線,不能只是宣告一個`limit`欄位**:`Block.near_limit`(取`len(self.value) >= int(self.limit * threshold)`,`threshold`預設0.8)是`SleepTimeAgent.run`決定要不要呼叫`block.summarize(...)`的唯一判斷依據——`main()`刻意讓`task`block在兩次`block_append`之後衝到`194/220`(超過176這個0.8上限),讓摘要真的被觸發,而不是像許多demo那樣把`limit`寫得很大、讓這條邏輯永遠不會走到。
- **摘要退化成幾乎清空一個block,比塞爆更糟**:`_summarize`原本(也是課程沒有明說、但production系統一定會踩到的細節)只按句號切句子,如果block內容像`task`一樣是分號/逗號串接的一串事實、完全沒有句號,「整段文字」會被當成單一個超長句子,直接超過`target_len`而被整段丟棄,回傳值只剩一個句點——這支程式在`_summarize`裡加了`_truncate_at_word_boundary`當fallback:找不到能塞進`target_len`的句子邊界時,退回到word-boundary截斷加`...`,確保consolidation是「壓縮」而不是「銷毀」。
- **Silent drift需要主動偵測,不能假設primary agent會自己發現**:課程原文點名的失效模式是「sleep-time agent改了一個block,primary agent完全不知道」,建議的解法是「version blocks的同時把diff秀在trace裡」。這支程式把這個建議做成可執行、可觀察的機制:`PrimaryAgent._snapshot_versions`在每輪處理完寫入後記錄看過的版本號,下一輪開場`_note_drift`比對目前版本跟上次記錄的版本——只要兩者不同,一定是`PrimaryAgent`自己以外的東西(這裡是`SleepTimeAgent`)改的,因為`PrimaryAgent`自己的寫入已經被同一輪的`_snapshot_versions`記錄過了。

## 核心原理
- **課程概念對應到程式碼**:

  | 課程概念 | 本程式對應 |
  |---|---|
  | 三層記憶(Core / Recall / Archival) | 只做了兩層——`BlockStore`是Core(永遠在prompt裡),`Archival`是Archival(要查才讀得到);Recall(可檢索的對話歷史)沒有獨立實作,`primary.trace`/`sleep.trace`只是print log,不是可查詢的store(見下方常見誤區) |
  | Memory block(`label`/`value`/`limit`/`description`) | `Block` dataclass——四個欄位一一對應,`version`+`history`是課程建議「version blocks」的具體實作 |
  | Human block / Persona block / 使用者自訂block | `main()`裡的`human`(使用者事實)、`persona`(agent自我風格)、`task`(當前任務範圍)三個block,對應課程說的「原始兩種canonical block,加上像Task這樣的自訂block」 |
  | Block工具surface(`block_append`/`block_replace`/`block_read`/`block_summarize`) | `Block.append`/`Block.replace`/`BlockStore.get`(讀取)/`Block.summarize`——命名跟課程列的四個工具一一對應,`PrimaryAgent.turn`裡`kind`字串(`"block_append"`/`"block_replace"`)直接沿用工具名 |
  | Sleep-time compute(離峰時段、不佔延遲、可用更強模型) | `SleepTimeAgent`——是跟`PrimaryAgent`完全獨立的class,`run`只在所有`primary.turn`跑完之後才被呼叫一次;「可以用更強模型」這件事本身沒有真的接LLM,靠的是架構上兩個agent互相獨立這件事來成立 |
  | Native reasoning(Letta V1,原生推理channel取代`send_message`/heartbeat) | 沒有實作——這支程式的`PrimaryAgent`/`SleepTimeAgent`都是純Python方法,不涉及任何ReAct風格的`Thought:`字串或原生推理channel,課程把這個列為2026年的另一個獨立演進,不在這一課的Build範圍內 |

- **`_seen_versions`是`PrimaryAgent`用來分辨「這是我自己寫的」還是「別人偷改的」的唯一依據**:`turn()`的執行順序是`_note_drift()`(比對進來時的版本)→處理這一輪自己的寫入→`_snapshot_versions()`(記錄離開時的版本)。因為`_snapshot_versions`是在自己的寫入處理完之後才記錄,下一輪`_note_drift`比對到的版本落差,一定不是這一輪`PrimaryAgent`自己造成的——只可能是兩輪之間跑的`SleepTimeAgent`(或未來任何其他寫入者)改的。
- **`SleepTimeAgent.run`一次做兩件互相獨立的事**:先掃過所有block,`near_limit()`回傳`True`的就呼叫`block.summarize(...)`;再掃過所有archival紀錄,任何`valid`且文字包含`claim`(不分大小寫)的紀錄就`invalidate`。這兩段邏輯不共用任何狀態,`main()`的demo裡也刻意讓它們分別被觸發(task block摘要 vs. Berlin時區紀錄失效),但底層設計上兩者可以只觸發其中一種、或都不觸發。
- **矛盾判定是substring比對,不是語意理解**:`claim.lower() in record.text.lower()`——`main()`傳入的`claim`是`"berlin"`,`Archival`裡唯一包含這個字(不分大小寫)的紀錄是turn 1寫入的CET時區備註,所以只有它會被標記失效;如果矛盾事實用完全不同的字眼描述同一件事(例如寫成「歐洲中部時區」而不提到Berlin),這段邏輯完全抓不到。

## 程式碼導覽
| 函式/類別 | 對應到理論的哪個部分 |
|---|---|
| `Block` | 一個memory block:`label`/`value`/`limit`/`description`,外加`version`+`history`供drift追蹤 |
| `Block.append` / `Block.replace` | `block_append`/`block_replace`兩個編輯工具,各自更新`version`並把舊值推進`history` |
| `Block.summarize` | `block_summarize`工具:呼叫`_summarize`產生壓縮後的內容,再透過`rewrite`寫回並升版本 |
| `Block.near_limit` | 判斷是否該觸發`summarize`的門檻檢查,`SleepTimeAgent`用它決定要不要動某個block |
| `BlockStore` | Core tier的容器:`create`/`get`/`labels`/`render`,`render`把所有block格式化成模型會看到的樣子 |
| `ArchivalRecord` / `Archival` | Archival tier:`insert`寫入、`invalidate`標記失效、`valid_records`/`all_records`供不同用途查詢 |
| `PrimaryAgent.turn` | 處理一輪對話:先呼叫`_note_drift`,再依`writes`列表dispatch到`block_append`/`block_replace`/`archival_insert`,最後`_snapshot_versions` |
| `PrimaryAgent._note_drift` / `_snapshot_versions` | 版本追蹤機制的兩端:進場比對、離場記錄,兩者合起來讓「別人偷改了block」變成可觀察的trace訊息 |
| `SleepTimeAgent.run` | 離峰時段整理:掃描near-limit的block並摘要,再依`contradictions`清單invalidate矛盾的archival紀錄 |
| `_summarize` | 優先按句號切句子、湊到`target_len`為止的摘要邏輯 |
| `_truncate_at_word_boundary` | `_summarize`找不到任何能塞進`target_len`的句子邊界時的備援:退到最近的空白截斷,而不是回傳幾乎清空的結果 |
| `main` | 四輪對話demo(含一次`block_replace`、三個block都被寫到)→pre-consolidation狀態→sleep pass→post-consolidation狀態→再一輪demo drift偵測 |

**實作細節 / 容易看漏的地方:**
- **`task`是這支demo裡唯一真的被摘要的block**:`human`(38字元/180上限)跟`persona`(70字元/160上限)全程都遠低於0.8的門檻,只有`task`在第二次`block_append`之後衝到`194/220`(超過176)。這是刻意的安排,不是巧合——如果把三個block的內容平均分散,近乎不可能在只有四輪的demo裡讓任何一個真的越過門檻。
- **`_summarize`對沒有句號的文字,行為分兩段**:先照句號切,`task`block的內容(`"...target senior eng audience=...; weekly office-hours; graded capstone project..."`)其實整段都沒有句號,所以`sentences`只有一個元素——這一個「句子」本身(194字元)就已經超過`target_len`(`220 // 2 = 110`),迴圈第一輪就`break`,`picked`維持空list;程式接著落到`_truncate_at_word_boundary`,在110字元內找最後一個空白截斷,再補上`...`,結果是`107/220`,而不是被砍成只剩一個句點。
- **`PrimaryAgent`能分辨「自己的寫入」跟「別人的偷改」,靠的是`_snapshot_versions`呼叫的時間點**:它在`turn()`裡處理完`writes`**之後**才執行,所以下一輪`_note_drift`比對到的版本落差,保證發生在兩輪之間、不是這一輪自己造成的。如果不小心把`_snapshot_versions()`搬到`_note_drift()`之前呼叫,`[drift]`訊息就會對自己的每一次寫入都誤報。
- **`archival_insert`的矛盾判定範圍是所有紀錄,不分`valid`與否以外的任何條件**:`SleepTimeAgent.run`裡`if record.valid and claim.lower() in record.text.lower()`——只排除已經失效的紀錄,不看是誰寫的、什麼時候寫的;`a002`(使用者偏好)剛好不包含`"berlin"`這個字,所以不會被誤傷,但這是文字內容剛好對得上,不是程式做了任何「這筆跟這個矛盾無關」的判斷。
- **`Block.history`存在,但目前沒有任何地方讀取它**:`append`/`replace`/`rewrite`都會把舊值推進`self.history`,理論上足夠支撐課程Exercise 3要求的`block_history(label)`查詢,但目前`BlockStore`/`SleepTimeAgent`/`main`都沒有任何呼叫路徑讀取這個list——它是留給讀者接上去的掛勾,不是已經在用的功能。

## 使用時機 / 優缺點
| 情境 | 建議 |
|---|---|
| 需要讓不同種類的長期記憶(使用者事實、agent自我風格、當前任務)各自有獨立的大小上限跟編輯規則 | `Block`/`BlockStore`這組介面直接可用,`label`+`limit`+`description`已經是完整的最小schema |
| 記憶整理(摘要/去重/矛盾判定)不能拖慢使用者看到回應的時間 | `SleepTimeAgent`跟`PrimaryAgent`的分離就是為了這件事——把整理邏輯搬到`sleep.run`,不要塞進`primary.turn` |
| 需要偵測「background整理agent動了什麼,而foreground完全不知道」 | 這支程式的`_seen_versions`/`_note_drift`機制可以直接借用,只要background寫入者會讓`version`遞增 |
| 需要真正語意層級的矛盾偵測(而不是關鍵字比對) | `SleepTimeAgent.run`的`claim.lower() in record.text.lower()`只是佔位邏輯,真正上線需要換成embedding相似度或LLM自己判斷是否矛盾 |

- ✅ 想搞懂memory block(具名、可編輯、有上限)怎麼具體對應到程式碼,並且看到`block_append`/`block_replace`/`block_summarize`三種寫入各自被demo一次——`main()`印出的兩段trace(primary/sleep-time)可以逐行對照。
- ✅ 需要一個不接LLM、100%可重現的骨架,驗證「block超過上限就該被摘要」「矛盾事實就該被標記失效」「background改了東西,foreground下一輪要能發現」這三個行為本身,而不是驗證某個真正的摘要模型好不好。
- ✅ 想確認`_summarize`在「文字沒有句號」這種邊界情況下不會整段消失——這是任何用句子分割器做摘要的實作都會踩到的真實bug,這支程式已經處理了。
- ❌ 只做了Core+Archival兩層,沒有獨立的Recall tier:對話歷史本身(`primary.trace`)不是一個可以被`conversation_search`之類工具查詢的store,只是給人看的print log。
- ❌ 沒有Native reasoning:`PrimaryAgent`/`SleepTimeAgent`都是純Python方法呼叫,不涉及任何原生推理channel或`Thought:`字串,Letta V1那一段演進完全不在這支程式的範圍內。
- ❌ 沒有處理「poisoned consolidation」:`SleepTimeAgent.run`會把任何`near_limit`的block內容原封不動送進`_summarize`,不檢查內容是否像是攻擊者植入、企圖影響下一次consolidation結果的指令。
- ❌ 矛盾判定只是substring比對:`claim.lower() in record.text.lower()`抓不到換句話說或用同義詞描述的矛盾,也沒有archival去重(同一件事被`insert`兩次會產生兩筆獨立紀錄)。

## 常見誤區
1. **以為`persona`block在這支demo裡沒被用到**:第二輪對話裡`block_append`確實寫入了`persona`(`"style=concise, citation-heavy, no filler..."`),只是它的內容(70字元)離160的上限還很遠,不會被`SleepTimeAgent`摘要——「被寫入」跟「被摘要」是兩件獨立的事,沒被摘要不代表沒被用到。
2. **以為`SleepTimeAgent.run`每次都會摘要至少一個block**:它只對`near_limit()`回傳`True`的block呼叫`summarize`——這支demo裡只有`task`真的越過0.8門檻,`human`跟`persona`全程都遠低於各自的上限,`sleep-time trace`裡完全不會出現它們的`consolidate`訊息。
3. **以為`_summarize`永遠是「挑幾個完整句子」的摘要**:只有原文按句號切得出多個句子、且至少有一個能塞進`target_len`時才會走這條路;像`task`block這種用分號/逗號串接、完全沒有句號的內容,會落到`_truncate_at_word_boundary`的word-boundary截斷,結果是「前半段+`...`」,不是「挑出來的完整句子」。
4. **以為`[drift]`訊息會把`PrimaryAgent`自己這一輪的寫入也標記出來**:`_snapshot_versions()`在每輪`writes`處理完之後才執行,所以`_note_drift`在下一輪開場比對到的版本落差,保證是兩輪之間發生的變化(這支demo裡只可能是`SleepTimeAgent`),不會把agent自己剛做的寫入誤判成drift。
5. **以為這支程式做了完整的MemGPT/Letta三層記憶(Core/Recall/Archival)**:它只做了Core(`BlockStore`)跟Archival(`Archival`)兩層,沒有獨立的Recall tier——`primary.trace`/`sleep.trace`只是給`main()`印出來看的list,不是可以被查詢的對話歷史store,這點跟[`../Agent_Memory`](../Agent_Memory/README.md)裡`conversation_search`可以查`evicted`訊息是不同的能力。

## 關鍵詞速查
| 詞彙 | 常聽到的說法 | 實際上是什麼 |
|---|---|---|
| Memory block | 「可編輯的prompt區塊」 | 有型別、持久、可被工具編輯的core-tier區段;`Block`的`label`/`value`/`limit`/`description`四個欄位 |
| Human block / Persona block | 「使用者記憶 / agent人設」 | MemGPT論文原始的兩種canonical block;這支程式的`human`跟`persona` |
| Sleep-time compute | 「非同步記憶工作」 | 跑在背景、離critical path的第二個agent,理論上可以用更貴更慢的模型;這支程式的`SleepTimeAgent` |
| Core / Recall / Archival | 「三層記憶」 | 永遠可見(`BlockStore`)/對話歷史(這支程式**沒有**獨立實作)/外部事實庫(`Archival`)——只做了兩層 |
| Block limit | 「上限」 | 每個block的字元上限,`near_limit`超過門檻就該觸發摘要,強迫內容被壓縮而不是無限累積 |
| Native reasoning | 「思考channel」 | Letta V1(2026)拿掉`send_message`/heartbeat、改用provider原生推理channel的重寫;這支程式沒有實作 |
| Silent drift | 「background偷改東西」 | Sleep-time agent改了block,primary agent沒發現;這支程式用`version`+`_seen_versions`具體示範怎麼偵測 |
| Block bloat | 「記憶被灌爆」 | 無限制`append`很快就會撞到`limit`;要在撞到之前就有摘要機制接住,而不是任由`append`失敗或溢出 |

## 延伸閱讀
- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks):block pattern的原始說明,`label`/`value`/`limit`/`description`這組schema的出處。
- [Letta, Sleep-time Compute blog](https://www.letta.com/blog/sleep-time-compute):非同步記憶整理的production案例,跟本程式`SleepTimeAgent`的設計動機相同。
- [Letta, Rearchitecting the Agent Loop](https://www.letta.com/blog/letta-v1-agent):Letta V1原生推理重寫,說明本程式刻意沒有實作的Native reasoning是什麼。
- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560):MemGPT原始論文,[`../Agent_Memory`](../Agent_Memory/README.md)完整實作了這篇的兩層架構。

## 複習自問
1. 課程Exercise 1問的是「`block_summarize`該用什麼門檻觸發,才能同時最小化摘要呼叫次數跟block溢出次數」。目前`near_limit`的預設`threshold=0.8`是寫死的,如果把它調成一個依block寫入頻率動態調整的值(寫得越快、門檻設得越低,提早留緩衝空間),`SleepTimeAgent.run`要怎麼取得「這個block最近寫入頻率」這個額外資訊?
2. 課程Exercise 2要你在archival上做sleep-time去重:兩筆文字token overlap超過90%就合併成一筆,而且只能在sleep pass裡做,不能佔用critical path。`Archival.all_records()`已經能拿到全部紀錄,但目前沒有任何相似度計算——要在`SleepTimeAgent.run`裡加這段邏輯,你會選在matching前(阻止重複寫入)還是matching後(定期清理)做去重?兩者對`ArchivalRecord.rid`的穩定性有什麼不同影響?
3. 課程Exercise 3要你替每次寫入都記錄diff,並開放`block_history(label)`讓operator偵錯「agent為什麼忘記了X」。`Block.history`已經在`append`/`replace`/`rewrite`裡累積舊值,但整支程式沒有任何地方讀取它——要加一個`BlockStore.block_history(label)`,回傳的該是完整的`(old, new)`配對列表,還是只回傳`Block.history`本身?如果要重建「第N版時block長什麼樣子」,現在的`history`(只存舊值,不存操作類型)夠不夠用?
4. 課程Exercise 4要你把sleep-time agent當成不受信任的寫入者,碰到Persona或Safety block的改動要先過第二個agent審核才能真的commit。以目前`SleepTimeAgent.run`的結構,如果要在`block.summarize(...)`真正寫入前插入一道「先送審、審核通過才`rewrite`」的關卡,這道關卡該放在`Block.summarize`內部,還是`SleepTimeAgent.run`呼叫它之前?哪一種設計比較不會被未來新增的呼叫路徑繞過?
5. 課程Exercise 5要你把demo換成真正的Letta API(`letta_v1_agent`),並記錄block schema跟native reasoning怎麼改變trace的形狀。以這支程式的`Block`/`PrimaryAgent`為例,如果`PrimaryAgent.turn`不再是dispatch寫死的`writes`列表,而是真的呼叫一個LLM、由它自己決定要不要呼叫`block_append`/`block_replace`,現在靠`_seen_versions`/`_note_drift`做的drift偵測邏輯還適用嗎——原生推理的思考channel會不會讓「這個block是不是被sleep-time agent改的」這件事,變成模型自己就能在推理過程裡看到,而不需要額外的版本比對?
