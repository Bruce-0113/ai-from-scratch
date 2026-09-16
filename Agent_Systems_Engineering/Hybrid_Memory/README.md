# Hybrid Memory — 向量 + KV + 圖譜的融合式記憶(Mem0)

對應程式: [`./hybrid_memory.py`](./hybrid_memory.py)

參考: [ai-engineering-from-scratch – Hybrid Memory: Vector + Graph + KV 課程](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/14-agent-engineering/09-hybrid-memory-mem0/docs/en.md)

前置概念: Phase 14·07([`../Agent_Memory`](../Agent_Memory/README.md),MemGPT)確立了「main context vs. 外部store」這個分野,Phase 14·08([`../Memory_Blocks_and_Sleep_Time_Compute`](../Memory_Blocks_and_Sleep_Time_Compute/README.md),Letta)把外部store拆成具名block、外加離峰整理。這支程式站在同一條演進線的下一步,但問題性質不同:前兩課解決的是「該存在哪一層」(main context / archival / block),這一課解決的是「該用哪一種store查」——單一store沒辦法同時處理三種查詢形狀:語意檢索("我們聊過關於agent drift的什麼")要vector、精確查值("使用者的電話號碼是什麼")要KV、關聯推理("哪些顧客共用帳單")要graph。三種store並行寫入、查詢時再融合排序,是Mem0(Chhikara et al., 2025, arXiv:2504.19413)的核心設計。

## TL;DR
單一store架構對不同查詢形狀各有天生弱點:vector對精確值查詢會引入不必要的語意模糊、KV對關聯推理無能為力、graph對自由文字語意檢索效率低。Mem0的解法不是選一種store,而是三種**平行**運作——每次寫入(`add`)同時進vector、KV(如果有結構化的`kv_triples`)、graph(如果有`graph_triples`),每次查詢(`search`)則用一個**融合分數**(`relevance + importance + recency`,各自可調權重)把vector跟KV的命中結果合併成一份排序清單。本程式用純Python刻出`VectorStore`(token-overlap模擬embedding)、`KVStore`(`(user_id, fact_type, entity)`三元組鍵值)、`GraphStore`(帶時序衝突偵測的typed edge)三個store,外加`Mem0`facade統一寫入跟融合查詢,並實作課程明確要求的**scope taxonomy**(user/session/agent三種隔離範圍)。`main()`跑一段ava(多輪對話、搬家)跟bob(帳務)的demo,展示語意檢索、圖譜的時序衝突處理(Berlin edge在Lisbon edge寫入後被標記失效而非刪除)、KV的精確查值、融合排序、以及user-scope跟session-scope兩種隔離範圍分別被驗證。

## 為什麼需要它
- **三種查詢形狀需要三種天生合適的資料結構,不是三種都硬要塞進同一種**:`VectorStore.search`用token-overlap(Jaccard相似度)找語意相關的文字,`KVStore.get`用`(user_id, fact_type, entity)`三元組做O(1)精確查值,`GraphStore.neighbors`沿著typed edge做關聯遍歷——`Mem0.add`把同一次寫入視情況同時送進三者(有`kv_triples`才寫KV,有`graph_triples`才寫graph),不強迫每筆記憶都要有完整的結構化資訊。
- **查詢時不能只問一種store,要融合排序**:`Mem0.search`不是「先試vector,不行再試KV」的fallback鏈,而是把`VectorStore.search`跟`KVStore.by_user`的命中結果都轉成同一個分數量綱(`relevance + importance + recency`,見`Mem0Config`)後合併進同一個`fused`字典再排序——這樣即使某筆記憶的字面相關度較低,仍然可能因為importance權重較高而排到前面,這正是`main()`裡`m002`(curriculum,importance 0.9)在`"where does ava live"`這個查詢下反而排在字面相關度更高的`m004`(Lisbon,importance 0.8)、`m003`(Berlin,importance 0.6)前面的原因(三筆記憶寫入時間相差不到一秒,recency幾乎打平,真正拉開差距的是importance)。
- **矛盾事實要能被非破壞性地標記失效,不能單純覆寫或刪除**:`GraphStore.add_edge`在寫入新edge前,先把同一個`(subject, relation)`底下所有還valid的舊edge標記成`valid=False`,而不是刪除——`main()`裡`ava --lives_in--> Berlin`在`ava --lives_in--> Lisbon`寫入後變成`INVALID`但仍然存在於`all_edges()`,`neighbors(valid_only=False)`可以查到完整的居住歷史,這是課程說的「Mem0g conflict detector」跟temporal reasoning的具體實作。
- **記憶不是全域可見,要照user/session/agent三種範圍隔離**:`Record.scope`搭配`Mem0._visible`實作課程要求的scope taxonomy——user-scoped記憶跨session持久但只對該user可見,session-scoped記憶只在寫入的那個session內可見,agent-scoped記憶(本程式沒有demo,但邏輯上支援)對任何呼叫者可見。`main()`用bob的帳務記錄驗證user-scope隔離、用ava的HR提醒驗證session-scope隔離,兩種都實際跑過`Mem0.search`並印出隔離確實生效。

## 核心原理
- **課程概念對應到程式碼**:

  | 課程概念 | 本程式對應 |
  |---|---|
  | Vector store(語意相似度) | `VectorStore`——`search`用token-overlap(Jaccard相似度)取代真正的embedding餘弦相似度,零重疊的紀錄直接丟棄而非排到最後 |
  | KV store(`(user_id, fact_type, entity)`三元組,O(1)精確查值) | `KVKey` + `KVStore`——`get`是O(1)點查,`by_user`是列出某user所有KV事實;`entity`欄位存的是事實的「值」而非「名稱」(見下方常見誤區) |
  | Graph store(typed edge,多跳推理,Mem0g時序衝突偵測) | `Edge` + `GraphStore`——`add_edge`寫入新edge前把同`(subject, relation)`的舊edge標記`valid=False`,`neighbors(valid_only=False)`可查完整歷史 |
  | Fusion score(`relevance + importance + recency`,per-product權重) | `Mem0Config`(`w_relevance`/`w_importance`/`w_recency`/`recency_halflife_s`)+`Mem0._recency_score`(指數衰減)+`Mem0.search`裡的加權和 |
  | Scope taxonomy(user / session / agent) | `Record.scope` + `Mem0._visible`——user依`user_id`隔離、session依`(user_id, session_id)`隔離、agent對所有呼叫者可見 |
  | Importance(寫入時標記或學到的權重) | `Record.importance`,寫入時由呼叫端指定(`Mem0.add`的`importance`參數),本程式沒有實作「從使用行為學到權重」這一半 |
  | Recency(相對上次寫入/讀取的指數衰減) | `Mem0._recency_score`——`0.5 ** (elapsed / half_life)`,半衰期預設一天,只算「上次寫入以來」,沒有實作「上次讀取以來」那一半(見下方常見誤區) |
  | Mem0(2025)benchmark:LoCoMo 91.6 / LongMemEval 93.4 / BEAM-1M 64.1 | 沒有實作——本程式是骨架示範,不接真正的embedding模型,無法重現論文的benchmark數字,見下方延伸閱讀 |

- **`Mem0.search`的兩條路徑共用同一個`_visible`關卡,這是這次整理修掉的一個真實不一致**:改動前,vector命中路徑會檢查`scope`參數,但KV命中路徑(`self.kv.by_user(user_id)`那段)完全沒有檢查`scope`——如果呼叫端指定`scope="session"`只想要當前session的記憶,KV路徑仍然會把該user所有scope的KV事實都塞進融合結果。現在`_visible`被兩條路徑共用,任何未來新增的命中來源只要呼叫`_visible`就能保證隔離規則一致。
- **KV路徑用一個扁平的`_KV_PSEUDO_RELEVANCE`(0.4)取代真正的relevance**:KV的比對邏輯是精確鍵值匹配,天生沒有「相似度」這個概念——`_KV_PSEUDO_RELEVANCE`只是為了讓KV事實能在融合排序裡有一席之地(高到不會被importance/recency直接蓋過、低到不會贏過真正語意相關的vector命中),不是模擬出來的真實分數。
- **`_visible`的session隔離是漸進式的,不是強制的**:`record.scope == "session"`時,只有在呼叫端**有給**`session_id`參數才會比對`record.session_id`;不給`session_id`則退化成只檢查`user_id`(等同user-scope的隔離力度)。這對應「查詢時不見得知道要限定哪個session,但一定知道是哪個user」這種實務情境——`main()`裡`scope="session", session_id="s002"`的呼叫展示的是完整隔離,而不帶`session_id`的呼叫(本程式沒有demo到)則是較寬鬆的行為,值得在自己延伸時留意。

## 程式碼導覽
| 函式/類別 | 對應到理論的哪個部分 |
|---|---|
| `Record` | 一次記憶寫入,三個store共用的資料形狀:`text`(給vector/KV用)、`scope`+`user_id`+`session_id`(隔離範圍)、`importance`+`ts`(融合分數用) |
| `VectorStore.search` | Vector store的語意檢索,token-overlap(Jaccard)相似度取代真正embedding |
| `KVKey` / `KVStore` | KV store:`(user_id, fact_type, entity)`三元組鍵值,`get`是O(1)點查,`by_user`列出某user全部KV事實 |
| `Edge` / `GraphStore.add_edge` | Graph store的寫入路徑:寫新edge前,把同`(subject, relation)`底下還valid的舊edge標記失效,非破壞性衝突處理(Mem0g) |
| `GraphStore.neighbors` / `all_edges` | Graph store的讀取路徑:`neighbors`預設只回傳valid edge,`valid_only=False`可查含失效紀錄的完整歷史 |
| `Mem0Config` | 融合分數的權重表(`w_relevance`/`w_importance`/`w_recency`)跟recency半衰期,per-product可調 |
| `Mem0.add` | 統一寫入入口:一定寫vector,`kv_triples`非空才寫KV,`graph_triples`非空才寫graph |
| `Mem0._recency_score` | Recency維度:指數衰減,`0.5 ** (elapsed / recency_halflife_s)` |
| `Mem0._visible` | Scope taxonomy的隔離關卡,vector跟KV兩條命中路徑共用同一份規則 |
| `Mem0.search` | 融合查詢:vector命中(真實relevance)跟KV命中(`_KV_PSEUDO_RELEVANCE`)都算出`relevance+importance+recency`分數,經`_visible`過濾後合併排序 |
| `main` | Demo:ava多輪對話(vector/KV/graph都寫到)+bob帳務記錄→vector-only檢索→graph時序衝突→KV點查與列表→融合排序→user-scope隔離→session-scope隔離 |

## 使用時機 / 優缺點
| 情境 | 建議 |
|---|---|
| 需要同時支援「語意檢索」「精確查值」「關聯推理」三種查詢,且不想為了其中一種犧牲另外兩種 | `Mem0`這組facade直接示範三store並行寫入、融合查詢的介面設計,可以直接參考`add`/`search`的參數形狀 |
| 記憶排序需要依產品需求調整「多看重新鮮度」vs.「多看重要性」 | `Mem0Config`四個欄位就是可調的介面,聊天助理可以把`w_recency`調高,合規/稽核場景可以把`w_importance`調高 |
| 需要「新事實推翻舊事實,但舊事實仍要可查」(例如稽核軌跡) | `GraphStore.add_edge`的非破壞性標記失效直接可用;但目前只有graph層有這個能力,KV層沒有(見下方常見誤區) |
| 需要照user/session/agent三種範圍隔離記憶,避免跨user或跨session洩漏 | `Record.scope`+`Mem0._visible`是最小可用的骨架,`main()`裡user-scope跟session-scope都有demo驗證 |
| 需要真正的語意相似度、而不是字面上的token重疊 | `VectorStore.search`目前是Jaccard相似度佔位,換成真正的embedding模型(sentence-transformers、Ollama等)是課程Exercise 1的範圍,不在本程式內 |

- ✅ 想搞懂「三store並行寫入、融合排序」這個設計本身怎麼落地成程式碼,`Mem0.add`/`Mem0.search`的兩條路徑可以逐行對照課程的fusion公式。
- ✅ 需要一個不接LLM、100%可重現的骨架,驗證「新事實讓舊事實失效但不刪除」「不同scope互相隔離」這兩個行為本身,而不是驗證某個真正的embedding模型好不好。
- ✅ 想確認scope taxonomy(user/session/agent)在程式碼裡具體長什麼樣子——`_visible`一個函式就是完整的隔離邏輯,沒有分散在多處。
- ❌ Vector store是token-overlap佔位,不是真正的embedding語意檢索;字面完全不重疊但語意相關的查詢(同義詞、換句話說)完全抓不到。
- ❌ 只有GraphStore有非破壞性的衝突處理;KVStore對同一`fact_type`的新舊值各自產生獨立的key,不會互相標記失效,`by_user`會把Berlin跟Lisbon兩筆都列出來(見下方常見誤區)。
- ❌ 沒有實作`as_of=timestamp`的時序查詢(課程Exercise 2)——雖然GraphStore保留了失效edge跟其寫入時間戳,但`search`/`neighbors`都沒有「回到某個時間點」的查詢介面。
- ❌ Importance完全靠寫入時人工指定(`Mem0.add`的`importance`參數),沒有「從使用者回饋或存取頻率學到權重」這一半(課程Exercise 4提到的`user_feedback`維度)。
- ❌ Agent-scope只有邏輯支援(`_visible`裡有處理這個分支),`main()`沒有實際demo到——目前的demo只驗證了user跟session兩種範圍。

## 常見誤區
1. **以為`KVKey`的`entity`欄位存的是「實體名稱」**:實際上它存的是事實的**值**(例如`"Lisbon"`),不是像`"ava"`這樣的實體識別碼。這個命名直接沿用課程原文的`(user_id, fact_type, entity)`三元組寫法,但拆解到程式碼裡,`entity`其實扮演的是「這筆事實的內容」——這也是下一條誤區的根源。
2. **以為KV store會像graph store一樣,新值自動讓舊值失效**:`mem.add(..., kv_triples=(("city", "Lisbon"),))`跟先前寫入的`kv_triples=(("city", "Berlin"),)`會產生**兩個不同的`KVKey`**(`entity`不同),`KVStore.put`是用key覆寫,不同key不會互相覆蓋。`main()`裡`mem.kv.by_user("ava")`會同時列出`m003`(Berlin)跟`m004`(Lisbon)——這是KVStore目前的真實限制,不是bug,因為`(user_id, fact_type, entity)`把值本身編進了key,天生就不是「同一個fact_type只保留最新值」的設計。要做到last-write-wins,需要額外一層以`(user_id, fact_type)`為key(不含`entity`)的索引,本程式沒有實作。
3. **以為`Mem0.search`回傳的排序完全由字面相似度決定**:融合分數是`relevance + importance + recency`的加權和,`Mem0Config`預設權重是`0.6/0.2/0.2`——`main()`裡查詢`"where does ava live"`,實際上`m003`(Berlin)跟`m004`(Lisbon)的token-overlap relevance都比`m002`(curriculum)高,但`m002`的importance(0.9)比`m004`(0.8)、`m003`(0.6)都高,兩者加權後`m002`還是以些微差距(0.430 vs 0.427)排到第一。看到排序「反直覺」時,先檢查是不是importance權重在主導(三筆記憶在demo裡幾乎同時寫入,recency差距可忽略),而不是預設relevance算錯了。
4. **以為`GraphStore`的失效edge已經從系統裡消失**:`add_edge`只是把舊edge的`valid`標記成`False`,`all_edges()`跟`neighbors(valid_only=False)`都還查得到——`main()`的「graph recall」demo刻意用`valid_only=False`把`INVALID`的Berlin edge也印出來,證明它還在,只是被排除在預設查詢(`valid_only=True`)之外。
5. **以為`scope="session"`永遠會限定到單一session**:只有呼叫`Mem0.search`時**同時**帶`scope="session"`跟具體的`session_id`,才會做到`(user_id, session_id)`的完整隔離;`_visible`裡`session_id`參數是`None`時,session-scoped記憶只會依`user_id`過濾,實際上等同user-scope的隔離力度。`main()`的session-scope demo刻意兩個呼叫都帶了`session_id`,分別展示同session命中、跨session不命中。

## 關鍵詞速查
| 詞彙 | 常聽到的說法 | 實際上是什麼 |
|---|---|---|
| Hybrid memory | 「混合式記憶」 | 多種store(vector/KV/graph)並行運作、查詢時融合排序的架構,不是「換一種更好的單一store」 |
| Fusion score | 「排序分數」 | `relevance + importance + recency`的加權和,`Mem0Config`定義權重,`Mem0.search`計算 |
| Scope taxonomy | 「記憶的可見範圍」 | user(跨session持久)/session(限單一thread)/agent(所有呼叫者共享)三種隔離層級;本程式的`Record.scope` |
| Mem0g | 「圖譜版的Mem0」 | Mem0的graph元件,負責多跳推理跟時序衝突偵測(新事實讓舊edge失效但不刪除);本程式的`GraphStore` |
| Embedding drift | 「向量檢索變不準」 | 語料庫變大後,原本表現好的vector檢索品質逐漸下降;本程式的token-overlap佔位完全沒有這個問題,但也沒有真正的語意能力 |
| KV schema creep | 「KV欄位越加越亂」 | 沒有治理的`fact_type`會無限增生,難以維護;本程式的`kv_triples`目前由呼叫端自由指定字串,沒有任何schema驗證 |
| Graph explosion | 「圖譜爆炸」 | 低品質的抽取器每則訊息塞進大量edge,拖垮查詢效能;本程式`add_edge`沒有per-call的數量上限或信心分數過濾 |
| Non-destructive invalidation | 「軟刪除」 | 新事實讓舊事實標記失效但保留在store裡,可支援時序查詢;本程式`GraphStore.add_edge`的核心行為,`KVStore`沒有對應機制 |

## 延伸閱讀
- [Chhikara et al., Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413):Mem0原始論文,vector+KV+graph混合架構跟fusion scoring的出處,論文報告LoCoMo 91.6、LongMemEval 93.4、BEAM-1M 64.1,均優於full-context LLM、單純vector、單純KV等baseline 10分以上。
- [Mem0 docs](https://docs.mem0.ai):production API跟managed cloud文件,自架版本可搭Postgres+Qdrant+Neo4j。
- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560):Mem0在課程譜系裡的前置設計,[`../Agent_Memory`](../Agent_Memory/README.md)完整實作了這篇的兩層架構。
- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks):課程點名的另一種「三層記憶」姊妹設計(具名block+自訂vector/graph後端),對應[`../Memory_Blocks_and_Sleep_Time_Compute`](../Memory_Blocks_and_Sleep_Time_Compute/README.md)。

## 複習自問
1. 課程Exercise 1要你把`VectorStore`換成真正的embedding(sentence-transformers或Ollama),並在1000次寫入後量recall@10。以目前`VectorStore.search`的介面(輸入query字串、輸出`(score, Record)`列表)為例,換成真正embedding後,`Mem0.search`裡跟KV路徑共用的`_visible`關卡跟融合公式需要跟著改嗎,還是vector store內部的實作細節可以整個替換而不影響`Mem0`facade其他部分?
2. 課程Exercise 2要你實作`as_of=timestamp`的時序查詢。`GraphStore`的`Edge`已經有`ts`欄位,失效的edge也還留在`_edges`裡沒被刪除——要讓`neighbors`支援「回到某個時間點看到的關係圖長怎樣」,你會怎麼定義「在`as_of`當下,這條edge是否算valid」?只看`ts <= as_of`夠嗎,還是需要記錄edge失效的時間點(目前`Edge`沒有這個欄位)?
3. 課程Exercise 3要你加入衝突偵測:記錄矛盾事實的log,並讓舊edge失效。`GraphStore.add_edge`已經做到「讓舊edge失效」,但完全沒有log「這是因為跟哪筆新事實矛盾」這件事——要加這個log,應該記在`Edge`本身(多一個欄位存「被誰取代」)還是另外開一個獨立的衝突紀錄list?兩種設計對「事後追查某個事實為什麼消失」的除錯體驗有什麼差別?
4. 課程Exercise 4要你替融合分數加上`user_feedback`維度,並討論這會不會被使用者惡意「洗」出想要的排序(gaming risk)。如果`Mem0Config`加一個`w_feedback`權重、`Record`加一個`feedback_score`欄位,你會如何限制這個分數的更新頻率或幅度,才不會讓使用者對同一筆記憶連續按讚而把它洗到榜首?
5. 課程Exercise 5要你把demo換成真正的Mem0 client library,並在20個測試查詢上比較recall。以本程式`Mem0.add`/`Mem0.search`的參數形狀為例(`user_id`/`session_id`/`scope`/`kv_triples`/`graph_triples`),如果要無痛切換成真正的Mem0 SDK,現在的介面設計哪些地方會跟SDK的參數對不上(例如SDK通常沒有本程式這種手動指定`kv_triples`/`graph_triples`的寫法,而是靠LLM自動抽取)?
