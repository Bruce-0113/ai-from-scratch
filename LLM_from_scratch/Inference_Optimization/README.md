# Inference Optimization (KV Cache / Batching / Speculative Decoding)

對應程式: [`./inference_optimization.py`](./inference_optimization.py)

參考:[ai-engineering-from-scratch – Inference Optimization 課程](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/10-llms-from-scratch/12-inference-optimization/docs/en.md)

## TL;DR
LLM inference分成兩個體質完全不同的階段:**prefill**一次平行處理整個prompt,是compute-bound(GPU算力吃滿);**decode**一次只生一個token、還要把整份模型權重從顯存搬進SM一次,是memory-bound(算幾微秒、等資料等更久)。所有inference優化技巧,拆開來看都是在打這兩個階段各自的瓶頸:KV cache省掉decode階段重算舊token key/value的浪費、continuous batching讓GPU在decode階段不要因為等batch裡最慢的那個request而空轉、prefix caching讓共用前綴(system prompt、few-shot)的多個request不用各自重算一遍attention、speculative decoding用一個小model先猜、大model一次verify多個token來降低「每個token都要完整跑一次大model」的memory-bound稅。本程式用numpy把這四個機制的核心資料結構跟時序邏輯都攤開實作一遍,外加一個記憶體容量規劃的計算機,讓你能拿真實model的參數量代進去,算出「這張GPU到底能同時撐幾個使用者」。

## 為什麼需要它
- **Naive inference浪費掉九成以上的GPU算力**:單一使用者跑Llama 3 70B大約50 tokens/秒,但如果每個request都各自跑、互不重疊,GPU在decode階段的每一步都是「算幾微秒、等權重從顯存搬過來等更久」,算力大部分時間是閒置的——這不是「model不夠快」,是排程方式的問題,而排程問題可以純靠工程解決,不必換更貴的硬體。
- **KV cache是decode階段唯一的「免費午餐」**:不快取的話,生成第N+1個token時,前面N個token的key/value會被完整重算一次,而它們的值其實從第一次算出來就不會再變——這是純粹的浪費。以Llama 3 70B、4K context為例,快取每個request大約要付1.25GB顯存(本程式`kv_cache_memory`算出來的數字,跟課程教材給的~1.28GB數字對得上),但換到的是每個decode step省下重算全部歷史token attention的成本。
- **Continuous batching是吞吐量的槓桿,不是錦上添花**:同一份硬體、同一個model,光是排程方式從「等整批做完才收下一批」(static batching)換成「一個slot空出來就馬上塞下一個request」(continuous batching),吞吐量可以差到2-5倍——這是本程式`demo_batching`可以親自跑出來驗證的數字,不是紙上談兵。
- **Speculative decoding把「準確率」跟「延遲」的取捨變成免費的**:它不是近似算法,draft model猜錯的token會被reject並用target model重新採樣,數學上輸出分佈跟target model單獨跑完全一致(exactly correct),但實際跑起來只要接受率夠高(70-85%),就能拿到2-3倍的decode加速——這打破了「要快就要犧牲品質」的直覺。
- **KV cache才是真正卡住「能服務幾個使用者」的瓶頸,不是model權重本身**:一張80GB GPU扣掉Llama-3-8B的模型權重跟系統開銷後,剩下的顯存還能撐上百個4K-context並發使用者(`memory_budget`算出來~114個);但同一張GPU連Llama-3-70B的權重(fp16需要~130GB)都放不下——這說明「這個model能不能單卡跑」跟「這個model能同時服務幾個人」是兩個要分開算的問題,前者決定要不要上多卡tensor parallelism,後者決定同一份硬體投資能攤到多少流量上。

## 核心原理
- **Prefill(compute-bound)vs. decode(memory-bound)**:prefill是一次矩陣乘法處理整段prompt,arithmetic intensity高,GPU核心持續busy;decode每一步都要把完整的模型權重從HBM搬進SM做一次矩陣乘法,但因為只算一個token,算的量遠小於「搬資料」的量,GPU算完馬上又要等下一批權重搬進來。這個「ops:byte比值」(每從記憶體搬1 byte能配上幾次運算)是判斷該用什麼優化手段的框架:比值低(memory-bound,decode就是這種)該做的是quantization或加大batch size把搬進來的權重攤給更多token共用;比值高(compute-bound,prefill是這種)該做的是kernel fusion或降精度。
- **KV Cache(`KVCache` / `MultiHeadAttention`)**:把每一層attention算出來的key/value寫進預先配置好的陣列裡,之後的每個decode step只需要算「新token的Q」跟「全部歷史K/V」的attention,不必重算歷史K/V本身。程式裡`update()`負責寫入、回傳「目前為止的全部K/V」,`advance()`負責把`seq_len`往前推——這兩步必須每次forward都成對呼叫,且推進的token數要跟這次forward實際處理的token數一致(prefill一次推進整段prompt長度,decode一次推進1),順序或數量錯了,下一次`update()`就會從錯的位置開始覆寫,悄悄把已經快取的token蓋掉。
- **Continuous batching(`simulate_static_batching` vs. `simulate_continuous_batching`)**:static batching把一批request綁在一起處理,批次裡跑得最快的那個也得等最慢的那個生完才能離開——短request被長request拖累,GPU在等待期間的算力是浪費的。continuous batching每個時間步都檢查有沒有request做完,一旦有空位立刻塞下一個排隊中的request,短request不會被卡住,GPU使用率因此明顯提升。
- **Prefix caching(`TrieNode` / `PrefixCache`)**:把token id序列組成一棵trie,每個節點存這個位置的KV狀態。多個request如果共用前綴(同一份system prompt、同一組few-shot範例),`lookup()`能找出目前為止最長的已快取前綴,只有前綴之後的部分需要重新計算——這正是vLLM/SGLang生態裡「RadixAttention / prefix caching」在做的事:prefix越長、共用的request越多,省下的重算量越可觀。
- **Speculative decoding(`DraftModel` / `TargetModel` / `speculative_decode`)**:小的draft model一次便宜地生出`num_speculative`個候選token,大的target model一次forward pass(`verify_cost`只需付一次,不隨候選數線性增加)平行驗證所有候選;驗證用的是rejection sampling(`target_p[token] / draft_p[token]`,程式裡有算出`acceptance_prob`示範這個公式,但實際accept/reject用的是模擬用的`acceptance_rate`參數,細節見下方「實作細節」),一路接受直到遇到第一個被拒絕的token為止,若整批都被接受還能再白拿一個target model直接採樣的bonus token。
- **容量規劃(`MODEL_CONFIGS` / `kv_cache_memory` / `memory_budget`)**:KV cache大小只跟`num_layers × num_kv_heads × head_dim`有關,跟query head數無關——這正是GQA(grouped-query attention)省顯存的原理:多個query head共用同一組KV head,`num_kv_heads`比實際head數少很多(Llama 3系列在32個query head背後只用8個KV head),KV cache直接按比例縮小。反例是`GPT-4-est`(估計沒用GQA、`num_kv_heads`跟query head數一樣多),同樣4K context下KV cache是Llama-3-8B的45倍(23GB vs. 512MB)——GQA/MQA對KV cache大小的影響,比任何其他單一設計選擇都大。

## 程式碼導覽
| 函式/類別 | 對應到理論的哪個部分 |
|---|---|
| `KVCache` | 預先配置好的per-layer K/V陣列,`update`寫入新token、`advance`推進游標,`memory_bytes`/`used_bytes`分別回報「配置了多少」跟「實際用了多少」 |
| `scaled_dot_product_attention` | 標準causal attention;`seq_len_q > 1`(prefill)才套causal mask,`seq_len_q == 1`(decode)不需要,因為新token本來就排在最後面 |
| `MultiHeadAttention` | 把KV cache接進標準multi-head attention的forward pass裡,示範prefill(一次處理整段prompt)跟decode(一次處理一個token)如何共用同一份程式碼路徑 |
| `Request` / `simulate_static_batching` | Naive batching:一批request綁死在一起,整批人等最慢的那個 |
| `simulate_continuous_batching` | 逐步(per-timestep)排程:誰做完了立刻讓下一個排隊中的request補進來 |
| `batching_stats` | 把completed request列表整理成平均/p50/p99延遲跟throughput,讓兩種batching策略能量化比較 |
| `TrieNode` / `PrefixCache` | 用trie結構快取token前綴對應的KV資料,`lookup`找最長已快取前綴、`insert`寫入新前綴、`hit_rate`回報整體命中率 |
| `DraftModel` / `TargetModel` | Speculative decoding的draft/target model替身,不是真的小model/大model,而是用參數化的機率分佈模擬它們的行為 |
| `speculative_decode` | 一輪「draft多個token → target一次verify → 逐個accept/reject」的完整流程,回傳這一次模擬的成本與speedup |
| `compare_speculation_strategies` | 用幾組真實文獻上的接受率(draft-target 78%、EAGLE 85%、n-gram 50%)跑多次試驗取平均,比較不同drafting策略的實際加速倍率 |
| `MODEL_CONFIGS` | 幾個真實model的層數/KV head數/head_dim/參數量,GQA與否直接影響KV cache大小 |
| `kv_cache_memory` | 給定model shape跟context長度,算出單一request的KV cache要佔多少顯存 |
| `memory_budget` | 給定GPU顯存,先扣掉模型權重跟開銷,剩下的顯存換算成「還能撐幾個並發使用者」 |
| `demo_kv_cache` / `demo_batching` / `demo_prefix_cache` / `demo_speculative_decoding` / `demo_memory_budget` | 對應五個章節的可執行demo,`main()`依序跑完全部,直接印出量化結果 |

**實作細節 / 容易看漏的地方:**
- 原始版本裡`KVCache.advance()`只在`seq_len == 1`(decode步)時才被呼叫,prefill那一次(`seq_len` = prompt長度,通常大於1)完全沒有推進`cache.seq_len`。這是一個真的會讓輸出錯掉的bug:下一次decode呼叫`update()`時,會誤以為快取還是空的,直接把prefill寫進去的K/V從位置0開始覆寫掉,等於讓model在生成階段完全「忘記」剛剛讀過的prompt。整理後的版本把`advance(seq_len)`改成每次forward都呼叫、且用實際處理的token數(而不是寫死的1)——`demo_kv_cache()`印出來的`cache.seq_len`會依序是6(prefill 6個token後)→9(再decode 3步後),可以直接跑一次確認修正有效。
- `speculative_decode`裡`acceptance_prob`(用`target_p[token] / draft_p[token]`算出的教科書rejection-sampling公式)有算出來,但實際accept/reject判斷用的是`r < draft_model.acceptance_rate`這個模擬參數,並沒有真的拿`acceptance_prob`去做比較——這是刻意的簡化(沒有真正的draft/target model可以算出有意義的機率分佈),但如果沒注意到這行,會誤以為程式在做正牌的rejection sampling驗證。
- 因為每一輪只要遇到第一個被拒絕的token就整輪提前結束,回傳的`acceptance_rate`(`avg_accepted / num_speculative`)天生就會比輸入的`draft_model.acceptance_rate`低——例如`acceptance_rate=0.78`、`num_speculative=5`時,理論期望的`avg_accepted`約`0.78+0.78²+0.78³+0.78⁴+0.78⁵ ≈ 2.52`,對應的「這一整輪的接受率」只有約50%,而不是78%。這不是bug,是幾何分佈truncation的必然結果,但拿`compare_speculation_strategies`回傳的`acceptance_rate`直接當成「per-token接受機率」來解讀就會誤判——它衡量的是「平均每一輪省下幾個token」,更接近speedup的直接成因。
- 這支程式沒有實作PagedAttention(vLLM的核心技巧):`KVCache`用的是每個request一段連續配置好的陣列,不是課程教材裡提到的「像作業系統virtual memory一樣切成固定大小的page」。連續配置在request長度變化很大、或多個request並發時,容易產生記憶體碎片化(教材數字:60-80%的碎片率);page-based配置搭配copy-on-write才能把碎片壓到接近零(~4%)並支援多個request共享同一段前綴的實體記憶體。這塊如果要繼續往下實作,可以參考vLLM論文的PagedAttention設計。

## 使用時機 / 優缺點
- ✅ 想搞懂「vLLM/SGLang/TensorRT-LLM這些inference server底層到底在幹嘛」,而不是只知道要裝哪個套件:KV cache的寫入時序、continuous batching的排程邏輯、prefix cache的trie結構、speculative decoding的accept/reject流程,這裡每一步都是用純Python/numpy攤開寫的,沒有藏在CUDA kernel或Rust runtime裡。
- ✅ 需要對「這個model能不能上線、能撐幾個使用者」做back-of-envelope估算:`memory_budget`把「模型權重佔多少、KV cache佔多少、還剩多少給並發使用者」三件事拆開算,面試或做容量規劃時可以直接套用這個框架,不用等到真的把model部署上去才發現放不下。
- ✅ 想直覺理解GQA/MQA為什麼重要:改一下`MODEL_CONFIGS`裡`num_kv_heads`,重跑`kv_cache_memory`,馬上能看到KV cache大小怎麼隨著GQA分組數線性縮放。
- ❌ 不要把這裡的`KVCache`/`PrefixCache`直接搬進真正的inference server:沒有PagedAttention式的page配置(見上方實作細節)、沒有多request共享實體記憶體的機制、`PrefixCache`是單機記憶體內的trie而非跨request/跨GPU的radix tree,正式場景請直接用vLLM、SGLang或TensorRT-LLM(選型細節見下方「生產環境怎麼選」)。
- ❌ `speculative_decode`的draft/target model是機率分佈的模擬,不是真的小model猜大model:程式沒有示範「怎麼訓練/選一個draft model」或EAGLE那類「用target model的hidden state直接預測」的做法,只示範了「給定一個接受率,加速倍率長什麼樣子」這個上層效果;真的要接draft model,需要另外接一個真實的小model或n-gram lookup。
- ❌ 不適合拿來做真正的效能測試或benchmark:所有的「cost」都是寫死的常數(`draft_cost=1.0`、`target_cost=10.0`、`verify_cost=12.0`),不是量測出來的真實硬體數字,拿這裡的`speedup`去做論文或報告裡的效能宣稱是不合理的,它只適合拿來建立「加速倍率跟接受率、投機token數量之間大致是什麼關係」的直覺。

## 生產環境怎麼選:vLLM / SGLang / TensorRT-LLM
2026年的LLM inference serving生態基本上由這三套engine主導,各自的優化重點不同,選型應該跟著workload特性走,而不是選「最新」或「最紅」的那個:

| Engine | 核心技術 | 最適合的情境 |
|---|---|---|
| vLLM | PagedAttention、continuous batching | 通用型serving,相容性最廣 |
| SGLang | RadixAttention(prefix caching)、結構化生成 | 多輪對話、constrained decoding |
| TensorRT-LLM | NVIDIA kernel fusion、FP8量化 | NVIDIA硬體上的單卡最大吞吐量 |

- **vLLM是預設起手式**:支援的model範圍最廣,任何GPU廠商(NVIDIA/AMD/Intel)都能跑,靠PagedAttention(本程式`KVCache`沒有實作的部分,見上方「實作細節」)加上continuous batching(對應`simulate_continuous_batching`的概念)拿到不錯的吞吐量,又有OpenAI相容API可以直接當替代品接上既有系統。不確定workload特性、想要通用性跟最廣硬體支援時,先選它。
- **SGLang在有前綴重疊的場景特別有優勢**:建立在跟vLLM類似的基礎上,額外加了RadixAttention(對應本程式`PrefixCache`想模擬的概念,但`PrefixCache`只是單機記憶體內的trie,不是跨request/跨GPU的radix tree)跟一套描述結構化LLM程式的DSL。如果workload是多輪對話(同一個對話歷史被反覆延伸)、大量request共用同一份system prompt/few-shot範例、或需要constrained decoding(JSON輸出、regex-guided generation、tool use),SGLang靠前綴重用往往能比vLLM快2-5倍。
- **TensorRT-LLM把model編譯成針對NVIDIA GPU優化過的kernel**:融合多個運算(attention + linear + activation揉進同一個kernel)、在H100上用FP8量化、並整合NVIDIA Triton Inference Server做生產部署。在NVIDIA硬體上能拿到最高的單卡吞吐量,但只能用在NVIDIA GPU上,而且部署設定比另外兩套麻煩。

實務上簡單的判斷順序:不確定就先上vLLM;一旦發現workload是「同一個對話歷史反覆被下一輪引用」或「大量request共用同一份system prompt」這種有前綴重疊的模式,就值得評估換SGLang;如果整個部署環境本來就是NVIDIA GPU、又在乎單卡極限吞吐量,才值得投入TensorRT-LLM的編譯與部署成本。

## 常見誤區
1. **以為KV cache省的是算力(FLOPs),其實它省的是「不用重算」這件事本身,代價是顯存**:KV cache用顯存換掉重算的浪費,但顯存不是無限的——這正是為什麼會有`memory_budget`這個計算機:KV cache太大,直接壓縮「能同時服務幾個人」這個數字,顯存從來就是inference serving真正的稀缺資源,不是算力。
2. **以為batch size越大,continuous batching就一定線性提升吞吐量**:batch size提升確實能讓GPU的decode階段做更多平行工作,但KV cache會跟著batch size線性成長(每個並發request都要有自己的一份K/V),`memory_budget`算出來的`max_users_at_Xk`就是這條線的上限——超過這個數字,不是排程問題,是純粹放不下。
3. **把speculative decoding當成「犧牲一點準確率換速度」的近似方法**:它的accept/reject機制在數學上保證輸出跟target model單獨生成的分佈完全一致,不是近似——真正決定它值不值得用的,是draft model多快、接受率多高,而不是「要不要犧牲品質」這個假的取捨。
4. **看到`compare_speculation_strategies`回傳的`acceptance_rate`,直接當成draft model「猜對的機率」**:如上方「實作細節」所說,這個數字因為「一輪遇到第一次拒絕就提前結束」而系統性地低於真正的per-token接受機率,拿它去反推draft model品質會低估。
5. **以為`PrefixCache`就是PagedAttention**:兩者都跟「省掉重複計算/記憶體管理」有關,但`PrefixCache`解決的是「不同request共用前綴時,前綴的KV要不要重算」,PagedAttention解決的是「同一份KV cache本身要怎麼配置記憶體、減少碎片化」——是兩個互補但不同層次的問題,production系統(如SGLang)通常兩個一起用。

## 複習自問
- 為什麼`KVCache.advance()`一定要在每次forward之後、依實際處理的token數呼叫,而不能簡化成「每次decode呼叫時固定+1」?如果prefill之後忘記推進,下一次`update()`會實際造成什麼後果?
- Prefill是compute-bound、decode是memory-bound,這兩句話如果要落地成「該對哪個階段做什麼優化」,你會分別想到哪些具體技巧?為什麼同一個技巧(例如加大batch size)對兩個階段的效果不一樣?
- GQA把`num_kv_heads`從等於query head數降到遠小於它,對KV cache大小、模型品質(attention表達能力)分別是什麼方向的取捨?如果`num_kv_heads`降到1(MQA的極端版本),會發生什麼?
- Speculative decoding「數學上輸出分佈跟target model一致」這件事,具體是靠rejection sampling的哪一步保證的?如果拿掉「被拒絕時要用target分佈重新採樣」這一步,直接丟棄被拒絕的token,還能保證分佈一致嗎?
- 如果你的服務同時遇到「延遲要求很嚴格的即時對話」跟「可以等、但要求高吞吐量的批次摘要」兩種流量,你會如何用這篇裡的技巧(continuous batching的batch size、KV cache預算、要不要開speculative decoding)分別調整這兩種流量的服務策略?
