# Prompt Caching (Context Caching)

對應程式:[`./prompt_caching.py`](./prompt_caching.py)

## TL;DR
把「每次請求都重算一次」的穩定前綴(system prompt、工具定義、few-shot範例、檢索文件)換成「算一次、之後都用讀的」。Provider會把prefix對應的KV-cache存起來,下次遇到位元組完全相同的前綴就直接讀取,不用重跑attention——省的是成本(cache read通常只要原價的10%~50%)也省的是time-to-first-token。一個15,000 token的system prompt,一天重複用在10,000次對話,不做cache每天要燒約$9,000,做了之後多數請求只需要付一折的price。

## 為什麼需要它
- Agent loop、多輪對話、RAG都有同一個共同結構:一段很大但不常變的前綴(system prompt/工具列表/檢索文件),加上一段很小但每次都不一樣的後綴(使用者這一輪的問題)。沒有caching,每次都得把整個前綴連同後綴一起送進模型重新計算。
- Caching不是「幫你把回應存起來」——同樣的system prompt配上不同的user message,answer完全不同,cache救的是**輸入端的attention計算**,不是輸出。
- 三大provider都做,但機制、折扣、TTL完全不同,選錯model或選錯TTL,cache可能整組打不中。

## 核心原理
- **Caching是prefix match,不是「有沒有出現過這段文字」**:cache key來自「從開頭到`cache_control`斷點為止,每一個byte」。斷點之後的內容差多少都不影響——但斷點*之前*只要有一個byte不同(時間戳記、被打亂順序的JSON key、換了一個工具),斷點之後全部失效。
- **渲染順序是固定的:`tools` → `system` → `messages`**。這決定了「什麼東西該放前面」——工具定義幾乎不變,應該最早;system prompt次之;會話歷史、當前問題最容易變,應該放最後、放在最後一個斷點之後。
- **三家provider的機制對照**:

| | Anthropic (Claude) | OpenAI | Gemini |
|---|---|---|---|
| 觸發方式 | 顯式 `cache_control` 斷點 | 自動(≥1,024 token前綴自動比對) | 顯式,建立具名的 `CachedContent` 物件 |
| Cache read折扣 | ~90%(付原價0.1x) | ~50% | ~75%(依模型) |
| Cache write加價 | ~1.25x(5分鐘TTL)/ ~2x(1小時TTL) | 無額外加價 | 依儲存時間計費(storage billing) |
| TTL | 預設5分鐘,可延長到1小時 | 由系統決定,不可調 | 使用者自訂(範例:3600s) |
| 適合場景 | 單一對話/agent loop,同一前綴短時間內反覆讀 | 無狀態serverless端點,懶得管cache細節 | 同一份大corpus要在數小時到數天內被多次呼叫重用 |

- **Anthropic的breakeven計算**:5分鐘TTL下,寫入付1.25x、讀取付0.1x——兩次請求((1.25x + 0.1x) = 1.35x)就打平不cache的2x。1小時TTL寫入要付2x,至少要3次請求(2x + 0.2x = 2.2x < 3x)才划算,原因是換取的是「跨越5分鐘以上的空檔仍保持warm」。
- **驗證cache真的有打中,不能只看有沒有報錯**:回應的`usage`會回報`cache_creation_input_tokens`(這次寫入的量,付加價)、`cache_read_input_tokens`(這次命中讀取的量,付折扣價)、`input_tokens`(沒被cache覆蓋、全價付的剩餘量)。如果重複打同一個前綴,`cache_read_input_tokens`一直是0,代表有東西在悄悄讓prefix每次都不一樣。

## 程式碼導覽
| 函式 | 對應到理論的哪個部分 |
|---|---|
| `anthropic_explicit_cache_demo` | Anthropic的顯式breakpoint:`cache_control`放在system區塊,示範第一次呼叫是cache write(`cache_creation_input_tokens`),第二次呼叫同一前綴變成cache read(`cache_read_input_tokens`) |
| `anthropic_extended_ttl_block` | 同一組`cache_control`但改用`ttl: "1h"`,對應「請求間隔5-60分鐘該用哪種TTL」的決策 |
| `openai_automatic_cache_demo` | OpenAI的自動快取:程式碼裡完全沒有`cache_control`,折扣透過`usage.prompt_tokens_details.cached_tokens`回報,不是靠設定觸發 |
| `gemini_explicit_cache_demo` | Gemini的具名`CachedContent`:先呼叫`client.caches.create`建立一個有自己TTL的cache物件,之後每次生成都用`cached_content=cache.name`參照它,而不是每次都在請求裡塞`cache_control` |

**實作細節 / 容易看漏的地方:**
- 四個函式都是示範單一provider機制長什麼樣子的獨立片段,不是同一個pipeline的四個步驟;每個函式需要呼叫方傳入真正的prompt內容(`rubric`、`code_a`/`code_b`、`system_prompt`/`user_msg`、`few_shot_examples`),而且都需要對應provider的真實API key才能實際執行。
- `anthropic_explicit_cache_demo`裡`cache_control`放在**system區塊**而不是放在user訊息上——這是關鍵位置,因為review的code每次都不同(放在後綴),只有rubric是穩定前綴(該被斷點鎖住的部分)。
- `openai_automatic_cache_demo`拿到的`cached_tokens`只是「這次省了多少」的觀測值,程式碼本身不需要為了觸發cache而做任何事——這也是自動快取跟顯式breakpoint最大的差異:少了控制權,但也少了「忘記加cache_control」這種失誤。

## 使用時機 / 優缺點
- ✅ Agent loop、多輪對話,同一個大system prompt在幾分鐘內被反覆讀取:Anthropic預設5分鐘TTL,搭配自動的top-level `cache_control`處理成長中的對話尾巴。
- ✅ Batch工作或請求間隔拉長到5-60分鐘:改用Anthropic的1小時TTL,前提是同一前綴在這1小時內至少會被讀3次以上,否則加價划不來。
- ✅ Serverless/無狀態端點,不想管cache細節:OpenAI自動快取,只要prompt前綴穩定超過1,024 token就會自動生效。
- ✅ 同一份大corpus(法規文件、大型codebase context)要被數小時到數天內反覆呼叫:Gemini的具名`CachedContent`,一次建立多次重用。
- ❌ Prompt從一開始就每次都不一樣(前1K token就已經因人而異):不要加`cache_control`,只會白付write premium卻永遠讀不到。
- ❌ 需要在對話中途切換model或大幅調整工具列表:model切換、工具增刪都會讓整個cache失效且沒有escape hatch,設計時應該讓main loop固定用同一個model,有變動需求改用其他管道(例如把「模式」當成訊息內容傳遞,而不是換工具集)。

## 常見誤區
1. **把易變內容放在system prompt最前面**:在system prompt裡塞入`目前時間: ...`、`使用者ID: ...`這類每次都不同的欄位,會讓它後面的所有內容(包含原本很穩定的rubric、工具定義)全部无法cache,因為斷點前的任何一個byte改變就會讓斷點失效。
2. **以為斷點放在最後就好**:如果一個請求前綴裡有「大段共用內容 + 一小段每次都不同的問題」,把`cache_control`放在整個訊息的最後面,等於把斷點放在會變動的那一小段之後——結果是每次都在寫入新的cache entry,卻從來沒有機會讀到。斷點應該放在「共用部分」的結尾,而不是整個prompt的結尾。
3. **序列化不是deterministic**:`json.dumps()`沒有`sort_keys=True`、或是疊代一個`set`,都會讓同樣的邏輯內容序列化出不同的byte順序——結果是tools或context每次long得都不一樣,cache永遠對不上,而且這種bug不會報錯,只會讓`cache_read_input_tokens`一直是0。
4. **只看有沒有報錯,不檢查`usage`欄位**:Caching壞掉是「靜默」的失敗模式——請求照樣成功,只是帳單變貴,沒有任何錯誤訊息會提醒你。應該把「重複請求時`cache_read_input_tokens > 0`」當成一個持續監控或整合測試的斷言,而不是設定完就不管了。

## 複習自問
- 為什麼「渲染順序是`tools → system → messages`」這件事,會直接決定你該把哪些內容放在prompt最前面?如果把一段會變動的內容誤放進tools定義裡,會發生什麼事?
- Anthropic的5分鐘TTL跟1小時TTL,分別在什麼樣的「請求間隔」下才划算?如果請求間隔通常小於5分鐘,選1小時TTL會發生什麼問題?
- OpenAI的自動快取跟Anthropic的顯式`cache_control`,在「控制權」與「出錯風險」上分別犧牲/換到了什麼?
- 如果`cache_read_input_tokens`在重複請求中一直是0,你會依序檢查哪些地方?為什麼「diff兩次請求的完整JSON payload」是最終手段而不是第一步?
