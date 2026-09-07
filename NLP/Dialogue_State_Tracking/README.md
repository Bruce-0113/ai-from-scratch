# Dialogue State Tracking（規則抽取 + 狀態更新 + LLM Structured Output + JGA評估）

對應程式：[`./dialogue_state_tracking.py`](./dialogue_state_tracking.py)

## TL;DR
「我要一間便宜的餐廳，在北區……啊不對，改成中等價位，然後加義大利菜」——三句話，三次狀態更新。Dialogue State Tracking（DST）要做的事，就是把使用者的訂位意圖，隨著每一輪對話維護成一個像`{cuisine: italian, area: north, price: moderate}`這樣的slot-value字典，且必須維護得完全正確，因為訂位系統最後是拿這個字典直接呼叫後端API——slot錯一個，訂到的就是錯的餐廳、錯的班機、扣錯的卡。這份程式用五個小區塊示範DST從最簡單到最現代的做法：用同義詞字典做規則式slot抽取、寫一個保留未提及slot、且能處理否定的狀態更新迴圈、用Pydantic schema約束LLM輸出讓模型每輪重新生成整個state、用Joint Goal Accuracy（JGA）評估「每個slot都對」的嚴格指標、以及用關鍵詞偵測使用者的更正（correction）意圖。

## 為什麼需要它
- Task-oriented對話系統的最終目的是呼叫後端API（訂餐廳、訂機票、辦銀行業務），API需要的是結構化參數而不是一段自然語言，DST正是「使用者說的話」跟「後端要執行的動作」之間的橋樑——這一步做錯，後面no matter接的是規則引擎還是LLM agent，結果都是錯的。
- 2026年即使LLM已經很強，DST仍然沒有消失：banking、醫療、航班訂位這類compliance-sensitive領域，法規要求slot值必須是deterministic、可稽核的，不能讓LLM自由生成；就算是LLM agent架構，呼叫工具（tool use）前也仍然需要先把slot resolve出來，才知道要塞什麼參數給API。
- 多輪對話裡的「更正」比看起來難處理：「actually，改成星期四」這句話必須理解成「覆蓋掉day這個slot」，而不是「新增一個day」或者被當成一句無關的閒聊忽略掉——append跟overwrite的判斷邏輯錯了，state就髒了。
- 現代作法是「經典DST概念＋LLM抽取器＋structured output防護欄」的混合：完全捨棄規則式方法在compliance場景行不通，完全依賴LLM生成又可能在敏感slot上出現無法預期的輸出，兩者的權衡正是這份程式想示範的重點。

## 核心原理

### 1. 規則式slot抽取（`CUISINE_SYNONYMS` / `extract_cuisine`）
最基本的作法：針對每個canonical slot值（如`italian`）維護一份同義詞清單（`italian`、`pasta`、`pizza`、`italy`都算），逐一檢查使用者的utterance裡有沒有出現這些關鍵詞，命中就回傳對應的canonical值。這種regex/關鍵詞比對在narrow domain（例如只做「訂義式餐廳」這種單一垂直服務）通常能覆蓋七成左右的典型說法，而且完全可debug——出錯時可以直接看是哪個詞沒被收錄，不像神經網路模型是個黑盒子。代價是「brittle」：只要使用者的講法不在同義詞清單裡（比如講「碳水炸彈」代稱義大利麵），就完全抓不到。

### 2. 狀態更新迴圈（`update_state`）
把新抽取到的slot值合併進現有state時，必須遵守三個不變量（invariant）：
- **沒被這句話提到的slot不能被清空**——`update_state`複製一份`state`後，只在抽取器回傳非`None`時才覆寫該slot，其餘slot維持原值。
- **明確的否定（negation）必須清空該slot**——例如「不用管cuisine了」——這在程式裡是`update_state`跑完一般抽取之後，再用`NEGATION_CLEARS`／`is_negated`額外掃一輪，把被否定的slot設回`None`。
- **使用者更正（correction，「actually……」）必須覆寫而非新增**——這件事本身在規則式架構下不容易做對（見第5節`is_correction`），是LLM-driven DST（第3節）想直接繞過的問題。

程式裡的`update_state`是示範用的框架程式碼：它引用的`SLOT_EXTRACTORS`（一個`{slot名稱: extractor函式}`的字典）、`NEGATION_CLEARS`（哪些slot允許被否定清空）、`is_negated`（判斷某個slot是否被否定的函式）都沒有在檔案裡定義，需要呼叫端自行組裝——重點是理解「先抽取覆寫、後否定清空」這個更新順序，而不是直接執行這段程式。

### 3. LLM-driven DST + Structured Output（`RestaurantState` / `llm_dst`）
用`pydantic.BaseModel`定義一個`RestaurantState` schema，把每個slot能取的值鎖死（`cuisine`只能是`italian`/`chinese`/`indian`/`thai`/`any`其中之一或`None`，`price`只能是`cheap`/`moderate`/`expensive`），再透過`instructor`把這個schema當成LLM呼叫的`response_model`，讓底層LLM API保證輸出的一定是通過validation的`RestaurantState`物件——不會有多餘欄位、不會有schema外的字串、不需要自己寫JSON parsing與例外處理。

這裡最關鍵的設計決定是：`llm_dst`每一輪都是把「完整對話歷史」丟給LLM、要求它重新生成整個state，而不是只丟「這一輪新增了什麼」讓LLM做增量更新。這個做法自然而然解決了append-vs-overwrite的模糊地帶跟correction的處理——LLM每次都是從頭看過整段對話重新判斷「現在的完整狀態應該是什麼」，不需要額外規則去區分「這是新增還是更正」。代價會在第7節的實作細節裡說明。

### 4. Joint Goal Accuracy評估（`joint_goal_accuracy`）
DST最標準的評估指標。對每一輪對話，只有當predicted state「整包」跟gold state完全相等（`p == g`），這一輪才算對；只要有一個slot錯，這一輪就算零分——JGA是all-or-nothing的指標，不是逐個slot分開算分後平均。這比單看「per-slot accuracy」嚴格得多：假設一個state有5個slot，即使模型4個都對、只錯1個，JGA上這一輪還是記為錯誤，不會拿到80%的部分分數。2026年MultiWOZ 2.4 leaderboard上頂尖系統大約落在80–83%左右，這代表就算是SOTA系統，平均每5輪對話仍有接近1輪會有某個slot出錯。

### 5. 更正線索偵測（`CORRECTION_CUES` / `is_correction`）
用一組固定短語（`actually`、`no wait`、`on second thought`、`change that to`）比對使用者最新的utterance，命中就代表這是一句「更正」——後續的狀態更新邏輯應該把這句話解讀成「覆寫某個slot」而不是「附加一筆新資訊」。這是一個純關鍵詞heuristic，覆蓋率有限（見常見誤區第2點），現代做法通常乾脆讓LLM對整段歷史重新生成state（第3節），這樣就不需要額外偵測「這句話是不是更正」，correction的處理隱含在「每次都重新推論完整state」這個策略裡。

## 程式碼導覽
| 函式 / 區塊 | 對應到理論的哪個部分 |
|---|---|
| `CUISINE_SYNONYMS` | 規則式slot抽取的同義詞字典：canonical值 → 同義詞清單 |
| `extract_cuisine(utterance)` | 逐一比對同義詞字典，回傳命中的canonical cuisine，或`None` |
| `update_state(state, utterance)` | 狀態更新迴圈：先用`SLOT_EXTRACTORS`覆寫有抽到值的slot，再用`NEGATION_CLEARS`／`is_negated`清空被否定的slot |
| `RestaurantState(BaseModel)` | LLM structured output的Pydantic schema：用`Literal`鎖住每個closed-vocabulary slot的合法值 |
| `llm_dst(history, llm)` | 把完整對話歷史丟給LLM，透過`instructor`的`response_model=RestaurantState`拿回驗證過的完整state |
| `joint_goal_accuracy(predicted_states, gold_states)` | JGA評估：逐輪比對predicted/gold state是否完全相等，算出「整輪全對」的比例 |
| `CORRECTION_CUES` / `is_correction(utterance)` | 更正線索的關鍵詞清單與偵測函式 |

**實作細節 / 容易看漏的地方：**
- `update_state`跟`llm_dst`都是示範用的框架程式碼（對應原始教材的示意片段），直接執行會丟出`NameError`：`update_state`用到的`SLOT_EXTRACTORS`、`NEGATION_CLEARS`、`is_negated`，以及`llm_dst`用到的`render`，都沒有在這個檔案裡定義，需要呼叫端自行提供。
- `extract_cuisine`的`CUISINE_SYNONYMS`只涵蓋`italian`跟`chinese`兩種，但`RestaurantState.cuisine`這個Pydantic schema卻宣告了`italian`/`chinese`/`indian`/`thai`/`any`五種合法值——這正是規則式抽取器最常見的維護問題：schema擴充了，同義詞字典卻沒有同步跟上，`indian`跟`thai`目前完全抽不出來，會被規則式pipeline遺漏。
- `joint_goal_accuracy`用`==`直接比較兩個state物件；如果`predicted_states`跟`gold_states`裡混用dict與`RestaurantState`實例，或兩邊字典的key集合不一致（例如一邊有`day: None`、另一邊完全沒有`day`這個key），`==`會判定不相等，即使「語意上」兩者代表相同的狀態——正式使用前務必統一predicted跟gold的state表示法。
- `joint_goal_accuracy`在`predicted_states`為空list時，`len(predicted_states)`是0，會丟出`ZeroDivisionError`，沒有額外的防呆判斷。
- `instructor`這個import在檔案裡沒有被直接呼叫——它的角色是patch底層LLM client（例如OpenAI/Anthropic的SDK client），讓該client支援`response_model`參數；`llm_dst`收到的`llm`參數，預期已經是這樣patch過的client，而不是`instructor`模組本身。
- `is_correction`的比對邏輯跟`extract_cuisine`一樣是不分大小寫的substring比對，所以`CORRECTION_CUES`裡任何一個短語只要以任何形式（含大小寫、含在更長的句子裡）出現在utterance中就會命中；但反過來說，沒收錄進`CORRECTION_CUES`的更正說法（例如單純講「不是，是禮拜四」而不講「actually」）就完全偵測不到。

## 使用時機 / 優缺點
2026年常見的決策對照表（依領域封閉程度、有無標註資料、合規要求挑方法）：

| 情境 | 建議做法 |
|---|---|
| 窄領域（一到兩個intent） | 規則式 + regex／同義詞字典 |
| 廣領域、有標註資料 | LDST（LLaMA + LoRA，MultiWOZ風格資料微調） |
| 廣領域、沒有標註資料、要上production | LLM + Instructor + Pydantic schema |
| 語音介面 | ASR + 正規化 + LLM-DST |
| 多領域訂位流程（餐廳+飯店+交通） | Schema-guided LLM，每個領域各自一個Pydantic model |
| Compliance-sensitive（銀行、醫療、航班） | 規則式為主，LLM作為fallback並加上人工確認流程 |

- ✅ 規則式抽取器完全deterministic、零延遲、零API成本、每個判斷都可以逐行debug，是compliance-sensitive領域的首選——法規通常要求slot值的產生過程可稽核、可重現。
- ✅ LLM + Pydantic schema（`RestaurantState` + `llm_dst`）的做法只需要幾行程式碼，就能同時處理open-vocabulary slot（人名、日期）、correction、跨輪coreference，這些規則式方法各自都需要額外邏輯才能處理的問題，LLM靠「重新生成整個state」一次解決。
- ✅ Instructor的`response_model`保證輸出一定通過schema validation，不會有無法解析的JSON、不會有schema外的幻覺欄位，比自己寫prompt要求「請輸出JSON」再手動parse可靠得多。
- ✅ Joint Goal Accuracy雖然嚴格，但正是這種「全對才算對」的特性，讓它能真實反映「這個state能不能安全拿去呼叫後端API」——後端API通常也是要嘛拿到完全正確的參數、要嘛整筆訂單都是錯的，沒有「部分正確」這種中間狀態。
- ❌ 規則式抽取器完全無法泛化到同義詞字典之外的講法（見實作細節裡`indian`/`thai`缺漏的例子），維護成本會隨著領域擴大而線性甚至更快增加。
- ❌ `llm_dst`這種「每輪都用完整歷史重新生成state」的模式，隨對話輪數增加，總token用量是O(n²)——第n輪要重新讀一次前面n-1輪的完整歷史，對話一長，成本與延遲都會顯著上升，需要靠限制歷史長度或摘要舊輪次來控制。
- ❌ JGA是turn-level的全有全無指標，看不出「哪一個slot最常出錯」——正式評估報告通常需要額外搭配slot-level precision/recall，才能定位問題出在哪個slot。
- ❌ 純keyword-based的correction偵測（`is_correction`）覆蓋率有限，遇到沒收錄的更正說法就會誤判成一般新增，這也是LLM-driven DST傾向乾脆放棄「偵測更正」、改成「每輪整包重新生成」的原因。

## 常見誤區
1. **只要換成LLM + Pydantic，就完全不用管state update的規則邏輯了**：在compliance-sensitive領域（銀行、醫療、航班訂位），法規通常要求關鍵slot必須走deterministic、可稽核的規則式流程，LLM即使準確率很高，也不能單獨作為唯一的決策來源——常見架構是「規則式為主、LLM作為fallback」而不是完全捨棄規則。
2. **`is_correction`這種關鍵詞偵測已經能可靠抓到所有更正意圖**：`CORRECTION_CUES`只收錄了幾個固定短語，任何沒有講出這些詞、但語意上是更正的說法（例如直接重新講一次新答案）都偵測不到；這正是為什麼現代做法傾向讓LLM對整段歷史重新推論state，而不是依賴一個關鍵詞清單去「偵測」更正。
3. **Joint Goal Accuracy高，代表每個slot個別的準確率也一定高**：JGA是「整輪全對才算對」的all-or-nothing指標，如果某個罕見slot經常出錯、但其他slot都很準，JGA仍然可能被拉低；反過來說，只看JGA也看不出「是哪一個slot在扯後腿」，需要額外拆解成slot-level的precision/recall才看得出來。
4. **讓LLM每輪重新生成整個state沒有額外成本**：這個做法雖然乾淨地解決了append/overwrite與correction的模糊地帶，但token成本會隨對話長度呈O(n²)成長——一段50輪的長對話，光是「重新讀歷史」這件事本身的總token用量就會遠高於「只處理新增的這一輪」，正式系統需要限制歷史窗口或做摘要壓縮。
5. **規則式抽取器裡的同義詞字典，會自動跟著Pydantic schema的合法值同步更新**：這是這份程式本身就示範出來的落差——`RestaurantState`宣告了5種cuisine，`CUISINE_SYNONYMS`卻只維護了2種；schema擴充跟同義詞字典維護是兩件需要分開追蹤的事，不會自動同步。

## 複習自問
- `update_state`為什麼要先做「抽取覆寫」、再做「否定清空」，而不是相反的順序？如果順序反過來，會出現什麼樣的bug？
- 為什麼`llm_dst`選擇讓LLM每輪重新生成「整個」state，而不是只生成「這一輪的更新」？這個設計決定解決了哪些問題，又付出了什麼代價？
- Joint Goal Accuracy跟「逐個slot分開算accuracy再平均」相比，兩者衡量的是不一樣的什麼？如果一個系統在10個slot裡有9個永遠對、1個永遠錯，這兩種指標分別會給出大約多少分？
- `extract_cuisine`目前只認得出`italian`跟`chinese`，但`RestaurantState`的schema允許`indian`／`thai`／`any`。如果不修改`CUISINE_SYNONYMS`就直接拿這個抽取器餵資料，會對後續的JGA評估結果造成什麼樣的偏誤？
- 為什麼在banking、醫療這類compliance-sensitive領域，即使LLM-driven DST的JGA分數比規則式方法高，仍然建議採用「規則式為主、LLM為輔」的架構，而不是直接all-in LLM？
- 承接原始教材的Exercise 3：如果要設計一個「規則式為主、LLM fallback」的路由邏輯，你會用什麼訊號（例如規則式抽取器一次抽到的slot數量、抽取器回傳的信心程度）來決定什麼時候該呼叫LLM？

## 延伸閱讀
- [Budzianowski et al. (2018). MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz Dataset for Task-Oriented Dialogue Modelling](https://arxiv.org/abs/1810.00278) — DST最經典的多領域benchmark資料集。
- [Rastogi et al. (2020). Towards Scalable Multi-Domain Conversational Agents: The Schema-Guided Dialogue Dataset](https://arxiv.org/abs/1909.05855) — SGD資料集，示範「schema-guided」、可泛化到未見過服務的ontology-free DST。
- [Heck et al. (2020). TripPy — A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://arxiv.org/abs/2005.02877) — pre-LLM時代的copy-based DST主流架構。
- [Feng et al. (2023). Towards LLM-driven Dialogue State Tracking (LDST)](https://arxiv.org/abs/2310.14970) — LLaMA + LoRA instruction tuning做DST。
- [King & Flanigan (2024). Unsupervised End-to-End Task-Oriented Dialogue with LLMs: The Power of the Noisy Channel](https://arxiv.org/abs/2404.10753) — 用EM演算法做無監督的task-oriented dialogue。
- [MultiWOZ leaderboard](https://github.com/budzianowski/multiwoz) — 標準DST結果彙整。
