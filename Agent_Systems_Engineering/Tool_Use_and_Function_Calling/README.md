# Tool Use and Function Calling — Schema 驗證、Argument Coercion 與 tool_use_id 關聯性

對應程式: [`./tool_use_and_function_calling.py`](./tool_use_and_function_calling.py)

參考: [ai-engineering-from-scratch – Tool Use and Function Calling 課程](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/14-agent-engineering/06-tool-use-and-function-calling/docs/en.md)

前置概念: 課程列的前置課是Phase14·01(Agent Loop,ReAct風格的agent控制迴圈)與Phase13·01(Function Calling Deep Dive,不在本repo收錄範圍內的更早期課程)——這支程式假設你已經知道agent怎麼跑一輪推理、什麼時候該發出一次tool call,重點放在再往前一步的問題:model吐出來的那個tool call,要怎麼變成一次真正、安全、可被驗證的function執行。跟本repo同屬Phase14的[`../Rewoo_Plan_and_Execute`](../Rewoo_Plan_and_Execute/README.md)裡也有一個`ToolRegistry.dispatch`,兩者共用同一個「查表呼叫、例外包成字串、絕不往外拋」的骨架,但ReWOO那支的呼叫方是一個完全寫死的`ScriptedPlanner`,永遠不會生出型別錯誤或不存在的工具名;這支程式反過來假設呼叫方是一個真正的、會犯錯的LLM,所以在同一個骨架上多疊了一層schema validation、type coercion,以及一份要進到模型prompt裡的`catalog()`。這支程式也是兩支已經寫好的lesson明確列出的前置概念:[`../Agent_Memory`](../Agent_Memory/README.md)(MemGPT記憶體工具)跟[`../Stateful_Graph_Orchestration`](../Stateful_Graph_Orchestration/README.md)(LangGraph狀態機)都假設你已經懂「agent呼叫工具的基本格式」——講的正是這支程式要示範的東西。

## TL;DR
Toolformer(Schick et al., NeurIPS 2023, arXiv:2302.04761)最早示範工具呼叫可以自監督學:只要一個候選API呼叫執行後能降低後續文字的next-token loss,就把它留在訓練語料裡當標註,完全不需要人工標記——論文也指出這個訊號是規模依賴的,小模型加了工具標註反而變差,大模型才會變好,這解釋了為什麼2026年的前沿模型工具呼叫能力普遍比多數7B模型強上一截。Berkeley Function Calling Leaderboard V4(Patil et al., ICML 2025)是建立在這條基線上的2026年評測標準,五個類別佔比40% agentic、30% multi-turn、10% live、10% non-live、10% hallucination,結論是single-turn的function calling已經接近solved,剩下的失敗集中在跨20步以上的長鏈呼叫、動態工具選擇、跨輪記憶,以及「該不該完全不呼叫任何工具」這幾個問題。這支程式不碰模型那一側,只刻model tool call落地前必須經過的那層管線:`ToolDef`把`name`/`description`/`input_schema`/`executor`包成一個工具,`_coerce`+`validate`實作JSON Schema一個子集(required、五種基本型別、enum、min/max)的檢查與型別修補,`ToolRegistry.catalog()`吐出跟Anthropic `input_schema`、OpenAI `function.parameters`同形狀的清單給模型看,`dispatch`/`dispatch_many`則把一次呼叫(或一批呼叫)變成`ToolResult`——不管是工具不存在、參數validation失敗、還是executor內部丟例外,一律包成一句可讀的錯誤字串,`tool_use_id`原封不動地繫在對應的結果上。`main()`一次示範五種call:乾淨的加法呼叫、需要把字串`"4"`修成整數`4`才能做乘法的呼叫、一個踩到enum之外的值(`"in_progress"`)因而被拒絕的呼叫、一個合法enum值通過的呼叫,以及一個呼叫了從未註冊過的工具`subtract`的呼叫——五個結果印出來,沒有一個是Python例外。

## 為什麼需要它
- **description是模型選工具的依據,不是給人看的註解**:`add`的description特別寫「Use for any integer addition」,`multiply`寫「Prefer multiplication over looped addition」——這兩句話會原封不動出現在`catalog()`回傳給模型的清單裡,是模型用來決定「這個任務該選哪個工具」的線索之一;課程的說法是「poor descriptions cause wrong-tool-picked failures most frequently」,`input_schema`寫得再精確,description含糊一樣容易選錯工具。
- **每一種失敗都必須是模型讀得懂的字串,不能是例外**:`dispatch`裡三個回傳失敗的位置——`call.name`查無此工具、`validate`回傳非空的`errors`、`tool.executor(**validated)`丟例外——全部被包成`ToolResult(ok=False, content="...")`,沒有一條路徑會讓例外真的往外傳。u03呼叫`classify(status="in_progress")`跟u05呼叫不存在的`subtract`,就是這兩種失敗實際印出來的樣子:模型理論上可以把這句錯誤訊息當成下一輪的observation重新嘗試,呼叫端也不需要另外包一層try/except去接`ToolRegistry.dispatch`丟出來的例外。
- **`tool_use_id`要在所有失敗路徑上都保持不變,批次dispatch才不會對錯結果**:`dispatch`三個失敗分支跟一個成功分支,全部用同一個`call.tool_use_id`建構`ToolResult`;`dispatch_many`只是把這個函式對一串call跑一輪。`main()`裡u01~u05一次送進去,裡面混著乾淨呼叫、enum錯誤、unknown tool錯誤,印出來的每一行`tool_use_id`都精準對回原本那個`ToolCall`,不會因為中間有呼叫失敗就把後面的id弄亂。

## 核心原理
- **課程概念對應到程式碼**:

  | 課程概念 | 本程式對應 |
  |---|---|
  | Tool schema(name/description/input_schema) | `ToolDef` + `ToolRegistry.catalog()`——吐出的shape跟Anthropic `input_schema`、OpenAI `function.parameters`一致 |
  | Argument validation checklist 的前三項(type coercion、enum validation、required fields) | `_coerce`+`validate`——數字字串會被coerce,enum不在清單直接拒絕,required缺漏直接報錯,三者的錯誤共用同一個`errors`清單 |
  | Argument validation checklist 的第四項(format validation:date/email/URL) | 沒有實作——module docstring開頭就寫明這是「required fields, string/int/number/bool/array/object, enum, minimum/maximum」這個子集,format validation不在範圍內 |
  | Parallel tool calls(model在一輪裡發出多個帶`tool_use_id`的call) | `ToolCall`/`ToolResult`/`dispatch_many`——id correlation被完整保留,但底層是循序的`list comprehension`,不是concurrent runtime(見下方實作細節) |
  | Sandboxing(每個工具的read/write範圍、network存取、timeout、memory上限) | `ToolDef.timeout_s`——只有這一個欄位存在,而且從未被`dispatch`/`dispatch_many`讀取或強制執行 |
  | Hallucination detection(BFCL V4類別:該拒絕呼叫任何工具時真的拒絕) | 沒有實作——`dispatch`對「工具不存在」(u05)有處理,但那是名稱層級的錯誤,不是「決定不呼叫任何工具」這個決策層級的行為(見使用時機) |

- **`_coerce`只做一種修補,而且是不對稱的**:數字型別(`integer`/`number`)碰到字串會嘗試`int(value)`/`float(value)`,失敗才報錯;但`boolean`沒有對應的字串修補路徑——`isinstance(value, bool)`檢查失敗就直接報`"expected boolean, got str"`,模型回傳字串`"true"`給boolean欄位的call會被拒絕,不會像數字字串那樣被coerce成`True`。同時因為Python的`bool`是`int`的subclass,`integer`/`number`兩個分支都先用`not isinstance(value, bool)`把`True`/`False`擋在外面,避免`True`被誤判成合法的`1`。
- **一次call裡,不同field的錯誤會全部累積,但同一個field只回報第一個踩到的檢查**:`validate`對每個field依序跑type coercion→enum→min/max,任何一步失敗就`continue`到下一個field,不會對同一個field疊加第二條錯誤訊息;但跨field之間`errors`是同一份list,一次call如果有兩個field各自出問題,`dispatch`回傳的`content`會用`"; "`把兩條訊息接在一起,模型一次就能看到全部問題,不必一次修一個、來回試好幾輪。
- **enum檢查在型別確認正確之後才會跑,不會幫忙修正**:u03的`classify(status="in_progress")`——`"in_progress"`本身是合法的字串型別,`_coerce`不會報錯,錯誤是後面`"enum" in prop and coerced not in prop["enum"]`這一步踩到的,回傳的訊息是`"status: 'in_progress' not in ['open', 'closed', 'pending']"`,程式不會嘗試猜測「你是不是想打pending」,只是原原本本地拒絕。
- **`dispatch_many`示範的是「一輪多個tool_use block」,不是「同時執行」**:`main()`印出的字樣是「parallel dispatch (5 calls in one turn)」,但`dispatch_many`的本體就是`[self.dispatch(c) for c in calls]`,一個一個循序跑完。課程原文把「model一輪送出多個call」跟「runtime怎麼執行它們(parallel if independent)」分成兩件獨立的事,這支程式只示範前者——底層要不要換成concurrent執行,是runtime自己的選擇,不影響`tool_use_id`correlation這個核心機制是否成立。

## 程式碼導覽
| 函式/類別 | 對應到理論的哪個部分 |
|---|---|
| `ToolDef` | 一個工具的完整宣告:`name`/`description`/`input_schema`/`executor`,外加一個從未被讀取的`timeout_s` |
| `ToolCall` | 一次模型發出的呼叫:`tool_use_id`+`name`+`args`(還沒被驗證/coerce過的原始參數) |
| `ToolResult` | 一次呼叫的結果:`tool_use_id`原封不動地帶回來,`ok`+`content`描述成功或失敗 |
| `_coerce` | 對單一value/schema pair做型別檢查跟數字字串的coercion,回傳`(value, error)` |
| `validate` | 對一次call的全部`args`做required/unknown field/coercion/enum/min-max檢查,回傳`(out, errors)` |
| `ToolRegistry.catalog` | 把每個已註冊工具的`name`/`description`/`input_schema`包成list,是模型實際會看到的東西 |
| `ToolRegistry.dispatch` | 查表→validate→呼叫executor,三種失敗全部包成字串,絕不往外拋例外 |
| `ToolRegistry.dispatch_many` | 對一串`ToolCall`跑`dispatch`,循序執行,`tool_use_id`保持對應 |
| `add`/`multiply`/`classify` | 三個toy executor,`description`刻意寫成「什麼時候該用」而不是「這是什麼」 |
| `main` | 註冊三個工具→印出`catalog()`→送出5個call(含一個unknown tool跟一個enum違規)→印出每個`ToolResult` |

**實作細節 / 容易看漏的地方:**
- **`ToolDef.timeout_s`只是宣告,整支程式沒有任何地方讀取它**:`dispatch`跟`dispatch_many`都不檢查時間、不設定deadline,`timeout_s=5.0`這個預設值目前純粹是文件用途——真正落地timeout加per-tool circuit breaker是課程Exercise 3要讀者自己補的功課,不是這支demo已經做到的sandboxing。
- **`dispatch_many`是循序`list comprehension`,`main()`裡「parallel dispatch」這句話講的是「模型一輪送出幾個call」,不是「runtime同時執行幾個call」**:兩者是獨立的事實。這支程式的id correlation設計(`ToolResult`永遠帶著原本的`tool_use_id`)本身就足以支援真正的並發dispatch(例如把`[self.dispatch(c) for c in calls]`換成`ThreadPoolExecutor.map`),但目前的實作沒有這麼做。
- **u05的「unknown tool」錯誤,示範的是名稱層級的容錯,不是BFCL V4的hallucination類別**:`dispatch`對`self._tools.get(call.name)`回傳`None`的處理,只是「模型說了一個從沒註冊過的工具名字」這種情況;BFCL V4的hallucination(10%)量的是另一件事——給模型一個「其實不該呼叫任何工具」的情境,看它會不會忍住不呼叫。這支程式完全沒有建模「不呼叫任何工具」這個分支,`ToolCall`永遠代表「模型已經決定呼叫某個工具」這件事已經發生了。
- **boolean欄位沒有字串coercion,這跟integer/number不對稱**:如果替某個工具設計一個`{"type": "boolean"}`的property,模型用JSON回傳原生的`true`/`false`會被Python的json解析器轉成`True`/`False`,可以直接通過`_coerce`;但如果模型(少數情況下)吐出字串`"true"`,`_coerce`會直接判定`"expected boolean, got str"`並拒絕,不會像`"4"` -> `4`那樣被修好。

## 使用時機 / 優缺點
| 情境 | 建議 |
|---|---|
| Provider的tool schema本身就有明確的typed properties/required/enum(Anthropic `input_schema`、OpenAI `function.parameters`) | 這支程式的`catalog`/`validate`形狀可以直接對應,不需要另外設計一層 |
| 需要偵測「這個情境其實不該呼叫任何工具」(BFCL V4 hallucination類別) | 這支demo沒有涵蓋,要照課程Exercise 1額外加一個no-op工具讓模型可以明確拒絕 |
| 工具本身有真正的read/write/network side effect,需要sandbox | `timeout_s`只是metadata,真正的執行邊界(timeout、circuit breaker)要照Exercise 3自己補上 |
| 要同時支援多個provider(Anthropic/OpenAI/Gemini/Bedrock) | 各家的tool schema形狀不完全一樣,需要額外一層translation adapter,這支demo只示範單一內部shape |

- ✅ 想搞懂tool schema(name/description/input_schema)怎麼具體對應成程式碼,並且看見「工具不存在」「validation失敗」「executor丟例外」三種失敗怎麼變成模型可以讀的字串,而不是Python例外——整支程式含`main()`不到220行。
- ✅ 需要一個不接LLM、100%可重現的骨架,驗證JSON Schema子集(required/五種型別/enum/min-max)的validate/coerce邏輯本身,或當成寫單元測試/教學示範的起點。
- ✅ 想直接看`tool_use_id`correlation在「一批call裡混著成功跟失敗」時如何仍然保持正確配對——`main()`的u01~u05一次示範五種不同結果,每一行都能對回原本的`ToolCall`。
- ❌ 沒有實作BFCL V4的hallucination類別要求的「決定不呼叫任何工具」:u05的「unknown tool」只是名稱層級的容錯,不是決策層級的拒絕呼叫。
- ❌ `dispatch_many`是循序執行,不是真的並發——「parallel dispatch」這個講法容易被誤解成執行時機,而不是「一輪送出多個call」這件事本身。
- ❌ `timeout_s`、read/write範圍、network存取等sandboxing相關的metadata都只是宣告,沒有任何強制執行邏輯。
- ❌ 沒有format validation(date/email/URL)——課程checklist的第四項,這支demo刻意只做最小子集,不是遺漏。

## 常見誤區
1. **以為main()印的「parallel dispatch」代表底層真的同時執行五個tool call**:`dispatch_many`其實是`[self.dispatch(c) for c in calls]`,一個一個循序跑完;「parallel」這裡對應的是課程說的「模型在同一輪assistant turn裡送出多個tool_use block」,不是runtime的執行時機。
2. **以為u05那個「unknown tool」錯誤就是BFCL V4測的hallucination能力**:那個類別測的是「模型該不該完全不呼叫任何工具」,u05測的是「模型呼叫了一個從未註冊過的工具名字」,兩者是完全不同層級的錯誤——這支程式沒有任何地方讓模型表達「這次不需要呼叫工具」。
3. **以為`_coerce`對所有型別都會嘗試修補**:只有數字字串(整數/浮點數)會被coerce,`boolean`沒有對應的字串修補路徑,`array`/`object`型別不對就直接拒絕,不會嘗試轉換或解析。
4. **以為`ToolDef.timeout_s`代表這支程式真的有超時保護**:它只是一個宣告的欄位,`dispatch`/`dispatch_many`完全沒有讀取或使用它;真正的timeout enforcement跟circuit breaker是課程Exercise 3要讀者自己加的功課。
5. **以為`validate`一次call只會回報一個錯誤**:如果一次call裡有兩個以上的field各自出問題,`errors`會把每個field(各自的第一個)錯誤都收集進同一個list,`dispatch`用`"; "`把它們接成一句話——一次observation可能包含好幾條錯誤訊息,不是只有一條。

## 關鍵詞速查
| 詞彙 | 常聽到的說法 | 實際上是什麼 |
|---|---|---|
| Function calling | 「Tool use」 | 有validated schema的structured-output工具呼叫,兩個詞在Anthropic/OpenAI的文件裡混用 |
| Toolformer | 「自監督工具標註」 | Schick et al. 2023——只留下能降低next-token loss的候選API call,不需要人工標記;工具使用能力隨模型規模浮現 |
| BFCL V4 | 「工具呼叫的benchmark」 | Berkeley Function Calling Leaderboard V4(Patil et al., ICML 2025)——40% agentic/30% multi-turn/10% live/10% non-live/10% hallucination |
| Tool schema | 「給模型看的函式簽名」 | `name`+`description`+JSON Schema(`properties`/`required`/`enum`/...);Anthropic用`input_schema`,OpenAI用`function.parameters`,兩者都是JSON Schema |
| tool_use_id | 「correlation ID」 | 把一次tool call跟它的result綁在一起;`dispatch`所有分支都保留它不變,是批次/並發dispatch時不能弄丟的那個id |
| Hallucination detection | 「知道什麼時候不要呼叫」 | BFCL V4的一個類別:該拒絕呼叫任何工具時真的拒絕;這支demo沒有實作,只有「工具名字不存在」這種名稱層級的容錯 |
| Argument coercion | 「字串轉數字的小修補」 | 針對可預期的schema不匹配做窄範圍修補(數字字串);含糊或型別完全不對的情況直接拒絕,不猜 |
| Sandboxing | 「工具執行的邊界」 | 每個工具各自的read/write範圍、network存取、timeout、memory上限;這支demo只留了`timeout_s`這個宣告,沒有實際圍欄 |

## 延伸閱讀
- [Schick et al., *Toolformer: Language Models Can Teach Themselves to Use Tools* (arXiv:2302.04761)](https://arxiv.org/abs/2302.04761):Toolformer原始論文,自監督工具標註的訓練訊號跟「工具使用隨規模浮現」的觀察都出自這裡。
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html):BFCL V4(Patil et al., ICML 2025)的官方leaderboard,agentic/multi-turn/live/non-live/hallucination五個類別的評測介面。
- [Anthropic, Tool use documentation](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview):`input_schema`/`tool_use`/`tool_result`/parallel tool use在Claude API裡的實際介面設計。
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/):`function_tool`型別與guardrails的官方文件,[`../Self_Refine_and_Critic`](../Self_Refine_and_Critic/README.md)的output guardrail小節也引用同一份文件。

## 複習自問
1. 課程Exercise 1要你加一個「no-op」工具,讓模型可以明確表示「這次不需要呼叫任何工具」,再拿類似BFCL hallucination的測試集量測效果。以現在`ToolCall`/`ToolRegistry`的介面來看,一個no-op工具要長什麼樣的`input_schema`?它跟u05示範的「呼叫了不存在的工具」錯誤路徑,在`dispatch`裡應該共用同一段程式碼,還是需要一條全新的分支?
2. 課程Exercise 2要你替int-as-string、float-as-string實作argument coercion,並指出「coercion從哪裡開始其實是在掩蓋真正的bug」。`_coerce`目前已經做了這兩種coercion——如果把coercion的範圍再放寬,比如允許`"4.0"`被coerce成`integer`(目前`int("4.0")`會丟`ValueError`而被拒絕),這樣的寬鬆會在什麼情境下讓一個真正的型別錯誤被誤判成合法輸入?
3. 課程Exercise 3要你替每個工具加上per-tool timeout,並在連續失敗3次後60秒內拒絕呼叫(circuit breaker),記錄這如何改變模型的recovery行為。`ToolDef.timeout_s`欄位已經存在但從未被讀取——要讓它生效,`dispatch`需要在呼叫`tool.executor`的地方加上什麼機制?circuit breaker的失敗計數要記在`ToolDef`上還是`ToolRegistry`上,才不會被下一次`register`同名工具時意外重置?
4. 課程Exercise 4要你讀BFCL V4的說明,挑一個類別(例如multi-turn)、跑10個範例prompt,回報通過率。以這支程式的`dispatch_many`為例,要模擬「multi-turn」情境,是不是要讓`ToolCall`的`args`可以引用前一輪`ToolResult.content`(類似[`../Rewoo_Plan_and_Execute`](../Rewoo_Plan_and_Execute/README.md)裡`#E<n>`那種evidence reference),而不是像現在`main()`裡五個call完全獨立、互不依賴?
5. 課程Exercise 5要你把這支stdlib validator換成Pydantic或Zod,並記錄兩者抓到了什麼toy實作沒抓到的問題。對照`_coerce`目前明確不做的事——array的item型別檢查(只確認是`list`,不檢查裡面每個元素的型別)、object的巢狀properties檢查(只確認是`dict`,不遞迴驗證裡面的欄位)、跟date/email/URL的format validation——換成Pydantic之後,這三類會不會是最先浮現出來的差距?
