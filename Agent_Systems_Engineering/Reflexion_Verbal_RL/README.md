# Reflexion — 不動權重的口語式強化學習（Verbal Reinforcement Learning）

對應程式: [`./reflexion_verbal_rl.py`](./reflexion_verbal_rl.py)

參考: [ai-engineering-from-scratch – Reflexion: Verbal Reinforcement Learning 課程](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/14-agent-engineering/03-reflexion-verbal-rl/docs/en.md)

前置概念: 課程列的前置課是Phase14·01(Agent Loop,ReAct風格的推理-行動-觀察交錯迴圈)與Phase14·02(ReWOO,把規劃跟執行拆成獨立角色)——這支程式假設你已經知道一個agent怎麼跑一輪推理、怎麼呼叫工具,以及「把角色拆開」本身的價值,重點放在更前面一步的問題:**同一個agent,失敗一次之後,要怎麼在完全不動任何權重的情況下,讓下一次嘗試變得更好。**跟本repo同屬Phase14的[`../Agent_Memory`](../Agent_Memory/README.md)(MemGPT風格的兩層記憶)雖然都叫「記憶」,處理的其實是不同維度的問題:Agent Memory解決的是「一個很長的任務裡,context window裝不下全部對話歷史,哪些該留、哪些該搬到外部儲存」的**空間**問題;這支程式的episodic memory解決的是「一個任務失敗之後,要記住『這次為什麼失敗、學到了什麼』,讓下一次重新嘗試時能參考」的**時間/學習**問題——兩者可以疊在同一個系統裡(失敗的反思本身也是一種要塞進或搬出context的資訊),但關注點不同。

## TL;DR
標準RL修一個失敗模式,要跑幾千次trial、算gradient、更新權重——貴、慢,而且大多數production agent根本沒有「每次失敗都重新訓練」的預算。Reflexion(Shinn et al., NeurIPS 2023 / arXiv:2303.11366)反過來問:如果agent失敗後只是**用自然語言想一下為什麼失敗**,把這段話存起來,下一次嘗試的prompt裡帶著它,不做任何梯度更新,會怎樣?論文在ALFWorld上贏過ReAct等未微調的baseline,在HotpotQA上優於ReAct,在HumanEval/MBPP程式碼生成上當時創下SOTA——全部都不需要一次梯度下降。本程式用純stdlib刻出四個角色:`Actor`(生成嘗試)、`binary_evaluator`(判斷成功與否,回傳有號距離)、`SelfReflector`(針對失敗寫一句診斷)、`EpisodicMemory`(有上限的FIFO緩衝區,存放歷次反思)。Demo任務很單純:從1~9挑三個整數,湊出和為20——`main()`把同一個任務跑兩次:`use_memory=False`的baseline(Actor永遠看不到任何反思,卡死在第一次的爛猜測)對照`use_memory=True`的Reflexion(Actor看到反思數量遞增,第3次收斂成功),兩條trace並排印出來,一眼看出「有沒有把反思餵回prompt」造成的行為差異。

## 為什麼需要它
- **驗證錯誤不需要梯度,只需要一句話**:`run_reflexion`裡`memory.add(Reflection(trial=t, text=text))`是整個demo的核心動作——把一次失敗變成一段可以在下一輪嘗試中被讀到的文字,不涉及任何模型參數的變化。這正是課程開頭那句話的具體實作:「gradient-based RL需要幾千次trial和一整個GPU cluster,Reflexion用自然語言在幾次trial內做到」。
- **這是一個被重複發明很多次的通用模式,不是一個孤立的技巧**:課程明確點名——Letta的sleep-time compute(把`SelfReflector`這一步搬到離峰時段跑,讓主線agent維持低延遲)、Claude Code自己的`CLAUDE.md`/「記住這件事」的記憶機制(把反思寫成可以被下一次session讀到的learnings)、pro-workflow的`/learn-rule`(把修正意見捕捉成明確規則)、LangGraph的reflection node(算分後決定要不要繞回去重做)——全部都是「用自然語言承載『從失敗中學到什麼』,跨run傳遞」這同一個洞見的變形。你現在讀的這個repo本身用來記錄User回饋的auto memory系統,結構上也是同一件事。
- **把「要不要相信這次的反思」跟「用什麼訊號判斷成功失敗」分開處理**:課程列出三種Evaluator——scalar(外部binary訊號,像這支程式的`binary_evaluator`)、heuristic(預先定義的失敗特徵,例如連續兩次選了同一個action)、self-evaluated(LLM自己幫自己打分,沒有ground truth時的次選方案)。這支程式只刻了最強訊號的scalar版本,但把它跟`SelfReflector`拆成獨立函式/類別,是為了讓heuristic或self-eval可以直接替換進來,不用動`run_reflexion`迴圈本身。

## 核心原理
- **四個角色/一個資料結構,一一對應到程式碼**:

  | 課程概念 | 本程式對應 |
  |---|---|
  | Actor | `Actor.act`——站在「LLM讀question+memory,生出下一次嘗試」這次呼叫的位置上,這裡是根據`len(memory.items)`回傳寫死的猜測 |
  | Evaluator(scalar類型) | `binary_evaluator`——回傳`(success, delta)`,`delta = sum(attempt) - target`是有號距離,不只是true/false |
  | Self-Reflector | `SelfReflector.reflect`——站在「LLM讀失敗的trajectory,寫一句自然語言診斷」這次呼叫的位置上,這裡依`delta`正負套一句寫死的模板 |
  | Episodic memory | `EpisodicMemory`(`items`+`max_len`的FIFO緩衝區)+`Reflection`(單筆`trial`/`text`紀錄) |
  | 把反思接回下一次prompt | `EpisodicMemory.as_prompt`——把`items`轉成條列字串,是「reflection prepended to the next trial's prompt」這句話的具體形狀 |
  | 一次完整的trial | `TrialResult`——把Actor的猜測、Evaluator的判定、Self-Reflector的診斷,包成一筆紀錄 |

- **Actor只讀反思的「數量」,不讀反思的「內容」**:`Actor.act`唯一用到的輸入是`len(memory.items)`,`Reflection.text`裡實際寫了什麼字,`act`完全沒有解析——`n==0`回傳`[1, 2, 3]`,`n==1`回傳`[5, 6, 7]`,`n>=2`回傳`[6, 7, 7]`。這是刻意的簡化:真正的Reflexion裡,LLM是因為**讀到**反思文字才調整行為;這支demo用「反思筆數」當替身指標,不用接LLM就能示範「memory變長,行為跟著變」這件事本身,但也因此,`EpisodicMemory.as_prompt()`這個方法在整支程式裡從未被呼叫——它是留給「如果Actor換成真正讀prompt的LLM」時要接上的那個掛勾,目前只是沒被用到的方法。
- **`use_memory=False`不是「不產生反思」,而是「Actor看不到反思」**:`run_reflexion`裡,無論`use_memory`是`True`還是`False`,`memory.add(...)`都會在每次失敗後執行,`memory`這個物件本身照樣累積反思;真正被`use_memory`開關的是傳給`actor.act(...)`的是哪個記憶——`True`時傳入會累積的那個`memory`,`False`時每一輪都重新蓋一個全新的`EpisodicMemory()`給它。也就是說baseline模式下反思仍然被寫下來,只是Actor每次拿到的都是一個剛蓋好、`len(items)==0`的空記憶,永遠等同於「第一次嘗試」——這才是baseline「Actor永遠學不會」的真正原因,不是記憶體被關掉了。
- **`Actor.act`裡`n>=3`的分支,在目前`main()`的參數下永遠不會被執行到**:因為`TARGET=20`固定,`n==2`那個分支的`[6, 7, 7]`本身就已經是成功解,`run_reflexion`一旦`success`就`break`,`n`永遠不可能在同一次呼叫裡走到3。這一行`return [6, 7, 7]`只是讓函式在假設性的「memory開著、卻連跑超過3次失敗」情境下也有定義良好的回傳值,是防呆,不是demo實際會走到的路徑。

## 程式碼導覽
| 函式/類別 | 對應到理論的哪個部分 |
|---|---|
| `Reflection` | 一筆反思:寫在哪個`trial`之後、內容`text`是什麼 |
| `EpisodicMemory` | 有`max_len`上限的FIFO反思緩衝區;`add`寫入並在超過上限時丟掉最舊的一筆,`as_prompt`轉成條列字串 |
| `Actor.act` | 站在「一次Actor LLM呼叫」的位置上,依`len(memory.items)`回傳寫死的猜測 |
| `binary_evaluator` | 課程三種Evaluator裡的scalar類型:回傳成功與否,以及有號距離`delta` |
| `SelfReflector.reflect` | 站在「一次Self-Reflector LLM呼叫」的位置上,依`delta`正負套用寫死的診斷模板 |
| `TrialResult` | 一次trial的完整紀錄:猜測、判定、診斷三者包在一起 |
| `run_reflexion` | 跑Actor→Evaluator→Self-Reflector的主迴圈,失敗就把反思寫入`memory`並重試,直到成功或`max_trials`用完 |
| `summarize` | 把一組`TrialResult`印成逐行trace,附上最終成功/失敗結論 |
| `main` | 用「三個1~9的整數湊成20」demo,並排跑`use_memory=False`與`use_memory=True`兩次,印出trial數對比 |

**實作細節 / 容易看漏的地方:**
- **`EpisodicMemory.as_prompt()`是目前整支程式裡唯一沒被任何呼叫路徑用到的方法**:`run_reflexion`和`main`都不呼叫它,`Actor.act`也只看`len(memory.items)`。它不是死code——它示範了「如果Actor是真正的LLM,memory該怎麼被序列化塞進prompt」——但在這支scripted demo的實際執行路徑裡完全不影響輸出,跟`Rewoo_Plan_and_Execute`裡`resolve_references`那條「找不到引用就原樣保留」的fallback是同一種性質:防呆/示範用途,不是active path。
- **`memory`物件的寫入跟Actor能不能看到它,是兩件被刻意分開的事**:讀`run_reflexion`容易誤以為`use_memory=False`等於「這次跑完全不建立`EpisodicMemory`」,但實際上`memory = EpisodicMemory()`在函式開頭就建立好、且不論開關與否都持續被`memory.add(...)`寫入——差別只在`actor.act(memory if use_memory else EpisodicMemory())`這一行,傳給Actor的是不是**同一個**物件。
- **這支demo只示範了三種Evaluator裡的一種**:課程提到scalar(ground truth pass/fail)、heuristic(預先定義的失敗特徵,例如連續動作重複)、self-evaluated(LLM自評,無ground truth時的次選方案)三種,`binary_evaluator`只實作了scalar這一種——heuristic和self-eval被留給Exercise 3自己動手加。
- **`max_len=6`這個上限,在demo的規模下從未被真正觸發**:`main()`兩次呼叫都是`max_trials=4`,`Reflexion`分支最多累積2筆反思(第3次trial就成功、迴圈提前`break`),遠低於`max_len=6`——這是課程提到「memory rot」問題(反思累積、部分過時)的其中一種緩解手段(TTL/定期壓縮),但這支demo的trial數太少,沒有機會真的把FIFO的丟棄行為秀出來。

## 使用時機 / 優缺點
依課程的判斷條件(不是「永遠比gradient-based RL好」):

Reflexion有效,當:
- 失敗訊號清楚(測試失敗、工具報錯、答案錯誤)
- 任務類型可重現(同一類問題可以再問一次)
- 反思還有改善空間(還有足夠的action budget)

Reflexion沒有幫助,當:
- Agent第一次就成功了
- 失敗來自外部因素(網路斷線、工具本身壞掉)——反思「網路斷線」對下一次沒有意義
- 反思變成迷信——把一次性的偶發flaky run編成一段敘事存起來

- ✅ 想搞懂Reflexion四個角色(Actor/Evaluator/Self-Reflector/Episodic memory)怎麼具體對應到程式碼:全部加起來不到130行,`main()`印出的兩條trace可以逐行對照「有沒有把反思餵回下一次嘗試」造成的差異。
- ✅ 需要一個不接LLM、100%可重現的骨架,拿來驗證「記憶如何影響下一步決策」這個結構本身,或當成寫單元測試/教學示範的起點。
- ✅ 想確認`use_memory`開關的精確語意(它關掉的是Actor的可見性,不是反思的產生),避免把這支demo的baseline誤讀成「完全沒有反思」。
- ❌ Actor是完全寫死的,而且只讀反思的**筆數**、不讀反思的**內容**:這支程式示範的是「有memory vs沒有memory」的行為差異,不是「LLM讀了反思文字之後,真的因為那段文字改變了策略」。
- ❌ 只實作了三種Evaluator裡的scalar一種:沒有heuristic(失敗特徵偵測)也沒有self-evaluated(LLM自評),兩者都留給讀者在Exercise裡自己加。
- ❌ 沒有處理課程點名的「memory rot」:`max_len`存在但demo的trial數太少從未真正觸發FIFO丟棄,也沒有TTL、去重、或sleep-time的清理agent。
- ❌ `EpisodicMemory.as_prompt()`目前是孤兒方法:接一個真正的LLM Actor進來之前,它不會被任何呼叫路徑用到。

## 常見誤區
1. **以為`use_memory=False`代表這次執行完全沒有建立或寫入`EpisodicMemory`**:實際上`memory`物件從函式一開始就建立,且不論`use_memory`為何都持續被`memory.add(...)`寫入——被開關控制的只是Actor能不能看到這個累積中的物件,還是每次都拿到一個全新的空殼。
2. **以為Actor是因為「讀懂了」反思的內容才調整猜測**:`Actor.act`唯一用到的輸入是`len(memory.items)`這個數字,`Reflection.text`裡寫了什麼字完全沒被程式邏輯讀取——那段文字只用來給人看(`summarize`的輸出),或是給一個真正的LLM Actor當prompt用(`as_prompt`),但目前的`Actor`不是那個真正的LLM。
3. **以為`Actor.act`裡`n>=2`跟`n>=3`是兩個真的會走到不同結果的分支**:兩者回傳完全相同的`[6, 7, 7]`,而且因為`TARGET=20`固定、`run_reflexion`成功就`break`,`n`在目前這支程式的任何呼叫路徑裡都不可能真的走到3——`n>=3`那行只是防呆,不是demo會展示的行為。
4. **以為這支程式示範了課程三種Evaluator的全部**:目前只有`binary_evaluator`(scalar類型)一種,heuristic和self-evaluated都沒有實作,對應課程Exercise 3的「加一個heuristic evaluator」還是留白的部分。
5. **以為`EpisodicMemory.max_len=6`這個上限在這支demo裡有實際作用**:兩次`main()`呼叫都只跑到最多4個trial、Reflexion分支最多累積2筆反思,遠低於6,FIFO的「丟掉最舊一筆」邏輯在這支demo的輸出裡從未被觸發過。

## 關鍵詞速查
| 詞彙 | 常聽到的說法 | 實際上是什麼 |
|---|---|---|
| Reflexion | 「自我修正」 | Shinn et al. 2023——Actor、Evaluator、Self-Reflector三角色加上episodic memory |
| Verbal reinforcement | 「不用梯度的學習」 | 用自然語言反思,接到下一次trial的prompt前面 |
| Episodic memory | 「每個任務的反思紀錄」 | 針對單一任務類型、有上限的反思緩衝區 |
| Scalar evaluator | 「binary成功訊號」 | 來自ground truth的pass/fail或數值分數 |
| Heuristic evaluator | 「規則型偵測器」 | 預先定義的失敗特徵(例如卡住重複同一動作、步數過多) |
| Self-evaluator | 「LLM自己當裁判」 | 沒有ground truth時的次選方案,訊號較弱,通常搭配工具驗證一起用 |
| Memory rot | 「反思過時」 | episodic buffer裡累積了過時或錯誤的反思,重跑會變慢;靠定期壓縮/TTL緩解 |
| Sleep-time reflection | 「離峰時段的非同步反思」 | 把Self-Reflector搬到主agent的hot path之外執行,維持主線的低延遲 |

## 延伸閱讀
- [Shinn et al., *Reflexion: Language Agents with Verbal Reinforcement Learning* (arXiv:2303.11366)](https://arxiv.org/abs/2303.11366):Reflexion的原始論文,本程式Actor/Evaluator/Self-Reflector三角色與ALFWorld/HotpotQA/HumanEval的結果都出自這裡。
- [Letta, Sleep-time Compute](https://www.letta.com/blog/sleep-time-compute):把Self-Reflector搬到離峰時段執行的production案例。
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents):把episodic buffer當成context管理的一部分來討論。
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview):reflection node模式在真正的框架裡怎麼實作。

## 複習自問
1. 課程Exercise 1要你把evaluator從binary換成回傳距離的scalar分數。`binary_evaluator`其實已經回傳`delta`這個有號距離,只是`run_reflexion`只用了`success`來決定要不要`break`——如果改成用`delta`的大小(而不是只有true/false)來影響`SelfReflector.reflect`寫出的診斷細節,或影響`run_reflexion`判斷「有沒有在往對的方向前進」,收斂速度會不會不一樣?
2. 課程Exercise 2要你幫反思加上10個trial的TTL。現在`EpisodicMemory.max_len`是靠筆數(FIFO)限制,不是靠`Reflection.trial`欄位判斷「這筆反思是幾個trial之前寫的」——`Reflection`已經記錄了`trial`,要怎麼利用這個欄位,在`add`或`as_prompt`裡把「超過10個trial前」的反思排除掉?
3. 課程Exercise 3要你實作一個heuristic evaluator:同一個action連續出現兩次就判定卡住。這個新的判定邏輯該插在`binary_evaluator`旁邊還是取代它?`SelfReflector.reflect`現在只處理「超過/不足/成功」三種情況,要怎麼加一種新的診斷模板來對應「卡住」這個新的失敗類型?
4. 課程Exercise 4要你跑一個「無視反思」的對抗性Actor,觀察最少要怎麼設計反思的prompt才能強迫Actor真的注意到它。但現在的`Actor.act`本來就只讀`len(memory.items)`、不讀`Reflection.text`——要讓「無視反思」這個實驗有意義,是不是要先把`Actor`換成一個真正會讀`EpisodicMemory.as_prompt()`輸出的版本,「無視」才會是一個相對於「有讀」的、有意義的對照組?
5. 課程Exercise 5要你讀論文Section 4關於ALFWorld的部分,想清楚Reflexion相對於原始ReAct的關鍵差異在哪裡,並嘗試從概念上重現130%的成功率提升。對照這支程式的結構——`run_reflexion`比起單純重複呼叫`Actor.act`的迴圈,多出來的`SelfReflector`與`EpisodicMemory`兩塊,如何具體對應到論文裡「ReAct+Reflexion」相對於「純ReAct」多出來的那一段機制?
