# Self-Refine and CRITIC — 自我修正迴圈與外部驗證的對照

對應程式: [`./self_refine_and_critic.py`](./self_refine_and_critic.py)

參考: [ai-engineering-from-scratch – Self-Refine and CRITIC: Iterative Output Improvement 課程](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/14-agent-engineering/05-self-refine-and-critic/docs/en.md)

前置概念: 課程列的前置課是Phase14·01(Agent Loop,ReAct風格的推理-行動-觀察迴圈)與Phase14·03(Reflexion,失敗後寫一段自然語言反思、餵回下一次嘗試)——這支程式假設你已經知道agent怎麼跑一輪推理,重點放在同一個task內部要怎麼收斂:generate出一版輸出後,critique要從哪裡來(自己評自己,還是外部工具核對),refine又要怎麼利用這段critique。跟本repo同屬Phase14的[`../Reflexion_Verbal_RL`](../Reflexion_Verbal_RL/README.md)處理的是相鄰但不同維度的問題:Reflexion的反思是**跨trial**的——同一個任務失敗一次、整個trial結束,反思被存進episodic memory,下一次trial重新開始時才讀到;這支程式的generate→critique→refine是**同一個trial內部**的迴圈——同一個output被反覆修改,history是這次迴圈自己的產物,沒有獨立的trial邊界。另一個更直接的親戚是[`../Anthropic_Workflow_Patterns`](../Anthropic_Workflow_Patterns/README.md)裡的`evaluator_optimizer`,那支程式的README明講它是「Self-Refine的一般化版本」(`proposer`+`evaluator`,evaluator哪裡來、怎麼評完全由呼叫端決定);這支程式要示範的正是「`evaluator`那個位置放什麼東西會造成天壤之別的結果」——放一個自評的`feedback_self`,還是放一個查表比對的`verify_external`。

## TL;DR
Self-Refine(Madaan et al., NeurIPS 2023 / arXiv:2303.17651)讓同一個LLM身兼三個角色——generate產出、feedback批評、refine修正——在7個任務上平均拿到+20絕對分數的提升,完全不需要訓練或外部工具。但論文也點出它的罩門:LLM對自己輸出的**事實性**錯誤,自我批評並不可靠。CRITIC(Gou et al., arXiv:2305.11738)把feedback這一步換成外部工具驗證(搜尋、程式解譯器、計算機、測試),讓refine拿到的critique是有根據的,不是模型自己對自己打分。這支程式用純stdlib刻出兩條迴圈共用同一個`generate`:任務是產生一份3條bullet的摘要,每條不超過60字元,且不含`KNOWN_WRONG_FACTS`裡列的已知錯誤(巴黎是德國首都、聖母峰在歐洲、太陽繞著地球轉)。差別只在`run_loop`裡`use_critic`這個開關,餵給`generate`的critique文字要從哪裡來:`feedback_self`(自評)還是`verify_external`(查表)。實際跑一次`main()`,Self-Refine那條迴圈4次iteration印出**一模一樣**的critique文字(「first bullet reads wrong, double-check capital」)跟**一模一樣**的輸出,4次用完直接宣告`did not converge`;CRITIC那條迴圈每次critique文字都不同,第1次修正巴黎/德國、第2次修正聖母峰/歐洲,第3次verifier回傳`ok`,3次收斂成功。造成這個差異的不是「哪個critique講得比較對」——`feedback_self`其實正確指出了德國那個錯誤——而是`generate`裡接手critique的那段程式碼,只認得critique文字裡有沒有字面出現`"germany"`或`"everest"`這兩個關鍵字,`verify_external`的用詞剛好踩中,`feedback_self`的用詞剛好踩不中。

## 為什麼需要它
- **critique要「可執行」,不是「正確」就夠**:`feedback_self`回傳的「first bullet reads wrong, double-check capital」在語意上完全正確,但`generate`裡負責把critique轉成下一版輸出的邏輯,只用`"germany" in last.critique.lower()`這種字面比對來決定要修哪個事實——一句寫得再正確的人話,只要沒有精準命中關鍵字,refine就無法採取行動。這正是CRITIC論文想解決的另一半問題:不只是「判斷本身準不準」,「判斷產出的形式能不能被下一步消化」同樣重要。
- **`KNOWN_WRONG_FACTS`不是靠關鍵字比對逐條核對的**:`verify_external`裡`for fact in KNOWN_WRONG_FACTS`這個迴圈,迴圈本體完全沒有用到`fact`或算出來的`key`——三次迭代做的是同一組寫死的`if`判斷(巴黎/德國、聖母峰/歐洲),跟`fact`實際指到哪一條字串無關。這代表清單裡第三條「the sun orbits the earth」從頭到尾不會被檢查到,`verify_external`目前只驗證了前兩條。
- **self-refine在這支demo裡不是「收斂得比較慢」,是結構性卡死**:因為`generate`在critique不含關鍵字時`return history[-1].output`(原樣不變),而`feedback_self`的兩句寫死模板都不含「germany」/「everest」,所以只要第一版輸出踩到Germany錯誤,`feedback_self`就會一路回傳同一句critique、`generate`就會一路回傳同一份輸出,直到`max_iters`用完——這不是機率性的「有時候學不會」,是給定這組寫死文字後100%可重現的死結。

## 核心原理
- **generate/feedback/verify/refine四個角色,對應到程式碼**:

  | 課程概念 | 本程式對應 |
  |---|---|
  | Generate | `generate`——站在「LLM讀topic+history生出下一版output」這次呼叫的位置上,第一次沒有history時回傳寫死的、有兩個事實錯誤的版本 |
  | Feedback(Self-Refine版,自評) | `feedback_self`——依序檢查Germany/Paris、Europe/Everest兩種已知錯誤pattern,回傳`(critique文字, 是否通過)` |
  | Verify(CRITIC版,外部工具) | `verify_external`——比對`KNOWN_WRONG_FACTS`、檢查bullet數量、檢查每行長度,回傳`(critique文字, 是否通過)` |
  | Refine | `refine`——名義上接收`prev`、`critique`兩個參數,但函式本體完全沒用到它們,直接呼叫`generate(topic, history)`,真正驅動下一版輸出的是`generate`內部對`history[-1].critique`的關鍵字比對 |
  | 一次完整的attempt | `Attempt`——`iteration`/`output`/`critique`/`verified`四個欄位包成一筆紀錄 |
  | generate→verify→refine主迴圈 | `run_loop`——`use_critic`開關決定`verify`要綁`feedback_self`還是`verify_external`,`verify`回傳`ok=True`就`break`,否則呼叫`refine`拿下一版`output`再繼續 |

- **`refine`是一層沒有實際作用的轉發**:簽章上`refine(topic, prev, critique, history)`看起來會用到`prev`跟`critique`,但函式本體是`return generate(topic, history)`——完全沒碰到這兩個參數。真正「讀critique、決定怎麼改」的邏輯,其實寫在`generate`裡對`history[-1].critique`的字串比對,不是在`refine`裡。`refine`這個名字掛在一個空殼函式上,實際工作被`generate`身兼二職做掉了。
- **`generate`只讀`history`的最後一筆,不是整段歷史序列**:`last = history[-1]`是`generate`唯一讀取`history`的地方——跟Self-Refine論文強調「refine prompt要看到完整history,拿掉歷史品質會明顯下降」的重點不完全對應,這支demo的決策只看「最新一筆critique裡有沒有出現關鍵字」,不管更早的critique寫了什麼。
- **`generate`的三個分支彼此互斥,且順序有意義**:`if not history`(第一次呼叫)→`if "germany" in ...`(優先修Germany)→`if "everest" in ...`(修Everest,且同時把第三條bullet的用詞從「Water boils at 100C」改成「Water boils at 100C at sea level」,即使這句本來就沒有錯)→都不中就原樣傳回。這代表就算verifier一次抓到兩個問題,`generate`一次也只會修一個——CRITIC那條迴圈能在3次內收斂,靠的是verifier剛好按照「Germany修好後才輪到抓Everest」這個順序逐一回報,不是一次性告訴refine全部問題。
- **`run_loop`裡`verify = verify_external if use_critic else (lambda o: feedback_self(o))`的lambda包裝是多餘的**:`feedback_self`的簽章`(output: str) -> tuple[str, bool]`跟`verify_external`完全一樣,直接寫`verify = verify_external if use_critic else feedback_self`效果相同,目前的lambda只是多一層沒有必要的間接呼叫。

## 程式碼導覽
| 函式/類別 | 對應到理論的哪個部分 |
|---|---|
| `KNOWN_WRONG_FACTS` | CRITIC查核用的參考資料;目前只有前兩條會被`verify_external`實際使用到 |
| `Attempt` | 一次iteration的完整紀錄:第幾輪、輸出、critique文字、是否通過 |
| `generate` | 站在「一次generate LLM呼叫」的位置上;首輪回傳寫死的錯誤版本,之後依`history[-1].critique`裡有沒有出現"germany"/"everest"決定要不要修、修哪一條 |
| `feedback_self` | Self-Refine的self-critique角色:純看`output`字串本身抓已知錯誤pattern,不查任何外部資料 |
| `verify_external` | CRITIC的verifier角色:對照`KNOWN_WRONG_FACTS`、驗bullet數量、驗每行長度,三種檢查任一失敗就打回 |
| `refine` | 名義上的refine角色,實際上是`generate(topic, history)`的轉發,不使用`prev`/`critique`參數 |
| `run_loop` | 主迴圈:`generate`出第一版→迴圈裡`verify`→通過就`break`,不通過就`refine`拿下一版,直到`max_iters`用完 |
| `print_run` | 把一組`Attempt`印成逐行trace,`OK`/`...`標示是否通過 |
| `main` | 並排跑`use_critic=False`(Self-Refine)與`use_critic=True`(CRITIC)兩次,印出trace跟收斂與否的總結 |

**實作細節 / 容易看漏的地方:**
- **`KNOWN_WRONG_FACTS`第三條從未被檢查**:`for fact in KNOWN_WRONG_FACTS`的迴圈體完全不依賴`fact`(算出來的`key`也沒被用到),三次迭代做的是同一組寫死判斷——`"the sun orbits the earth"`這條清單項目實質上是死資料,是這支demo最容易被誤讀成「已涵蓋全部已知錯誤」的地方。
- **第3條bullet在Everest修正分支裡被順手改了用詞,但跟「修正錯誤」無關**:`generate`修Everest的那個分支,把「Water boils at 100C」換成「Water boils at 100C at sea level」——這句話本來就不在`KNOWN_WRONG_FACTS`裡,兩種寫法對`verify_external`而言都算通過,這只是demo文字上的巧合變化,不是為了修任何已知錯誤。
- **`run_loop`最後一輪(`i=max_iters`)裡,最後一次`refine`呼叫的結果會被算出來但從未使用**:第4輪`verify`判定不通過後,程式依然會執行一次`output = refine(...)`算出第5版output,但迴圈接下來因為`range(1, max_iters+1)`用盡而直接結束——這個剛算出來的新版本從未被`verify`,也不會進入`history`,單純是白算的一次呼叫,不影響已經印出的trace(`print_run`只印`history`裡記錄的4筆)。

## 使用時機 / 優缺點
依課程給的判準:

| 情境 | 建議模式 |
|---|---|
| 有明確的外部驗證管道(單元測試、型別檢查器、linter、事實查核API) | CRITIC——critique有根據,refine才有東西可以真的修 |
| 沒有外部驗證管道的任務(創意寫作、格式、語氣) | Self-Refine仍然有用——CRITIC在這種情境會退化成Self-Refine本身 |
| 兩個角色用同一個prompt風格互評 | 兩者都要避免——容易變成「rubber-stamp」,critique永遠說沒問題 |

- ✅ 想搞懂generate/feedback/refine三角色怎麼具體對應到程式碼,並且用肉眼比較「自評」跟「外部驗證」在**同一組輸入**下的行為差異:整支程式加`main()`不到130行,兩條trace一次印出來,不用讀論文就能看出差別在哪裡。
- ✅ 需要一個不接LLM、100%可重現的骨架,拿來驗證「critique的可執行性(actionability)比critique的正確性更關鍵」這個論點,或當成寫單元測試/教學示範的起點。
- ✅ 想直觀感受CRITIC論文的核心主張——不是「外部工具比LLM聰明」,而是「外部工具的輸出格式剛好是下一步能直接消化的形式」:`verify_external`的critique文字被刻意設計成包含`generate`要找的關鍵字,`feedback_self`則刻意寫成人話,兩者的正確性其實相當,差別只在能不能被下游採取行動。
- ❌ `generate`跟`refine`是完全寫死的關鍵字比對,不是真正讀懂critique語意的LLM——這支demo示範的是「行為的形狀」,不是「LLM真的會怎麼處理一句自然語言critique」。
- ❌ `verify_external`只驗證了`KNOWN_WRONG_FACTS`裡的前兩條,第三條「太陽繞地球轉」從未被檢查到(見上方實作細節)——不要把這支demo的驗證覆蓋率當成CRITIC模式本身的限制,這是這支demo程式碼本身的一個遺漏。
- ❌ 一次只修一個事實:即使verifier一次能看出兩個問題,`generate`的分支設計也只會挑其中一個修,不是真的「一次性修完所有已知問題」。

## 常見誤區
1. **以為Self-Refine在這支demo裡「學得比較慢」,最終還是會收斂**:實際上給定這組寫死的文字,Self-Refine這條路徑100%卡死——`feedback_self`的兩句模板都不含字面的`"germany"`/`"everest"`,`generate`在critique不含這兩個詞時會原樣傳回上一版output,4次iteration後必定以`did not converge`結束,不是機率性的、也不是「多給幾次iteration就會好」。
2. **以為CRITIC贏過Self-Refine是因為`verify_external`「判斷得更準」**:兩者對同一版output的判斷其實一致——`feedback_self`同樣正確指出了Germany錯誤。差別在`verify_external`的critique文字字面上包含了`generate`要匹配的關鍵字,`feedback_self`的用詞沒有,是「格式能不能被下游採取行動」的差別,不是「誰更懂事實」的差別。
3. **以為`KNOWN_WRONG_FACTS`裡列的三條已知錯誤,`verify_external`都會逐一核對**:`for fact in KNOWN_WRONG_FACTS`這個迴圈體完全不用`fact`,三次迭代做的是同一組寫死判斷,第三條「the sun orbits the earth」從來不會被檢查到。
4. **以為`refine(topic, prev, critique, history)`真的用`prev`和`critique`這兩個參數來決定怎麼改**:函式本體是`return generate(topic, history)`,這兩個參數完全沒被用到——真正讀取critique字面內容的邏輯,寫在`generate`內部對`history[-1].critique`的比對裡。
5. **以為`run_loop`每一次`refine`產生的output都會被記錄進`history`**:最後一輪(`i=max_iters`)例外——`verify`判定不通過後依然會呼叫一次`refine`算出下一版`output`,但迴圈接下來因為`range`用盡而結束,這個剛算出來的新版本從未被`verify`、也從未進入`history`,單純是白算的一次呼叫,不影響已經印出的trace。

## 關鍵詞速查
| 詞彙 | 常聽到的說法 | 實際上是什麼 |
|---|---|---|
| Self-Refine | 「LLM自己修自己」 | Madaan et al. 2023——generate/feedback/refine三步驟,同一模型,靠history避免重蹈覆轍 |
| CRITIC | 「工具輔助驗證」 | Gou et al. 2023——把feedback換成外部工具(搜尋/程式解譯器/計算機/測試) |
| Evaluator-Optimizer | 「Anthropic工作流模式」 | 評審與優化兩角色循環直到評審通過;Self-Refine在Anthropic框架下的命名與一般化版本 |
| Output guardrail | 「事後檢查」 | OpenAI Agents SDK在agent輸出後跑的驗證器,可以是CRITIC式(呼叫工具)或Self-Refine式(純函式) |
| Actionable critique | 「講得對不代表能被執行」 | 這支程式的核心示範:`feedback_self`語意正確卻因為用詞不含關鍵字而無法驅動`generate`改寫,`verify_external`的用詞剛好可以 |
| Rubber-stamp loop | 「自我認證陷阱」 | 同prompt風格的自評容易收斂成「看起來沒問題」,要靠結構不同的prompt或外部grounding打破 |
| Refine history | 「上一步做了什麼」 | `generate`只讀`history[-1].critique`,不是把完整history全部讀入——只看最新一筆 |

## 延伸閱讀
- [Madaan et al., *Self-Refine: Iterative Refinement with Self-Feedback* (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651):Self-Refine的原始論文,generate/feedback/refine三角色與7個任務+20分的結果都出自這裡。
- [Gou et al., *CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing* (arXiv:2305.11738)](https://arxiv.org/abs/2305.11738):CRITIC的原始論文,三種驗證工具類別的分類出自Section 3。
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents):evaluator-optimizer工作流模式的原始出處,`../Anthropic_Workflow_Patterns`的`evaluator_optimizer`實作也引用同一篇。
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/):output guardrails的官方文件,`OutputGuardrailTripwireTriggered`等CRITIC式驗證器的介面設計。

## 複習自問
1. 課程Exercise 1要你把`max_iterations`設成1,觀察CRITIC是否還有幫助。以現在`run_loop`的結構來看,`max_iters=1`時迴圈只跑一輪就結束——`verify_external`第一次一定抓到Germany錯誤(`ok=False`),`refine`會算出下一版,但迴圈已經用盡,這個改進過的版本永遠不會被verify也不會被印出來。這樣的`max_iters=1`還算不算「CRITIC真的幫上忙」?要讓1次iteration也能看出CRITIC的價值,`run_loop`或`print_run`要怎麼改,才能把「被discard掉的最後一次refine結果」也納入判斷?
2. 課程Exercise 2要你把外部verifier換成一個有30%機率誤判的noisy版本。`verify_external`目前是100%決定性的(同一個output永遠得到同一個判定)——要模擬noisy verifier,是要在`verify_external`內部加一個隨機數,還是在`run_loop`傳入`verify`函式的這一層包一個「以30%機率翻轉`ok`」的wrapper?如果`ok`被錯誤翻轉成`True`,`history`最後一筆會記錄一個「verified」但其實還有已知錯誤的輸出,這跟`print_run`目前只看`verified`欄位來判斷「passed」的邏輯,會不會需要一起改?
3. 課程Exercise 3要你做一個「大模型生成、小模型評審」的變體,比較是否贏過同模型。以這支程式的介面來看,`generate`跟`feedback_self`/`verify_external`目前都是各自獨立的函式,沒有共享任何「模型」狀態——要模擬「不同模型」的差異,是不是要讓`feedback_self`換成一個故意比`generate`更保守/更容易漏判的版本(例如只檢查Germany、不檢查Everest),藉此觀察「評審能力比生成能力弱」時,`run_loop`還能不能收斂?
4. 課程Exercise 4要你讀CRITIC論文Section 3,指出三種驗證工具類別(搜尋引擎、程式解譯器、計算機/領域驗證器)並各舉一個例子。對照`verify_external`——它目前只做到「查表比對」(比較接近搜尋引擎類別裡最簡化的版本)跟「格式檢查」(bullet數量、行長度,比較接近領域驗證器),完全沒有涉及程式解譯器或計算機類別——要替這支demo新增一個計算機類別的verifier,可以加在什麼樣的任務上(例如讓某個bullet宣稱一個算式的答案,用`eval`或`ast`核對)?
5. 課程Exercise 5要你把OpenAI Agents SDK的`output_guardrails`對應到CRITIC的verifier角色,並指出SDK哪裡做對了、哪裡沒做到。對照這支程式,`verify_external`回傳`(critique文字, bool)`的形狀,跟guardrail「驗證失敗就丟`OutputGuardrailTripwireTriggered`」的形狀不完全一樣——`run_loop`目前是「驗證不過就靜默重試」,沒有SDK那種「頂多重試N次、超過就中斷並往外拋例外」的機制。要把這支程式改成guardrail風格,`run_loop`結束時(`max_iters`用完仍未`ok`)要不要改成拋例外,而不是像現在這樣直接把`history`原樣回傳給呼叫端自己判斷?
