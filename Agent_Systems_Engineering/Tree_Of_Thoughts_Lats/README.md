# Tree of Thoughts + LATS — 把推理當成搜尋,而不是一條直線

對應程式: [`./tree_of_thoughts_lats.py`](./tree_of_thoughts_lats.py)

參考: [ai-engineering-from-scratch – Tree of Thoughts and LATS: Deliberate Search 課程](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/14-agent-engineering/04-tree-of-thoughts-lats/docs/en.md)

前置概念: 課程列的前置課是Phase14·01(Agent Loop,ReAct風格的推理-行動-觀察交錯迴圈)與Phase14·03([`../Reflexion_Verbal_RL`](../Reflexion_Verbal_RL/README.md),失敗後寫一段自然語言反思、不動權重就讓下一次嘗試變好)——這支程式假設你已經知道一個agent怎麼跑一輪推理、怎麼呼叫工具,以及「失敗之後可以怎麼修正」,重點放在更前面一步的問題:**如果連「要不要相信這一步」都可能是錯的,agent要怎麼在多條路徑之間搜尋、比較、回頭,而不是被鎖死在一條線性的思路上。** LATS(本程式的第二個演算法)本身就是把ToT、ReAct、Reflexion三者統一在同一個MCTS迴圈底下的產物,所以讀完這支程式,前面三堂課的角色會在同一支code裡再看到一次。

## TL;DR
Chain-of-thought是一條直線:只要某一步想錯,後面全部跟著錯,而且沒有回頭路。Tree of Thoughts(Yao et al., NeurIPS 2023, arXiv:2305.10601)把推理攤開成一棵樹——每個節點是一個「想法(thought)」,展開成多個候選,一個value function幫每個候選打分,再用BFS/DFS/beam search探索分數高的分支。論文數字很誇張:Game of 24從GPT-4 CoT的4%準確率,靠ToT衝到74%。LATS(Zhou et al., ICML 2024, arXiv:2310.04406)接著把ToT、ReAct、Reflexion三個角色統一進Monte Carlo Tree Search——Policy(提議下一步,ReAct風格)、Value function(幫partial trajectory打分,ToT風格)、Self-Reflector(對失敗寫診斷,Reflexion風格,這支程式沒刻)——四階段跑Select→Expand→Simulate→Backpropagate,論文在HumanEval上報出92.7% pass@1,逼近當時的SOTA,一樣不需要梯度更新。本程式用純stdlib在同一個玩具任務(Game of 24:給定`[4, 6, 4, 1]`,用`+ - * /`湊出`24`)上刻出`tot_bfs`(ToT的beam-flavored BFS)和`mcts`(LATS的MCTS迴圈),共用同一組`expand`(想法生成)與`value`(打分)函式,讓兩種搜尋策略的差異只來自「怎麼探索」,不是「怎麼評分」。

## 為什麼需要它
- **單一trajectory一旦錯,沒有回頭路,而搜尋有**:CoT的核心限制是它把每一步都焊死在前一步上,一旦某個中間結論錯了,模型沒有機制發現「這條路走錯了,換一條」。`tot_bfs`每一層都同時展開整個frontier、幫每個候選打分,只留分數最高的幾個繼續走——這就是「探索多條路徑、比較、放棄弱的」這個能力的最小實作。
- **LATS不是又發明一個新演算法,是把已經學過的三個角色接進同一個搜尋迴圈**:課程明確點名——`mcts`裡的Select步驟本質上就是ReAct的「決定下一步」,`value`函式本質上就是ToT的self-evaluation,而`simulate`的隨機rollout加上backprop,是「跑一次、看結果、更新信念」這個Reflexion式回饋迴圈的搜尋版本。理解LATS,等於把前面三堂課的積木重新組裝一次。
- **搜尋要花的token是CoT的100-1000倍,所以「什麼時候值得用」跟「演算法怎麼寫」一樣重要**:`main()`最後印出的`Cost: ToT uses 100-1000x the tokens of CoT. Use with intent.`不是裝飾——這支程式的`tot_bfs`光是三層就跑了152次expansion、`mcts`跑了80個iteration就產生286次expansion,對照CoT只需要一次forward pass。課程的判斷準則(見下方「使用時機」)比演算法本身更值得記住。
- **一個看似合理的value function,可能系統性地騙過兩種搜尋演算法**:這是這支程式跑起來之後才會發現、而不是讀論文能看到的事——見下方「實作細節」,`value`對未完成狀態的打分方式,會讓ToT跟LATS在這個具體demo裡都錯過真正的解答,原因跟樹狀搜尋本身無關,純粹是評分函式的盲點。

## 核心原理
- **四個角色/兩種搜尋策略,對應到程式碼**:

  | 課程概念 | 本程式對應 |
  |---|---|
  | Policy(提議下一步,ReAct風格) | `expand`——列出所有合法的下一步(不是抽樣K個,是全部合法組合),對每一對剩餘數字套用四種運算 |
  | Value function(對partial trajectory打分,ToT風格self-evaluation) | `value`——依「剩餘數字裡最接近TARGET的距離」給分,終局狀態額外判斷是否精確命中 |
  | Self-Reflector(對失敗寫自然語言診斷,Reflexion風格) | 沒有實作——本程式標題就寫明"no LLM",這一角色留白,見下方優缺點 |
  | ToT的BFS/beam search | `tot_bfs`——每層展開整個frontier、打分、只留`max_expansions_per_level`個繼續下一層 |
  | LATS的MCTS四階段 | `select`(Select,獨立版本,未被呼叫)/`mcts`內建的while迴圈(實際被使用的Select)、`expand`(Expand)、`simulate`(Simulate)、`backprop`(Backpropagate) |
  | UCT(Upper Confidence bound for Trees) | `uct`——`child.q`(exploitation)加上`c * sqrt(log(parent.visits)/child.visits)`(exploration) |

- **`tot_bfs`跟`mcts`共用同一組`expand`/`value`,差異只在「怎麼探索」**:這是刻意的設計——如果兩種演算法各自帶一套打分邏輯,就沒辦法乾淨地比較「BFS式的beam剪枝」跟「MCTS式的UCT+隨機rollout」誰在同一個問題上表現更好。程式碼層面,這也是為什麼`value`函式的任何盲點都會同時影響兩種演算法(見下面「實作細節」)。
- **`expand`裡只算`a >= b`的減法與除法,從來不算反過來的順序**:因為`state`每次都重新排序成由大到小,`itertools.combinations`給出的`i < j`必然對應`state[i] >= state[j]`,所以`a - b`與`a / b`永遠是唯一被嘗試的順序,`b - a`與`b / a`從未出現過。這是真實的搜尋空間限制:某些Game of 24的合法解需要一個負的中間值(例如`4 - (4 * (1 - 6))`裡的`1 - 6`),在這支程式的搜尋空間裡永遠不可達——但對`[4, 6, 4, 1]`這組數字來說,保序的解依然存在(下面會證明),所以這個限制沒有讓demo變成無解,只是讓搜尋空間比完整版Game of 24小一些。
- **`NUMBERS`裡有兩個`4`,`expand`不會去重,兩個索引位置產生的相同數值會變成兩個獨立節點**:例如把`6`分別跟「第一個4」和「第二個4」組合,都會產生一個`state`完全相同的`6*4=24`節點,但它們是兩個不同的`Node`物件,各自佔用`tot_bfs`beam裡的一個名額。這不是bug,但是理解下面「實作細節」裡beam為什麼會被塞滿的關鍵前提。

## 程式碼導覽
| 函式/類別 | 對應到理論的哪個部分 |
|---|---|
| `Node` | 樹上一個狀態:剩餘數字`state`、到這裡的操作紀錄`trace`,加上MCTS要用的`visits`/`value_sum`/`children`,`q`是平均回饋 |
| `evaluate` | 對兩個數字套用一個運算子,除以零回傳`None`讓呼叫端跳過這個候選 |
| `expand` | Policy步驟:列出所有合法的下一步(所有位置配對 × 四種運算) |
| `value` | Value function:對終局狀態判斷是否精確命中,對未完成狀態用「剩餘數字裡誰離24最近」打分 |
| `tot_bfs` | ToT的beam-flavored BFS:展開整層、打分、剪枝到`max_expansions_per_level`,重複`max_depth`次 |
| `uct` | UCT公式:未訪問過的子節點回傳無限大,強迫至少被選過一次 |
| `select` | MCTS Select階段的獨立實作,**未被`mcts`呼叫**(見下方實作細節) |
| `simulate` | Simulate階段:從某節點隨機rollout到深度上限,回傳終點的`value` |
| `backprop` | Backpropagate階段:把reward往回加到路徑上每個節點的`visits`/`value_sum` |
| `mcts` | LATS主迴圈:每次iteration做一次Select→Expand→Simulate→Backpropagate |
| `_all_leaves` | 收集樹上所有目前還沒展開過的節點(葉節點),供`mcts`挑選最終答案 |
| `main` | 在同一組數字上分別跑`tot_bfs`與`mcts`,印出兩條trace與論文對照數字 |

**實作細節 / 容易看漏的地方(以下皆為實際執行`python tree_of_thoughts_lats.py`驗證過的行為,不是推測):**

- **這支demo跑出來的兩個答案,其實都沒有真正湊出24——原因出在`value`的評分方式,不是搜尋演算法本身**:`value`對未完成狀態的打分是`-best_distance/100`,其中`best_distance`是「剩餘數字裡離24最近的那一個」,而不是「這個狀態是否還能被合法地湊成24」。這代表只要狀態裡剛好還留著一個等於24的殘值(例如`6*4=24`之後剩下的`(24, 4, 1)`),它就會被打成`0.0`分——跟一個真正只差一步就完成的節點同分,即使剩下的`4`和`1`不管怎麼運算組合進去,除非用到`+0`或`*1`這種恆等運算,否則結果一定會偏離24。
- **ToT的beam因此被「看起來已經解決」的假分支塞滿,真正能走到24的分支被剪掉**:實際展開第一層,`6*4=24`(留下`24,4,1`)打分`0.000`,是最高分,而真正能走向解答的`6+1=7`(留下`7,4,4`)打分`-0.170`,排名第7,只是勉強擠進`max_expansions_per_level=8`的beam。但因為`NUMBERS`有兩個`4`,`6*4=24`這個節點被複製了兩份,展開到第二層時,光是這兩份`6*4=24`底下「把剩下的`4`跟`1`做四則運算」的8種組合就佔滿了整個beam寬度——`6+1=7`那條真正的解答路徑(`6+1=7, 7*4=28, 28-4=24`,已用窮舉搜尋驗證存在且可達)連進入第二層beam的機會都沒有。`tot_bfs`最後只能在`(27,)`(`value=-0.030`)收尾,不是24。
- **LATS回報的「最佳」答案,狀態其實還沒湊完,只是看起來像**:`main()`跑出的LATS最終葉節點停在`(24, 4)`兩個數字,`trace`是`['6*4=24', '24*1=24']`,`value=0.000`——但這不是終局(終局要求`len(state)==1`),而`24`和`4`的四種組合(`28`、`20`、`96`、`6`)沒有一個等於24,這條路徑已經走進死巷,只是`value`函式看不出來。
- **`mcts`最終選答案的方式,是直接對每個葉節點重新打`value`分,不是看MCTS學到的統計量**:`best_leaf = max(_all_leaves(root), key=value, default=root)`用的是`value`這個「當下狀態」的symbolic分數,不是`node.q`(平均backprop回饋)或`node.visits`(被造訪次數)——這跟教科書AlphaZero式MCTS「選最常被造訪的分支」不一樣。結果就是一個只被隨機rollout幸運造訪過一次的葉節點,可以贏過一個被反覆驗證、visits很高的節點,這也是為什麼上面那個`(24, 4)`死巷會被選為「最佳答案」。
- **`select`函式定義了卻從未被呼叫**:`mcts`需要完整的root-to-leaf路徑才能做`backprop`,所以它把跟`select`一模一樣的while迴圈直接寫在`mcts`內部,而不是呼叫`select`(因為`select`只回傳葉節點,不回傳路徑)。`select`被保留下來,單純是為了讓讀者能看到「Select階段」被抽出來、不摻雜路徑記錄邏輯的乾淨版本——它不影響`main()`的任何輸出。
- **`mcts`裡`children[0]`是刻意選第一個,不是隨機也不是挑分數最高的**:一個葉節點第二次被選中(`visits > 0`)且還有超過一個數字剩下時,才會呼叫`expand`產生它的所有子節點,而`mcts`只對`children[0]`(依`itertools.combinations`與`OPS`固定順序產生的第一個)做這次的simulate/backprop,其餘手足節點維持`visits=0`,等之後某次UCT選擇因為`float("inf")`而被挑到時才會真正被造訪。這是標準MCTS「一次只深入一個新分支」的做法,只是「選哪一個新分支」在這裡是固定的,不是隨機或按分數排序。

## 使用時機 / 優缺點
依課程的判斷條件(不是「搜尋永遠比CoT好」):

搜尋值得用,當:
- 單一trajectory已經證實會失敗的複雜推理任務
- 有便宜、可靠的value function(單元測試、明確的目標值)
- 正確性比wall-clock時間重要

搜尋沒有幫助,當:
- 單一trajectory加上一個雜訊很大的evaluator——這種組合下,搜尋反而更容易找到「評分很高但答案是錯的」的節點,而不是真正正確的答案(這支程式的`value`盲點正是這個問題的具體案例)
- 任務本身簡單到CoT就能穩定答對
- 2026年的現實是:大部分production agent用的是「ReAct + 工具驗證」,不是LATS——搜尋主要出現在特定場景:有測試回饋的coding、deep-research式的多查詢探索、LangGraph的subgraph規劃模式

- ✅ 想搞懂ToT的beam search跟LATS的MCTS(Select/Expand/Simulate/Backpropagate)具體長什麼樣子,而且要能在同一組`expand`/`value`上直接比較兩種策略的行為差異。
- ✅ 想要一個不接LLM、100%可重現的骨架,拿來驗證「一個評分函式的盲點,會怎麼同時破壞BFS跟MCTS兩種搜尋」這個現象本身——這支程式意外地是個很好的反面教材。
- ✅ 需要先把UCT公式、Select/Expand/Simulate/Backpropagate四階段、beam剪枝這些機制的最小實作跑過一次,再去讀真正接LLM的LATS實作。
- ❌ 沒有真的接LLM,`Self-Reflector`角色(LATS三角色之一)完全沒有實作——這支程式只示範Policy+Value兩個角色,失敗後不會產生自然語言反思。
- ❌ `value`函式的評分方式(剩餘數字裡有沒有已經等於TARGET的)是這支demo特有的簡化,會被「殘值恰好等於目標」的假陽性騙到——真正的LLM self-evaluation通常是對「這個部分解還有沒有機會湊出答案」做判斷,不會只看數字上是否巧合相等。
- ❌ `expand`只嘗試`a >= b`的減法/除法順序,結構上無法探索需要負中間值的解法,雖然對這支程式的具體輸入不影響可解性,但不是完整的Game of 24搜尋空間。

## 常見誤區
1. **以為`main()`印出的ToT/LATS結果就是24,因為`value`顯示`0.000`看起來像「完美」**:LATS那條`value=0.000`是`(24, 4)`兩個數字的未完成狀態,不是終局;ToT那條真正跑到終局`(27,)`,但`value=-0.030`,已經明確標示沒有命中——兩者都不是24,原因見「實作細節」。
2. **以為搜尋演算法(beam剪枝、UCT)本身有問題,才會找不到解**:用窮舉、不剪枝的BFS重新展開同一個問題可以找到真正的解(`6+1=7, 7*4=28, 28-4=24`),證明問題不在`tot_bfs`或`mcts`的搜尋邏輯,而在`value`函式對「未完成狀態」的打分方式,會系統性地讓假陽性(殘值巧合等於24)贏過真正通往解答的分支。
3. **以為`select`函式是`mcts`實際呼叫的Select實作**:`mcts`為了同時記錄root-to-leaf路徑,把同一段邏輯直接內聯在自己的迴圈裡,`select`函式本身在`main()`的任何呼叫路徑裡都不會被執行到。
4. **以為`expand`會嘗試兩個數字所有可能的運算順序**:因為`state`永遠維持由大到小排序,`a - b`與`a / b`只會用`a >= b`的順序計算,`b - a`、`b / a`不存在於搜尋空間裡。
5. **以為`NUMBERS`裡的兩個`4`只會被當成一個數字處理**:`expand`按索引位置配對,不按數值去重,所以任何跟「6」或「1」搭配的運算都會因為兩個`4`各自產生一份,實際上讓某些分數的節點在beam/樹裡被複製,擠壓其他分支的空間。

## 關鍵詞速查
| 詞彙 | 常聽到的說法 | 實際上是什麼 |
|---|---|---|
| Tree of Thoughts (ToT) | 「讓LLM想多條路」 | Yao et al. 2023——把推理攤成樹,每個節點是一個thought,配合BFS/DFS/beam search與self-evaluation |
| LATS | 「MCTS版的agent」 | Zhou et al. 2024——把ToT(Value)、ReAct(Policy)、Reflexion(Self-Reflector)統一進MCTS的Select/Expand/Simulate/Backpropagate四階段 |
| UCT | 「探索/利用的平衡公式」 | `Q值 + c * sqrt(log(父節點visits)/子節點visits)`,c越大越鼓勵造訪次數少的節點 |
| Value function / self-evaluation | 「幫每個想法打分」 | 對partial trajectory評分,分類(sure/likely/impossible)、數值評分、或投票都算,這支程式用的是symbolic距離評分 |
| Beam search / beam width | 「只留下最好的幾個」 | 每層只保留分數前N名繼續展開,對應這支程式的`max_expansions_per_level` |
| Rollout / simulate | 「模擬跑到底看結果」 | MCTS的Simulate階段,從某節點隨機走到終局(或深度上限)取得一個reward估計 |
| Backpropagate | 「把結果往回傳」 | 把simulate得到的reward往回加到root-to-leaf路徑上每個節點的統計量 |

## 延伸閱讀
- [Yao et al., *Tree of Thoughts: Deliberate Problem Solving with Large Language Models* (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601):ToT原始論文,Game of 24的4%→74%數字出自這裡。
- [Zhou et al., *Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models* (arXiv:2310.04406)](https://arxiv.org/abs/2310.04406):LATS原始論文,HumanEval 92.7% pass@1與WebShop 75.9%的數字出自這裡。
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview):subgraph模式在真正框架裡怎麼實作搜尋式的規劃。
- [AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131):用可程式化的evaluator做演化式搜尋,是「搜尋 + 明確可驗證的評分」這條路線更進一步的案例。

## 複習自問
1. 課程Exercise 1要你用UCT的`c=0.1`跟`c=2.0`分別跑一次,觀察trace的差異。把`main()`裡`uct`的預設值`c=1.4`改掉重跑,搜尋路徑會偏向哪一種節點(造訪次數少的,還是Q值高的)?這跟「實作細節」提到的`children[0]`固定選擇方式,兩者對探索廣度的影響是疊加還是互相抵銷?
2. 課程Exercise 2要你對value function加上隨機jitter,測試MCTS對雜訊的穩健度。如果對`value`回傳值加一點隨機擾動,`(24, 4, 1)`這種「殘值巧合等於24」的假陽性節點,分數會不會因此不再穩定贏過真正通往解答的分支?這是否代表jitter反而能修正這支程式目前的評分盲點?
3. 課程Exercise 3要你在token預算有限的情況下,比較beam search式ToT跟純BFS的表現。這支程式的`tot_bfs`本身已經是beam-flavored(每層剪到`max_expansions_per_level`),如果把它換成不剪枝的窮舉BFS(就像本README「常見誤區」第2點驗證用的那段),多花的expansion數量會是多少倍?
4. 「實作細節」指出`value`只看「殘值是否恰好等於目標」,不看「這個狀態是否還能被合法湊成目標」。如果要修正這個盲點,`value`需要額外知道什麼資訊(例如剩餘數字的個數與它們能不能透過恆等運算把殘值原封不動地帶到終局)?這個修正該放進`value`本身,還是應該變成`expand`要主動排除的候選?
5. 課程Exercise 5要你讀LATS論文Section 5.1,回推HumanEval的rollout次數,並嘗試從概念上重現92.7% pass@1的結果。對照這支程式——如果把`Self-Reflector`角色(目前完全沒實作)接進`mcts`的Backpropagate階段,讓每次失敗的rollout多寫一句自然語言診斷、影響下一次`simulate`的行為,這會最先改善demo裡的哪一個具體問題:ToT的beam被假陽性塞滿,還是LATS最終選答案時只看`value`不看`visits`?
