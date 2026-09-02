# Natural Language Inference (NLI)

對應程式：[`./NLI.py`](./NLI.py)

## TL;DR
NLI（Natural Language Inference）在做一件事：給定premise（`t`）和hypothesis（`h`），判斷`h`相對於`t`是entailment（蘊含）、contradiction（矛盾）還是neutral（無關）。這個三分類任務表面上很學術，但其實是「A有沒有支持/反駁B」這種判斷的通用引擎——摘要有沒有幻覺、RAG答案有沒有根據、文件屬於哪個類別，全部都能化簡成一次NLI推論。

## 為什麼需要它
- 你做了一個摘要器，摘要生出來了，但你怎麼知道它沒有胡編？把原文當premise、摘要當hypothesis跑一次NLI，不是entailment就是幻覺。
- 你做了一個RAG chatbot，它回答了「是」，你怎麼知道這個答案有被檢索到的文件support？premise是檢索到的context、hypothesis是生成的答案，not entailment就代表答案沒有根據。
- 你有一萬篇新聞要分類，但沒有訓練標籤。zero-shot classification把每個候選類別包裝成一句hypothesis（「這篇文章是關於{label}」），entailment分數最高的類別就是預測結果——不用訓練、不用fine-tune。
- 這三個場景背後是同一顆模型（`facebook/bart-large-mnli`），這也是為什麼幾乎所有RAG評估框架（RAGAS、DeepEval）底層都藏著一個NLI模型。

## 核心原理
- **三個標籤的定義**：entailment是「`t`成立的話`h`一定成立」（"貓在沙發上睡覺"蘊含"房間裡有貓"）；contradiction是「`t`成立的話`h`一定不成立」；neutral是兩者互不相干，`t`既不支持也不反駁`h`。
- **這不是嚴格邏輯蘊含**：NLI要判斷的是「一般人讀完`t`會不會合理推論出`h`」，不是形式邏輯的必然性。例如"John walked his dog"在NLI裡會被判成entail"John has a dog"，但嚴格邏輯上除非額外公理化「擁有」的定義，否則推不出來。
- **架構**：模型是transformer encoder（這裡用的是BART），輸入`[CLS] premise [SEP] hypothesis [SEP]`，`[CLS]`的表徵接一個3-way softmax，在MultiNLI等資料集上訓練，得到「兩段文字之間關係」的分類器。
- **Zero-shot classification的機制**：把每個候選標籤代入模板（預設是「This example is about {label}.」）變成一句hypothesis，對每個候選各跑一次entailment分數，取分數最高的當預測類別——本質上是把「分類」問題重寫成「entailment」問題，借用NLI模型的能力而不需要重新訓練分類頭。
- **RAG faithfulness check的機制**（對應`is_faithful`）：把檢索到的context當premise、生成的answer當hypothesis，entailment分數超過threshold才算「有根據」；分數低代表答案內容context裡找不到支持，也就是幻覺。這正是RAGAS faithfulness指標的核心：把答案拆成atomic claims，逐一對context跑NLI，統計entail的比例。

## 程式碼導覽
| 程式碼 | 對應到理論的哪個部分 |
|---|---|
| `pipeline("text-classification", model="facebook/bart-large-mnli", top_k=None)` | 建立NLI分類器；`top_k=None`代表回傳全部三個標籤的分數，不只是機率最高的那一個 |
| `nli({"text": premise, "text_pair": hypothesis})` | 直接的entailment判斷：`text`是premise、`text_pair`是hypothesis，回傳三個標籤（entailment/neutral/contradiction）各自的分數 |
| `pipeline("zero-shot-classification", model="facebook/bart-large-mnli")` | 建立zero-shot分類器；底層仍是同一顆NLI模型，只是pipeline會自動把每個候選標籤代入hypothesis模板 |
| `zs(text, candidate_labels=labels)` | 對`text`與每個候選標籤各跑一次entailment，回傳依分數排序後的`labels`與`scores` |
| `is_faithful(answer, context, threshold=0.5)` | RAG faithfulness check：`context`當premise、`answer`當hypothesis，取entailment分數與`threshold`比較，回傳answer是否被context「撐住」 |

**實作細節 / 容易看漏的地方：**
- `top_k=None`取代了已棄用的`return_all_scores=True`，兩者效果相同——回傳全部標籤而非只有top-1。
- `text`與`text_pair`的順序不能顛倒：NLI不是對稱關係，`t`蘊含`h`不代表`h`蘊含`t`（"貓在沙發上"蘊含"房間有貓"，反過來不成立），所以`is_faithful`裡一定是context在前（premise）、answer在後（hypothesis）。
- `is_faithful`用`next(...)`從三個標籤裡挑出`entailment`那一筆，因為pipeline回傳的順序不保證固定，不能直接假設entailment分數在`result[0]`。
- zero-shot classification的hypothesis模板是可以自訂的（`hypothesis_template`參數），這份程式用的是預設模板；模板措辭會顯著影響準確率，是常見的調參項。

## 使用時機 / 優缺點
- ✅ 需要判斷「兩段文字之間的邏輯關係」（摘要是否忠於原文、答案是否有根據），直接用NLI而不用另外呼叫一次LLM去問「這個對不對」——更便宜、更快、結果可重現。
- ✅ 沒有訓練標籤但需要文字分類，zero-shot classification可以先當baseline，不用等標註資料到位。
- ✅ RAG系統的faithfulness監控，`is_faithful`這種pattern可以直接接進evaluation pipeline，對每個生成的claim逐一檢查。
- ❌ premise是長文件（多段落）時，句子級NLI模型的準確率會明顯下降，需要換成document-level NLI模型（如DocNLI訓練出來的模型），不能直接套用這裡的`bart-large-mnli`。
- ❌ zero-shot classification的候選標籤語意如果彼此重疊或太抽象（例如"good"、"positive"），entailment判斷會不穩定，效果不如有標籤資料時做fine-tune。
- ❌ 特定領域（法律、醫療）的文字，通用MNLI模型的判斷不可靠，需要領域專用的NLI模型（如MedNLI、SciNLI）。

## 常見誤區
1. **NLI能完全消除幻覺，不只是降低幻覺**：`is_faithful`只是一層過濾，entailment分數本身也可能誤判（尤其是claim很長、包含多個子陳述時）。NLI降低幻覺風險，不代表它是100%可靠的真相檢測器。
2. **hypothesis-only shortcut**：模型有時候光看hypothesis就能猜對標籤（例如"沒有"、"從不"這類詞常跟contradiction掛勾），這代表訓練資料裡有label leakage，不是模型真的在做entailment推理——這是評估NLI模型時要特別檢查的陷阱。
3. **zero-shot分類的模板措辭會大幅影響準確率**：「This example is about {label}.」跟直接用「{label}」在同一組資料上可能有10個百分點以上的差距，換模型或換任務時應該重新測過，不能假設預設模板永遠最好。
4. **threshold不是universal常數**：`is_faithful`裡的`threshold=0.5`只是示範值，實際場景要拿一批人工標註的(context, answer, 是否忠實)三元組校準，不同模型、不同領域的合理threshold可能差很多。

## 複習自問
- 為什麼`is_faithful`裡`context`一定要當`text`（premise）、`answer`一定要當`text_pair`（hypothesis），如果順序顛倒會發生什麼問題？
- zero-shot classification本質上是把分類問題轉換成什麼問題？如果candidate_labels裡有兩個語意高度重疊的標籤，模型的entailment分數會怎麼表現？
- entailment分數超過threshold就代表答案「一定正確」嗎？NLI能保證什麼、不能保證什麼？
- 如果要把這份程式的faithfulness check用在段落長度（而非單句）的RAG答案上，需要先做什麼前處理，理論根據是什麼？
