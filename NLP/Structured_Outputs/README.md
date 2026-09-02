# Structured Outputs & Constrained Decoding

對應程式：[`./structured_outputs.py`](./structured_outputs.py)

## TL;DR
跟LLM要JSON，大部分時候會拿到JSON——但production系統不能接受「大部分時候」。Constrained decoding在每一步生成時直接把不合法token的logit改成`-inf`，讓模型在數學上不可能生成不符合schema的輸出，把「大部分時候正確」變成「保證正確」。

## 為什麼需要它
- Free-form生成不是contract，是建議。跟模型說「只回傳JSON」，模型還是可能回「情感是正面的——這則評論非常正面因為顧客明確表示...」，parser直接crash。
- 三種解法，可靠度與綁定成本互相取捨：**prompting**（拜託模型，frontier模型~80%成功率，小模型更低）→ **原生structured output API**（OpenAI `response_format`、Anthropic tool use，可靠但綁死vendor）→ **constrained decoding**（改logits，100%合法，可以用在任何本地模型上，但要自己管grammar/FSM）。
- 這份程式把三層都示範一遍，並額外加上Instructor這個「不改logits、靠重試」的第四種折衷方案，讓四種做法能直接對照。

## 核心原理
- **Constrained decoding的核心動作**：每一個生成step，logit processor算出「在目前grammar狀態下，哪些token合法」，把其餘token的logit設成`-inf`，softmax後這些token的機率就是0——不是「大機率不選」，是「不可能被選到」。
- **FSM（有限狀態機）**：把JSON Schema或regex編譯成FSM，每個token的合法性查詢是O(1)。Outlines用的就是這個方法；缺點是遞迴schema（例如巢狀結構）需要被攤平成固定深度，處理不了真正的遞迴。
- **CFG（context-free grammar）**：比FSM更有表達力，能處理遞迴schema，代價是decoding開銷比FSM略高。XGrammar、llguidance走這條路，OpenAI 2025年的structured output實作也採用了llguidance。
- **Instructor的機制完全不同**：它不碰logits，而是把schema寫進prompt、解析輸出、驗證失敗就重試（預設3次）。好處是provider-agnostic，任何LLM都能用；代價是重試帶來的延遲與成本，且無法達到100%保證合法。
- **反直覺的結果：constrained decoding往往比不受限生成更快**。原因有二：一是合法token集合變小，搜尋空間跟著縮小；二是像`{"name": "`這種被schema完全決定的scaffolding部分，實作良好的引擎可以直接跳過token-by-token生成。
- **Schema欄位順序不是格式問題，是邏輯問題**：JSON物件是無序的，但生成是有序的——如果把`answer`放在`reasoning`前面，模型會先「承諾」一個答案，再回頭生成看似合理的理由；JSON合法，答案卻可能是錯的，而且沒有任何validation能抓到這種錯誤。永遠讓reasoning欄位在前、结论欄位在後。

## 程式碼導覽
| 函式/類別 | 對應到理論的哪個部分 |
|---|---|
| `mask_logits` | Constrained decoding的最小核心：把不在`valid_token_ids`裡的位置設成`-inf`，讓sampler不可能選到 |
| `generate_constrained` | 完整的FSM-guided decoding迴圈：每步查FSM合法token、mask logits、取樣、用取到的token推進FSM狀態，直到FSM進入accept state |
| `Review` | 對應Outlines示範：用Pydantic schema定義「情感分類」的目標結構，讓`outlines.generate.json`把它編譯成FSM |
| `outlines.generate.json(model, Review)` | 把schema編譯成FSM並綁定到本地HF模型；每次生成都保證輸出是合法的`Review` JSON，不需要事後再驗證 |
| `Invoice` | 對應Instructor示範：一樣是Pydantic schema，但走的是「prompt + 重試」而非「改logits」的路線 |
| `instructor.from_anthropic(...)` | 把schema塞進prompt指示模型輸出對應JSON，解析失敗時自動重試，provider換成OpenAI/Gemini等也能重複使用同一套`response_model`模式 |
| `client.responses.create(..., text={"format": {"type": "json_schema", ...}})` | 對應原生vendor API：schema直接送給OpenAI，由server端做constrained decoding，不需要本地模型或FSM編譯 |

**實作細節 / 容易看漏的地方：**
- `generate_constrained`裡的`sample(logits)`跟`fsm`、`model.next_token_logits`都是示意用的介面，不是這個檔案裡實際定義的函式——這段程式碼的重點是「decoding迴圈長什麼樣子」，不是可以直接執行的完整實作，真正生產環境會用Outlines/XGrammar這類套件。
- `mask_logits`回傳的list長度跟輸入`logits`一樣，只是把不合法位置蓋成`-inf`；合法位置的原始分數完全不變，代表在合法token之間，模型原本的偏好排序仍然保留，constrained decoding只砍掉不合法的選項，不改變合法選項間的相對機率。
- 四段demo（FSM手刻、Outlines、Instructor、OpenAI原生API）彼此獨立，不是同一個pipeline的四個步驟；`model`、`client`變數在後面的區塊被重新賦值覆寫，是刻意示範四種不同機制個別長什麼樣子，不是bug。
- `Invoice.total_usd`用`Field(ge=0)`加了數值下限限制，這種欄位級validation是Pydantic本身的能力，跟constrained decoding的token級限制是兩層不同的保證：constrained decoding保證「JSON結構合法」，Pydantic的`Field`約束保證「數值語意合理」。

## 使用時機 / 優缺點
- ✅ 本地模型、需要100%合法輸出、schema不含遞迴：Outlines（FSM）是首選，延遲低、實作直觀。
- ✅ 本地模型但schema有遞迴結構（巢狀留言串、AST）：改用XGrammar或llguidance（CFG-based）。
- ✅ 需要跨provider共用同一套Pydantic模型、能接受偶爾重試的延遲成本：Instructor。
- ✅ 用OpenAI/Anthropic/Gemini且schema不複雜：直接用vendor原生structured output API，不用自己管本地模型。
- ❌ 巨大enum（例如上萬個選項）用FSM編譯會很慢甚至timeout，這種情況該先用retriever縮小候選集合，再對候選做constrained decoding。
- ❌ Grammar定太死會讓模型無路可走：例如把日期欄位強制成`YYYY-MM-DD` regex，模型遇到「未知日期」的情況時只能編造一個日期，因為schema不允許它說「不知道」——一定要留`null`或sentinel值當逃生門。

## 常見誤區
1. **JSON mode ≠ 有schema保證**：vendor的「純JSON模式」只保證語法上是合法JSON，不保證符合你要的欄位結構（沒有`required`、沒有`enum`限制），一定要額外提供完整schema，不能只開JSON mode就當作結構化輸出解決了。
2. **Instructor不是constrained decoding的一種實作**：它完全不碰logits，本質是「prompt engineering + 驗證重試」，跟Outlines/XGrammar的「數學上不可能生成不合法token」是不同等級的保證——Instructor無法達到100%成功率，只是把失敗率壓得很低。
3. **schema欄位順序會影響推理品質，不只是美觀問題**：把要下結論的欄位（`answer`、`decision`）放在最前面，等於強迫模型在還沒推理前就先輸出答案，之後的`reasoning`欄位只是在幫已經定案的答案找理由。永遠把讓模型「先想後答」的欄位放在schema前面。
4. **FSM能處理的grammar是有限的**：Outlines的FSM對遞迴schema只能攤平成固定深度處理，不是真的支援任意深度的巢狀結構；需要真正遞迴（例如評論串的巢狀回覆）時，要換成CFG-based的XGrammar或llguidance，不是加大FSM深度就能解決。

## 複習自問
- 為什麼「把不合法token的logit設成`-inf`」比「生成後再重新解析、驗證失敗就丟掉重來」更能保證100%合法輸出？兩者在失敗率上的本質差異是什麼？
- Instructor跟Outlines都能接受同一個Pydantic schema，但保證的強度不一樣，差別具體是在哪一步？
- 為什麼schema欄位順序會影響模型的推理品質，而不只是最終JSON長什麼樣？如果要設計一個「分類 + 理由」的schema，欄位順序該怎麼排？
- 什麼情況下FSM（Outlines）不夠用、一定要換成CFG（XGrammar/llguidance）？程式裡的`Review`、`Invoice`兩個schema分別屬於哪一種情況？
