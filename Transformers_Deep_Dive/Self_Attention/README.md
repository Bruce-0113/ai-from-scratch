# Self-Attention from Scratch

對應程式: [`./self_attention.py`](./self_attention.py)

參考:[ai-engineering-from-scratch – Self-Attention from Scratch 課程](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/07-transformers-deep-dive/02-self-attention-from-scratch/docs/en.md)

> 多頭(multi-head)/分組查詢(GQA)/多查詢(MQA) attention 已經搬到獨立主題,見 [`../Multi_Head_Attention/`](../Multi_Head_Attention/README.md)。

## TL;DR
RNN處理序列時,token 1的資訊要被硬擠過49次壓縮才能傳到token 50,長距依賴關係在這個過程裡逐漸失真——這是LSTM/GRU的gating機制也無法根治的結構性瓶頸。2017年"Attention Is All You Need"給出的答案是:把「recurrence」整個拿掉,只留attention。Self-attention讓每個token在**同一個平行步驟**裡直接跟序列中任何位置對話:每個token投影出Query(我在找什麼)、Key(我提供什麼標籤)、Value(我實際攜帶的內容),Query對所有Key做內積算出相似度、除以`sqrt(dk)`避免softmax飽和、再用softmax轉成一組相加為1的權重,最後用這組權重對所有Value做加權和——這就是`softmax(QK^T / sqrt(dk)) @ V`。本程式重現原教材"Build It"五步驟(softmax、scaled dot-product attention、SelfAttention class、玩具句子demo、ASCII熱力圖),並把教材Exercise提到但沒寫出完整程式碼的**causal mask**(把雙向attention轉成decoder用的自回歸attention)補齊實作。

## 為什麼需要它
- **RNN的序列瓶頸是結構性的,不是調參能解決的**:每個時間步的hidden state是固定大小的向量,無論句子多長,所有歷史資訊都要塞進同一個瓶頸,距離越遠的token,梯度要傳回去的路徑越長,long-range dependency在實務上經常學不到——這也是2014年Bahdanau attention(讓decoder回頭看encoder每個位置)出現的原因,但那時attention還只是掛在RNN旁邊的輔助機制。
- **不scale,softmax會飽和,梯度會消失**:當`dk`變大(例如64),Q、K兩個高維向量的內積期望的量級會隨`dk`線性成長,原始分數動輒落在十位數,這時softmax的輸出會趨近one-hot(某個位置機率接近1,其餘接近0),對應的梯度在非最大值位置幾乎是0——模型學不到「哪些token其實也有點關係」這種漸進式的訊號。除以`sqrt(dk)`把分數量級拉回softmax梯度還有效的區間。
- **訓練decoder(自回歸生成)時,沒有causal mask模型會直接偷看答案**:訓練時input是完整的目標序列,如果attention能看到未來的token,模型只要「抄下一個字」就能把loss壓到接近0,但推論時模型必須一個字一個字生成,看不到未來——這種train/inference不一致會讓模型在真正生成時完全失效。causal mask把每個位置「未來」的attention分數設成`-inf`,softmax後這些位置權重變成0,強迫每個token只能看到自己與更早的token。

## 核心原理
- **Q/K/V是同一份embedding的三種投影,對應資料庫查詢的三個角色**:輸入`X`(shape `(n, d_model)`)分別乘上`Wq`、`Wk`、`Wv`得到`Q`、`K`、`V`。可以把它想成「軟性的資料庫查詢」:傳統資料庫是`Query -> 精準比對 -> 一筆結果`,attention是`Query -> 跟所有Key算相似度 -> 對所有Value做加權混合`。

  | 角色 | 一句話白話 | 實際意義 |
  |---|---|---|
  | Query (Q) | 「我在找什麼」 | 這個token拿去跟所有Key比對相似度的投影 |
  | Key (K) | 「我能提供什麼標籤」 | 用來被Query比對、決定要分配多少注意力的投影 |
  | Value (V) | 「我實際攜帶的內容」 | 依attention weight被加權混合、真正進入輸出的投影 |

- **公式與shape**:`Scores = Q @ K^T`得到`(n, n)`的相似度矩陣,每一列代表「這個token對序列中每個位置的原始關注程度」。除以`sqrt(dk)`做scaling後,對每一列做`softmax`得到attention weights(每列相加為1),最後`weights @ V`得到`(n, dv)`的輸出——每個token的輸出是「所有token的Value,依照這個token算出來的權重混合而成」。完整公式:`Attention(Q, K, V) = softmax(Q @ K^T / sqrt(dk)) @ V`,對應[`scaled_dot_product_attention`](./self_attention.py)。
- **Causal mask是加在softmax之前的一個布林矩陣**:`causal_mask(n)`回傳一個上三角(不含對角線)為`True`的`(n, n)`布林矩陣,代表「位置j在位置i的未來」。`scaled_dot_product_attention`在算完scaled scores後,把mask為`True`的位置設成`-inf`,再做softmax——`exp(-inf) = 0`,這些位置的權重被強制歸零,而且softmax做的是「先減去該列最大值再取exp」,只要對角線(自己看自己)一定沒被mask,這個減法就不會用到`-inf`,不會出現`NaN`。
- **這裡只實作單一head**:一組`Wq/Wk/Wv`只能學出一種關係模式(見下方常見誤區)。把`d_model`切成多個獨立子空間、平行跑多組attention的做法(multi-head)、以及production模型為了省KV cache而讓K/V共用投影的變體(GQA/MQA),是下一個主題[`Multi_Head_Attention`](../Multi_Head_Attention/README.md)的內容,這裡的`SelfAttention`正是那邊`mha_forward`裡每一個head在做的事。

## 程式碼導覽
| 函式/類別 | 對應到理論的哪個部分 |
|---|---|
| `softmax` | 對最後一軸做數值穩定的softmax(先減去該列最大值再取exp),把scaled scores轉成attention weights |
| `scaled_dot_product_attention` | 核心公式`softmax(QK^T / sqrt(dk)) @ V`,額外支援`mask`參數,把mask為`True`的位置設成`-inf`後再softmax |
| `causal_mask` | 建構`(n, n)`的上三角布林mask,配合`scaled_dot_product_attention`的`mask`參數把雙向attention轉成decoder用的自回歸attention |
| `SelfAttention` | 單一head的self-attention:用Xavier-like scaling初始化`Wq/Wk/Wv`,`forward`把`X`投影成`Q/K/V`後呼叫`scaled_dot_product_attention` |
| `print_attention_weights` | 把attention矩陣印成對齊的文字表格,單一header row + 每個token一列,同時被self-attention與causal mask的demo重複使用 |
| `ascii_heatmap` | 把attention weight依`weights.max()`的比例映射到` ░▒▓█`五級字元,提供一個不需要畫圖套件的視覺化 |
| `demo_softmax` | 重現教材Step 1:對`[2.0, 1.0, 0.1]`跑softmax,驗證輸出相加為1 |
| `make_toy_sentence` | 建立`["The", "cat", "sat", "on", "the", "mat"]`與對應的隨機embedding`X`,供後面所有demo共用同一份輸入 |
| `demo_self_attention` | 重現教材Step 3-5:跑單一head self-attention、印文字表格與ASCII熱力圖,並回傳訓練好的`attn`供causal mask demo重用 |
| `demo_causal_mask` | 教材Exercise 1的完整實作:重用同一個`attn`的`Wq/Wk/Wv`,只加上mask,對照雙向與causal attention的差異 |

**實作細節 / 容易看漏的地方:**
- 原教材的"Build It"只寫了softmax、scaled dot-product attention、單一head的`SelfAttention`,以及玩具句子/ASCII熱力圖demo;`causal_mask`是本程式依照教材的Learning Objectives與Exercise 1補上的完整實作,教材本身在這塊只有文字描述跟習題敘述,沒有附程式碼。Multi-head attention(Exercise 2)獨立搬到了[`Multi_Head_Attention`](../Multi_Head_Attention/)主題,不在這支程式裡。
- `demo_causal_mask`刻意重用`demo_self_attention`回傳的同一個`attn`實例,而不是重新建一個新的`SelfAttention`——這樣兩次印出的attention矩陣差異只來自mask本身,而不是「剛好兩組隨機初始化的權重不一樣」,對照才有意義。
- Windows的預設終端機編碼通常是`cp950`(繁體中文Big5),而`ascii_heatmap`用到的`░▒▓█`是需要UTF-8才能正確輸出的字元——`if __name__ == "__main__":`區塊在呼叫`main()`前會檢查並視需要把`sys.stdout`重設成UTF-8,否則在這類終端機上直接執行會丟出`UnicodeEncodeError`而整個程式中斷。

## 使用時機 / 優缺點
- ✅ 想搞懂transformer最核心的機制,而不是只會呼叫`nn.MultiheadAttention`或`model.forward()`:整個`Q@K^T -> scale -> softmax -> @V`的資料流,加上causal mask怎麼接上去,全部攤開成純NumPy,沒有任何一步是黑盒。
- ✅ 需要對「為什麼要scale」「causal mask怎麼生效」建立具體的直覺:程式裡的demo刻意把雙向vs.causal的attention矩陣印出來對照,而不是只給公式。
- ❌ 不要把這裡的`SelfAttention`直接套進真正的模型:沒有反向傳播、沒有dropout、沒有殘差連接與LayerNorm、也沒有相對位置編碼(RoPE等),`Wq/Wk/Wv`從頭到尾是隨機初始化、不會被訓練——這裡的重點是「一次forward pass的機制」,不是可訓練的模組。
- ❌ 不適合拿來評估效能或做長序列實驗:`scaled_dot_product_attention`是`O(n^2)`的純Python/NumPy實作,沒有做FlashAttention式的kernel融合或分塊計算,序列一長記憶體跟運算量會爆炸——這正是production inference engine(如vLLM、TensorRT-LLM)要另外解決的問題,不在本程式範圍內。

## 常見誤區
1. **以為self-attention跟cross-attention是同一件事**:本程式的`Q/K/V`全部來自同一個`X`(所以叫self-attention);encoder-decoder架構裡的cross-attention是Query來自decoder、Key/Value來自encoder輸出,兩者的角色分工不同,不能混用同一組直覺。
2. **以為`sqrt(dk)`只是「隨便乘的normalization常數」,拿掉也沒差**:拿掉scaling在`dk`小的玩具範例裡看不出差異,但`dk`一旦變大(例如真實模型的64或128),不scale會讓softmax輸出趨近one-hot、梯度消失,這是一個會隨模型規模變嚴重的問題,不是可有可無的細節。
3. **以為causal mask是訓練跟推論都要加的東西**:causal mask只有decoder(自回歸生成)場景需要,像BERT這種encoder-only、做雙向理解任務的模型完全不用mask——要不要加mask取決於「這個attention layer的token在生成時看不看得到未來」,不是所有transformer都需要。
4. **以為單一head已經能學到句子裡所有的關係型態**:一組`Wq/Wk/Wv`只能產生一組`(n, n)`的attention weight,主詞-動詞、代名詞指涉、單純相鄰位置這些不同性質的pattern會被迫擠進同一組分數裡互相干擾——這正是[`Multi_Head_Attention`](../Multi_Head_Attention/)要解決的問題,不是這支程式的`SelfAttention`能做到的。

## 複習自問
- 為什麼causal mask要加在softmax**之前**(把分數設成`-inf`),而不是在softmax算完之後直接把對應的weight設成0?這兩種做法對其餘(未被mask)位置的權重分佈,分別會有什麼不同的影響?
- 如果拿掉`sqrt(dk)`的scaling,在`dk=4`的玩具範例裡跟`dk=512`的真實模型規模裡,分別會觀察到什麼程度的softmax飽和?
- 本程式的`scaled_dot_product_attention`是`O(n^2)`的實作,如果序列長度從6拉到6000,記憶體與運算量會怎麼變化?FlashAttention這類方法大致是從哪個角度去緩解這個問題?
- 如果一句話裡同時存在「主詞-動詞一致性」與「代名詞指涉」兩種關係,單一head的attention weight要怎麼在這兩者之間取捨?這個限制跟`Multi_Head_Attention`要解決的問題是同一件事嗎?
