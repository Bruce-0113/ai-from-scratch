# Self-Attention from Scratch

對應程式: [`./self_attention.py`](./self_attention.py)

參考:[ai-engineering-from-scratch – Self-Attention from Scratch 課程](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/07-transformers-deep-dive/02-self-attention-from-scratch/docs/en.md)

## TL;DR
RNN處理序列時,token 1的資訊要被硬擠過49次壓縮才能傳到token 50,長距依賴關係在這個過程裡逐漸失真——這是LSTM/GRU的gating機制也無法根治的結構性瓶頸。2017年"Attention Is All You Need"給出的答案是:把「recurrence」整個拿掉,只留attention。Self-attention讓每個token在**同一個平行步驟**裡直接跟序列中任何位置對話:每個token投影出Query(我在找什麼)、Key(我提供什麼標籤)、Value(我實際攜帶的內容),Query對所有Key做內積算出相似度、除以`sqrt(dk)`避免softmax飽和、再用softmax轉成一組相加為1的權重,最後用這組權重對所有Value做加權和——這就是`softmax(QK^T / sqrt(dk)) @ V`。本程式除了重現原教材"Build It"五步驟(softmax、scaled dot-product attention、SelfAttention class、玩具句子demo、ASCII熱力圖),額外把教材Exercise提到但沒寫出完整程式碼的兩塊補齊:**causal mask**(把雙向attention轉成decoder用的自回歸attention)與**multi-head attention**(從零實作切頭/平行attention/合併/投影),並在最後跟`torch.nn.MultiheadAttention`做shape與機制的對照。

## 為什麼需要它
- **RNN的序列瓶頸是結構性的,不是調參能解決的**:每個時間步的hidden state是固定大小的向量,無論句子多長,所有歷史資訊都要塞進同一個瓶頸,距離越遠的token,梯度要傳回去的路徑越長,long-range dependency在實務上經常學不到——這也是2014年Bahdanau attention(讓decoder回頭看encoder每個位置)出現的原因,但那時attention還只是掛在RNN旁邊的輔助機制。
- **不scale,softmax會飽和,梯度會消失**:當`dk`變大(例如64),Q、K兩個高維向量的內積期望的量級會隨`dk`線性成長,原始分數動輒落在十位數,這時softmax的輸出會趨近one-hot(某個位置機率接近1,其餘接近0),對應的梯度在非最大值位置幾乎是0——模型學不到「哪些token其實也有點關係」這種漸進式的訊號。除以`sqrt(dk)`把分數量級拉回softmax梯度還有效的區間。
- **訓練decoder(自回歸生成)時,沒有causal mask模型會直接偷看答案**:訓練時input是完整的目標序列,如果attention能看到未來的token,模型只要「抄下一個字」就能把loss壓到接近0,但推論時模型必須一個字一個字生成,看不到未來——這種train/inference不一致會讓模型在真正生成時完全失效。causal mask把每個位置「未來」的attention分數設成`-inf`,softmax後這些位置權重變成0,強迫每個token只能看到自己與更早的token。
- **單一attention head被迫把所有關係類型平均混在一組權重裡**:一句話裡「主詞-動詞」的對應關係、「代名詞-指涉對象」的對應關係、單純的「相鄰位置」關係,這些pattern在向量空間裡的幾何形狀通常不一樣。如果只有一組Q/K/V投影,模型必須把這些不同性質的關係擠進同一組相似度分數裡,等於被迫做平均、彼此互相干擾。Multi-head把`d_model`切成`n_heads`組獨立的、維度更小的子空間,讓每個head自由學出不同的關係模式,最後再concatenate、投影回`d_model`。

## 核心原理
- **Q/K/V是同一份embedding的三種投影,對應資料庫查詢的三個角色**:輸入`X`(shape `(n, d_model)`)分別乘上`Wq`、`Wk`、`Wv`得到`Q`、`K`、`V`。可以把它想成「軟性的資料庫查詢」:傳統資料庫是`Query -> 精準比對 -> 一筆結果`,attention是`Query -> 跟所有Key算相似度 -> 對所有Value做加權混合`。

  | 角色 | 一句話白話 | 實際意義 |
  |---|---|---|
  | Query (Q) | 「我在找什麼」 | 這個token拿去跟所有Key比對相似度的投影 |
  | Key (K) | 「我能提供什麼標籤」 | 用來被Query比對、決定要分配多少注意力的投影 |
  | Value (V) | 「我實際攜帶的內容」 | 依attention weight被加權混合、真正進入輸出的投影 |

- **公式與shape**:`Scores = Q @ K^T`得到`(n, n)`的相似度矩陣,每一列代表「這個token對序列中每個位置的原始關注程度」。除以`sqrt(dk)`做scaling後,對每一列做`softmax`得到attention weights(每列相加為1),最後`weights @ V`得到`(n, dv)`的輸出——每個token的輸出是「所有token的Value,依照這個token算出來的權重混合而成」。完整公式:`Attention(Q, K, V) = softmax(Q @ K^T / sqrt(dk)) @ V`,對應[`scaled_dot_product_attention`](./self_attention.py)。
- **Causal mask是加在softmax之前的一個布林矩陣**:`causal_mask(n)`回傳一個上三角(不含對角線)為`True`的`(n, n)`布林矩陣,代表「位置j在位置i的未來」。`scaled_dot_product_attention`在算完scaled scores後,把mask為`True`的位置設成`-inf`,再做softmax——`exp(-inf) = 0`,這些位置的權重被強制歸零,而且softmax做的是「先減去該列最大值再取exp」,只要對角線(自己看自己)一定沒被mask,這個減法就不會用到`-inf`,不會出現`NaN`。
- **Multi-head attention是「切開、各自attention、接回去」,不是額外參數**:把`d_model`維的`Wq/Wk/Wv`(每個都是`(d_model, d_model)`)算出的`Q/K/V`,沿最後一維切成`n_heads`份、每份維度`d_head = d_model / n_heads`,每個head獨立做一次完整的scaled dot-product attention(各自的`(n, n)`權重矩陣),把`n_heads`個`(n, d_head)`輸出沿最後一維concatenate回`(n, d_model)`,再乘上輸出投影`Wo`得到最終輸出。關鍵是:總參數量與單一大head的`Wq/Wk/Wv`相當(切開的是這幾個矩陣,不是疊加更多),多出來的只有`Wo`——用同樣的參數預算換來「多組獨立關係模式」而不是「一組更精細的關係模式」。
- **本程式跟`torch.nn.MultiheadAttention`的關係是shape/機制對照,不是數值對照**:兩邊都是「切頭→各自attention→concatenate→輸出投影」,shape完全對得上(`(seq_len, d_model)`進、`(seq_len, d_model)`出、weights是`(n_heads, seq_len, seq_len)`),但兩邊的`Wq/Wk/Wv/Wo`是各自獨立初始化,數值不會一樣,對照的重點是「架構等價」而非「輸出相同」。

## 程式碼導覽
| 函式/類別 | 對應到理論的哪個部分 |
|---|---|
| `softmax` | 對最後一軸做數值穩定的softmax(先減去該列最大值再取exp),把scaled scores轉成attention weights |
| `scaled_dot_product_attention` | 核心公式`softmax(QK^T / sqrt(dk)) @ V`,額外支援`mask`參數,把mask為`True`的位置設成`-inf`後再softmax |
| `causal_mask` | 建構`(n, n)`的上三角布林mask,配合`scaled_dot_product_attention`的`mask`參數把雙向attention轉成decoder用的自回歸attention |
| `SelfAttention` | 單一head的self-attention:用Xavier-like scaling初始化`Wq/Wk/Wv`,`forward`把`X`投影成`Q/K/V`後呼叫`scaled_dot_product_attention` |
| `MultiHeadAttention` | 多頭attention:`Wq/Wk/Wv`各是`(d_model, d_model)`,`_split_heads`把投影結果切成`(n_heads, n, d_head)`,每個head獨立跑attention後concatenate、乘`Wo`投影回`(n, d_model)` |
| `print_attention_weights` | 把attention矩陣印成對齊的文字表格,單一header row + 每個token一列,同時被self-attention與causal mask的demo重複使用 |
| `ascii_heatmap` | 把attention weight依`weights.max()`的比例映射到` ░▒▓█`五級字元,提供一個不需要畫圖套件的視覺化 |
| `demo_softmax` | 重現教材Step 1:對`[2.0, 1.0, 0.1]`跑softmax,驗證輸出相加為1 |
| `make_toy_sentence` | 建立`["The", "cat", "sat", "on", "the", "mat"]`與對應的隨機embedding`X`,供後面所有demo共用同一份輸入 |
| `demo_self_attention` | 重現教材Step 3-5:跑單一head self-attention、印文字表格與ASCII熱力圖,並回傳訓練好的`attn`供causal mask demo重用 |
| `demo_causal_mask` | 教材Exercise 1的完整實作:重用同一個`attn`的`Wq/Wk/Wv`,只加上mask,對照雙向與causal attention的差異 |
| `demo_multi_head_attention` | 教材Exercise 2的完整實作:建立2個head的`MultiHeadAttention`,印出input/output/weights的shape,並逐head印出各自的attention矩陣 |
| `demo_pytorch_comparison` | 對應教材"Use It"段落:跟`torch.nn.MultiheadAttention`對照shape與機制 |

**實作細節 / 容易看漏的地方:**
- 原教材的"Build It"只寫了softmax、scaled dot-product attention、單一head的`SelfAttention`,以及玩具句子/ASCII熱力圖demo;`causal_mask`、`MultiHeadAttention`跟`demo_pytorch_comparison`是本程式依照教材的Learning Objectives與Exercise補上的完整實作,教材本身在這幾塊只給了文字描述跟習題敘述,沒有附程式碼。
- `demo_causal_mask`刻意重用`demo_self_attention`回傳的同一個`attn`實例,而不是重新建一個新的`SelfAttention`——這樣兩次印出的attention矩陣差異只來自mask本身,而不是「剛好兩組隨機初始化的權重不一樣」,對照才有意義。
- `MultiHeadAttention`的`n_heads=2`、`d_model=8`刻意跟單一head demo的`dk=dv=4`對齊(`d_head = d_model / n_heads = 4`),方便直接比較「同樣的關注力粒度,切成兩個head vs. 只有一個head」的差異。
- Windows的預設終端機編碼通常是`cp950`(繁體中文Big5),而`ascii_heatmap`用到的`░▒▓█`是需要UTF-8才能正確輸出的字元——`if __name__ == "__main__":`區塊在呼叫`main()`前會檢查並視需要把`sys.stdout`重設成UTF-8,否則在這類終端機上直接執行會丟出`UnicodeEncodeError`而整個程式中斷。

## 使用時機 / 優缺點
- ✅ 想搞懂transformer最核心的機制,而不是只會呼叫`nn.MultiheadAttention`或`model.forward()`:整個`Q@K^T -> scale -> softmax -> @V`的資料流,加上causal mask跟multi-head怎麼接上去,全部攤開成不到200行純NumPy,沒有任何一步是黑盒。
- ✅ 需要對「為什麼要scale」「causal mask怎麼生效」「multi-head是切開不是疊加」建立具體的直覺:程式裡的demo刻意把雙向vs.causal、單頭vs.多頭的attention矩陣印出來對照,而不是只給公式。
- ❌ 不要把這裡的`SelfAttention`/`MultiHeadAttention`直接套進真正的模型:沒有反向傳播、沒有dropout、沒有殘差連接與LayerNorm、也沒有相對位置編碼(RoPE等),`Wq/Wk/Wv`從頭到尾是隨機初始化、不會被訓練——這裡的重點是「一次forward pass的機制」,不是可訓練的模組。
- ❌ 不適合拿來評估效能或做長序列實驗:`scaled_dot_product_attention`是`O(n^2)`的純Python/NumPy實作,沒有做FlashAttention式的kernel融合或分塊計算,序列一長記憶體跟運算量會爆炸——這正是production inference engine(如vLLM、TensorRT-LLM)要另外解決的問題,不在本程式範圍內。

## 常見誤區
1. **以為self-attention跟cross-attention是同一件事**:本程式的`Q/K/V`全部來自同一個`X`(所以叫self-attention);encoder-decoder架構裡的cross-attention是Query來自decoder、Key/Value來自encoder輸出,兩者的角色分工不同,不能混用同一組直覺。
2. **以為`sqrt(dk)`只是「隨便乘的normalization常數」,拿掉也沒差**:拿掉scaling在`dk`小的玩具範例裡看不出差異,但`dk`一旦變大(例如真實模型的64或128),不scale會讓softmax輸出趨近one-hot、梯度消失,這是一個會隨模型規模變嚴重的問題,不是可有可無的細節。
3. **以為causal mask是訓練跟推論都要加的東西**:causal mask只有decoder(自回歸生成)場景需要,像BERT這種encoder-only、做雙向理解任務的模型完全不用mask——要不要加mask取決於「這個attention layer的token在生成時看不看得到未來」,不是所有transformer都需要。
4. **以為head數越多一定越好**:`n_heads`增加代表`d_head = d_model / n_heads`跟著變小,每個head能表達的子空間維度變少,head數過多時單一head的表達力可能反而不夠——這是一個要跟`d_model`搭配權衡的超參數,不是越大越好。
5. **以為multi-head比single-head多很多參數**:`MultiHeadAttention`的`Wq/Wk/Wv`跟單一大head的`SelfAttention`的`Wq/Wk/Wv`維度是一樣的(都是`(d_model, d_model)`量級),差別只在切割方式跟多了一個`Wo`輸出投影——不是每個head各自擁有一整組`(d_model, d_model)`的投影矩陣。

## 複習自問
- 為什麼causal mask要加在softmax**之前**(把分數設成`-inf`),而不是在softmax算完之後直接把對應的weight設成0?這兩種做法對其餘(未被mask)位置的權重分佈,分別會有什麼不同的影響?
- 如果拿掉`sqrt(dk)`的scaling,在`dk=4`的玩具範例裡跟`dk=512`的真實模型規模裡,分別會觀察到什麼程度的softmax飽和?
- Multi-head attention「切開`d_model`成`n_heads`份」與「訓練`n_heads`組各自獨立、維度不變的`SelfAttention`後把輸出接起來」,這兩種做法在參數量、運算量、可學到的表達能力上分別有什麼差異?
- 本程式的`scaled_dot_product_attention`是`O(n^2)`的實作,如果序列長度從6拉到6000,記憶體與運算量會怎麼變化?FlashAttention這類方法大致是從哪個角度去緩解這個問題?

## Further Reading
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) — 原始transformer論文
- [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/) — 圖解transformer架構的經典文章
- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) — 逐行對照論文與PyTorch實作
