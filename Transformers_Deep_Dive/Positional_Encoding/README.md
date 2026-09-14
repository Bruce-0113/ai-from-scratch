# Positional Encoding (Sinusoidal / RoPE / ALiBi)

對應程式: [`./positional_encoding.py`](./positional_encoding.py)

參考:[ai-engineering-from-scratch – Positional Encoding 課程](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/07-transformers-deep-dive/04-positional-encoding/docs/en.md)

前置閱讀:[`../Self_Attention/`](../Self_Attention/README.md)、[`../Multi_Head_Attention/`](../Multi_Head_Attention/README.md)——這兩支程式的`scaled_dot_product_attention`/`mha_forward`本身完全不知道token的順序,順序訊號要靠這裡的三種方法額外注入。

## TL;DR
Attention是permutation-invariant的:把輸入的token打亂順序,`softmax(QK^T / sqrt(dk)) @ V`算出來的還是同一組結果,只是被重新排列而已——對語言這種順序本身帶有意義的資料來說,這是致命的缺陷。位置訊息必須額外注入,而且有兩條完全不同的路線:**絕對位置**(告訴模型「這是第幾個token」,如sinusoidal)跟**相對位置**(告訴模型「這兩個token差幾格」,如RoPE、ALiBi)。本程式用純Python(不依賴NumPy/PyTorch)重現教材"Build It"四步驟:sinusoidal絕對位置編碼、RoPE用旋轉角度把相對距離編進Q/K的內積、ALiBi直接在attention分數上做線性距離懲罰,以及驗證RoPE「只認得相對距離、不認得絕對位置」這個關鍵性質。

## 為什麼需要它
- **Attention本身沒有任何機制知道token的順序**:`Q @ K^T`只看每個token的內容(embedding投影出的Q、K向量),完全不看它在序列裡的位置索引——把"the cat sat"跟"sat the cat"的token集合丟進同一組`Wq/Wk/Wv`,只要維持每個token內容不變、只打亂順序,attention算出來的整組weight只是被重新排列,數值本身不會變。RNN/LSTM靠遞迴結構天生帶有順序,attention拿掉了遞迴,順序就必須用別的方式補回來。
- **絕對位置編碼(sinusoidal、learned embedding)有個共同的先天限制:學不會沒看過的位置**:sinusoidal是一張固定的`(max_len, d_model)`查表,數學上可以算出任意`pos`的值(公式對`pos=100000`一樣算得出來),但問題不在「算不算得出來」,而在模型的**其他權重**(尤其是後面每一層attention/FFN)在訓練時只看過`pos`落在`[0, max_len)`區間的組合——超出訓練時看過的範圍,模型沒有學過該怎麼反應,這就是「extrapolation失敗」的真正意思。GPT-2/GPT-3的learned absolute embedding更極端:超過`max_len`的位置連數值都沒有,直接查表越界。
- **相對位置編碼解決的正是這個推廣問題**:RoPE跟ALiBi的共同想法是——與其編碼「這是絕對第幾個位置」,不如編碼「這兩個token差幾格」。距離`m - n`是一個可以自然外推的量(訓練時看過距離3、距離10,不代表沒看過距離3000就不能處理,只是精確度會隨距離增加而衰減),這是為什麼2023年後長context模型幾乎全部轉向RoPE或ALiBi,而不是繼續用sinusoidal/learned absolute embedding。
- **RoPE和ALiBi解決同一個問題,但手段完全相反**:RoPE修改Q、K本身(用旋轉矩陣),讓內積在數學上自動只剩相對距離這個變數;ALiBi完全不碰Q/K/embedding,只在算完`QK^T/sqrt(dk)`之後、softmax之前,直接減去一個跟距離成正比的懲罰值。兩種做法殊途同歸,都讓模型「知道」哪些token比較近,但實作與訓練後行為(尤其是外推時的穩定度)有明顯差異。

## 核心原理
- **Sinusoidal:用不同頻率的sin/cos當作每個位置的「指紋」**
  ```
  PE[pos, 2i]   = sin(pos / 10000^(2i/d_model))
  PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))
  ```
  `i`越小,頻率越高(週期越短),`pe[pos]`裡低維度的值隨`pos`變化得很快;`i`越大,頻率越低,高維度的值幾乎是緩慢漂移的直流分量。這組固定向量直接加到輸入embedding上(`X' = X + PE[:N]`),不需要訓練,但也正因為是「加」上去而非模型自己學出來的關係,模型必須從訓練資料裡自己歸納出「這組sin/cos指紋代表位置」這件事,對訓練時沒出現過的`pos`沒有把握。程式裡的`sinusoidal(N, d)`逐字對應教材公式,`demo_sinusoidal`把整張表印成ASCII熱力圖,可以直接看到「左邊窄條紋、右邊寬色塊」的頻率遞減現象(對應Exercise 1)。

- **RoPE:不加東西,而是把Q、K本身旋轉一個跟位置成正比的角度**
  ```
  [q'_2i    ]   [ cos(pos·θ_i)  -sin(pos·θ_i) ] [q_2i   ]
  [q'_2i+1  ] = [ sin(pos·θ_i)   cos(pos·θ_i) ] [q_2i+1 ]

  θ_i = base^(-2i / d),  base = 10000
  ```
  把每一對相鄰維度`(x_2i, x_2i+1)`看成2D平面上的一個點,旋轉`pos * θ_i`角度——`apply_rope(x, pos, base)`就是逐字對應這個公式的實作。關鍵性質是:如果Query在位置`m`被旋轉、Key在位置`n`被旋轉(各自獨立旋轉,不是對內積做旋轉),兩個旋轉後向量的內積`q'_m . k'_n`會化簡成一個只跟`m - n`有關的函式——這是旋轉矩陣的性質(`R(m)^T R(n) = R(n - m)`),不是巧合也不是額外設計出來的技巧。`demo_rope_relative_property`直接用數字驗證這件事:同一組`(q, k)`分別在`(pos_q=5, pos_k=2)`跟`(pos_q=105, pos_k=102)`算出來的內積完全相同(`m-n`都是3),對應教材Build It Step 4的驗證步驟。
- **ALiBi:完全不碰embedding或Q/K,直接在attention分數上扣分**
  ```
  attn_score[i, j] = (q_i · k_j) / sqrt(d)  -  m_h · |i - j|
  ```
  `m_h`是每個head專屬的斜率,`slopes[h] = 2 ** (-8 * (h+1) / n_heads)`(注意`h`照論文慣例是從1開始算,不是從0),幾何級數分布——有些head斜率大,幾乎只關注非常靠近的token;有些head斜率極小,行為接近沒有距離懲罰。`alibi_bias(n_heads, seq_len)`回傳每個head各自的`(seq_len, seq_len)`懲罰矩陣,`bias[h][i][j] = -m_h * |i - j|`,加到該head的原始attention分數上(softmax之前)即可,不需要改任何投影矩陣。因為懲罰是`|i-j|`的線性函式,對任意`i, j`都有定義,不像sinusoidal的表被`max_len`綁死,這是ALiBi外推能力好、且訓練成本為零的原因。

  | 方法 | 絕對/相對 | 需要修改的地方 | 外推能力 | 代表模型 |
  |---|---|---|---|---|
  | Sinusoidal | 絕對 | 加到輸入embedding | 差 | 原始Transformer、早期BERT |
  | Learned absolute | 絕對 | 加到輸入embedding(可訓練) | 幾乎沒有 | GPT-2、GPT-3 |
  | RoPE | 相對 | 旋轉Q、K | 搭配scaling後不錯 | Llama 2/3/4、Qwen 2/3、Mistral、DeepSeek-V3、Kimi |
  | RoPE + YaRN | 相對 | 旋轉Q、K + 微調`base`/per-dim scaling | 優秀 | Qwen2-1M、Llama 3.1 128K |
  | ALiBi | 相對 | 直接改attention分數 | 優秀 | BLOOM、MPT、Baichuan |

- **RoPE的`base`是長context微調的旋鈕,不是隨便選的常數**:`base`越大,同樣的`i`對應的`θ_i`越小,旋轉得越慢,能在不「繞圈混淆」的前提下分辨的相對距離範圍就越大。NTK-aware scaling把`base`重新縮放成`base * scale_factor^(d/(d-2))`,YaRN則是對每個維度分別做內插、盡量維持attention分佈的熵不被打亂——兩者都只是在動`base`這一個超參數背後的精神,沒有改變RoPE旋轉的基本形式,這也是教材說「RoPE贏過ALiBi」的關鍵原因:`base`給了一個乾淨、事後就能調的旋鈕,不需要改架構。本程式的`apply_rope`保留`base`參數但只示範預設值10000,NTK/YaRN的per-dimension scaling留給有興趣的人自己延伸(見下方複習自問)。

## 程式碼導覽
| 函式 | 對應到理論的哪個部分 |
|---|---|
| `sinusoidal` | Build It Step 1:絕對位置編碼公式,回傳`(N, d)`的固定表 |
| `apply_rope` | Build It Step 2:對一個Q/K向量做位置相關的旋轉,`pos`分別套在Q、K上才會產生相對距離性質 |
| `alibi_bias` | Build It Step 3:每個head的斜率`m_h`與對應的`(seq_len, seq_len)`線性距離懲罰矩陣 |
| `ascii_heatmap` | 把任意浮點數矩陣(可正可負,如PE的`[-1,1]`或ALiBi的`<=0`)印成ASCII熱力圖,依矩陣自身的min/max正規化,跟`Self_Attention`只假設非負attention weight的版本不同 |
| `demo_sinusoidal` | Exercise 1:印出PE表的熱力圖,對照「低維度變化快、高維度變化慢」的頻率遞減現象 |
| `demo_rope` | 把同一個向量在不同`pos`下旋轉,`pos=0`應該完全不變,直接當作`apply_rope`是「真旋轉」的sanity check |
| `demo_rope_relative_property` | Build It Step 4:同一組`(q, k)`在位移前後(`m-n`相同、絕對位置不同)算出的內積必須一致,印出`PASS: True/False` |
| `demo_alibi` | Build It Step 3的展示:印出每個head的斜率,並把head 0的懲罰矩陣畫成熱力圖(對角線最亮、四角最暗) |

**實作細節 / 容易看漏的地方:**
- 本程式全部用純Python(`list`/`math`),沒有依賴NumPy或PyTorch——教材原始程式碼片段本身就是純Python寫的,`sinusoidal`/`apply_rope`/`alibi_bias`三個函式的邏輯逐字對應教材"Build It"給的程式碼,只加上docstring與型別/shape說明,沒有更動運算邏輯。`ascii_heatmap`跟所有`demo_*`是本程式依教材Exercise 1與Step 4描述補上的完整實作,教材本身這幾塊只給了文字敘述,沒有附程式碼。
- `sinusoidal`的`d`如果是奇數,最後一個維度沒有cos搭檔,會留在初始化時的`0.0`——這不是bug,而是原始公式`2i`/`2i+1`成對出現的自然結果,真實模型的`d_model`幾乎不會是奇數,所以教材與本程式都沒有特別處理。
- `apply_rope`必須分別套用在Q(位置`m`)跟K(位置`n`)上,**不能**先算出`q . k`再對內積本身做旋轉——相對距離性質是旋轉矩陣`R(m)^T R(n) = R(n-m)`的代數結果,如果順序顛倒(比如對合併後的分數做旋轉),這個性質就不成立了。`demo_rope_relative_property`特意保留`q`、`k`兩個獨立向量,分別旋轉後才內積,對應這個容易寫錯的地方。
- `alibi_bias`裡的head索引`h`是`h+1`(對應論文「`h`從1算到`n_heads`」的慣例),如果誤用`h`從0開始算斜率公式,算出來的整組斜率會系統性偏大——這是教材Common Pitfalls明確點出的地方。
- 這支程式**沒有**實作NTK-aware scaling、YaRN、LongRoPE,也沒有實作learned absolute positional embedding——教材把這些列為"Modern Extensions"與Exercise 2/3(需要訓練一個小模型比較困惑度),本程式聚焦在三個方法各自的核心機制與RoPE的關鍵性質驗證,不含訓練迴圈。

## 使用時機 / 優缺點
- ✅ 想搞懂「位置編碼」實際上在改什麼:是加到embedding上(sinusoidal)、旋轉Q/K(RoPE),還是直接扣attention分數(ALiBi),三種方法動的是完全不同的地方,不是同一招的三種變體。
- ✅ 需要對「為什麼RoPE能外推、sinusoidal不行」建立具體直覺:`demo_rope_relative_property`用實際數字證明RoPE的內積只跟`m-n`有關,不是只講「RoPE是相對位置編碼」這句抽象的話。
- ❌ 不要把這裡的三個函式直接套進真正的模型:沒有跟`Self_Attention`/`Multi_Head_Attention`的`scaled_dot_product_attention`/`mha_forward`接起來,也沒有batch維度、沒有反向傳播——重點是「位置訊息怎麼被編碼」,不是一個可訓練、可部署的模組。
- ❌ 這裡驗證的是RoPE的數學性質,不是「模型訓練後真的能外推到多長」:`demo_rope_relative_property`證明的是`apply_rope`這個函式本身的代數性質(旋轉矩陣的結構),不代表任何用RoPE訓練出來的真實模型在超出訓練長度後品質不會下降——教材Exercise 2/3要驗證的是後者,需要實際訓練小模型比較困惑度,超出本程式範圍。
- ❌ 沒有實作NTK-aware/YaRN/LongRoPE:如果要理解production長context模型實際怎麼把`base`重新縮放,需要另外讀對應論文或Hugging Face `transformers`裡RoPE scaling的實作(見下方複習自問)。

## 常見誤區
1. **以為sinusoidal的公式在位置很大時會「算不出來」或「報錯」**:`sinusoidal(N, d)`在數學上對任意`pos`都算得出`sin`/`cos`值,不會有數值錯誤——extrapolation失敗指的是模型**其他層的權重**在訓練時只看過`pos < max_len`的組合,沒學過怎麼處理沒看過的位置訊號,不是這個函式本身的計算限制。
2. **以為RoPE的「相對位置」是額外設計出來的技巧**:`q'_m . k'_n`只跟`m-n`有關,是旋轉矩陣`R(m)^T R(n) = R(n-m)`的代數必然結果——只要`apply_rope`确实是對每個維度對做2D旋轉,這個性質自動成立,不需要另外加約束或正則化去「教」模型學會相對位置。
3. **搞混ALiBi的「加bias」跟causal mask的「設為`-inf`」**:causal mask(見`Self_Attention`)是把未來位置的分數設成`-inf`,softmax後強制變成0(完全遮蔽);ALiBi的`bias`是一個連續的懲罰值(`-m_h * |i-j|`),越遠懲罰越重但不會歸零,兩者可以疊加使用(decoder模型常常同時需要causal mask擋住未來,又用ALiBi讓「近的token」得到更高權重)。
4. **以為`alibi_bias`的斜率`h`該從0開始算**:`slopes[h] = 2 ** (-8 * (h+1) / n_heads)`裡的`h+1`是刻意的,對應論文「head編號從1算到`n_heads`」的慣例——如果改成`h`不加1,算出來的整組斜率會偏大,跟論文/教材的數值對不上。
5. **以為RoPE跟ALiBi可以直接互換、效果一樣**:兩者都是相對位置編碼,但RoPE改的是Q/K本身(需要模型架構支援旋轉),ALiBi改的是attention分數(架構改動更小、幾乎零訓練成本)——選哪一個會影響模型能不能直接套用現有的`nn.MultiheadAttention`等函式庫,也會影響外推時的行為細節,不是純粹的效能/品質取捨。

## 複習自問
- 為什麼`apply_rope`一定要分別套用在Q（位置`m`）跟K（位置`n`）上,而不能等算出`q . k`之後才對這個純量做「旋轉」?如果你把旋轉搬到內積算完之後,`demo_rope_relative_property`的驗證還會通過嗎?
- `sinusoidal`與`alibi_bias`都可以對任意`pos`/`seq_len`算出數值,但教材說sinusoidal外推差、ALiBi外推好——如果兩者在「公式能不能算」這件事上是一樣的,真正造成外推能力差異的是什麼?
- 如果要把NTK-aware scaling接到`apply_rope`,你會怎麼修改`base`參數的傳入方式?這個改動會不會影響`demo_rope_relative_property`驗證的相對距離性質?
- ALiBi的`bias[h][i][j] = -m_h * |i-j|`只用了`|i-j|`(距離),沒有用`i-j`(方向)——如果一個任務需要區分「這個token在我前面」跟「在我後面」,ALiBi這個設計夠用嗎?RoPE呢?
