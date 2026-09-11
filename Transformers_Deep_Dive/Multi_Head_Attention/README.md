# Multi-Head Attention (MHA / GQA / MQA)

對應程式: [`./multi_head_attention.py`](./multi_head_attention.py)

參考:[ai-engineering-from-scratch – Multi-Head Attention 課程](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/07-transformers-deep-dive/03-multi-head-attention/docs/en.md)

前置閱讀:[`../Self_Attention/`](../Self_Attention/README.md)——這裡的每一個head做的都是那支程式裡`SelfAttention`在做的事,只是換成同時跑好幾組。

## TL;DR
單一attention head只會產生一組`(n, n)`的attention weight,主詞-動詞一致性、代名詞指涉、單純相鄰位置——這些性質不同的關係型態,全部被迫擠進同一組softmax分數裡互相干擾。2017年"Attention Is All You Need"的解法很直接:與其硬要一個head學會所有關係,不如平行跑好幾個維度更小的head,各自自由學出不同的模式,最後再concatenate、投影回原本的維度——`d_model`切成`n_heads`份,總參數量不變,表達力卻不是線性疊加而是「多套獨立的假設」。本程式用純NumPy重現這個「切開→各自attention→合併→投影」的流程(`mha_forward`),並補上2023年後幾乎所有正式模型都在用的變體:**Grouped-Query Attention (GQA)**與**Multi-Query Attention (MQA)**——讓K/V的head數少於Q的head數、用`repeat`補齊,換取推論時KV cache記憶體的大幅縮減。程式最後用實際數字(依Llama 3 70B的設定)量出GQA/MQA到底省了多少記憶體,並跟`torch.nn.MultiheadAttention`/`scaled_dot_product_attention(enable_gqa=True)`做shape對照。

## 為什麼需要它
- **單一head被迫把所有關係類型平均混在一組權重裡**:一句話裡「主詞-動詞」的對應關係、「代名詞-指涉對象」的對應關係、單純的「相鄰位置」關係,這些pattern在向量空間裡的幾何形狀通常不一樣。如果只有一組Q/K/V投影,模型必須把這些不同性質的關係擠進同一組相似度分數裡,等於被迫做平均、彼此互相干擾。2019-2024年的probing研究也證實不同head確實會分工:有專門處理位置關係的head、專門盯著前一個token的head、複製(copy)head、命名實體head,甚至是被認為是in-context learning底層機制的induction head。
- **切頭幾乎是免費的午餐**:`split_heads`只是一次reshape加一次transpose,沒有額外的迴圈;在真實硬體上,`Qh @ Kh.transpose(...)`是一次`(heads, N, d_head) x (heads, d_head, N) -> (heads, N, N)`的batched matmul,GPU一個kernel就處理完所有head——多加head多花的是FLOPs,不是額外的複雜度或kernel launch次數。
- **KV cache是推論時的真實記憶體瓶頸,而且只跟K/V的head數有關**:自回歸生成時,每個decoder layer都要把已生成token的K、V存起來(KV cache),cache大小是`n_layers × seq_len × n_kv_heads × d_head × dtype_bytes × 2`(K跟V各一份)——注意這裡完全不含Q,Query只用在當下這一步,不需要被快取。當`n_heads`衝到64、128甚至更多時,KV cache會變成推論記憶體的主要開銷,長context場景下這個問題更明顯。
- **GQA/MQA是「只縮小K/V的head數」這個單一想法的兩個極端**:MQA(2019, Shazeer)把K/V直接砍到只剩1個共用head,cache最小但品質有感下降;GQA(2023, Ainslie et al.)取中間值——`n_kv_heads`介於1與`n_heads`之間,例如Llama 3 70B用64個Q head配8個KV head,cache縮到1/8,品質幾乎不掉。這正是2023年之後(Llama 2起)幾乎所有正式模型的預設選擇,不再是MHA。

## 核心原理
- **Split → Attend in parallel → Concatenate & project**:輸入`X`(shape `(n, d_model)`)分別投影成`Q/K/V`(各自`(n, d_model)`),`split_heads`把最後一維切成`n_heads`份、reshape成`(n_heads, n, d_head)`(`d_head = d_model / n_heads`);每個head獨立做一次完整的scaled dot-product attention,彼此在attention計算過程中完全不交談;`combine_heads`把`n_heads`個`(n, d_head)`輸出接回`(n, d_model)`,乘上輸出投影`W_o`——這是唯一一個讓不同head的資訊互相混合的地方。
- **切頭不是多疊參數,是把同一包參數切開用**:單一大head的`Wq/Wk/Wv`是`(d_model, d_model)`,`n_heads`個小head的`Wq/Wk/Wv`合起來也是`(d_model, d_model)`——差別只在`reshape`的方式,不是每個head各自擁有一整組獨立的投影矩陣。多出來的只有`W_o`(`(d_model, d_model)`),用同樣的參數預算換到「多組獨立關係模式」而不是「一組更精細的關係模式」。

  | Variant | Q heads | K/V heads | 代表模型 |
  |---|---|---|---|
  | MHA | N | N | GPT-2, BERT, T5 |
  | MQA | N | 1 | PaLM, Falcon |
  | GQA | N | G(例如N/8) | Llama 2 70B、Llama 3+、Qwen 2+、Mistral |
  | MLA | N | 壓縮到低秩latent | DeepSeek-V2、V3 |

- **GQA/MQA只改K、V怎麼投影,Q完全不變**:`gqa_project(X, W, n_kv_heads, n_heads)`把`X`投影到`n_kv_heads`組(`W`的shape是`(d_model, n_kv_heads * d_head)`,比MHA的`(d_model, d_model)`窄,這正是省參數的地方),再用`np.repeat`把每一組複製`n_heads // n_kv_heads`次,補齊成跟Q一樣的`n_heads`份——`n_kv_heads == n_heads`時就是還原成MHA,`n_kv_heads == 1`就是MQA。`gqa_forward`把這個投影接上跟`mha_forward`完全一樣的attention計算,唯一差異是K/V的來源函式換掉。
- **KV cache省下多少,直接由`n_kv_heads`決定**:`kv_cache_bytes`量的是`n_layers × seq_len × n_kv_heads × d_head × dtype_bytes × 2`,這個算式裡沒有`n_heads`——這就是為什麼「Q head數多不影響cache,K/V head數少才是真正省記憶體的關鍵」。程式用Llama 3 70B的設定(80層、64個Q head、8個KV head、d_head=128)實際算出GQA是8倍cache縮減、MQA是64倍,對應教材裡「N/G cache shrink」的說法。
- **MLA(Multi-head Latent Attention)只在本程式裡帶過,沒有實作**:DeepSeek-V2/V3把K/V壓縮進一個低秩的latent向量、真正attend時才解壓回來,比GQA更進一步省cache,但要多付出運算量——教材把它列為「2026 lineage」的最新一環,本程式聚焦在MHA/GQA/MQA,MLA留給有興趣的人自己延伸(見下方複習自問)。
- **本程式跟PyTorch的關係是shape/機制對照,不是數值對照**:`nn.MultiheadAttention`是`mha_forward`的一行版本;PyTorch 2.5+的`scaled_dot_product_attention(..., enable_gqa=True)`則是`gqa_forward`的一行版本——它接受K/V的head數比Q少,內部自動處理repeat並派發到融合kernel(例如CUDA上的Flash Attention),不會真的把K/V materialize成`n_heads`份再算,這是本程式`np.repeat`版本跟production實作的差異所在。

## 程式碼導覽
| 函式/類別 | 對應到理論的哪個部分 |
|---|---|
| `softmax` | 跟Self_Attention的版本邏輯相同,但多了`axis`參數:attention weight這裡是`(n_heads, n, n)`的三維陣列,需要指定沿哪一軸做normalize |
| `split_heads` / `combine_heads` | Step 1:`(n, d_model) <-> (n_heads, n, d_head)`的reshape+transpose,一次到位、沒有per-head迴圈 |
| `mha_forward` | Step 2:完整的multi-head attention——split、per-head的scaled dot-product attention(以batched matmul一次算完所有head)、combine、乘`W_o` |
| `gqa_project` | Step 3:把K或V投影到`n_kv_heads`組,再用`np.repeat`補齊到`n_heads`份,對齊Q的head數 |
| `gqa_forward` | 接上`gqa_project`的完整GQA/MQA forward,`n_kv_heads`傳`n_heads`還原MHA,傳1就是MQA |
| `kv_cache_bytes` | KV cache記憶體公式,量化GQA/MQA相對MHA省下多少 |
| `print_attention_weights` | 跟Self_Attention同款的文字表格印法,逐head重複使用 |
| `make_toy_sentence` | 建立`["The", "cat", "sat", "on", "the", "mat"]`與隨機embedding`X`,本程式獨立維護,不跨檔案import Self_Attention的版本 |
| `demo_split_combine_roundtrip` | 驗證`combine_heads(split_heads(X, n_heads))`精確等於`X`,對應教材「一次reshape+transpose,沒有迴圈」的說法 |
| `demo_mha` | Step 2+4:跑4個head的MHA,印出input/output/weights的shape,並逐head印出各自學到的attention矩陣(即使是隨機初始化,不同head的pattern也明顯不同) |
| `demo_gqa_vs_mqa` | Step 3+Exercise 2:重用`demo_mha`的`W_q`,分別跑GQA(`n_kv_heads=2`)跟MQA(`n_kv_heads=1`),對照`W_k/W_v`的shape縮減 |
| `demo_kv_cache_savings` | Exercise 2的記憶體量化:依Llama 3 70B設定算出MHA/GQA/MQA的KV cache大小與縮減倍數 |
| `demo_pytorch_equivalents` | 對應教材"Use It":`nn.MultiheadAttention`跟`scaled_dot_product_attention(enable_gqa=True)`的shape對照 |

**實作細節 / 容易看漏的地方:**
- `split_heads`/`combine_heads`/`mha_forward`/`gqa_project`四個函式的邏輯逐字對應教材"Build It"裡給的程式碼片段,只加上docstring跟型別/shape說明,沒有更動運算邏輯。`gqa_forward`、`kv_cache_bytes`跟所有`demo_*`是本程式依教材Exercise 2與"Use It"段落補上的完整實作,教材本身這幾塊只給了文字敘述。
- `gqa_project`的`W`參數形狀容易搞錯:不是MHA那種`(d_model, d_model)`,而是`(d_model, n_kv_heads * d_head)`——`d_head`要用Q那邊的`d_model / n_heads`,不是`n_kv_heads`去除。這支程式`demo_gqa_vs_mqa`裡特意把`W_k`/`W_v`的shape印出來(例如`n_heads=4, n_kv_heads=2`時是`(8, 4)`而不是`(8, 8)`),用來對照「K/V投影矩陣本身就變窄了」這個容易被忽略的重點。
- `demo_gqa_vs_mqa`刻意重用`demo_mha`回傳的`W_q`,讓GQA/MQA/MHA三次demo的Query投影完全相同,對照的變因只有K/V的分組方式。
- `kv_cache_bytes`算的是**一個序列**、**所有layer**加總的K+V cache大小,不含batch size;真實部署時還要再乘上同時服務的sequence數,GQA/MQA省下的比例不會變,但絕對值會隨batch線性放大。
- PyTorch的`scaled_dot_product_attention(..., enable_gqa=True)`是2.5版之後才有的參數,`nn.MultiheadAttention`本身沒有原生支援GQA/MQA(它假設Q/K/V的head數相同)——這也是本程式在`demo_pytorch_equivalents`裡分別用兩個不同API對照MHA跟GQA的原因。

## 使用時機 / 優缺點
- ✅ 想搞懂「多頭」實際上是怎麼從一個`SelfAttention`變出來的,而不是只知道`num_heads`這個超參數要填多少:`split_heads`/`combine_heads`把整個切開/合併的reshape邏輯攤開成兩行,沒有藏在框架的C++/CUDA kernel裡。
- ✅ 需要對「為什麼2023年後幾乎所有模型都用GQA而不是MHA」建立具體直覺:`demo_kv_cache_savings`直接算出Llama 3 70B配置下MHA/GQA/MQA的實際記憶體數字(GiB),不是只講「GQA省記憶體」這句抽象的話。
- ❌ 不要把`mha_forward`/`gqa_forward`直接套進真正的模型:沒有反向傳播、沒有dropout、沒有殘差連接與LayerNorm、`W_q/W_k/W_v/W_o`從頭到尾是隨機初始化——重點是「一次forward pass裡切頭/分組怎麼運作」,不是可訓練、可部署的模組。
- ❌ 這裡的attention計算仍是`O(n^2)`的純NumPy版本,沒有做FlashAttention式的kernel融合:`demo_pytorch_equivalents`裡的`scaled_dot_product_attention`在CUDA上會自動派發到融合kernel,但本程式自己的`mha_forward`/`gqa_forward`不會,只適合拿來理解機制,不適合拿來測效能。
- ❌ Multi-head Latent Attention(MLA)沒有實作:只在README與程式docstring裡帶過概念,如果要深入DeepSeek-V2/V3怎麼把K/V壓縮進低秩latent,需要另外讀對應論文(見下方複習自問)。

## 常見誤區
1. **以為head數越多一定越好**:`n_heads`增加代表`d_head = d_model / n_heads`跟著變小,每個head能表達的子空間維度變少,`d_head`一旦低於32左右,`sqrt(d_head)`的scaling效果會開始跟頭的表達力打架;`d_head`太大(超過256)又會失去「多組小型專家」的效果。教材建議`d_head`落在64或128,這是要跟`d_model`一起權衡的超參數,不是越大越好。
2. **以為multi-head比single-head多很多參數**:`mha_forward`的`Wq/Wk/Wv`跟單一大head的`SelfAttention`維度是一樣的(都是`(d_model, d_model)`量級),差別只在切割方式跟多了一個`W_o`——不是每個head各自擁有一整組`(d_model, d_model)`的投影矩陣。
3. **以為GQA/MQA省的是參數量或運算量**:GQA/MQA主要省的是**推論時的KV cache記憶體**,不是訓練參數量或FLOPs——`W_k`/`W_v`確實變窄了(參數量略減),但真正的價值在於推論時只需要快取`n_kv_heads`份K/V,不是`n_heads`份,這在長context、大batch的推論場景差異巨大。
4. **搞混`gqa_project`裡`W`的shape跟MHA的`Wk`**:如果誤用`(d_model, d_model)`去餵`gqa_project`,`split_heads`會用錯的`d_head`去切,得到的head數或維度會跟Q對不上——`W`的第二維必須是`n_kv_heads * d_head`,`d_head`要用Q那邊算出來的`d_model / n_heads`,不是憑`n_kv_heads`重新算一次。
5. **以為`nn.MultiheadAttention`可以直接拿來做GQA**:PyTorch的`nn.MultiheadAttention`假設Q/K/V的head數相同,沒有`n_kv_heads`這種參數;要用GQA得改用`scaled_dot_product_attention(..., enable_gqa=True)`並自行管理K/V的head數,或直接用有原生GQA支援的模型函式庫(如Hugging Face transformers)。

## 複習自問
- 為什麼KV cache的大小公式裡沒有`n_heads`,只有`n_kv_heads`?如果只看這個公式,你會怎麼跟人解釋「Q head數再多也不會讓cache變大」這件事?
- MQA(`n_kv_heads=1`)跟GQA(`1 < n_kv_heads < n_heads`)都是縮小K/V head數,兩者在品質與cache大小之間的取捨分別落在什麼位置?如果你要幫一個新模型選`n_kv_heads`,你會依據哪些因素(部署硬體、context長度、對品質的容忍度)決定?
- `split_heads`/`combine_heads`是「一次reshape+transpose,沒有迴圈」,如果改成寫一個Python for迴圈、逐head手動切片再疊在一起,對正確性有影響嗎?對真實GPU上的執行效率呢?
- 如果要把本程式的`gqa_forward`延伸成Multi-head Latent Attention(MLA):K、V不是被repeat補齊,而是先壓縮成一個低秩latent向量、attend前再解壓——你預期`kv_cache_bytes`這個函式要怎麼改,才能正確反映MLA的cache大小?
