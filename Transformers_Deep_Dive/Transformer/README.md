# The Full Transformer — Encoder + Decoder

對應程式: [`./transformer.py`](./transformer.py)

參考:[ai-engineering-from-scratch – The Full Transformer 課程](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/07-transformers-deep-dive/05-full-transformer/docs/en.md)

前置閱讀:[`../Self_Attention/`](../Self_Attention/README.md)、[`../Multi_Head_Attention/`](../Multi_Head_Attention/README.md)、[`../Positional_Encoding/`](../Positional_Encoding/README.md)——這支程式裡的`multi_head_attention`就是MHA那支程式的`mha_forward`,只是同一份邏輯被重複用在三種場合(self-attention、causal self-attention、cross-attention);位置編碼(sinusoidal/RoPE/ALiBi)在這支程式裡刻意留白,`src`/`tgt`直接當成「已經加過位置訊號的embedding」餵進去,細節見Positional_Encoding那支程式。

## TL;DR
一個attention層只是feature extractor,不是模型——2017年"Attention Is All You Need"真正的貢獻,是把attention包裝成一個可以疊深、疊了不會壞掉的**block**:self-attention之外還要有殘差連接(不然梯度過約6層就消失)、正規化(穩定residual stream)、逐位置的feed-forward network(attention之外唯一做非線性變換的地方),decoder還要多一個cross-attention把encoder的資訊接進來。這個「六件套」骨架從2017年沿用至今——BERT(encoder-only)、GPT(decoder-only)、T5(encoder-decoder)全部長在同一個骨架上,2026年真正變的只是骨架裡每個零件的實作細節:LayerNorm換成RMSNorm、ReLU-FFN換成SwiGLU、post-norm換成pre-norm。本程式用純Python(沿用Self_Attention/Multi_Head_Attention的`Matrix`型別與attention邏輯)把encoder block跟decoder block的wiring都接出來,而且**同一套`encoder_block`/`decoder_block`程式碼**分別跑一次2017古典設定(LayerNorm+ReLU-FFN)跟一次2026現代設定(RMSNorm+SwiGLU-FFN),最後接上final norm+輸出投影得到vocab logits,用實際數字證明骨架不變、只是換了兩個函式。

## 為什麼需要它
- **單一attention層的表達力撐不起深度,但不加保護地疊深會直接壞掉**:一層attention只做了一次「依內容混合」,論文本身的實驗與後續研究都顯示,沒有殘差連接的深層網路梯度過幾層就消失或爆炸(這裡約6層是常被引用的經驗值)——疊深帶來的容量提升,前提是網路本身撐得住深度,這正是2017年那組設計要解決的問題。
- **2017年打包的六個決定,是此後每一種transformer的共同骨架**:embedding+位置訊號、self-attention、feed-forward network、殘差連接、normalization,以及(僅decoder)cross-attention——BERT只用encoder那一半、GPT只用decoder那一半(拿掉cross-attention)、T5兩半都用,但拆開看都是同一組零件的排列組合,不是各自發明新架構。
- **pre-norm在2019年後成為預設,不是美學選擇**:原始論文是`LN(x + sublayer(x))`(post-norm);Xiong et al. (2020)指出post-norm在沒有精心設計warmup的情況下很難訓練超過十幾層,`x + sublayer(LN(x))`(pre-norm)讓每個sublayer看到的輸入尺度穩定,深層堆疊不需要warmup也能收斂——Llama、Qwen、GPT-3之後、Mistral全部採用pre-norm。
- **2026年的「現代化」是骨架裡零件的替換,骨架本身沒有變**:LayerNorm→RMSNorm(少一個mean-centering運算,穩定度至少打平)、ReLU-FFN→SwiGLU-FFN(在Llama/PaLM/Qwen論文裡穩定贏過ReLU/GELU約0.5個ppl點)、sinusoidal→RoPE、MHA→GQA/MLA——每一項換的都是「某個sublayer裡用哪個函式」,不是把encoder_block/decoder_block的wiring打掉重練,這也是本程式故意讓兩種設定共用同一份`encoder_block`/`decoder_block`程式碼的原因。

## 核心原理
- **六個零件,對照到程式碼**:

  | 零件 | 作用 | 對應的程式 |
  |---|---|---|
  | Embedding + 位置訊號 | token→向量,注入順序資訊 | 本程式刻意跳過,`src`/`tgt`直接視為已加好位置訊號的向量(見`Positional_Encoding`) |
  | Self-attention | 每個位置看所有位置(decoder裡是遮住未來) | `multi_head_attention`,`causal`參數控制遮罩 |
  | Feed-forward network | 逐位置的兩層(或三層)MLP,attention之外唯一的非線性變換 | `ffn_relu` / `ffn_swiglu` |
  | 殘差連接 | `x + sublayer(x)`,讓梯度能穿透深層堆疊 | `add` |
  | Normalization | 穩定residual stream的數值尺度 | `layer_norm` / `rms_norm` |
  | Cross-attention(僅decoder) | Query來自decoder,Key/Value來自encoder輸出 | `multi_head_attention(..., kv_source=enc_out)`,搭配`CrossAttentionParams` |

- **Encoder block(雙向,BERT/T5-encoder這一型用)**:
  ```
  x → norm → MHA(self) → +x → norm → FFN → +x → out
  ```
  沒有遮罩,每個位置都看得到所有位置。

- **Decoder block(GPT/T5-decoder這一型用)**:
  ```
  x → norm → MHA(masked self) → +x → norm → MHA(cross to encoder) → +x → norm → FFN → +x → out
  ```
  比encoder多一個sublayer——中間的cross-attention是唯一讓encoder資訊流進decoder的管道。純decoder-only架構(GPT)會整段拿掉cross-attention,只剩masked self-attention+FFN。

- **Pre-norm vs. post-norm**:原始論文是`LN(x + sublayer(x))`(post-norm);2026預設是`x + sublayer(LN(x))`(pre-norm)。本程式的`encoder_block`/`decoder_block`寫死走pre-norm wiring,兩種年代設定的差異只在`norm`這個函式指向`layer_norm`還是`rms_norm`,wiring本身不變。

- **2026現代化block長什麼樣**:

  | 零件 | 2017 | 2026 |
  |---|---|---|
  | Normalization | LayerNorm | RMSNorm |
  | FFN activation | ReLU | SwiGLU |
  | FFN expansion | 4× | 2.6×(SwiGLU用三個矩陣,總參數量打平) |
  | 位置編碼 | Sinusoidal絕對位置 | RoPE |
  | Attention | 完整MHA | GQA(或MLA) |
  | Bias項 | 有 | 無 |

  本程式demo了前三列(norm、FFN activation、FFN expansion),位置編碼跟attention變體留給`Positional_Encoding`/`Multi_Head_Attention`兩支程式。

- **為什麼`ffn_expansion`兩種設定不一樣(4.0 vs. 2.6)**:`ffn_relu`用兩個矩陣,參數量是`2·d·h`;`ffn_swiglu`用三個矩陣,參數量是`3·d·h`——同樣的`h`,SwiGLU會比ReLU-FFN多50%參數。要讓兩者總參數量打平,`h`要打2/3折:`2·d·(4d) = 3·d·(r·d)`解出`r = 8/3 ≈ 2.67`,教材取的2.6正是這個折算值的近似,不是隨便選的數字。

- **單一block的參數量**(`d_model=d`,FFN比例`r`):MHA是`4d²`(Q/K/V/O四個投影);FFN(SwiGLU)約`3rd²`;normalization的參數量可忽略。以`d=4096, r=2.6, 32層`(接近Llama 3 8B的規模)估算,單層約`4·4096² + 3·2.6·4096² ≈ 150M`,32層疊起來加上embedding/輸出層,量級對得上公開的參數量——這也是教材Exercise 1要你動手驗證的計算。

## 程式碼導覽
| 函式/類別 | 對應到理論的哪個部分 |
|---|---|
| `Matrix` / `randn` / `matmul` / `transpose` / `add` | 沿用`Self_Attention`同款的最小矩陣型別,純Python、無NumPy依賴;`add`就是殘差連接 |
| `softmax_rows` | 跟`Self_Attention`同款softmax,多了`mask`參數支援decoder的causal mask |
| `layer_norm` | 2017 LayerNorm:逐row減mean、除以std |
| `rms_norm` | 2026 RMSNorm:逐row除以RMS,不減mean,少一個運算 |
| `silu` / `ffn_relu` / `ffn_swiglu` | FFN的兩種實作:2017 ReLU-FFN(2個矩陣)vs. 2026 SwiGLU-FFN(3個矩陣,`silu(X·W1) * (X·W3)`再接`W2`) |
| `scaled_dot_product_attention` / `multi_head_attention` | 直接沿用`Multi_Head_Attention`程式的邏輯(換成本檔的`Matrix`型別);`causal`參數控制要不要遮未來,`kv_source`參數讓K/V來自另一個輸入(cross-attention用這個) |
| `BlockParams` | 一個block的self-attention+FFN權重,`use_swiglu`/`use_rmsnorm`兩個flag決定接哪一種FFN/norm |
| `CrossAttentionParams` | decoder專用的cross-attention Q/K/V/O權重,跟`BlockParams`分開——因為encoder block完全用不到這組權重 |
| `encoder_block` | Encoder的wiring:pre-norm self-attention+residual,再pre-norm FFN+residual |
| `decoder_block` | Decoder的wiring:masked self-attention、cross-attention、FFN三個sublayer,各自pre-norm+residual |
| `output_projection` | 「decoder之後接final norm、再投影到vocab logits」這一步,對應教材"Add a final LN before the output projection" |
| `run_transformer` | encoder跑完整個stack→decoder(帶cross-attention)跑完整個stack→`output_projection`,一次到位地接出教材的`encode`/`decode` |
| `demo_transformer_stack` / `main` | 用同一組`src`/`tgt`,先跑一次2017古典設定、再跑一次2026現代設定,印出encoder/decoder/logits的shape跟數值,驗證「換函式,shape不變」 |

**實作細節 / 容易看漏的地方:**
- encoder block跟decoder block共用同一個`BlockParams`(self-attention+FFN),但cross-attention的權重獨立成`CrossAttentionParams`,只有`decoder_block`會用到——這對應教材架構圖裡「cross-attention只存在於decoder」這件事,不是把一份鬆散的權重硬塞給每個block,也省下encoder block完全用不到的4個`(d, d)`矩陣。
- `ffn_expansion`在古典/現代兩組demo裡刻意設成不同值(4.0跟2.6),見上方「核心原理」的折算——如果兩邊都用同一個`ffn_expansion`,比較的就不是「換了SwiGLU」,而是「換了SwiGLU還多了50%參數」,不是公平的對照。
- `output_projection`把教材「decode之後接final LN再接輸出投影」跟「驗證shape是`(tgt_len, vocab)`」兩件事一次做完;`run_transformer`回傳的`dec_out`是**投影前**的hidden state,`logits`才是投影後的`(tgt_len, vocab_size)`結果,兩者分開回傳是為了讓demo兩個都印得出來對照。
- `demo_transformer_stack`跑古典/現代兩組設定時用不同的`seed`,是刻意的:如果共用同一個`rng`,兩組權重的隨機序列會不一樣長(SwiGLU比ReLU-FFN多一個`W3`矩陣要抽),後面所有權重都會跟著錯位,兩組demo就不是「只換了norm/FFN」的乾淨對照。
- 這支程式的`src`/`tgt`是直接random出來的向量,不是真的token embedding lookup加位置編碼——這個抽象層級跟`Multi_Head_Attention`程式的`make_toy_sentence`一致,重點是block本身怎麼wiring,不是完整的輸入pipeline(真正的embedding+位置編碼見`Positional_Encoding`)。
- 沒有訓練迴圈、沒有loss,`W_out`跟所有attention/FFN權重一樣是隨機初始化——教材本身在這一課也明講這一課是講架構,不是講loss。

## 使用時機 / 優缺點
- ✅ 想搞懂「一個transformer block裡到底疊了幾層東西,順序是什麼」,而且想親眼看到「把LayerNorm+ReLU換成RMSNorm+SwiGLU,shape完全不變、只是數值不同」這件事是怎麼發生的:`demo_transformer_stack`同一組input分別跑兩次,兩份print可以逐行對照。
- ✅ 需要一個對照組來理解「decoder為什麼比encoder多一個sublayer」:`decoder_block`的三段pre-norm+residual跟`encoder_block`的兩段並排看,中間那段cross-attention就是差異所在。
- ❌ 不要拿來訓練或部署:沒有backprop、沒有dropout,`Wq/Wk/...`從頭到尾隨機初始化,`vocab_size=12`只是為了讓`output_projection`的shape看得出來,不是真的字表。
- ❌ RoPE、GQA/MLA、post-norm都沒有實作或對照:位置編碼固定用「已經加好的隨機向量」代替(見`Positional_Encoding`);attention固定是plain MHA,沒有做GQA/MQA的head數縮減(見`Multi_Head_Attention`);normalization固定走pre-norm wiring,沒有pre-norm/post-norm的穩定度對照(教材Exercise 2留的延伸,見下方複習自問)。
- ❌ 沒有自動算參數量:教材Exercise 1要求數`encoder_block`在特定設定下的參數量,這支程式沒有寫`count_parameters`之類的函式去自動算,只在「核心原理」放了公式,有興趣可以照公式自己動手驗證。

## 常見誤區
1. **以為decoder一定要有cross-attention**:GPT系列是純decoder-only,壓根沒有cross-attention這個sublayer,只有masked self-attention+FFN;本程式的`decoder_block`是encoder-decoder架構(如T5)的版本,cross-attention是這種架構特有的,不是所有decoder都要。
2. **以為pre-norm/post-norm只是把normalization搬個位置,數值行為一樣**:`x + sublayer(LN(x))`(pre-norm)跟`LN(x + sublayer(x))`(post-norm)訓練起來的穩定度差很多——post-norm在沒有warmup的情況下疊到十幾層容易activation爆炸,這正是2019年後(Xiong et al.)幾乎所有正式模型都改用pre-norm的原因。本程式`encoder_block`/`decoder_block`寫死pre-norm,沒有做post-norm的對照。
3. **以為SwiGLU的`ffn_expansion`應該跟ReLU-FFN一樣是4**:SwiGLU用三個矩陣(`W1`/`W2`/`W3`)而不是兩個,同樣的`ffn_expansion`會讓SwiGLU多出約50%參數量,教材用2.6取代4正是為了讓兩者總參數量打平,不是隨便選的數字。
4. **以為`decoder_block`回傳的就是vocab logits**:`decoder_block`回傳的是hidden state(shape`(tgt_len, d)`),要再過一次`output_projection`(final norm+`W_out`)才會變成`(tgt_len, vocab_size)`的logits——這兩步在教材裡是分開的動作,程式裡也刻意讓`run_transformer`把`dec_out`跟`logits`分開回傳,方便對照。
5. **以為RMSNorm/LayerNorm數值上大同小異、可以互換不影響結果**:兩者都會讓每個row的尺度穩定下來,但RMSNorm不做mean-centering,對同一組輸入算出來的normalized值並不相同——這是選normalization函式時要意識到的實質差異,不只是「少一步運算」那麼單純。

## 複習自問
1. 如果把`decoder_block`裡的cross-attention sublayer整個拿掉,只留masked self-attention+FFN,會變成哪一種現實中存在的架構?那種架構下,`CrossAttentionParams`這個類別還需要存在嗎?
2. 教材Exercise 2要求把pre-norm換成post-norm,疊12層後量activation的norm。如果要在這支程式上做這個實驗,你會改`encoder_block`/`decoder_block`裡的哪一行?你預期兩種wiring疊到第12層時,`x`的數值量級大概會差多少個量級?
3. `ffn_expansion`在古典設定用4.0、現代設定用2.6,如果兩邊都固定用4.0,`ffn_swiglu`那組權重的總參數量會比`ffn_relu`多出多少百分比?(提示:`ffn_relu`是`2·d·h`,`ffn_swiglu`是`3·d·h`)
4. 教材Exercise 1要你算`d_model=512, n_heads=8, ffn_expansion=4, swiglu=True`時`encoder_block`的參數量。試著自己寫一個`count_parameters(p)`函式,把`BlockParams`裡每個`Matrix`的`rows*cols`加總,驗證跟「核心原理」的`4d²+3rd²`公式算出來的數字是否一致。
5. 教材Exercise 3要求在一個toy copy task上訓練4層encoder-decoder。如果要在這支程式上加訓練迴圈,`Matrix`型別完全沒有反向傳播——你會怎麼設計最小可行的梯度計算(手動反傳,還是乾脆換成NumPy/PyTorch重寫這幾個函式)?
