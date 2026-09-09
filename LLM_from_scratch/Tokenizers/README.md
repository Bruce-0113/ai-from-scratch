# Tokenizers (BPE / WordPiece / SentencePiece)

對應程式: [`./tokenizers.py`](./tokenizers.py)

參考:[ai-engineering-from-scratch – Tokenizers 課程](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/10-llms-from-scratch/01-tokenizers/docs/en.md)

## TL;DR
LLM不讀文字,只讀整數——tokenizer就是那個「文字→整數」的轉換規則,而且這個規則一旦訓練好就固定死,之後model看到的世界形狀就這樣定了。三種直覺做法(word-level、character-level)在正式場景都撐不住:word-level碰到沒看過的詞只能`[UNK]`,character-level序列長到把attention capacity都拿去學"t"+"h"+"e"="the"。所有現代LLM都改用**subword tokenization**,常見字保持完整、罕見字拆成有意義的片段,vocabulary大小落在可控範圍(30K~200K)。本程式用最小可行版本的**byte-level BPE**(GPT-2/3/4背後的演算法)示範這套機制:從256個raw byte開始,不斷數相鄰pair、合併最常出現的那一對,重複到達目標vocabulary size為止;訓練出來的merge table本身就是tokenizer。

## 為什麼需要它
- **[UNK]是不可接受的失敗模式**:word-level遇到沒看過的詞(新詞、拼字錯誤、程式碼、外語)就吐`[UNK]`,model等於瞎了一塊;subword保證任何輸入都能被某種方式表示出來,不會有真正的「看不懂」。
- **序列長度直接吃掉context window與推論成本**:同一段文字,character-level要50個token,subword可能只要10個——context window是有限資源,tokenizer效率差,等於少了一大截可用容量;而且每個token都要跑一次forward pass,token數越多,inference越慢、API bill越貴。
- **vocabulary大小是一個真實的工程取捨,不是隨便挑的超參數**:vocab越大,平均每個詞需要的token數越少(壓縮率越好),但embedding matrix(vocab_size × hidden_dim)線性變大——128K vocab配4096維embedding,光embedding就有5.24億參數,32K vocab只要1.31億,差距高達4億參數,完全來自tokenizer的選擇。
- **不同語言吃的虧不一樣("multilingual tax")**:訓練語料以英文為主的tokenizer,對韓文、中文的平均fertility(每個詞要拆成幾個token)明顯更高,同樣的context window,非英語使用者實際能塞的資訊量更少、付一樣的錢卻得到更差的體驗——這也是Llama 3把vocab從32K衝到128K的主因之一。

## 核心原理
- **BPE訓練迴圈**:給定corpus,先encode成byte序列(0~255,基礎vocab就是這256個值)。每一輪:數出所有相鄰pair的出現次數→挑出現次數最多的那一對→合併成一個新token id(`256 + i`)→把這個merge記進merge table→重複,直到達到`num_merges`或沒有pair可合併為止。訓練結束時vocabulary大小是`256 + num_merges`。
- **Merge table本身就是tokenizer,而且「順序」是核心不變量**:encode新文字時,必須按照merge被學到的**同一個順序**依序套用,不能任意排序或平行套用——如果merge 1先把"t"+"h"合併成"th",merge 5才能把"th"+"e"合併成"the";順序錯了,"the"就永遠合不出來。這也是為什麼`self.merges`要用一般dict(在Python 3.7+保留插入順序)而不是set。
- **BPE vs. WordPiece的merge判準不同**:BPE問「這對pair出現次數最多是哪個?」(`count(A,B)`),WordPiece(BERT用)問「這對pair一起出現的頻率,是不是明顯高於各自獨立出現時『純巧合』下的期望值?」(`count(AB) / (count(A) * count(B))`)。BPE找的是「常見」,WordPiece找的是「共現度顯著」——結果是不同的vocabulary。WordPiece另外用`##`前綴標記「這個片段是接續前一個token」,例如`"unhappiness" -> ["un", "##happi", "##ness"]`。
- **SentencePiece更進一步,連「先用空白分詞」這一步都拿掉**:它把輸入視為原始Unicode字元流(包含空白本身),不做任何語言相關的pre-tokenization規則,因此對中文、日文、泰文這種字詞之間沒有空白的語言天生友善。SentencePiece支援兩種演算法:BPE模式(邏輯同上,只是套用在原始字元上)與**Unigram模式**(反過來做:先給一個很大的候選vocabulary,每輪移除「拿掉後對整體likelihood影響最小」的token,是BPE的鏡像——用剪枝取代合併)。
- **Byte-level是關鍵的工程細節**:在原始byte(0~255)而不是Unicode字元上做BPE,保證了基礎vocabulary只有256個、天生涵蓋任何語言/編碼、**永遠不會有`[UNK]`**——即使是訓練語料完全沒見過的byte pattern,最差情況也能退化回一個一個byte輸出(對應到程式裡"unhappiness"這種未出現在訓練corpus裡的詞,壓縮率明顯變差的現象)。
- **業界vocabulary size與tokenizer類型對照**:

| Model | Vocab Size | Tokenizer類型 | 平均每個英文字的token數 |
|---|---|---|---|
| BERT | 30,522 | WordPiece | ~1.4 |
| GPT-2 | 50,257 | Byte-level BPE | ~1.3 |
| Llama 2 | 32,000 | SentencePiece BPE | ~1.4 |
| GPT-4 | ~100,256 (cl100k_base) | Byte-level BPE | ~1.2 |
| Llama 3 | 128,256 | Byte-level BPE (tiktoken) | ~1.1 |
| GPT-4o | 200,019 (o200k_base) | Byte-level BPE | ~1.0 |

## 程式碼導覽
| 函式/類別 | 對應到理論的哪個部分 |
|---|---|
| `CharTokenizer` | Character-level baseline:每個Unicode code point就是一個token,沒有訓練、沒有`[UNK]`,但序列長度是subword的好幾倍,用來對照BPE的壓縮效果 |
| `BPETokenizer._get_pairs` | 「數相鄰pair出現次數」——BPE訓練迴圈的第一步 |
| `BPETokenizer._merge_pair` | 「把某個pair合併成新token」——由左到右掃描,合併後pair的第二個元素被吃掉,不會跟下一個pair重疊 |
| `BPETokenizer.train` | 完整訓練迴圈:重複「數pair→挑最大→合併→記錄merge與vocab」,對應到「merge table就是tokenizer」這個核心概念 |
| `BPETokenizer.encode` / `decode` | encode依`self.merges`的學習順序重播所有merge;decode反查`self.vocab`把token id還原成byte再解成UTF-8字串 |
| `demo_roundtrip` | 對每個測試句子做encode→decode,驗證roundtrip無損,並印出「token數/原始byte數」的壓縮率 |
| `demo_tiktoken_comparison` | 拿同樣的句子餵給OpenAI的`tiktoken`(`cl100k_base`),對照「40次merge、單一段落訓練」的玩具tokenizer跟「~10萬次merge、上百GB語料訓練」的正式tokenizer差距有多大 |
| `analyze_vocabulary` | 統計vocabulary的使用分布(Zipf分布的具體呈現):最常用的10個token、多少比例的vocabulary在這批測試文字裡完全沒被用到 |

**實作細節 / 容易看漏的地方:**
- 原始教材版本定義了`analyze_vocabulary`卻沒有在demo流程裡呼叫它;這裡整理後在`main()`裡把`test_sentences`與`comparison_texts`一起餵給它,讓三個demo(roundtrip、tiktoken比較、vocabulary分析)在同一次執行裡完整跑完,而不是留一個「寫了但沒人用」的函式。
- `_merge_pair`合併時用`i += 2`跳過整個pair,所以連續重複的pattern(例如`"aaa"`要合併`(a,a)`)只會從左到右不重疊地合併,不會合併出重疊的結果——這跟正式BPE實作的語意一致,但如果不注意`i`的遞增方式很容易寫成會重疊合併的錯誤版本。
- `decode`用`errors="replace"`而不是讓例外往外丟:因為任意一段token id組成的byte序列,不保證會落在合法的UTF-8字元邊界上(尤其是還在訓練中途、或輸入是adversarial資料時),用`U+FFFD`取代掉不合法的byte比整個程式crash更務實。
- `train`裡沒有做任何pre-tokenization(不像正式tokenizer通常會先照空白/標點粗切一次再進BPE),所以像`"unhappiness"`這種不在corpus裡的詞,merge沒學過對應pattern,退化成接近char-level的10個token——這正好示範了「訓練語料的分布,直接決定了新文字的壓縮效果」。

## 使用時機 / 優缺點
- ✅ 想理解「LLM怎麼把文字變成整數」的底層機制,而不是只會呼叫`tokenizer.encode()`:這支程式把整個訓練/推論迴圈攤開來,沒有任何一行是隱藏在函式庫裡的黑盒。
- ✅ 需要對「vocabulary size怎麼影響壓縮率與embedding參數量」做直覺判斷:程式裡的`analyze_vocabulary`與跟`tiktoken`的對照,直接把「40 merges vs. 10萬merges」的差距量化出來。
- ❌ 不要把這支程式的`BPETokenizer`直接拿去訓練真正的model:沒有pre-tokenization、沒有特殊token(`<pad>`/`<eos>`/`<unk>`)、`train`是純Python迴圈(`O(merges × len(tokens))`,沒有用heap/優先佇列優化),語料一大就會慢到不可用——正式場景請用`tiktoken`(推論用,Rust實作)或Hugging Face `tokenizers`函式庫(訓練用,Rust實作,GB級語料幾秒內訓練完)。
- ❌ 不適合語言邊界不靠空白的場景(中文、日文、泰文)做進一步實驗:這支程式的BPE是在UTF-8 byte上做,理論上可以處理任何語言,但沒有實作SentencePiece式的語言無關pre-tokenization,拿中文語料訓練出來的merge品質不會太好,想驗證這塊建議直接讀SentencePiece論文或用其函式庫。

## 常見誤區
1. **以為「訓練BPE」跟「壓縮字串」是同一件事,merge數越多必然越好**:merge數是vocabulary size的另一種說法,vocab越大確實平均token數越少,但embedding matrix大小是`vocab_size × hidden_dim`,vocab每長大一截都要多花實打實的參數量與訓練成本——不是「無腦調大就贏」,是有tradeoff的工程決策(見上面vocab size表格)。
2. **忽略merge順序,以為`merges`是一個無序集合**:如果拿掉Python dict保留插入順序這個前提(例如換成存進`set`或做了排序),`encode`重播merge的順序就會跟訓練時不一致,結果是「同一段文字,不同次執行encode出不同的token序列」這種難以察覺的bug。
3. **拿玩具tokenizer(小語料、少量merge)的token數,直接跟`tiktoken`之類的正式tokenizer比較優劣**:程式裡故意把兩者並排印出來,就是要凸顯「同一個演算法,語料規模與merge次數差了幾個數量級,結果天差地遠」——差距來自訓練資料量,不是演算法本身有問題。
4. **以為decode永遠不會出錯,直接假設`bytes.decode("utf-8")`一定成功**:如果token id組合起來的byte序列剛好切在多位元組UTF-8字元中間(理論上訓練良好的tokenizer很少發生,但拿任意token id序列手動組合、或跟其他tokenizer的token id混用時可能發生),不做`errors="replace"`就會直接丟`UnicodeDecodeError`。

## 複習自問
- 為什麼BPE的merge table「必須按照學習順序重播」,而不能先蒐集完所有merge規則、再依「pair長度」或「字母順序」重新排序後套用?如果打亂順序會發生什麼具體錯誤?
- vocabulary size從32K調到128K,對「壓縮率」「embedding參數量」「非英語文字的fertility」三者分別會造成什麼方向的影響?這三個影響彼此之間有沒有互相矛盾的地方?
- BPE跟WordPiece的merge判準(`count(A,B)` vs. `count(AB)/(count(A)*count(B))`)分別容易在什麼樣的語料特性下,選出「不符合語言直覺」的merge?
- 如果你在正式產品裡發現同一段文字在不同時間點被encode出不同的token id序列,你會依序檢查哪些環節?(提示:merge順序、tokenizer版本是否被熱更新、輸入是否經過看不見的normalization)
