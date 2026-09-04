# Text Summarization（TextRank 抽取式 + BART 生成式 + ROUGE 評估）

對應程式：[`./text_summarization.py`](./text_summarization.py)

## TL;DR
摘要有兩種完全不同的做法：抽取式（extractive）從原文挑幾句「原封不動」貼出來，生成式（abstractive）則是用模型「重新寫」一段新文字。這份程式示範抽取式的代表演算法TextRank（把句子當節點、相似度當邊權重，跑PageRank排序）、生成式的代表模型BART（`facebook/bart-large-cnn`，用`transformers.pipeline`直接生成摘要），以及業界最常用的自動評估指標ROUGE（用n-gram/最長共同子序列重疊比對生成摘要與參考摘要）。

## 為什麼需要它
- 摘要是搜尋結果預覽、會議記錄、新聞導讀、法律/財報重點整理等應用的共同底層需求——把長文壓縮成短文，但「怎麼壓縮」直接決定使用者能不能信任這段輸出。
- 抽取式與生成式是兩種本質不同的風險取捨：抽取式因為句子逐字取自原文，文法一定通順、也不會捏造事實，但讀起來可能像「劃重點」而非流暢摘要；生成式讀起來更像人寫的摘要，但模型是「生成」而非「複製」，存在憑空捏造內容（hallucination）的風險，法律、醫療、金融等受監管領域若不做事實性檢查就直接上線生成式摘要，風險很高。
- ROUGE只比對字面上的n-gram重疊，量不到「這句話事實正不正確」——生成式摘要就算ROUGE分數很高，也可能把「上漲5%」寫成「上漲50%」或把「A公司」寫成「B公司」，這類錯誤是抽取式摘要不可能犯，卻是生成式摘要的常見失敗模式。
- 摘要沒有單一「最佳解」：新聞導讀、學術論文、對話記錄、多文件彙整，各自需要不同的模型與評估策略，選型前要先想清楚「抽取式夠用，還是真的需要生成式」。

## 核心原理

### 1. 抽取式 vs 生成式：兩種摘要典範
抽取式系統告訴你「原文說了什麼」——排序後直接挑出原文裡的句子，不生成任何新文字，所以文法保證通順、也不會無中生有。生成式系統告訴你「作者想表達什麼」——用encoder-decoder模型逐token生成一段全新文字，能做到抽取式做不到的改寫、濃縮與跨句整合，但也因此有可能生成原文沒說過的內容。這份程式的第一部分（TextRank）示範抽取式，第二部分（BART）示範生成式。

### 2. TextRank：把文件當成句子相似度圖跑PageRank（`sentence_split` / `similarity` / `textrank`）
TextRank（Mihalcea & Tarau, 2004）把每個句子當成圖上的一個節點，句子與句子之間的邊權重是兩句共享詞彙的多寡（`similarity`：詞交集數量除以兩句字數的log和，log正規化避免長句只因為字多就跟每句都很像）。有了這張完整圖之後，`textrank`跑PageRank的冪迭代（power iteration）：每一輪，每個句子把自己目前的分數依照邊權重比例分給鄰居，並乘上`damping`（沿邊走訪的機率，`1-damping`是跳到隨機句子的機率，標準值0.85），直到所有分數變化小於`epsilon`或跑滿`iterations`輪為止。分數最高的`top_k`句被選出後，會依照它們在原文裡的原始順序重新排列再回傳——如果直接依分數高低輸出，讀起來會像被打散的句子清單，而不是連貫的摘要。

### 3. BART 生成式摘要（`facebook/bart-large-cnn`）
BART是encoder-decoder架構的transformer：encoder讀入完整文章，decoder逐token生成新的摘要文字，而不是從原文複製句子。`facebook/bart-large-cnn`是在CNN/DailyMail新聞語料上微調過的版本，程式裡用`transformers.pipeline("summarization", model=...)`一行就能載入並推論。`max_length` / `min_length`控制輸出token數的上下限，`do_sample=False`代表用貪婪/beam解碼而非隨機採樣，同樣的輸入每次會得到同樣的輸出。

### 4. ROUGE 評估：n-gram / 最長共同子序列重疊
ROUGE（Recall-Oriented Understudy for Gisting Evaluation, Lin 2004）是摘要任務最常見的自動評估指標，靠比對生成摘要與人工參考摘要之間的文字重疊程度打分，而不是理解語意。原始論文的核心公式是以**召回率（recall）**為主：ROUGE-N = (所有參考摘要中，與生成摘要相符的n-gram數量總和) / (所有參考摘要的n-gram總數)——分子分母都可以對多份參考摘要加總，讓評分不會只依賴單一一種「標準答案」寫法。像`rouge_score`這樣的現代套件則更進一步，同時算出precision、recall、F1（`fmeasure`），這份程式最後只取用`v.fmeasure`（F1）。

**ROUGE-1（unigram）範例**：參考摘要1「police killed the gunman」（4字）、參考摘要2「the gunman was shot down by police」（7字）、生成摘要「police ended the gunman」。生成摘要與參考摘要1相符的字有3個（police、the、gunman），與參考摘要2相符的也是3個，所以ROUGE-1 = (3+3)/(4+7) = 6/11 ≈ 0.545。

**ROUGE-2（bigram）範例**：同樣三句話，只有「the gunman」這個bigram同時出現在生成摘要與兩份參考摘要中，所以ROUGE-2 = (1+1)/(3+6) = 2/9 ≈ 0.222（4字的句子有3個bigram，7字的句子有6個bigram）。bigram重疊比unigram更能反映語序，但也因此分數通常比ROUGE-1低很多，兩者不能直接放在一起比大小。

**ROUGE-L（最長共同子序列，LCS）**：LCS允許字詞之間不連續也算數，只要相對順序一致即可，因此不需要像ROUGE-N一樣事先決定n-gram的長度。以參考摘要「police killed the gunman」為例：候選句「police ended the gunman」的LCS是「police the gunman」（跳過killed/ended這組不同的詞），得分0.75；另一候選句「the gunman murdered police」的LCS只有「the gunman」，得分0.50，正確反映出詞序被打亂、語意也更失真。這是ROUGE-L相對ROUGE-2的優勢：這兩句在ROUGE-2下可能被打成一樣的分數（因為bigram重疊數量相近），但ROUGE-L能分辨出詞序被保留與被打亂的差異。ROUGE-L的限制是只取「最長的那一段」共同子序列，其餘沒被選中、但同樣有意義的重疊片段會被忽略。

**ROUGE-W（加權LCS）**：LCS本身分不出「連續3個字都對上」和「對上的3個字分散在句子各處」有什麼差別——兩者算出來的LCS長度一樣。ROUGE-W替LCS加上一個依連續長度加權的函數（例如長度的平方），讓連續命中的區段得分更高，藉此獎勵詞序保留得更完整的句子。

**ROUGE-S / ROUGE-SU（skip-bigram）**：不要求兩個詞連續出現，只要詞序一致、中間可以跳過任意字詞，就算一組skip-bigram。以「police killed the gunman」（4個字）為例，共有C(4,2)=6組skip-bigram（police-killed、police-the、police-gunman、killed-the、killed-gunman、the-gunman）。候選句「police ended the gunman」對到其中3組（police-the、police-gunman、the-gunman），得分0.50；「the gunman murdered police」只對到1組（the-gunman），得分約0.167。skip-bigram比LCS更寬鬆：它會檢查所有可能的詞對組合，而不是只挑一條最長的子序列，代價是計算量比LCS更大。

`use_stemmer=True`會先把單字做詞幹化（例如"running"與"run"視為同一個詞）再比對，否則單純的詞形變化就會拉低分數。原始論文的實驗結論也建議：ROUGE-2 / ROUGE-L / ROUGE-W / ROUGE-S在單文件摘要評估上表現較好，ROUGE-1 / ROUGE-L / ROUGE-W / ROUGE-SU4在「非常短的摘要」（如標題）上表現較好，去除停用詞（stopwords）與使用多份參考摘要都能提高跟人類評分的相關性。業界另一個常見的粗略經驗法則是「ROUGE-L到40分算不錯，50分算優異」，但這類數字高度依賴資料集與摘要長度，不能直接套用到任何語料上當作絕對門檻。

### 5. 生成式摘要的事實性問題（hallucination）
生成式摘要最大的風險是模型生成了原文沒有支持的內容，常見可歸納成四類錯誤：**實體替換**（人名/地名/公司名寫錯，例如把甲公司寫成乙公司）、**數字漂移**（數量級或數值改變，例如「500萬」變成「5000萬」）、**極性反轉**（正負面情緒寫反，例如把「否決」寫成「通過」）、**憑空捏造**（原文根本沒提到的事實）。偵測方法包括：用專門訓練的分類器（如FactCC）判斷摘要是否被原文支持、用QA方式驗證（對摘要提出問題，看原文與摘要能否給出一致答案）、以及用NER抽出摘要與原文中的實體做entity-level F1比對。這份程式沒有實作事實性檢查，正式上線生成式摘要前，這是不能省略的一步。

### 6. ROUGE之外：更貼近語意與人類判斷的評估方式
ROUGE只看字面重疊，摘要用不同措辭表達同樣意思時分數會被低估，因此業界發展出更進階的評估方法：**BERTScore**用contextual embedding算相似度，能抓到同義改寫；**BARTScore**把評估本身也框成一個生成任務，用生成機率當分數；**MoverScore**用Earth Mover's Distance衡量兩段文字的語意分佈距離；**G-Eval**則是設計prompt鏈讓大型語言模型（如GPT-4）直接當裁判評分，實測與人類判斷的相關性可達約80%，但也繼承了LLM評分本身的不穩定性與成本。這些方法同樣沒有在這份程式中實作。

## 程式碼導覽
| 函式 / 區塊 | 對應到理論的哪個部分 |
|---|---|
| `sentence_split(text)` | 抽取式前處理：用正則切句（`(?<=[.!?])\s+`），簡化版的句子邊界偵測 |
| `similarity(s1, s2)` | TextRank相似度公式：詞交集數量 / 兩句字數的log和 |
| `textrank(text, top_k, damping, iterations, epsilon)` | 建立句子相似度圖 → PageRank冪迭代 → 依原文順序回傳前`top_k`句 |
| `pipeline("summarization", model="facebook/bart-large-cnn")` | 載入微調過的BART encoder-decoder模型 |
| `summarizer(article, max_length=120, min_length=60, do_sample=False)` | 生成式摘要推論：貪婪解碼、輸出長度限制在60-120 token之間 |
| `rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)` + `scorer.score(...)` | ROUGE-1/2/L的F-measure評估，`use_stemmer=True`先做詞幹化再比對 |

**實作細節 / 容易看漏的地方：**
- `textrank`函式定義好之後，這份程式並沒有實際呼叫它去處理`article`——它是抽取式做法的獨立示範區塊，跟後面BART的生成式流程是兩條平行路線，不是串接在一起的pipeline。
- 檔案最後的ROUGE區塊是示意用的：`reference_summary`（人工參考摘要）與`generated_summary`（例如BART輸出的`summary[0]["summary_text"]`）並沒有在檔案裡被賦值，需要呼叫端自行提供。
- `scorer.score(reference_summary, generated_summary)`只接受單一參考摘要、單一生成摘要，不是原始ROUGE論文裡「對多份參考摘要加總分子分母」的寫法；若手上有多份人工參考摘要，需要自己對每份分別呼叫`.score()`再做彙整（例如取平均或取最高分）。
- 程式只讀取`v.fmeasure`（F1），沒有用到`rouge_score`同時算出的`precision`、`recall`——如果想知道摘要「漏了多少參考重點」（recall）或「摻了多少參考沒提到的內容」（precision，非事實性檢查，純字面比對），需要額外印出這兩個欄位。
- `rouge_score`套件只實作ROUGE-1/2/L（LCS-based），沒有ROUGE-W（加權LCS）或ROUGE-S/SU（skip-bigram）這兩種變體，這兩者仍停留在README的概念說明，沒有對應的程式碼可直接呼叫。
- `similarity`只有在兩句都是空字串時才會回傳0.0（此時分母的兩個log都是`log(1)=0`）；只有一句是空字串時分母不會是0，因為`log(0+1)`本身就是0，不影響非空的那一句。
- `textrank`裡`sim[i][j]`是對稱矩陣（`similarity(a,b)==similarity(b,a)`），所以雖然程式用「`i`把分數分給`j`」的有向寫法迭代，實質上跑的是無向圖版本的PageRank。
- 事實性檢查（FactCC / QA-based / entity-level F1）與ROUGE之外的評估方法（BERTScore、BARTScore、MoverScore、G-Eval）都只在README的核心原理裡說明概念，這份`.py`檔案沒有對應的程式碼。

## 使用時機 / 優缺點
依應用場景挑摘要模型（起始建議，仍需在自己的資料上驗證）：

| 應用場景 | 建議方向 |
|---|---|
| 新聞導讀 | BART/PEGASUS等在新聞語料微調過的生成式模型 |
| 學術論文 | 針對長文件微調的模型（如PEGASUS-arXiv、Longformer-Encoder-Decoder） |
| 對話/會議記錄 | 在對話語料微調過的生成式模型（如BART-SAMSum） |
| 多文件彙整 | 先各自摘要再彙整的map-reduce流程，通常搭配LLM |
| 法律/醫療/金融等受監管內容 | 優先考慮抽取式，或生成式搭配強制的事實性檢查關卡 |

- ✅ 抽取式（TextRank）：不需要GPU、不需要訓練資料、不會捏造事實，適合當快速上線的baseline，或事實準確性優先於流暢度的場景。
- ✅ 生成式（BART）：摘要讀起來更精煉、更像人寫的，能做抽取式做不到的跨句改寫與濃縮，適合新聞導讀、對話摘要等追求可讀性的場景。
- ✅ ROUGE：計算便宜、可重現、業界共同語言，適合當開發過程中快速迭代的自動化指標。
- ❌ 抽取式的天花板較低：句子只能整句選取或捨棄，沒辦法把兩句拆開重組成更精簡的新句子，壓縮率有限。
- ❌ 生成式在沒有事實性檢查的情況下，不適合直接用於法律、醫療、金融等錯誤成本很高的場景——見上方「事實性問題」小節的四種常見錯誤類型。
- ❌ ROUGE分數高不代表摘要事實正確、也不代表語意正確：同義改寫可能被低估分數，捏造內容如果字面上剛好重疊也可能拿到不錯的分數，正式評估建議搭配BERTScore、G-Eval等方法或人工抽樣檢查。

## 常見誤區
1. **抽取式摘要只是簡單的規則系統，效果一定比生成式差**：TextRank本質上是一個圖演算法（PageRank的變體），在很多場景下抽取式摘要的表現並不差，尤其是新聞這種「重點通常整句出現」的文體；是否需要生成式，要看場景是否真的需要改寫與濃縮。
2. **生成式摘要讀起來比較通順，所以品質一定比較好**：可讀性與正確性是兩件事，生成式摘要的流暢文字底下可能藏著實體替換、數字漂移等幻覺錯誤，不能用「讀起來順不順」當作品質判斷依據。
3. **ROUGE分數高就代表摘要品質好**：ROUGE只量字面上的n-gram/LCS重疊，量不到事實正確性，也可能低估用不同措辭表達相同意思的優質摘要。
4. **「ROUGE-L 40分算好、50分算優異」這種數字可以套用在任何資料集上**：這類經驗法則是特定資料集與任務下的觀察，摘要長度、資料領域不同，合理的ROUGE分數範圍也會不同，比較時務必對齊同一個資料集與基準。
5. **忘記設定`use_stemmer=True`不影響比較公平性**：不做詞幹化時，"summarize"與"summarizing"會被當成完全不同的詞，同一模型在有無詞幹化下的ROUGE分數不能直接互相比較。
6. **ROUGE-2分數一樣，代表兩份摘要的品質也一樣**：ROUGE-2只看bigram重疊「數量」，「police ended the gunman」與「the gunman murdered police」對同一份參考摘要可能算出相近的ROUGE-2分數，但後者詞序被打亂、語意也偏離原意更多——這正是ROUGE-L（LCS要求相對順序一致）存在的原因，只看單一ROUGE變體容易漏掉這類差異。
7. **只要用單一份參考摘要評分就足夠**：原始ROUGE論文的公式設計成可以對多份參考摘要加總分子分母，論文本身的實驗也指出使用多份參考摘要能提高跟人類評分的相關性；只用一份參考摘要（就像這份程式`scorer.score()`的預設用法）容易把「跟這位標註者寫法不同」誤判成「摘要品質不好」。
8. **只要換成生成式摘要，事後就不用再檢查了**：即使選用了效果很好的生成式模型，正式上線前仍需要事實性檢查（FactCC / QA-based / entity-level F1其中至少一種），這不是可以省略的步驟，尤其是內容錯誤成本高的領域。

## 複習自問
- `textrank`裡的`damping`參數代表什麼意義？如果把它設成1.0（沒有`1-damping`的隨機跳轉項），迭代過程會有什麼問題？
- 為什麼`textrank`選出`top_k`句之後，還要依照原文順序重新排列，而不是直接依分數高低輸出？
- `similarity`函式為什麼要用`log(len(w1)+1) + log(len(w2)+1)`當分母，而不是直接用交集詞數除以總詞數？
- 生成式摘要的四種常見幻覺錯誤（實體替換、數字漂移、極性反轉、憑空捏造）之中，哪一種是抽取式摘要在原理上就不可能發生的？為什麼？
- ROUGE只比對字面重疊，在什麼情境下會低估一段其實語意正確的摘要？BERTScore這類方法想解決的正是哪一個問題？
- 「police ended the gunman」與「the gunman murdered police」對參考摘要「police killed the gunman」算ROUGE-2時分數相近，但ROUGE-L明顯不同（0.75 vs 0.50）。為什麼ROUGE-L能分辨出這兩句的差異，而ROUGE-2不容易？
- ROUGE-W替LCS加上連續長度的加權，是為了解決LCS原本的哪一種盲點？如果兩句話的LCS長度相同、但一句是連續命中、另一句是分散命中，原始ROUGE-L能分辨出差異嗎？
- 這份程式的`scorer.score(reference_summary, generated_summary)`只用了一份參考摘要，如果要照原始ROUGE論文的公式改成支援多份參考摘要，分子分母大致要怎麼調整？
- 如果要替法律合約的重點摘要選一種做法，你會優先考慮抽取式還是生成式？為什麼？
