# Quantization (Number Formats / GPTQ / AWQ)

對應程式: [`./quantization.py`](./quantization.py)

參考:[ai-engineering-from-scratch – Quantization 課程](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/10-llms-from-scratch/11-quantization/docs/en.md)

## TL;DR
Llama 3 70B有700億參數,FP16下每個參數佔2 bytes,光權重就要140GB——單張80GB的A100根本放不下,得兩張卡才擠得進去。但16 bit其實浪費得離譜:神經網路的權重絕大多數集中在0附近,實測Llama 3 70B有95%的權重落在[-0.1, 0.1]之間,FP16完整的動態範圍(±65504)幾乎沒用到。Quantization做的事很單純:找一個scale factor,把浮點數壓進更少的bit裡,FP16→INT4能把140GB壓到35GB,單卡消費級GPU就塞得下。代價是精度——每砍一個bit就摧毀一些資訊,技術好壞的差別,就在於摧毀的是「反正不重要」的資訊,還是「其實很關鍵」的資訊。本程式用numpy把FP32/FP16/BF16/FP8的bit layout、對稱/非對稱/per-channel量化、GPTQ與AWQ的簡化版核心邏輯,全部從頭攤開實作一遍。

## 為什麼需要它
- **16 bit對神經網路權重來說是過度配置**:權重分布接近高斯、集中在0附近,FP16的動態範圍(涵蓋到±65504)絕大部分用不到——這不是「模型需要16 bit才能準」,是浮點格式本來就是為通用數值設計的,拿到權重這種特化分布上自然有大量浪費空間,而quantization就是把這塊浪費榨出來。
- **量化不是玩具技巧,是每個超過7B模型的標準部署路徑**:社群把Llama 3量化到INT4(GPTQ)後,WikiText上perplexity只掉1-2個點;Mistral釋出Mixtral 8x22B的FP8 checkpoint,在MMLU上量到的品質損失是0——llama.cpp能在MacBook上跑70B模型,靠的正是GGUF格式的量化。
- **省下來的不只是顯存,吞吐量也跟著漲**:FP16→FP8在H100上有30-50%的推論加速、品質損失小於0.1%;FP16→INT8(LLM.int8())記憶體減半,品質損失小於0.5%;FP16→INT4(GPTQ/AWQ)記憶體剩1/4,品質損失1-3%,讓70B模型塞進單張48GB GPU。
- **技術好壞決定量化能不能用,不是「量不量化」本身的取捨**:naive的INT4量化可以直接把模型弄壞,但技術得當的INT4能保留原模型95-99%的品質——差別在於怎麼決定scale、怎麼挑哪些權重要被更小心地對待,這正是本程式`quantize_symmetric`(naive)跟`simulated_gptq`/`simulated_awq`(講究技術)之間的對照。

## 核心原理
- **浮點數的三個部分(`float_to_fp32_bits` / `float_to_fp16_bits` / `float_to_bf16_bits` / `simulate_fp8_e4m3`)**:sign決定正負,exponent決定數值的範圍(多大多小),mantissa決定精度(有幾位有效數字)。FP32是`[1 sign][8 exponent][23 mantissa]`,精度約7位十進位數字;FP16砍到`[1 sign][5 exponent][10 mantissa]`,範圍大幅縮小(最大值約65504),對集中在0附近的權重還好,但拿來裝training時會爆掉的activation/gradient就很危險。BF16(`float_to_bf16_bits`)保留FP32的8-bit exponent(範圍不變)只砍mantissa到7 bit——Google的設計哲學是「對深度學習來說,範圍比精度重要」:FP16裡會underflow成0的極小梯度,BF16裝得下;FP32裡0.07342在BF16裡變成約0.0734,精度損失可以接受。FP8(`simulate_fp8_e4m3`)只剩4-bit exponent、3-bit mantissa(E4M3,推論常用;另有E5M2給訓練用的梯度,範圍優先於精度)。INT8/INT4沒有exponent/mantissa的區分,單純是均勻分布的整數格,精度完全靠外部的scale factor撐。
- **量化的核心操作只有兩步(`quantize_symmetric` / `dequantize_symmetric`)**:找一個`scale = max(abs(tensor)) / qmax`,把浮點值除以scale再四捨五入存成整數;還原時整數乘回scale。誤差來源就是這次四捨五入,單一數值的誤差最多是`scale / 2`。
- **Per-tensor vs. per-channel(`quantize_per_channel` / `dequantize_per_channel`)**:per-tensor整個weight矩陣共用一個scale,如果某一行(channel)數值特別大、另一行特別小,小的那行會被大的那行拖累,精度被犧牲掉;per-channel替每一行(或每一列)算各自的scale,多付出的成本只是要多存N個scale factor,但品質提升非常明顯——正式的量化方法幾乎都用per-channel或更細的粒度(如GPTQ/AWQ實際用的per-group)。
- **Asymmetric量化補上zero-point(`quantize_asymmetric` / `dequantize_asymmetric`)**:對稱量化假設數值分布以0為中心,但像ReLU之後的activation永遠不會是負的,對稱量化等於把一半的整數範圍浪費在永遠不會出現的負值上。Asymmetric量化改成`quantized = round(tensor / scale) + zero_point`,把實際的`[min, max]`對應到整個無號整數範圍,不留浪費。
- **敏感度不是均勻的(`simulate_transformer_layer` / `sensitivity_experiment`)**:同樣的bit數,量化模型的不同部位造成的傷害差很多。權重(訓練時緩慢變化、接近高斯分布)最耐量化;activation範圍更寬、常有outlier(單一attention head的activation可能是平均值的100倍),量化不慎會摧毀關鍵資訊;KV cache的誤差會沿著後續每一次attention累積,長context下影響更明顯;attention logits最脆弱,因為softmax會把pre-softmax的小誤差放大成attention分布的明顯偏移——這也是為什麼多數量化方案讓attention計算留在FP16/BF16,其他都量化。程式裡的`sensitivity_experiment`直接把這個階層量出來,依MSE排序印出「Weights only」「Activations only」「KV cache only」「Attention logits (5% noise)」四種對照。
- **PTQ vs. QAT**:Post-Training Quantization(PTQ)直接對訓練好的模型套用量化,不需要重新訓練,幾分鐘到幾小時就能做完,INT8/FP8下效果很好;但INT4的naive PTQ常常因為誤差累積而失敗,需要GPTQ、AWQ這類進階方法搭配calibration data。Quantization-Aware Training(QAT)在training的forward pass裡插入fake quantization,讓模型自己學會把權重放到「量化誤差小」的位置,梯度靠straight-through estimator(把四捨五入的梯度視為1)往回傳——品質比PTQ好,但要付出一次完整訓練的成本,Google在Gemini、Meta在部分Llama部署上都用了QAT。

| 面向 | PTQ | QAT |
|---|---|---|
| 成本 | 幾分鐘到幾小時 | 一次完整訓練 |
| INT8品質 | 極佳(<0.1%損失) | 極佳 |
| INT4品質 | 搭配GPTQ/AWQ尚可(1-3%損失) | 更好(<1%損失) |
| INT2品質 | 差 | 部分任務堪用 |
| Calibration資料 | 128-1024筆 | 完整訓練資料集 |
| 適用時機 | 部署、快速迭代 | 低bit下要求最高品質 |

- **GPTQ(`simulated_gptq` / `dequantize_gptq`)**:一次性PTQ方法,用少量calibration data(通常128筆)估計Hessian(這個權重對輸出有多敏感的二階資訊),逐column量化,把這一column的量化誤差按重要性補償到後面幾個column,讓「重要」的權重被照顧得更仔細。本程式的簡化版拿H的對角線(每個輸入特徵的平均平方activation)當重要性的替代品,而非真正的inverse-Hessian曲率,但抓住了「用calibration data決定量化順序與誤差補償」這個核心概念。
- **AWQ(`simulated_awq`)**:觀察到只有大約1%的權重格外重要——不是因為權重本身數值大,而是因為它們會被特別大的activation值相乘。AWQ用calibration data找出這些「salient」權重,量化前先把它們對應的input row放大(讓量化後的相對精度提升),量化後再把整體除回來抵銷這個縮放。品質通常跟GPTQ相當甚至略勝,而且套用速度快1.5-2倍。
- **GGUF**:llama.cpp生態用的檔案格式,支援mixed precision——第一層和最後一層(embedding、output head)通常留高精度,中間層用INT4或INT3;模型權重、tokenizer、metadata全部包在單一檔案裡,是為CPU/Apple Silicon推論設計的格式,`Q4_K_M`是目前最流行的GGUF量化變體。

## 量化方法效果對照(引用課程教材的量測數字)

| Model | Format | 大小 | Perplexity (WikiText-2) | MMLU | Tokens/sec (A100) |
|---|---|---|---|---|---|
| Llama 3 70B | FP16 | 140GB | 3.12 | 79.5% | 38 |
| Llama 3 70B | FP8 | 70GB | 3.14 | 79.3% | 55 |
| Llama 3 70B | GPTQ INT4 | 35GB | 4.32 | 77.8% | 72 |
| Llama 3 70B | AWQ INT4 | 35GB | 4.18 | 78.1% | 75 |
| Llama 3 70B | GGUF Q4_K_M | 40GB | 4.25 | 77.9% | 28 (CPU) |

規律很清楚:FP8幾乎是免費的午餐(品質幾乎不掉、速度提升明顯);INT4要付1-2個MMLU百分點的代價,換來吞吐量翻倍、記憶體砍到1/4——這筆交易在幾乎所有部署場景下都划算。

## 程式碼導覽
| 函式/類別 | 對應到理論的哪個部分 |
|---|---|
| `float_to_fp32_bits` / `float_to_fp16_bits` / `float_to_bf16_bits` / `simulate_fp8_e4m3` | 把數值拆解成sign/exponent/mantissa的實際bit,`simulate_fp8_e4m3`是因為numpy沒有原生FP8型別,手動重建E4M3的行為 |
| `display_format_comparison` | 印出同一個數值在FP32/FP16/BF16/FP8下分別存成什麼、誤差多大的對照表 |
| `quantize_symmetric` / `dequantize_symmetric` | Per-tensor對稱量化:整個張量共用一個scale,最基本、也最容易被outlier拖累的量化方式 |
| `quantize_per_channel` / `dequantize_per_channel` | 每個channel(row或column)各自算scale,是GPTQ/AWQ等正式方法的基礎粒度 |
| `quantize_asymmetric` / `dequantize_asymmetric` | 多一個zero-point,把非零中心的分布(如ReLU後的activation)對應到完整的無號整數範圍 |
| `quantization_error` | 算MSE、RMSE、最大誤差、SNR(dB)、cosine similarity,是貫穿全部demo的量化品質量尺 |
| `compare_quantization_methods` | 同一個張量,對稱/per-channel/asymmetric三種量化方式的誤差並排比較 |
| `bit_width_sweep` | 同一個張量在2/3/4/8/16 bit下的品質曲線,找出「品質懸崖」大概落在哪個bit數 |
| `simulate_transformer_layer` | 極簡的單層self-attention(QKV投影→scaled dot-product attention→輸出投影),暴露所有中間張量供敏感度實驗替換 |
| `sensitivity_experiment` | 依序只量化weights/activations/KV cache/attention logits其中一項,量出四者的敏感度階層 |
| `simulated_gptq` / `dequantize_gptq` | 簡化版GPTQ:用calibration data估計的Hessian對角線當重要性指標,逐column量化並把誤差補償到後面幾個column |
| `simulated_awq` | 簡化版AWQ:找出被大activation值相乘的salient weight,量化前放大、量化後除回來 |
| `full_quantization_comparison` | 在含outlier的合成權重矩陣上,把naive per-tensor、per-channel、GPTQ、AWQ四種方法的重建誤差與下游matmul誤差全部串起來比較 |
| `memory_calculator` / `print_memory_table` | 給定參數量(billions)與每參數bit數,算出實際佔用的顯存(GiB),印出7B~405B模型在FP32~INT2下的記憶體需求表 |

**實作細節 / 容易看漏的地方:**
- `simulated_gptq`裡真正拿來當「重要性」依據的是`np.diag(H)`(calibration activation的平均平方值),這只是inverse-Hessian曲率的粗略替代品,不是論文裡真正的二階最佳化;而且誤差補償只往後傳3個column(`col + 1`到`col + 4`),不是像真正GPTQ那樣傳給後面所有column。這支程式抓住的是「用calibration data決定量化順序與補償」的核心精神,不是可以直接拿去量化真實模型的實作。
- `quantize_per_channel`的`axis`參數容易搞混方向:`axis=0`是「沿著axis 1取abs max」,也就是每一row各自一個scale;`axis`給別的值則是每一column各自一個scale。`dequantize_per_channel`必須用同一個`axis`才能把`scales`reshape回正確的方向(row用`(-1, 1)`,column用`(1, -1)`),`axis`不一致會讓還原出來的值整個錯位。
- `simulate_fp8_e4m3`裡`exp = max(-6, min(8, exp))`這段刻意把exponent範圍夾在`[-6, 8]`——對應E4M3的4-bit exponent(bias為7,理論範圍0~15、實際去掉部分給特殊值後大約是-6~8)——如果把這個clamp拿掉,數值超出E4M3實際能表示的範圍時會算出不合理的bit pattern。
- `sensitivity_experiment`裡「KV cache only」跟「Weights only」用的都是`quantize_per_channel`,唯獨「Attention logits」那組沒有真的做量化,而是直接對`attn_scores`加5%的高斯雜訊來模擬量化誤差的效果——這是因為attention logits理論上要量化的是softmax前的分數本身,直接量化會牽涉到更複雜的處理,用雜訊模擬是簡化但足以示範「softmax會放大誤差」這個結論的做法。

## 使用時機 / 優缺點
- ✅ 想搞懂FP16/BF16/FP8這些格式實際上「省」在哪裡、量化的scale/zero-point是怎麼算出來的:這支程式把bit layout跟量化算式全部攤開成純numpy,沒有藏在PyTorch的`torch.quantize_per_tensor`或`bitsandbytes`底層C++/CUDA kernel裡。
- ✅ 需要對「量化不同部位的模型,誰比較危險」建立直覺:`sensitivity_experiment`直接把weights/activations/KV cache/attention logits的敏感度量出來排序,可以拿來解釋「為什麼production的量化方案常常讓attention留在高精度」。
- ✅ 想大致理解GPTQ跟AWQ「多做了什麼」,而不只是知道要裝哪個套件:`simulated_gptq`跟`simulated_awq`把兩者的核心思路(Hessian引導的誤差補償 vs. salient weight的先放大後除回)簡化到能一次讀完的程度。
- ❌ 不要把這裡的`simulated_gptq`/`simulated_awq`當成可以直接拿去量化真實模型的實作:沒有per-group量化(GPTQ/AWQ實際上是以128個權重為一組,不是整個column共用一個scale)、沒有真正的inverse-Hessian求解、也沒有處理真實模型裡數以千計column之間的耦合關係——正式場景請用`auto-gptq`、`autoawq`或`llama.cpp`的轉換工具。
- ❌ `simulate_fp8_e4m3`是純numpy模擬,不是真正的硬體FP8:沒有處理FP8的denormal數值、round-to-nearest-even等IEEE細節,拿來直覺理解「E4M3有多少mantissa bit」很夠用,但數值不會跟真正H100上跑出來的FP8結果完全一致。
- ❌ 這裡的每個demo都是合成資料(`np.random.randn`),不是真實模型的權重:`full_quantization_comparison`裡的outlier行是刻意注入的,用來示範per-tensor量化為什麼會被outlier拖垮,但不代表真實模型的outlier分布長這樣——想量測真實影響需要接上WikiText perplexity或MMLU這類benchmark。

## 常見誤區
1. **以為量化只是「把數字變小」,跟「量化哪裡」無關**:同樣是INT8,量化weights幾乎無損,量化attention logits可能讓輸出明顯跑掉——`sensitivity_experiment`量出來的階層(weights < activations < KV cache < attention logits)才是決定量化策略的關鍵,不是bit數本身。
2. **把GPTQ/AWQ當成近似算法,以為犧牲一點準確率是必然代價**:INT4配得當的GPTQ/AWQ只掉1-3%的品質,而naive per-tensor INT4可能直接讓模型輸出亂掉(`full_quantization_comparison`裡naive跟GPTQ/AWQ的SNR差距可以直接跑出來看)——差距來自技術好壞,不是「量化必然有損」這句話能一概而論的。
3. **搞混per-tensor跟per-channel的成本效益**:per-channel只是多存幾個scale factor(相對於整個weight矩陣,這點額外開銷幾乎可以忽略),換到的品質提升卻很顯著——這不是一個需要猶豫的取捨,幾乎所有正式量化方法都預設用per-channel或更細的粒度。
4. **以為BF16跟FP16只是「換個名字的16 bit」,可以隨意互換**:兩者的exponent/mantissa切法不同(BF16範圍同FP32、精度較低;FP16範圍窄、精度較高),同一個容易underflow的極小梯度,FP16會變成0、BF16可以保留——訓練場景通常要BF16,存權重給inference用則FP16/BF16皆可。
5. **看到perplexity delta很小就以為所有下游任務都沒事**:課程教材強調perplexity只是總體指標,數學/程式碼類任務對精度損失比一般常識類任務敏感得多——量化後務必額外用task-specific benchmark(MMLU、HumanEval、GSM8K)驗證,不能只看WikiText perplexity delta。

## 複習自問
- 為什麼`quantize_per_channel`要替每個channel算獨立的scale,而不是像`quantize_symmetric`一樣整個張量共用一個?如果weight矩陣裡有一行的數值是其他行的50倍,per-tensor量化會對哪些行造成傷害?
- BF16跟FP16的exponent/mantissa切法不同,分別適合什麼場景?如果拿BF16做inference、FP16做training,各自可能踩到什麼坑?
- `sensitivity_experiment`量出來的敏感度階層(weights最耐量化、attention logits最脆弱),背後的數學原因是什麼?為什麼softmax會放大量化誤差,而不是像線性層一樣讓誤差保持線性比例?
- GPTQ跟AWQ都是為了在INT4下維持品質,但解決的角度不一樣(Hessian引導的誤差補償 vs. salient weight的先放大後除回)——如果一個weight矩陣裡有明顯的outlier row,你預期兩者處理起來會有什麼不同?
- 如果你要把一個70B模型部署到單張48GB GPU上,會怎麼決定該用FP8、GPTQ INT4、還是AWQ INT4?除了模型大小之外,還有哪些因素(目標硬體、任務類型、是否能接受重新calibration)會影響這個決定?
