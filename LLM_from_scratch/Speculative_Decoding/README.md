# Speculative Decoding(Leviathan Rejection Sampling / EAGLE-3)

對應程式: [`./speculative_decoding.py`](./speculative_decoding.py)

參考:[ai-engineering-from-scratch – Speculative Decoding and EAGLE-3 課程](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/10-llms-from-scratch/15-speculative-decoding-eagle3/docs/en.md)

## TL;DR
LLM自迴歸解碼(autoregressive decoding)是memory-bound的:每生一個token,都要把整份模型權重從HBM搬進SM算一次,GPU算力大部分時間是閒置的——70B model在H100上大約只能跑35 tokens/秒,不是算力不夠,是「搬資料」比「算」慢太多。Speculative decoding把這個「算力閒置」問題轉換成一個可以解的吞吐量問題:一個便宜的draft先猜出N個候選token,verifier(目標大model)用一次forward pass平行驗證這N個候選,只要驗證通過就能一口氣拿到多個token,而不必為每個token都跑一次完整的大model forward pass。這裡真正的關鍵是Leviathan rejection rule(Leviathan、Kalman、Matias,ICML 2023):以機率`min(1, q/p)`接受draft token,一旦拒絕就改從殘差分佈`(q-p)+`(clip到非負再正規化)重新採樣——這個機制在數學上保證整體輸出的分佈跟「直接從verifier `q`採樣」完全一致,不是近似,是純粹的延遲優化,沒有任何品質代價。本程式把這條規則(`accept`/`residual`)、一輪完整的draft→verify→accept/reject流程(`spec_step`)、KV cache rollback該怎麼記帳(`SequenceWorker`)、以及「draft品質(acceptance rate α)如何決定加速倍率、進而決定最佳draft長度N」這幾件事都攤開實作,並用chi-square檢定實證這個理論不變量真的成立。

## 為什麼需要它
- **Decode階段的GPU算力浪費,不是靠換更貴的硬體能解決的**:decode每一步只算一個token,但要付出「把整份模型權重從顯存搬進運算單元」的全部代價——這個ops:byte比值低到GPU大部分時間在等資料而不是在算——speculative decoding用「一次verify多個候選token」把這個memory-bound的稅攤薄到多個token上,是純軟體/演算法層面的解法。
- **「加速不能犧牲品質」這件事,Leviathan rule給了嚴格的數學保證,不是工程上的妥協**:很多優化技巧都要在速度跟準確度之間取捨,但speculative decoding的accept/reject機制保證了輸出分佈跟target model單獨生成完全一致——這也是為什麼2026年所有主流inference server(vLLM、SGLang、TensorRT-LLM)預設都會開啟它,而不是當成一個「風險換效能」的選項。
- **Draft品質(acceptance rate α)是唯一真正決定加速倍率的槓桿,而且槓桿效應很大**:同樣的draft length,α從0.6(vanilla小model當draft)提升到0.9(EAGLE-3水準),`optimal_draft_length`算出來的最佳加速倍率會從約1.9倍跳到約4.7倍——這說明「要不要訓練一個更好的draft」比「調draft length的參數」重要得多。
- **EAGLE-1→EAGLE-2→EAGLE-3兩年的演進史,每一步都在解決上一步留下的具體限制**:EAGLE-1讓draft直接吃verifier的hidden state(不再是獨立訓練的小model);EAGLE-2加上dynamic draft tree,一次驗證多條候選路徑;EAGLE-3放棄「模仿hidden state」的訓練目標、改成直接練token prediction,並加入training-time test(TTT)讓訓練時跟推論時的輸入分佈一致——理解這個演進脈絡,比記住「EAGLE-3比較快」更有用,因為它告訴你下一個瓶頸大概會出現在哪裡。
- **KV cache rollback的成本結構,決定了「拒絕」這件事有多便宜**:被拒絕的候選token在verifier的KV cache裡已經寫入了(verify是一次batched forward pass,所有候選的K/V都會被算出來),但rollback不需要真的清除或重算這些位置,只需要調整一個logical length,之後的讀取自然會忽略已經寫入但邏輯上已作廢的部分——這代表「猜錯了」的代價幾乎可以忽略,真正貴的是「猜」跟「驗證」這兩件事本身。

## 核心原理
- **Leviathan rejection rule(`accept` / `residual`)**:設draft分佈為`p`、verifier分佈為`q`。從`p`採樣出候選token後,以`min(1, q(token)/p(token))`的機率接受;拒絕時改從殘差分佈`(q-p)+`正規化後採樣。這個規則不管`p`有多差都成立——`p`越差,拒絕率越高,但輸出永遠精確分佈成`q`,這是speculative decoding能被視為「純延遲優化」而不是「近似算法」的數學基礎。
- **一輪完整的speculative round(`make_draft_and_verifier` / `spec_step`)**:真實系統會用一次verifier forward pass平行驗證`prefix + d_1...d_N`這N個draft token;本程式用「每個位置各自的toy `(p, q)`分佈對」模擬這件事(用`noise`參數混合`q`跟均勻分佈來控制`p`離`q`有多遠,藉此控制acceptance rate,而不需要真的訓練一個draft/target model)。`spec_step`從左到右依序對每個draft token跑`accept`,一旦拒絕就用`residual`採樣出修正token並提前結束這一輪;如果全部N個都被接受,還能額外從下一個位置的`q`分佈白拿一個bonus token——這正是「一次verifier forward pass最多能拿到N+1個token」的來源。
- **KV cache rollback bookkeeping(`SequenceWorker`)**:verify pass會把N個候選位置的K/V都寫進verifier的cache裡,但如果在第j個位置被拒絕,j之後的K/V就是「寫了但邏輯上作廢」的資料。真實系統要嘛用一塊scratch buffer、接受後才真正提交(vLLM、TensorRT-LLM的做法),要嘛保留實體cache但另外維護一個logical length、之後的讀取只讀到這個長度為止(截斷式做法,本程式採用的簡化版)。`spec_step`回傳的`output_tokens`長度本身就已經是「這一輪真正該保留的token數」,所以`SequenceWorker.apply_round`不需要對accept/reject分兩種情況處理,直接把`kv_length`往前推進`len(output_tokens)`即可。
- **理論不變量的實證(`chi_square_critical_value_95` / `leviathan_invariant_check`)**:固定住一組`(p, q)`,讓大量trial都用同一對分佈跑accept/reject流程,把「speculative sampling產生的token」的經驗分佈拿去跟`q`本身做chi-square goodness-of-fit檢定,同時用同樣數量的「直接從`q`採樣」的結果當作對照組。因為專案沒有依賴`scipy`,95%信心水準的臨界值改用Wilson-Hilferty近似公式(立方根常態近似)手動算出來,跟標準查表值誤差在0.1%以內。
- **加速倍率數學(`expected_accepted_tokens` / `expected_speedup` / `optimal_draft_length`)**:給定per-token acceptance rate `α`跟draft length `N`,一輪期望能接受的token數是`E[accepted] = (1-α^(N+1))/(1-α)`(幾何衰減級數的封閉解);若draft跟verifier的成本比是`c = cost(draft)/cost(verifier)`,一輪的成本是`N*c+1`個verifier單位,換算成相對於「不做speculation」的加速倍率就是`E[accepted]/(N*c+1)`。這個式子在`N`上不是單調的——`N`太大時,verify成本線性增加的速度會超過`E[accepted]`趨緩(因為`α^(N+1)`很快趨近0)帶來的邊際好處,所以存在一個讓加速倍率最大化的最佳`N`,`optimal_draft_length`就是在做這個搜尋。

**EAGLE系列與其他draft策略(來自課程教材,2026年生態現況參考)**:

| 策略 | Draft類型 | α (acceptance rate) | 加速倍率 | 訓練成本 |
|---|---|---|---|---|
| Vanilla(Leviathan 2023) | 獨立訓練的小model | 0.55–0.70 | 1.8–2.3× | 無(重用現成小model) |
| Medusa | 在verifier上加額外的LM head | 0.65–0.75 | 2–3× | ~1B SFT tokens |
| EAGLE-1 | 吃verifier hidden state的單層transformer | 0.70–0.80 | 2.5–3× | ~60B tokens |
| EAGLE-2 | EAGLE-1 + dynamic draft tree | 0.80–0.88 | 3–4× | ~60B tokens |
| EAGLE-3 | 多層特徵融合 + training-time test(TTT) | 0.88–0.92 | 3.5–6.5× | ~60–200B tokens |
| Lookahead | 無draft model(Jacobi iteration) | N/A | 1.3–1.6× | 無 |

## 程式碼導覽
| 函式/類別 | 對應到理論的哪個部分 |
|---|---|
| `accept` | Leviathan accept/reject test:`u < min(1, q_prob/p_prob)`,`p_prob<=0`時直接接受 |
| `residual` | 拒絕後要重新採樣的殘差分佈,`(q-p)`clip到非負後正規化;`q<=p`處處成立時退化回`q`本身 |
| `make_draft_and_verifier` | 用`noise`參數把隨機生成的`q`跟均勻分佈混合出`p`,模擬「draft離verifier有多近」,藉此控制acceptance rate而不需要真model |
| `spec_step` | 一輪完整流程:依序drafts `draft_len`個token→逐一對照各自的`q`跑`accept`→第一次拒絕就用`residual`採樣修正token並提前結束→全部接受則多抽一個bonus token |
| `measure_acceptance_rate` | 對`spec_step`重複跑多輪取平均,量出這組參數下的經驗acceptance rate α |
| `SequenceWorker` | 追蹤單一sequence的logical KV cache長度,`apply_round`依這一輪`output_tokens`的實際長度推進`kv_length` |
| `chi_square_critical_value_95` | 用Wilson-Hilferty近似算出給定自由度下chi-square分佈的95%臨界值,不依賴`scipy` |
| `leviathan_invariant_check` | 核心實證:固定`(p,q)`跑50000次speculative sampling,拿輸出的token分佈跟`q`做chi-square檢定,並用直接從`q`採樣的結果當對照組 |
| `expected_accepted_tokens` | `E[accepted] = (1-α^(N+1))/(1-α)`,算出一次verifier forward pass平均能拿到幾個token |
| `expected_speedup` | 加上draft/verifier的成本比`cost_ratio`,把`expected_accepted_tokens`換算成實際wall-clock加速倍率 |
| `optimal_draft_length` | 掃過`1..max_len`的draft length,回傳讓`expected_speedup`最大的N與對應加速倍率 |
| `demo_leviathan_check` / `demo_spec_step` / `demo_kv_rollback` / `demo_speedup_curve` | 對應上面四個機制的可執行demo,`main()`依序跑完全部並印出量化結果 |

## 使用時機 / 優缺點
- ✅ 想搞懂「為什麼speculative decoding不是近似算法,而是有嚴格數學保證的優化」:`accept`/`residual`/`leviathan_invariant_check`把Leviathan theorem的每一步跟它的實證檢驗都攤開寫,不必只相信論文裡的一句話。
- ✅ 想有一個不需要真的接draft/target model,就能玩「加速倍率 vs. draft品質(α)、draft length(N)、成本比(c)」關係的sandbox:改一下`optimal_draft_length`的參數,馬上能看到EAGLE系列α每提升一階,最佳N跟加速倍率怎麼變。
- ✅ 想理解KV cache rollback為什麼便宜:`SequenceWorker`把「accept全部/中途reject」統一成同一種bookkeeping動作,直接對應到真實inference server裡的邏輯截斷做法。
- ❌ 這裡的draft/verifier都是用`noise`參數合成出來的toy分佈,不是真的訓練一個draft model或EAGLE-style的feature-fusion head——想知道EAGLE-1/2/3實際上怎麼訓練、網路長什麼樣子,要看論文原文,這裡只示範了「acceptance rate α」這個訓練結果如何影響下游的加速數學。
- ❌ 沒有實作tree attention/dynamic draft tree(EAGLE-2、EAGLE-3的關鍵機制之一):`spec_step`只做單一線性chain的draft(依序draft N個token),不是課程教材裡提到的「一次驗證一整棵候選樹」——想繼續往下實作,可以參考`README`複習自問或課程的Exercise 3(模擬`[2,2,2]`形狀的draft tree)。
- ❌ `expected_speedup`裡的`cost_ratio`是使用者自訂的常數,不是在真實硬體上量測出來的數字,不能拿這裡算出來的加速倍率去做效能宣稱或論文引用——它只適合建立「加速倍率跟α、N、c之間大致是什麼關係」的直覺。

## 常見誤區
1. **以為speculative decoding是「犧牲一點準確率換速度」的近似方法**:Leviathan rejection rule在數學上保證輸出分佈跟verifier單獨採樣完全一致,不是近似——真正決定它值不值得用的,是draft多快、acceptance rate多高,而不是「要不要犧牲品質」這個假的取捨(這點跟`Inference_Optimization`筆記裡speculative decoding段落的結論一致)。
2. **以為draft length(N)越大一定越快**:`expected_speedup`不是`N`的單調函數——`N`太大時,verify成本(`N*c`)線性增加的速度會超過`E[accepted]`趨緩帶來的邊際好處(`α^(N+1)`很快趨近0),存在一個讓加速倍率最大化的最佳`N`,`optimal_draft_length`就是在算這件事。
3. **把chi-square檢定的單次`FAIL`當成程式有bug**:`leviathan_invariant_check`是在95%信心水準下做真正的假設檢定,即使理論完全成立,大約每20個seed就會有1個因為隨機性而落在臨界值之外——這是統計檢定的正常行為,不是實作錯誤(程式的docstring裡也特別註記了這點)。
4. **以為acceptance rate α只取決於draft model「有多小」**:EAGLE系列的演進說明了α更取決於draft能不能直接吃到verifier的hidden state(feature-based,EAGLE-1起)、能不能一次驗證多條候選路徑(dynamic tree,EAGLE-2)、以及訓練時的輸入分佈跟推論時是否對齊(training-time test,EAGLE-3)——不是單純「model越小、猜得越差」這麼粗糙的關係。
5. **把KV cache rollback想成要整份重算或清空記憶體**:被拒絕位置之後的K/V資料確實已經寫入verifier的cache,但rollback只需要調整一個logical length(`SequenceWorker.kv_length`),之後的讀取自然會忽略邏輯上已作廢的部分——真正的成本是「draft」跟「verify」這兩個forward pass本身,rollback的bookkeeping代價可以忽略。

## 複習自問
- Leviathan rejection rule保證輸出分佈跟直接從verifier `q`採樣完全一致,具體是靠哪一步達成的?如果拿掉「拒絕時要從殘差分佈`(q-p)+`重新採樣」這一步,直接丟棄被拒絕的token改成不輸出任何東西,還能保持這個保證嗎?
- EAGLE-1、EAGLE-2、EAGLE-3三個世代,每一步分別拿掉了前一版的哪個具體限制?如果你只能為現有系統做一項升級,你會優先做哪一個,理由是什麼?
- 為什麼`optimal_draft_length`算出來的最佳draft length不是無限大,而是存在一個有限的最佳值?如果`cost_ratio`(draft相對verifier的成本)變得更便宜,你預期最佳`N`會怎麼變化?為什麼?
- `SequenceWorker.apply_round`為什麼不需要對「全部接受」跟「中途被拒絕」寫兩套不同的邏輯?這跟`spec_step`回傳的`output_tokens`已經先做了什麼處理有關?
- 如果你的acceptance rate α從0.6(vanilla draft水準)提升到0.9(EAGLE-3水準),在draft length固定不變的情況下,`expected_accepted_tokens`大概會怎麼變化?這對「該選多長的draft length」這個決策有什麼影響?
