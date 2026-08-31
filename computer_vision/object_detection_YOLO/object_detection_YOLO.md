# Object Detection — YOLO（Grid + Anchor + NMS）

對應程式：[`./object_detection_YOLO.py`](./object_detection_YOLO.py)

## TL;DR
物體偵測 = 在特徵圖的每個位置「同時」做分類 + 邊界框迴歸，一次前向傳播就產生所有候選框，最後用 Non-Max Suppression（NMS）把重複框清乾淨。

## 為什麼需要它
- 傳統做法要先產生大量候選區域（sliding window / selective search），再逐一丟進分類器，速度慢、無法即時。
- YOLO 把偵測問題重新表述成「dense prediction」：整張圖切成 grid，每格搭配幾個 anchor，網路一次前向傳播就同時輸出所有位置的類別、座標、objectness，屬於 one-stage detector，換取速度。
- 是自駕車、監控系統、文件版面解析、工廠瑕疵檢測等「要同時知道物體在哪裡＋是什麼」場景的基礎技術。
- 工程上真正要權衡的四件事：框要多準（box regression）、類別要不要對（classification）、有沒有物體要判斷得準（objectness）、以及每個真實物體只能對應一個預測（避免重複偵測）。

## 核心原理
- **Grid + Anchor**：圖片切成 `grid_size × grid_size` 個 cell，每個 cell 配置 `num_anchors` 個預先定義好長寬比的 anchor box。偵測問題因此變成「對每個 (cell, anchor) 都要預測一組值」的密集預測問題，而不是先找候選框再分類。
- **Output tensor**：`YOLOHead` 對每個 (cell, anchor) 輸出 `5 + num_classes` 個數值 `[tx, ty, tw, th, objectness, class_logits...]`：
  - `tx, ty`：中心點相對於該 cell 左上角的 offset，經 `sigmoid` 壓到 0~1，再乘上 `stride`、加上 cell 座標才是全圖 pixel 座標（見 `decode`）。
  - `tw, th`：寬高相對於該 anchor 的 log-scale 比例，換算回真實寬高要取 `exp` 再乘上 anchor 的 w/h。
  - `objectness`：這個位置「有沒有物體」的置信度（raw logit，用時要 `sigmoid`）。
  - `class_logits`：每個類別的機率（raw logit）；此實作用 **獨立 sigmoid**（multi-label 風格）而非 softmax，`postprocess` 裡用 `sigmoid(obj) * max(sigmoid(class))` 當最終信心分數。
- **IoU（Intersection over Union）**：`box_iou` 算兩框交集面積 / 聯集面積，用在兩個地方，意義不同：(1) target assignment 時只比較 **寬高**（anchor 本身沒有位置，只是形狀模板），挑出跟 ground truth 形狀最接近的 anchor；(2) NMS / 評估指標時比較的是 **有位置**的框，判斷是否為同一個物體的重複偵測。
- **Target assignment（`assign_targets`）**：每個 ground truth box 只指派給 (a) 中心點所在的 cell、(b) 寬高 IoU 最高的 anchor。指派到的位置寫入 `target` 並在 `has_obj` 標記 `True`，其餘所有 cell/anchor 視為「沒有物體」的負樣本。
- **Loss（`yolo_loss`）**，三個 component 分開算：
  1. **Box regression**：只在 `has_obj` 的位置上算 `(tx, ty, tw, th)` 的 MSE。
  2. **Objectness**：拆成 positive（`has_obj=True`）跟 negative 兩部分分別算 BCE，再用 `lambda_obj` / `lambda_noobj` 各自加權——因為一張圖裡絕大多數 cell/anchor 都沒有物體，若不用權重平衡，模型會被大量負樣本主導而傾向永遠預測「沒有物體」。
  3. **Classification**：只在有物體的位置上算 one-hot vs class logits 的 BCE。
  4. `lambda_coord`（預設 5.0）通常比其他權重大，因為座標 MSE 的數值量級天生比機率的 BCE 小，要主動放大它的重要性。
- **NMS（`nms`）**：inference 時先用 `conf_threshold` 篩掉低信心候選框，再依分數由高到低貪婪排序：每次保留最高分框、移除跟它 IoU 超過 `iou_threshold` 的其他框（視為重複偵測同一物體），重複直到沒有框剩下。
- **偵測指標**：`precision@0.5`（IoU>0.5 才算對的預測中，猜對類別的比例）、`recall`（真實物體中被偵測到的比例）、`mAP@0.5`（IoU 門檻 0.5 下，各類別 AP 取平均）、`mAP@0.5:0.95`（在 0.5~0.95 多個 IoU 門檻下取平均，對框的定位精準度要求更嚴格，能區分「分類對但框不準」跟「真的沒偵測到」）。

## 程式碼導覽
| 函式/類別 | 對應到理論的哪個部分 |
|---|---|
| `box_iou` | 計算 IoU，NMS 跟 anchor 挑選都靠它 |
| `nms` | 貪婪 NMS，篩掉重複偵測同一物體的框 |
| `sigmoid` | 把 `tx, ty` 與 objectness、class 的 raw logit 轉成機率/比例 |
| `encode` | 把一個 ground truth box 轉成 `(tx, ty, tw, th)` 訓練目標；概念上跟 `assign_targets` 裡的內聯計算重複，抽出來方便單獨理解 encode 的邏輯 |
| `decode` | `encode` 的反函數，把預測的 `(tx, ty, tw, th)` 還原成 pixel 座標框 |
| `YOLOHead` | 1x1 conv 檢測頭，把 feature map 轉成 `(H, W, num_anchors, 5+num_classes)` 的密集輸出 |
| `assign_targets` | 對一張圖的所有 ground truth boxes 做「指派到哪個 cell/anchor」，產生 `target` 與 `has_obj` mask |
| `yolo_loss` | 用 `assign_targets` 的結果算 box + objectness(pos/neg) + classification 三部分 loss，加權加總 |
| `postprocess` | inference 用：把網路 raw output 解碼成 pixel 座標框，依信心值篩選，再做 NMS，得到最終偵測結果 |

**實作細節 / 容易看漏的地方：**
- `tx, ty` 不是直接的座標，`tw, th` 也不是直接的寬高——都要照 `decode` 裡的公式（sigmoid + cell index + stride；exp + anchor 尺寸）換算，不要直接把 raw output 當座標讀。
- `assign_targets` 挑 anchor 時的 IoU 只用寬高（`min(bw,aw)*min(bh,ah)` 那段），跟 NMS/評估時「有位置」的 IoU 是不同計算，語意不要混在一起。
- `lambda_noobj` 是在處理類別不平衡問題（多數 cell 沒有物體），用的是 **loss 加權**而不是取樣（undersampling/oversampling）的方式。
- `postprocess` 裡重新手刻了一次跟 `decode` 相同的解碼算式（sigmoid/exp 換算），沒有直接呼叫 `decode`——邏輯等價，但如果要重構可以抽共用函式。

## 使用時機 / 優缺點
- ✅ 需要即時或低延遲偵測：單次前向傳播出所有框，比 two-stage（如 Faster R-CNN 先產生 proposal 再分類）快很多。
- ✅ 物體大小、長寬比相對可預期，能靠事先設計（如 k-means）的一組 anchor 涵蓋大部分情況的場景。
- ❌ 小物體、密集重疊物體效果較差：每個 cell/anchor 只能負責一個物體，同一 cell 內多個物體搶佔同一 anchor 時，其中一個會被漏掉。
- ❌ Anchor 需要針對資料集特性預先設計，設計不好會直接影響精準度；不像後續 anchor-free 方法（如 FCOS、YOLOX）省掉這一步手動調整。

## 常見誤區
1. **objectness 不是「有沒有物體的機率」的原始值**：網路輸出的是 logit，要過 `sigmoid` 才是機率；`postprocess` 用 `sigmoid(obj) * max(sigmoid(class))` 當最終信心分數，兩者都要過 sigmoid 才能相乘。
2. **conf_threshold 跟 iou_threshold 是兩個獨立旋鈕**：`conf_threshold` 決定多少候選框進入 NMS，`iou_threshold` 決定 NMS 判定「重複偵測」的嚴格程度，調任一個都會同時影響最終 precision 跟 recall，不要以為只調其中一個就夠。
3. **mAP@0.5 跟 mAP@0.5:0.95 回答的是不同問題**：前者只要求框「大致對上」，後者對定位精準度要求嚴格得多，能分辨模型是「類別分對但框畫不準」還是「真的沒找到物體」。
4. **YOLOv1/v2 風格 loss 跟 anchor-free 或 YOLOv8 系列的 loss（如 CIoU/DFL）不是同一套**，這裡實作的是最基礎的 grid+anchor+MSE 版本，理解機制用，不代表現代 YOLO 版本的實際細節。

## 複習自問
- Target assignment 挑 anchor 時為什麼只比較寬高（IoU on w/h），不看位置？如果考慮位置會發生什麼問題？
- `lambda_noobj` 的作用是什麼？如果把它拿掉（跟 `lambda_obj` 設一樣），模型的預測行為大概會往哪個方向壞掉？
- `decode` 裡 `tw, th` 為什麼要先取 `log`（在 `encode`）、還原時再取 `exp`，而不是直接讓網路輸出寬高的比例？
- 如果同一個 cell 裡有兩個真實物體，且它們的寬高 IoU 對到同一個 anchor 最高，`assign_targets` 會怎麼處理？這會造成什麼後果？
