# Instance Segmentation — Mask R-CNN

對應程式：[`./instance_segmentation.py`](./instance_segmentation.py)

## TL;DR
Instance segmentation = detection + 每個物體一個mask。Mask R-CNN 在 Faster R-CNN 的偵測骨架上，用 RoIAlign 取代會做座標捨入的 RoIPool，再加一個小小的 mask head，就能讓每個偵測到的物體都有自己的pixel-level輪廓，即使兩個物體是同一類別。

## 為什麼需要它
- Semantic segmentation只能回答「這個pixel是什麼類別」，兩個相鄰同類別物體（例如兩台車）會被合併成同一塊；要分辨「car #3」跟「car #5」需要instance segmentation。
- 計數個體、跨影格追蹤、量測單一物體的尺寸（磚牆裡每一塊磚、顯微鏡影像裡每一顆細胞），都需要「每個物體各自一個mask」。
- Mask R-CNN（He et al., 2017）把instance segmentation重新表述成「detection + mask」，架構乾淨到接下來五年幾乎所有instance segmentation論文都是它的變體；torchvision的實作至今仍是中小型資料集的production default。
- 真正的工程難題是取樣：proposal box的四個角落幾乎不會剛好落在pixel邊界上，要怎麼從feature map裡切出一塊固定大小的區域？答錯這題，到處都要付出幾個mAP百分點的代價，RoIAlign就是這題的答案。

## 核心原理
- **五個部件**：backbone（ResNet，抽feature）→ FPN（把不同解析度的feature都補齊語意）→ RPN（在FPN每個位置粗篩~1000個候選框）→ RoIAlign（把任意浮點座標的候選框轉成固定大小的feature patch）→ box head + mask head（分別做分類/框修正、跟每個proposal的28x28 mask預測）。
- **RoIAlign vs RoIPool**：RoIPool把proposal box的座標跟grid切分都四捨五入成整數，在stride越大的深層feature map上，這個捨入誤差會被放大成好幾個pixel的偏移；RoIAlign完全不做捨入，用bilinear interpolation在浮點座標上直接取樣，這一項改動就能讓COCO上的mask AP提升3-4個百分點。
- **RPN**：在feature map的每個位置放K個不同大小/長寬比的anchor，預測「這裡有沒有物體」跟「怎麼把anchor微調成更貼合的框」，取分數最高的~1000個框、以IoU 0.7做NMS，交給後面的heads。跟第6課YOLO的loss結構相同，只是這裡只有兩類（有物體/沒有物體）。
- **Mask head**：對每個RoIAlign後的proposal接一個小FCN（4個3x3 conv + 一次2倍反卷積上採樣 + 最後1x1 conv），輸出`num_classes`個channel、每個都是28x28的binary mask，最後只留下預測類別對應的那個channel——分類跟mask預測因此是解耦的，mask head不用學會分辨類別，只要學會畫出形狀。
- **四個loss相加**：`L = L_rpn_cls + L_rpn_box + L_box_cls + L_box_reg + L_mask`。前兩項是RPN的objectness/框迴歸；`L_box_cls`是box head對(C+1)類（含背景）的cross-entropy；`L_box_reg`是box修正的smooth L1；`L_mask`是28x28 mask輸出的per-pixel binary cross-entropy。
- **輸出格式**：torchvision的`maskrcnn_resnet50_fpn_v2`回傳每張圖一個dict：`boxes (N,4)`是pixel座標、`labels (N,)`是class ID（0是背景，實際偵測到的類別從1開始）、`scores (N,)`是信心分數、`masks (N,1,H,W)`是`[0,1]`之間的float mask，且已經是原圖解析度（內部已經把28x28的head輸出upsample回去了），要拿到binary mask得自己在0.5處threshold。

## 程式碼導覽
| 函式/類別 | 對應到理論的哪個部分 |
|---|---|
| `roi_align_single` | 從頭實作RoIAlign：把box座標轉進feature map座標系、切成`output_size × output_size`個bin、在每個bin中心用`grid_sample`做bilinear取樣，全程沒有任何四捨五入 |
| `roi_align`（torchvision） | 拿官方版本跟自己的實作互相對照，驗證數值一致性 |
| `maskrcnn_resnet50_fpn_v2` | 載入COCO預訓練的完整Mask R-CNN，示範inference輸出格式 |
| `binary_masks` | 把`[0,1]`的float mask在0.5處threshold成boolean mask |
| `build_custom_maskrcnn` | fine-tuning的標準作法：保留backbone/FPN/RPN，把box head跟mask head換成符合自己`num_classes`的版本 |
| `freeze_backbone_and_fpn` | 凍結backbone跟FPN參數，讓小資料集上只需要訓練RPN跟兩個head，降低overfitting風險 |

**實作細節 / 容易看漏的地方：**
- `roi_align_single`裡的座標轉換`c * spatial_scale - 0.5`跟torchvision`aligned=True`的行為對齊，這個`-0.5`是讓pixel中心對齊的關鍵，拿掉會跟官方版本的取樣位置差半個pixel。
- 對照官方`roi_align`時要設定`sampling_ratio=1, aligned=True`，因為`roi_align_single`每個bin只取一個中心點做bilinear取樣，跟官方預設的多點取樣平均不是同一件事，兩者參數要對齊才能比較。
- `build_custom_maskrcnn`裡`num_classes`必須把背景也算進去，資料集有4個真實類別就要傳`num_classes=5`；`freeze_backbone_and_fpn`凍結的是backbone/FPN參數（`model.backbone`已經把FPN包在裡面），不是box/mask head——head的參數本來就是新建的，一定要保持`requires_grad=True`才能被訓練到。
- `masks`張量的形狀是`(N, 1, H, W)`，中間那個維度是「只留下預測類別對應的channel」後留下的單一channel，不要誤以為是batch或其他維度，用`squeeze(1)`把它拿掉即可得到`(N, H, W)`。

## 使用時機 / 優缺點
- ✅ 需要分辨「同類別的不同個體」而不只是「這是什麼類別」的任務：個體計數、單一物體量測、跨影格追蹤。
- ✅ 中小型資料集上，torchvision的Mask R-CNN搭配凍結backbone/FPN的fine-tuning流程，是至今仍常見的production baseline，訓練/除錯都相對直觀。
- ❌ Two-stage架構（先RPN出proposal、再逐一過heads），推論速度比one-stage的YOLO系列/單階段分割模型慢，即時應用要另外考慮。
- ❌ Anchor-based RPN跟YOLO一樣，對於anchor涵蓋不到的極端長寬比或密集重疊物體效果會打折扣。

## 常見誤區
1. **RoIAlign不是「更精確版的RoIPool」這麼簡單的差異**：RoIPool的捨入誤差在淺層、小解析度圖上可能看不出來，但在Mask R-CNN常用的stride 16/32的深層feature map上會被放大成好幾個pixel，直接反映在mask邊界品質上，這是Mask R-CNN相對Faster R-CNN masking能力的關鍵差異，不是錦上添花的優化。
2. **`labels`是1-based，不是0-based**：class 0保留給背景，資料集自己的類別要從1開始編號，`build_custom_maskrcnn`的`num_classes`參數也要記得把背景算進去，這個位移很容易在mapping自己的類別名稱時忘記。
3. **mask head的28x28輸出跟最終回傳的mask解析度不是同一件事**：`predictions`裡`p['masks']`已經是upsample回原圖解析度的結果，如果要理解mask head本身的容量限制（例如28x28 vs 56x56的取捨），要看的是mask head內部的輸出，而不是最終API回傳的張量形狀。
4. **凍結backbone不代表凍結整個模型**：`freeze_backbone_and_fpn`只鎖住`model.backbone.parameters()`，RPN跟兩個head的參數預設仍是`requires_grad=True`，訓練時看trainable參數量下降的幅度是否符合預期，才能確認凍結真的生效在你以為的範圍。

## 複習自問
- RoIAlign裡「不做任何座標捨入」具體是靠哪一步做到的？如果拿掉`grid_sample`改成整數index直接切片，會退化回什麼行為？
- Mask head為什麼要對每個類別各輸出一個28x28 mask、再依預測類別挑一個channel，而不是直接輸出一個跟類別數無關的單一mask？這樣設計解決了什麼問題？
- `build_custom_maskrcnn`保留了backbone、FPN、RPN不變，只換了box head跟mask head，這個選擇的前提假設是什麼？如果新資料集的物體尺度分佈跟COCO差很多，這個假設還成立嗎？
- 為什麼`masks`回傳的是`(N, 1, H, W)`的機率而不是直接的binary mask？把threshold的決定權留給使用者，帶來什麼彈性？
