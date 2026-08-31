# Semantic Segmentation — U-Net

對應程式：[`./semantic_segmentation.py`](./semantic_segmentation.py)

## TL;DR
Segmentation 就是「每個 pixel 都做分類」。U-Net 用「先downsample抓全局context的encoder」搭配「再upsample還原解析度的decoder」，中間用skip connection把encoder的高解析度細節直接接給decoder，同時兼顧「這是什麼場景」跟「這個pixel精確屬於誰」。

## 為什麼需要它
- Classification 每張圖出一個label，detection 每張圖出幾個box，segmentation 每個pixel都要一個label——輸出從 `H×W` 張圖只有一個值，變成 `H×W` 個值，資訊量差好幾個量級。
- 醫學影像（腫瘤輪廓）、自駕車（道路/車道/障礙物）、衛星影像（建物範圍）、文件版面解析、機器人（可抓取區域），這些任務都需要「完整輪廓」而不是一個框就能解決。
- 架構上的難題：網路要同時看到「全局context」（這是什麼場景）跟「局部細節」（這個pixel是路還是人行道），但一般CNN為了取得context會不斷downsample、把細節壓縮掉。U-Net就是為了同時兼顧這兩者而設計的架構。

## 核心原理
- **Semantic vs instance vs panoptic**：semantic只分類別（兩台相鄰的車會被當成同一塊）；instance進一步分辨個體（car #3 vs car #5，只管前景物體）；panoptic兩者合一（每個pixel有class，每個instance有unique id，前景背景都涵蓋）。這份筆記對應的是semantic。
- **U-Net整體形狀**：encoder每downsample一次，空間解析度減半、channel數加倍，重複4次直到bottleneck（最深、解析度最小、channel最多）；decoder方向相反，每upsample一次解析度加倍、channel減半，也重複4次回到原始解析度。最後用一個1x1 conv把channel壓到`num_classes`。
- **Skip connections**：decoder在做pixel-level預測時，手上的feature map解析度已經被壓得很小，細節（邊界在哪裡）早就在encoder下採樣時被壓掉了。Skip connection把encoder在對應解析度算出的feature直接concat進decoder同解析度的feature，把那些高解析度細節「借」給decoder用。
- **Transposed conv vs bilinear upsample**：decoder要把空間維度變大，有兩種做法：(1) `nn.ConvTranspose2d`（可學習的上採樣，是U-Net原始論文的做法，但stride/kernel沒對齊好容易出現棋盤狀artifact）；(2) 先bilinear upsample再接一般conv（artifact少、參數少，現在較常用的預設做法）。這份程式用的是後者。
- **Pixel-wise cross-entropy**：輸出`(N, C, H, W)`的logits，target是`(N, H, W)`的整數class ID，等於在每個空間位置各自算一次分類的cross-entropy，`F.cross_entropy`原生支援這個shape，不用reshape。
- **Dice loss 與類別不平衡**：cross-entropy把每個pixel一視同仁，但當某個類別佔絕大多數（例如醫學影像99%背景、1%腫瘤）時，模型只要全部猜背景就能有99%準確率卻毫無用處。Dice loss直接優化預測mask跟真實mask的重疊程度：`Dice = 2*sum(p*y) / (sum(p)+sum(y)+eps)`，`loss = 1 - Dice`，因為是比例式計算，天生對類別不平衡不敏感。實務上常用**組合loss**：`L = CE + lambda * Dice`（lambda約1），CE提供訓練初期穩定的梯度，Dice則在訓練後段把重心放在「形狀是否真的疊得準」。
- **評估指標**：pixel accuracy（整體pixel答對比例，類別不平衡時會失真，跟分類任務的accuracy問題一樣）；IoU per class（每個類別各自算IoU，平均起來是mIoU）；Dice（跟IoU單調相關，`Dice = 2*IoU/(1+IoU)`，醫學界偏好Dice，自駕車界偏好IoU）；Boundary F1（只看邊界pixel的F1，針對邊界精準度要求高的任務，如半導體檢測）。**要報告每個類別各自的IoU，而不是只看mIoU**——mIoU可能把一個15%的類別跟九個85%的類別平均成一個好看但誤導的數字。
- **輸入解析度的取捨**：U-Net的encoder downsample幾次，輸入邊長就要能被2的那個次方整除（這份實作4次downsample，所以要能被16整除）。醫學影像常見512x512甚至1024x1024，記憶體用量跟`H*W*C_max`成正比，解析度越大、bottleneck channel越多，越吃顯存。常見解法：(1) 把輸入切成小tile分開處理再拼回去；(2) 用dilated convolution取代部分downsample，在維持感受野的同時保留較高解析度（DeepLab系列的做法）。

## 程式碼導覽
| 函式/類別 | 對應到理論的哪個部分 |
|---|---|
| `DoubleConv` | U-Net最基本的building block：兩組「3x3 conv → BatchNorm → ReLU」，`bias=False`是因為BN的beta已經扮演bias的角色 |
| `Down` | encoder的一步：`MaxPool2d(2)`把空間解析度減半，再接`DoubleConv`把channel數加倍 |
| `Up` | decoder的一步：先bilinear upsample放大解析度，若跟對應skip的解析度沒對齊（輸入邊長不是16的倍數時會發生）就用`F.interpolate`校正，再跟encoder傳來的skip feature做channel維度的concat，最後過`DoubleConv` |
| `UNet` | 組裝整個網路：`inc`是最高解析度的入口DoubleConv，`d1~d4`是4層encoder（`d4`同時也是最深的bottleneck），`u1~u4`是4層decoder依序接回`x4,x3,x2,x1`四個解析度的skip，`outc`是最後的1x1 conv把channel壓到`num_classes` |
| `dice_loss` | 把logits做softmax變機率、target轉one-hot，算Dice係數再取`1 - Dice`；`eps`避免除以零（某類別在該batch完全沒出現時） |
| `combined_loss` | `cross_entropy + lam * dice_loss`，回傳總loss跟拆開的個別數值方便logging |
| `iou_per_class` | 對每個類別各自算predicted mask跟true mask的IoU，該類別在batch中完全沒出現（union=0）時回傳`nan` |
| `synthetic_segmentation` | 產生「彩色背景+隨機圓形/方形」的合成資料集（class 0=背景、1=圓、2=方），逼網路學會辨認「形狀」而不是單純記顏色 |
| `SegDataset` | 把numpy陣列包裝成torch `Dataset`，做channel排列跟型別轉換 |
| `train_one_epoch` | 標準訓練迴圈：forward → `combined_loss` → backward → 累計loss跟每類別IoU（`nan_to_num(0)`把該batch缺席的類別暫時當0處理） |
| `run_demo` | 串起整個流程：建模型、smoke test一次forward確認輸出shape、在合成資料上訓練5個epoch，觀察各類別IoU上升 |

**實作細節 / 容易看漏的地方：**
- `Up.forward`裡的shape比對只比較`shape[-2:]`（空間維度），不比較channel數——這是刻意的：空間對不齊是輸入邊長非16倍數的正常情況，要用`F.interpolate`悄悄修正；但channel數對不上代表接線接錯，應該讓它直接報錯而不是被靜默attempt插值。
- 這份程式的`UNet`沒有額外獨立的「bottleneck」class，`d4`（最深的`Down`）本身就同時扮演bottleneck的角色，跟課程圖示裡「encoder四層 + 獨立bottleneck」在概念上是等價的，只是程式碼把它們合併實作。
- `iou_per_class`回傳的`nan`是刻意設計來標記「這個類別在這個batch裡沒出現」，`train_one_epoch`裡用`nan_to_num(0)`只是為了訓練時方便累計印出來看，正式算epoch/整個資料集的mIoU時應該用`nanmean`之類的方式跳過缺席的類別，而不是把它們當0平均進去（那樣會低估mIoU）。

## 使用時機 / 優缺點
- ✅ 需要精確輪廓、而不是一個框就夠的任務：醫學影像分割、自駕車道路/車道分割、衛星影像建物輪廓等。
- ✅ 資料集不大、想要一個結構清楚、容易讀懂/除錯的encoder-decoder baseline時，U-Net是很好的起點。
- ❌ 高解析度輸入（如1024x1024以上）非常吃顯存，需要tiling或改用dilated conv（DeepLab系列）之類的技巧才能在一般GPU上訓練。
- ❌ 這裡的標準U-Net只做semantic segmentation，無法區分同類別的不同個體（兩個相鄰的圓形都會被標成同一個class 1）；需要個體區分要換成instance/panoptic的架構（如Mask R-CNN、Mask2Former）。

## 常見誤區
1. **只看mIoU會被平均掉的資訊騙**：一個類別IoU只有15%,其他九個類別都85%,平均起來看起來還不錯,但那個15%的類別可能才是任務真正在意的（例如醫學影像裡的腫瘤類）。務必回報每個類別各自的IoU。
2. **Dice loss不是用來取代cross-entropy,而是互補**：純Dice在訓練初期梯度較不穩定,純CE在類別極度不平衡時會被多數類別主導,組合loss(`CE + lambda*Dice`)才是目前醫學/工業分割的常見預設做法。
3. **輸入邊長沒對齊2的次方(這裡是16)不會直接報錯,而是靠`F.interpolate`偷偷修正**：如果沒注意到這件事,可能會誤以為模型天生支援任意解析度輸入,而忽略了這個隱藏的插值可能造成的些微像素對齊誤差。
4. **Transposed convolution跟bilinear upsample+conv不是同一件事的兩種寫法而已**：transposed conv是可學習的參數,較容易出現棋盤狀artifact；bilinear upsample本身沒有可學習參數,是先做平滑放大再交給後面的conv學特徵,兩者訓練出來的邊界品質可能不同,不要隨意假設两者等價。

## 複習自問
- 為什麼decoder不能只靠bottleneck的feature做出精準的pixel-level預測,一定要靠skip connection?
- Dice loss的公式裡,為什麼它對class imbalance比cross-entropy不敏感?（提示：從「比例」跟「絕對pixel數量」的角度想）
- 如果`iou_per_class`裡某個類別回傳`nan`,在計算整個資料集的mIoU時應該怎麼處理,才不會低估或高估分數?
- 這份程式的輸入必須要能被16整除,這個「16」是怎麼來的？如果把`UNet`的encoder/decoder層數從4層改成3層,這個限制數字會怎麼變?
