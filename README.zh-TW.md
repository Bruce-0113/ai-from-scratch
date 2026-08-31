# AI From Scratch

*英文版請見 [README.md](README.md)。*

這是一個個人學習/複習用的 repo，目標是把 AI/ML 的核心概念——NLP、電腦視覺、LLM 工程——盡量從零開始實作出來，並搭配學習筆記，方便日後複習。

## 目的

每支 script 都是從零實作一個核心概念（word2vec、YOLO、RAG、LoRA……），盡量不依賴高階套件的一行呼叫，讓底層機制看得見，而不是被單一函式呼叫藏起來。部分 script 會額外搭配一份「學習筆記」(.md)，寫作方向是給日後複習用，而不是給第一次閱讀用。

## 目錄結構

每個概念都有自己的資料夾：`<主題>/<script_name>/<script_name>.py`，旁邊可能會有一份 `<script_name>.md` 放複習筆記。

```
NLP/
  word2vec/
    word2vec.py
    word2vec.md       # 學習筆記，複習導向
  BoW_TF_IDF/
    BoW_TF_IDF.py
  ...
LLM_engineering/
  RAG/
    RAG.py
  ...
computer_vision/
  object_detection_YOLO/
    object_detection_YOLO.py
  ...
```

筆記是陸續補上的，不是每支 script 都有——下面表格的「筆記」欄位是目前的完成狀況。

## 內容

### NLP
| Script | 筆記 | 說明 |
|---|---|---|
| [text_processing](NLP/text_processing/text_processing.py) | — | 斷詞、詞幹化/詞形還原、詞性標註基礎（nltk、spaCy）|
| [BoW_TF_IDF](NLP/BoW_TF_IDF/BoW_TF_IDF.py) | — | Bag-of-Words 與 TF-IDF 向量化，含手刻版與 scikit-learn 版 |
| [word2vec](NLP/word2vec/word2vec.py) | [筆記](NLP/word2vec/word2vec.md) | Skip-gram + negative sampling 詞向量，從零實作並與 gensim 對照 |
| [GloVe_FastText_BPE](NLP/GloVe_FastText_BPE/GloVe_FastText_BPE.py) | — | 共現矩陣建構（GloVe 概念）|
| [POS_tagging](NLP/POS_tagging/POS_tagging.py) | — | 最高頻詞性 baseline、bigram HMM + Viterbi 解碼，並與 spaCy 對照 |
| [NER](NLP/NER/NER.py) | — | BIO 標註、規則式（gazetteer）標註器、CRF、BiLSTM-CRF |
| [sentiment_analysis](NLP/sentiment_analysis/sentiment_analysis.py) | — | 小型玩具資料集上的情感分類 |
| [CNN_n_RNN](NLP/CNN_n_RNN/CNN_n_RNN.py) | — | TextCNN 與 BiLSTM 文字分類，並示範 RNN 梯度消失現象 |

### LLM Engineering
| Script | 筆記 | 說明 |
|---|---|---|
| [RAG](LLM_engineering/RAG/RAG.py) | — | 最小可行的 RAG pipeline（切段、TF-IDF、cosine 相似度檢索、prompt 組裝）|
| [advanced_RAG](LLM_engineering/advanced_RAG/advanced_RAG.py) | — | BM25、混合檢索（RRF）、重排序、HyDE 查詢擴展、parent-child 切段、忠實度/召回率評估 |
| [LoRA_QLoRA](LLM_engineering/LoRA_QLoRA/LoRA_QLoRA.py) | — | LoRA 注入/合併，以及簡化版 QLoRA 風格（NF4）量化示範 |
| [function_calling](LLM_engineering/function_calling/function_calling.py) | — | 模擬 LLM 的工具呼叫（function calling）流程 |
| [context_engineering](LLM_engineering/context_engineering/context_engineering.py) | — | token 預算控管、對話歷史壓縮、"lost in the middle" 緩解、情境組裝 |
| [caching_rate_limiting](LLM_engineering/caching_rate_limiting/caching_rate_limiting.py) | — | 語意快取、token bucket 限流、成本追蹤、模型路由 |
| [evaluation_testing](LLM_engineering/evaluation_testing/evaluation_testing.py) | — | LLM-as-judge 評分、ROUGE-L、信賴區間、baseline 比較 |
| [production_llm_app](LLM_engineering/production_llm_app/production_llm_app.py) | — | 模擬生產環境 LLM 服務：prompt A/B 測試、防護機制（guardrails）、重試/降級、串流輸出 |

### Computer Vision
| Script | 筆記 | 說明 |
|---|---|---|
| [convolution](computer_vision/convolution/convolution.py) | — | 2D 卷積基礎運算（padding 等）從零實作 |
| [classic_network](computer_vision/classic_network/classic_network.py) | — | 經典 CNN 架構（LeNet-5）|
| [image_classification](computer_vision/image_classification/image_classification.py) | — | 端到端影像分類：合成資料集、TinyResNet、mixup + cosine LR schedule |
| [transfer_learning_fine_tuning](computer_vision/transfer_learning_fine_tuning/transfer_learning_fine_tuning.py) | — | 用預訓練 ResNet18 backbone 做遷移學習/微調 |
| [object_detection_YOLO](computer_vision/object_detection_YOLO/object_detection_YOLO.py) | [筆記](computer_vision/object_detection_YOLO/object_detection_YOLO.md) | 從零實作的最小 YOLO 風格單階段物件偵測器 |
| [semantic_segmentation](computer_vision/semantic_segmentation/semantic_segmentation.py) | [筆記](computer_vision/semantic_segmentation/semantic_segmentation.md) | 從零訓練的 U-Net 語意分割 |

## 環境需求

Script 各自獨立，依賴套件混合使用：`numpy`、`torch`、`torchvision`、`scikit-learn`、`gensim`、`nltk`、`spacy`、`sklearn-crfsuite`。實際需要看各檔案最上方的 import，再自行安裝。
