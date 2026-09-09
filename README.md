# AI From Scratch

*Also available in [繁體中文](README.zh-TW.md).*

A personal study repo for implementing core AI/ML concepts from scratch — NLP, computer vision, and LLM engineering — and reviewing them later through paired learning notes.

## Goal

Each script reimplements a concept (word2vec, YOLO, RAG, LoRA, ...) with minimal reliance on high-level library calls, so the underlying mechanics stay visible instead of hidden behind a single function call. Selected scripts are paired with a Markdown "learning notes" file written for fast review later, not just for a first read.

## Layout

Each concept gets its own folder: `<topic>/<script_name>/<script_name>.py`, with an optional `<script_name>.md` next to it holding the review notes.

```
NLP/
  word2vec/
    word2vec.py
    word2vec.md       # learning notes, review-oriented
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

Notes are added incrementally, so not every script has one yet — the "Notes" column below shows current coverage.

## Contents

### NLP
| Script | Notes | Description |
|---|---|---|
| [text_processing](NLP/text_processing/text_processing.py) | — | Tokenization, stemming/lemmatization, and POS tagging basics (nltk, spaCy) |
| [BoW_TF_IDF](NLP/BoW_TF_IDF/BoW_TF_IDF.py) | — | Bag-of-Words and TF-IDF vectorization, from scratch and with scikit-learn |
| [word2vec](NLP/word2vec/word2vec.py) | [notes](NLP/word2vec/word2vec.md) | Skip-gram word embeddings with negative sampling, from scratch, compared against gensim |
| [GloVe_FastText_BPE](NLP/GloVe_FastText_BPE/GloVe_FastText_BPE.py) | — | Co-occurrence matrix construction (GloVe-style) |
| [POS_tagging](NLP/POS_tagging/POS_tagging.py) | — | Most-Frequent-Tag baseline, bigram HMM decoded with Viterbi, spaCy comparison |
| [NER](NLP/NER/NER.py) | — | BIO tagging, rule-based gazetteer tagger, CRF, and BiLSTM-CRF |
| [sentiment_analysis](NLP/sentiment_analysis/sentiment_analysis.py) | — | Sentiment classification on a small toy dataset |
| [CNN_n_RNN](NLP/CNN_n_RNN/CNN_n_RNN.py) | — | TextCNN and BiLSTM for text classification, plus a vanishing-gradient demo |

### LLM Engineering
| Script | Notes | Description |
|---|---|---|
| [RAG](LLM_engineering/RAG/RAG.py) | — | Minimal Retrieval-Augmented Generation pipeline (chunking, TF-IDF, cosine search, prompt construction) |
| [advanced_RAG](LLM_engineering/advanced_RAG/advanced_RAG.py) | — | BM25, hybrid search (reciprocal rank fusion), reranking, HyDE query expansion, parent-child chunking, faithfulness/recall metrics |
| [LoRA_QLoRA](LLM_engineering/LoRA_QLoRA/LoRA_QLoRA.py) | — | LoRA injection/merging and a simplified QLoRA-style (NF4) quantization demo |
| [function_calling](LLM_engineering/function_calling/function_calling.py) | — | Simulated LLM tool-calling / function-calling loop |
| [context_engineering](LLM_engineering/context_engineering/context_engineering.py) | — | Token budgeting, history compression, "lost in the middle" mitigation, context assembly |
| [caching_rate_limiting](LLM_engineering/caching_rate_limiting/caching_rate_limiting.py) | — | Semantic response caching, token-bucket rate limiting, cost tracking, complexity-based model routing |
| [evaluation_testing](LLM_engineering/evaluation_testing/evaluation_testing.py) | — | LLM-as-judge scoring, ROUGE-L, confidence intervals, baseline-vs-new comparison |
| [production_llm_app](LLM_engineering/production_llm_app/production_llm_app.py) | — | Simulated production LLM service: prompt A/B testing, guardrails, retry-with-fallback, streaming |

### LLM From Scratch
| Script | Notes | Description |
|---|---|---|
| [Tokenizers](LLM_from_scratch/Tokenizers/tokenizers.py) | [notes](LLM_from_scratch/Tokenizers/README.md) | Byte-level BPE tokenizer built from scratch, benchmarked against a char-level baseline and tiktoken |

### Computer Vision
| Script | Notes | Description |
|---|---|---|
| [convolution](computer_vision/convolution/convolution.py) | — | 2D convolution fundamentals (padding, etc.) implemented from scratch |
| [classic_network](computer_vision/classic_network/classic_network.py) | — | Classic CNN architecture (LeNet-5) |
| [image_classification](computer_vision/image_classification/image_classification.py) | — | End-to-end image classification: synthetic dataset, TinyResNet, mixup + cosine LR schedule |
| [transfer_learning_fine_tuning](computer_vision/transfer_learning_fine_tuning/transfer_learning_fine_tuning.py) | — | Transfer learning / fine-tuning on a pretrained ResNet18 backbone |
| [object_detection_YOLO](computer_vision/object_detection_YOLO/object_detection_YOLO.py) | [notes](computer_vision/object_detection_YOLO/object_detection_YOLO.md) | Minimal YOLO-style single-stage object detector, from scratch |
| [semantic_segmentation](computer_vision/semantic_segmentation/semantic_segmentation.py) | [notes](computer_vision/semantic_segmentation/semantic_segmentation.md) | U-Net for semantic segmentation, trained from scratch |

## Requirements

Scripts are standalone and pull from a mix of `numpy`, `torch`, `torchvision`, `scikit-learn`, `gensim`, `nltk`, `spacy`, and `sklearn-crfsuite`. Check the imports at the top of a script for what it needs before running it.
