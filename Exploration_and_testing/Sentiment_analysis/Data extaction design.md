| Date       | News                                                        | Open   | High   | Low    | Close  | Volume   | Label    |
|------------|-------------------------------------------------------------|--------|--------|--------|--------|----------|----------|
| YYYY-MM-DD | Content of news articles affecting Gold price                | 1234.5 | 1250.7 | 1222.3 | 1240.1 | 100000   | 1 / 0 / -1 |

**Column Description:**
- **Date**: The date the news was released
- **News**: The content of news articles that could potentially affect the Gold price
- **Open**: The Gold price (in Rs) at the beginning of the day
- **High**: The highest Gold price (in Rs) reached during the day
- **Low**: The lowest Gold price (in Rs) reached during the day
- **Close**: The adjusted Gold price (in Rs) at the end of the day
- **Volume**: Total volume traded during the day (Optional)
- **Label**: The sentiment polarity of the news content  
  - 1: positive  
  - 0: neutral  
  - -1: negative

**Sample Data Row:**
| 2024-06-01 | "Gold prices likely to rise amid global uncertainty." | 50000 | 50500 | 49500 | 50200 | 1500 | 1 |

Data Preprocessing
===================

1) During Preprocessing of data, find the mean, median and maximum length of text (sequence length). This would help to tune hyper parameter on sequence lenght.



Recommended Option 
1. Data Collection & Preprocessing
Collect historical and real-time financial data related to gold prices from reliable sources (e.g., market APIs, news websites, economic reports).
Perform preprocessing as per the design document: cleaning, deduplication, tokenization, normalization.
2. Pretrained Model Selection
Evaluate and select a transformer-based pretrained language model fine-tuned for financial applications (e.g., FinBERT, BloombergGPT, or LlamaFin).
If such models aren't available, use a general-purpose LLM like BERT, RoBERTa, or GPT-2/3, and specialize via fine-tuning.
3. Fine-Tuning on Domain Data
Fine-tune the selected model using domain-specific data related to gold prices (news headlines, financial reports, analyst commentary, etc.).
4. Embedding Strategy
Investigate using domain-relevant embeddings (Word2Vec, GloVe, or Sentence-BERT) to encode contextual financial terms if needed before passing to transformer.
5. RAG Architecture Integration (Optional)
Explore integrating a Retrieval-Augmented Generation (RAG) model:
Store gold price–related news and historical data in a vector database.
Use RAG to fetch relevant context during inference for better prediction.
6. Hyperparameter Tuning
Conduct hyperparameter tuning on validation set: learning rate, batch size, dropout, epochs, etc., to improve performance.
Integrate with Wanda for analysis

Less Recommended Option 
1. Data Collection & Preprocessing
Same as above.
2. Embedding Strategy First
Begin by testing Word2Vec, GloVe, or sentence embeddings to represent text, rather than starting with a transformer model directly.
3. Model Training from Scratch
Train a transformer model from scratch on the financial corpus — this is resource-intensive and may not be optimal unless massive high-quality labeled data is available.
4. Explore Base Software
Evaluate external or in-house software tools as discussed during the TA session for model building or orchestration.
5. Fine-Tune Hyperparameters
Tune based on performance, but without leveraging pretrained transformer knowledge, it may lead to lower baseline performance.
Recommendation
Option 1 (Refined) is strongly preferred for the following reasons:
Leverages pretrained models, reducing training time and data requirements.
Domain adaptation through fine-tuning is more efficient than training from scratch.
Combines retrieval (RAG) and generative models, which is ideal for handling diverse and dynamic financial data.
Uses embeddings where helpful but does not rely on them exclusively.


Recommendations for Fine-Tuning on Finance Data
Use Case
Recommended Base Model
Small GPU (< 24 GB)
mistralai/Mistral-7B-v0.1 or meta-llama/Meta-Llama-3-8B (4-bit)
Finance Specific Tasks
AI4Finance/FinGPT or NousResearch/Hermes-*
Instruction-Following Q&A
Open-Orca/Mistral-7B-OpenOrca or zephyr-7b-beta
MoE Fast Inference
mistralai/Mixtral-8x7B-Instruct-v0.1




