
# How to use

+ This implementation relies on hf-huggingface and ollama.
+ It uses a ColPali model for embedding and retrieval capabilities, and Llama3.2-Vision for generation.

+ Define the root folder where your documents are stored in utils.config_dict['data_path']
+ Run the $embeddings.py$ script to convert your .pdf files into images and embed them using ColPali
+ 
# Embedding Model

+ The same model is used for embedding documents and retrieving them, ensuring consistency across the pipeline.
+ I've decided to add the name of the file and the page number as metadata on the Vector DB. This helps extend the model later for info attribution (Perplexity-style).

# Limitations

+ For simplicity, we do not include a reranker. This step should be added after the top k docs are retrieved from vector search.
+ A more robust approach that uses rank fusion and relies on both keyword search (using an algorithm like BM25) and semantic search could be used.
+ !!!!!!!!!!!!!!!!!! Mention the chunking procedure and why length size is what it is !!!!!!!!!!!!!!!!!!

# Improvements

+ Beyond a standard RAG approaches, one could experiment with agentic workflows for data retrieval. This increases the complexity of the pipeline, as well as costs and latency. The tradeoff between increased complexity and costs must be considered.