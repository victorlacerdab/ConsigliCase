
# How to use

+ This repo is not meant to be used outside of my own environment, so I do not provide a requirements.txt file or Docker image. Nonetheless, I give a general overview of the embedding, retrieval, and generation flow.
+ This implementation relies on hf-huggingface and ollama.
+ It uses a ColPali model for embedding and retrieval capabilities, and Llama3.2-Vision for generation.
+ Define the root folder where your documents are stored in utils.config_dict['data_path']
+ Run the $embeddings.py$ script to convert your .pdf files into images and embed them using ColPali
+ Run the 'run.py' script in your terminal to start the CLI.

# Embedding Model

+ The same model is used for embedding documents and retrieving them, ensuring consistency across the pipeline.

# Limitations

+ For simplicity, we do not include a reranker. This step should be added after the top k docs are retrieved.
+ Also for simplicity, a full vector database is not used.
+ A more robust approach that uses rank fusion and relies on both keyword search (using an algorithm like BM25) and semantic search.

# Improvements

+ In order to improve reasoning in multi-entity queries, a useful approach could be that of synthetically generating questions about the documents and the rows on its tables, and having them answered in structured outputs. These texts could then be embedded and later retrieved.
+ Caching of questions and answers is not implemented, but would enhance speed and accuracy (assuming correct answers are cached).