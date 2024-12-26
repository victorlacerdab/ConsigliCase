import torch, json
from utils import config_dict, list_files, get_device
from vlm import DataHandler

pdf_file_ls = list_files(config_dict['data_path'], 'pdf')
kb_embedder = DataHandler(device=get_device(), config_dict=config_dict)

print(f'There are {len(pdf_file_ls)} documents to be uploaded.')

# Transforms each page of the documents into images
for file in pdf_file_ls:
    kb_embedder.preprocess_pdf_as_img(file)

# Stores a list with img file names
img_file_ls = list_files(config_dict['imgdata_path'], 'png')

# The main embedding loop.
# Iterates through the files one by one and updates the index store.
# In a real world scenario, this step would be handled in batches and the storage would be a vector database
# e.g. ChromaDB, Pinecone, PostgreSQL + pgvector, etc.
idx_store = {}
embeddings_list = []

for idx, file in enumerate(img_file_ls):
    if idx%10 == 0:
        print(f'Embedding file {idx+1}/{len(img_file_ls)}')
    idx_store.update({idx:file})
    embeddings = kb_embedder.embed_img_tensor(file)
    embeddings_list.append(embeddings)

docs_embs = torch.cat(embeddings_list, dim=0)

# Saves the generated embeddings as a tensor
torch.save(docs_embs,  config_dict['emb_path'])

# Saves the index store as .json file
with open('idx_store.json', 'w') as json_file:
    json.dump(idx_store, json_file, indent=4)

print('Documents successfully indexed to the vector database.')