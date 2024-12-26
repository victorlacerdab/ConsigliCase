import torch
from transformers import ColPaliForRetrieval, ColPaliProcessor
import os
import fitz
import json
import ollama
from PIL import Image
from typing import List

class Retriever:
    def __init__(self, device, config_dict: dict):
        self.emb_model, self.processor = self.init_colpali(device)
        self.data_path = config_dict['data_path']
        self.imgdata_path = config_dict['imgdata_path']
        self.doc_embs = self._init_doc_embs(config_dict['emb_path'])
        self.idx_store = self._init_idx_store(config_dict['idx_store_path'])

    def init_colpali(self, device):
        model_name = "vidore/colpali-v1.2-hf" # This base model can be further finetuned for application-specific solutions

        model = ColPaliForRetrieval.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        ).eval()

        processor = ColPaliProcessor.from_pretrained(model_name)

        return model, processor

    def score_similarity(self, query: str, top_k=1) -> torch.Tensor: # Tensor of dim (1, top_k)
        query_emb = self._embed_query_tensor(query)
        top_k_idcs = torch.topk(self.processor.score_retrieval(query_emb, self.doc_embs), k = top_k)
        return top_k_idcs[1] # Returns only the indices and not the actual distance values, see topk docs.
    
    def get_files_from_idx_store(self, top_k_args: torch.Tensor) -> List[str]:
        top_k_args = top_k_args.squeeze().tolist()
        # If it's a single integer, convert it to a list
        if isinstance(top_k_args, int):
            top_k_args = [top_k_args]
        relevant_docs = [self.idx_store[str(int(k))] for k in top_k_args]
        return relevant_docs

    def _embed_query_tensor(self, query: str) -> torch.Tensor:
        query = self.processor(text=query).to(self.emb_model.device)
        with torch.no_grad():
            query_embs = self.emb_model(**query).embeddings
        return query_embs

    def _init_doc_embs(self, emb_path: str) -> torch.Tensor:
        docs_embs = torch.load(emb_path)
        return docs_embs
    
    def _init_idx_store(self, idx_store_path: str) -> dict:
        with open(idx_store_path, 'r') as f:
            idx_store = json.load(f)
        return idx_store
    
class GenerationModel:
    def __init__(self, retrieval_model: Retriever, config_dict: dict):
        self.language_model = config_dict['language_model']
        self.retrieval_model = retrieval_model
        self.system_prompt = config_dict['system_prompt']
        self.chat_history = self._init_chat_history()
        self.idx_store = self.retrieval_model.idx_store

    def full_blown_rag(self, prompt: str, top_k: int):
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Invalid prompt provided")
        
        print('Analyzing many documents. This may take a while.')
        top_k_sim_idcs = self.retrieval_model.score_similarity(prompt, top_k=top_k)
        retrieved_docs = self.retrieval_model.get_files_from_idx_store(top_k_sim_idcs)
        final_summary = ''
        for doc in retrieved_docs:
            summary = ollama.chat(model=self.language_model, messages=[
                {'role': 'system',
                 'content': f'Summarize the document in order to help an expert answer this question: {prompt}',
                 'images': [doc]}
            ])
            final_summary = final_summary + ' [NEXT SUMMARY] ' + summary['message']['content']

        print(f'Finished analyzing the documents. Initiating answer.')

        # Appends the summaries for later
        self.chat_history.append({
            'role': 'system',
            'content': f'Useful facts for answering {prompt}: {final_summary}.'
        })
        # Appends the actual prompt
        self.chat_history.append({
            'role': 'user',
            'content': prompt
        })

        stream = ollama.chat(model=self.language_model, messages=self.chat_history, stream=True)
        
        full_response = ""
        for chunk in stream:
            content = chunk['message']['content']
            full_response += content
            yield content 

        self.chat_history.append({'role': 'assistant', 'content': full_response})
    
    def stream_rag_interact(self, prompt: str):
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Invalid prompt provided")
        
        top_k_sim_idcs = self.retrieval_model.score_similarity(prompt)
        retrieved_docs = self.retrieval_model.get_files_from_idx_store(top_k_sim_idcs)
        
        self.chat_history.append({
            'role': 'user',
            'content': prompt,
            'images': [retrieved_docs[0]]  # Llama Vision only accepts one image at a time
        })
        stream = ollama.chat(model=self.language_model, messages=self.chat_history, stream=True)
        
        full_response = ""
        for chunk in stream:
            content = chunk['message']['content']
            full_response += content
            yield content 

        self.chat_history.append({'role': 'assistant', 'content': full_response})

    def stream_interact(self, prompt: str):
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Invalid prompt provided")
        
        self.chat_history.append({'role': 'user', 'content': prompt})
        response = ''
        stream = ollama.chat(model=self.language_model, messages=self.chat_history, stream=True)
        for chunk in stream:
            content = chunk['message']['content']
            response += content
            yield content
        self.chat_history.append({'role': 'assistant', 'content': response})

    def regenerate_answer(self):
        self.chat_history.pop()
        response = ''
        stream = ollama.chat(model=self.language_model, messages=self.chat_history, stream=True)
        for chunk in stream:
            content = chunk['message']['content']
            response += content
            yield content
        self.chat_history.append({'role': 'assistant', 'content': response})
        
    def _init_chat_history(self):
        chat_history = []
        chat_history.append({'role': 'system', 'content': self.system_prompt})
        return chat_history

class DataHandler:
    def __init__(self, device, config_dict: dict):
        self.emb_model, self.processor = self._init_colpali(device)
        self.collection_name = config_dict['collection_name']
        self.data_path = config_dict['data_path']
        self.imgdata_path = config_dict['imgdata_path']

    def preprocess_pdf_as_img(self, fpath: str):
        doc = fitz.open(fpath)
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=300) # Change dpi for better or worse quality
            output_path = os.path.join(self.imgdata_path, f"{os.path.basename(fpath)}_page_{page_num + 1}.png")
            pix.save(output_path)
        
    def _init_colpali(self, device):
        model_name = "vidore/colpali-v1.2-hf" # This base model can be further finetuned for application-specific solutions

        model = ColPaliForRetrieval.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        ).eval()

        processor = ColPaliProcessor.from_pretrained(model_name)

        return model, processor
    
    def embed_img_tensor(self, img_path: str):
        image = Image.open(img_path)
        image = self.processor(images=image).to(self.emb_model.device)

        with torch.no_grad():
            image_embeddings = self.emb_model(**image).embeddings

        return image_embeddings
