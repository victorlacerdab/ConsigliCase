import os
import torch

config_dict = {'emb_model': 'sentence-transformers/multi-qa-mpnet-base-dot-v1',
               'language_model': 'llama3.2-vision',
               'data_path': './Data',
               'imgdata_path': './Data/Images',
               'emb_path': './pdf_embeddings.pt',
               'idx_store_path': './idx_store.json',
               'system_prompt': 'You are a professional assistant, tasked with carefully and truthfully analyzing tables and reports.'}

questions = [
    "What was BMW's total revenue in 2023?",
    "How much revenue did Tesla generate in 2023?",
    "What was Ford's revenue for the year 2020?",
    "Can you provide the revenue figures for BMW in 2017?",
    "What key economic factors influenced Ford's performance in 2021?",
    "Which Tesla product is currently in the development stage?",
    "What were BMW's profit figures for 2020 and 2023?",
    "Between Tesla and Ford, which company achieved higher profits in 2022?",
    "What were Tesla's profit numbers for 2022 and 2023?",
    "Which company recorded better profitability in 2022 overall?",
    "Provide a summary of revenue figures for Tesla, BMW, and Ford over the past three years.",
    "What were the growth trends for BMW's financial performance from 2020 to 2023?"
]

def list_files(dfolder_path: str, file_type: str) -> list[str]:
    pdf_files = []
    for root, _, files in os.walk(dfolder_path):
        for file in files:
            if file_type == 'pdf':
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
            elif file_type == 'png':
                if file.lower().endswith('.png'):
                    pdf_files.append(os.path.join(root, file))
    return pdf_files

def get_device():
    '''
    Returns the device to be used by the embedding and language models.
    '''
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU:", torch.cuda.get_device_name())
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS is available.")
    else:
        device = torch.device("cpu")
        print("Neither CUDA nor MPS is available. Using CPU.")
    
    return device