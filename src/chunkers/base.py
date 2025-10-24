from abc import ABC, abstractmethod
import pandas as pd
import torch
import os

class BaseChunker(ABC):
    def __init__(
        self,
        # chunking strategies can be one of the following: 
        #   full: full report
        #   sent: sentences using HuggingFace's SentenceSplitter.sentence_splitter
        #   noph: noun phrase using spacy's noun_chunks
        #   npvb: noun phrase with verb using spacy's noun_chunks appending verbs
        #   sbar: S/SBARS using Stanford's stanza
        #   mixd: union of sent, npvb, sbars 
        strategy: str, # full, sent, noph, npvb, sbar, mixd
        embedding_pt_path: str, 
        index_csv_path: str, 
        llm_embedder,  # Callable: List[str] -> np.ndarray or torch.Tensor
        split: str
    ):
        self.strategy = strategy
        self.embedding_pt_path = embedding_pt_path
        self.index_csv_path = index_csv_path
        self.llm_embedder = llm_embedder  # Inject external LLM API or object
        self.split = split  # 'test' or 'train'
        self.df_chunks = None
        self.embeddings = None

    def get_chunk_level(self) -> str:
        """
        Returns the name of the current chunking strategy as a string 
        (e.g., "sentence", "noun_phrase").
        """
        return self.strategy
    
    def load_or_compute(self, texts_df: pd.DataFrame, **kwargs):
        """
        Main interface: loads or computes chunk embeddings and sets self.df_chunks and self.embeddings.
        texts_df: DataFrame with at least a 'text' column, and identifying columns (e.g. report_id).
        """
        if os.path.exists(self.embedding_pt_path) and os.path.exists(self.index_csv_path):
            self.embeddings = torch.load(self.embedding_pt_path)
            self.df_chunks = pd.read_csv(self.index_csv_path)
        else:
            self.df_chunks = self.chunk(texts_df, **kwargs)
            self.embeddings = self.compute_embeddings(self.df_chunks['text'].tolist())
            torch.save(self.embeddings, self.embedding_pt_path)
            self.df_chunks.to_csv(self.index_csv_path, index=False)
    
    @abstractmethod
    def chunk(self, texts_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Returns a DataFrame with chunked text and identifying columns.
        Must produce a 'text' column, plus all needed metadata columns.
        """
        pass

    def compute_embeddings(self, texts: list) -> torch.Tensor:
        """
        Uses injected LLM API function to compute and return embeddings.
        """
        return self.llm_embedder(texts)  # Should return a tensor or np array
