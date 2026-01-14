from abc import ABC, abstractmethod
import logging
import pandas as pd

from src.utils.metrics import micro_f1_at_k, macro_f1_at_k

class BaseSolver(ABC):
    def __init__(self, name: str):
        self.name = name
        if not hasattr(self, "name") or self.name is None:
            raise NotImplementedError("Subclass must define self.name")
        self.logger = logging.getLogger(self.name)
		
    @abstractmethod
    def fit(self, reference_embeddings, reference_labels=None): 
        """
        Load and prepare reference embeddings, possibly storing associated labels or metadata for each vector. This is analogous to training for nonparametric solvers, setting up the reference index or structure.
        """
        pass

    @abstractmethod
    def predict(self, query_embeddings, top_k, distance_metric='cosine'): 
        """"
        For each query embedding, returns the index/label of the most similar reference entry (or top-k matches).

        
        query_embeddings : 
            Text embeddings.

        Returns
        -------
        y_pred : DataFrame
            Table of identifying columns, icd_code, distance, threshold_passed flag.
        """
        pass

    @abstractmethod
    def train(self, query_embeddings, distance_metric='macro_f1_at_k', top_k=5): 
        """"
        Compute optimal model parameter wrt metric.

        Returns
        """
        pass
    

    def scores(
        self,
        pred_labels: pd.DataFrame,
        true_labels: pd.DataFrame,
        top_k=1,
        level='leaf',
        granularity='sentence'
    ):
        """
        Computes quantitative evaluation metrics, such as precision, recall, 
        micro and macro F1 score at k by comparing true labels against the 
        top-k predicted labels for each text unit. 

        Parameters
        ----------
        pred_labels : pandas.DataFrame
            DataFrame containing model-predicted labels. Must include columns:
            - 'report_id': identifier for the report/document.
            - 'sentence_id': identifier for the sentence or text unit (indexed).
            - 'icd_code': predicted ICD-10 code(s) for each row.
            - 'distance': model output score or inverse-proximity for each label. 
            Each row represents one predicted label per text unit.

        true_labels : pandas.DataFrame
            DataFrame of ground truth labels. Must include:
            - 'report_id': identifier for the report/document.
            - 'sentence_id': identifier for the sentence or text unit.
            - 'icd_code': ground truth ICD-10 code(s) for each text unit.
            Each row denotes one reference label per text unit.

        top_k : int, default=1
            Number of top-ranked predictions per text unit to consider as "retrieved".

        level : {'leaf', 'subcategory', 'category', ''}, default='leaf'
            Granularity level for the evaluation (e.g., leaf ICD-10 codes). 
            As there can be multiple subcategories for a single code, the one 
            closest to the root is chosen. If a code is leaf node and highest 
            subcategory this is identical to setting level to leaf.
        
        granularity : {'sentence', 'report'}, default='sentence'
            Text granularity for which to aggregate labels.

        Returns
        -------
        scores : dict
            Dictionary with precision_at_k, recall_at_k, micro_f1_at_k, macro_f1_at_k, 
             per true code metrics (precision, recall, f1), and k (=top_k).
        """
        # TODO: replace codes with ancestor in case level != leaf

        # Select groupby keys depending on granularity
        if granularity == 'sentence':
            group_keys = ['report_id', 'sentence_id']
        elif granularity == 'report':
            group_keys = ['report_id']
        else:
            raise ValueError("granularity must be 'sentence' or 'report'")
        
        micro_metrics = micro_f1_at_k(
            pred_labels=pred_labels, 
            true_labels=true_labels, 
            top_k=top_k, 
            group_keys=group_keys
        )
        macro_metrics = macro_f1_at_k(
            red_labels=pred_labels, 
            true_labels=true_labels, 
            top_k=top_k, 
            group_keys=group_keys
        )
        return {**micro_metrics, **macro_metrics}


    @abstractmethod
    def get_neighbors(self, query_embeddings, top_k=10): 
        """
        Returns full information (distances, scores, or indices) of the k nearest reference embeddings for labeling or auditing. This allows storing and reviewing close matches.
        """
        pass

    @abstractmethod
    def set_parameters(self, **kwargs): 
        """
        Optionally set internal solver parameters like similarity threshold, distance metric, or tuning options.
        """
        pass

    @abstractmethod
    def get_parameters(self): 
        """
        Retrieve the current configuration of internal parameters for transparency and reproducibility.
        """

    @abstractmethod
    def partial_fit(self, new_reference_embeddings): 
        """
        Incrementally add more reference vectors without a full rebuild (important for dynamic datasets).
        """
	
    @abstractmethod
    def save(self, filename): 
        """
        Method to persist a model/index state.
        """
        pass

    @abstractmethod
    def load(self, filename): 
        """
        Method to reload model/index state.
        """
        pass