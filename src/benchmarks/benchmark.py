from abc import ABC, abstractmethod
import logging


class BenchmarkFactory:
    _registry = {}

    @classmethod
    def register(cls, key):
        def decorator(subclass):
            cls._registry[key] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, key, *args, **kwargs):
        if key not in cls._registry:
            raise ValueError(f"Unknown score type: {key}")
        return cls._registry[key](*args, **kwargs)


class Benchmark(ABC):
    def __init__(self, name: str):
        self.name = name
        if not hasattr(self, "name") or self.name is None:
            raise NotImplementedError("Subclass must define self.name")
        self.logger = logging.getLogger(self.name)

    @abstractmethod
    def get_test_set(self):
        """
        Return DataFrame containing chunked test reports and their labels.

        The function must use the mandatory configuration file to define
        report chunking.
        Returns a pandas DataFrame with columns:
            - "text": the textual content of each chunk,
            - "y_true": the true label(s) associated with each chunk.
        The DataFrame uses a multi-index consisting of a report ID from the
        test set and, optionally, a sentence or text chunk ID.
        """
        pass

    @abstractmethod
    def get_test_vectors(self):
        """
        Retrieve or generate embeddings for the corresponding test set.

        Loads precomputed embeddings from a fixed directory or cache, or
        computes and persists them if missing. Returns a PyTorch tensor,
        indexed identically to the test set, where each entry is a vector for
        a report chunk or sentence. Embeddings should be cached and versioned
        to match updates in the underlying test set.
        """
        pass

    @abstractmethod
    def get_train_set(self):
        """
        Return DataFrame containing chunked train reports and their labels.

        Behaviour analogous to get_test_set()
        """
        pass

    @abstractmethod
    def get_train_vectors(self):
        """
        Retrieve or generate embeddings for the corresponding train set.
        Behaviour analogous to get_test_vectors().
        """
        pass
