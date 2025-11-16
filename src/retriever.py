import faiss
from faiss import write_index, read_index
from pathlib import Path
import pandas as pd
import numpy as np
import gc

class Retriever:
    """
    Инициализация ретривера.

    Args:
        ranker_model (OVRanker): Инстанс модели ранжирования.
        embedder (Embedder): Инстанс эмбеддера.
        ranker_n_items (int): Количество эталонов, которое вернет модель ранжирования.
        cosine_batch (int): Размер батча индекса, который будет передан ранкеру после расчета косинусного сходства.

    Returns:
        Retriever: Инициализированный ретривер.
    """
    def __init__(self, ranker_model, embedder, ranker_n_items=3, cosine_batch=50):

        self.ranker = ranker_model
        self.embedder = embedder
        self.results = []
        self.loaded = None
        self.index = None  
        self.k = ranker_n_items
        self.cosine_batch = cosine_batch

    def _load_model(self):
        """Загрузка модели ранкера в память"""
        print("Загружаю модель ранжирования")
        self.loaded = self.ranker.load_model()
        return self.loaded 
    
    def _unload_model(self):
        """Выгрузка модели эмбеддера из памяти"""
        print("Выгружаю модель ранжирования")
        if self.ranker.pipe is not None:
            del self.ranker.pipe
            self.ranker.pipe = None
        gc.collect()
        self.loaded = None
        return self.loaded

    def _create_index(self):
        """Создаем индекс и сохраняем его в папочку к эмбеддингам"""
        print("Создаю индекс")
        embeddings = self.embedings.astype('float32')
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        print("Сохраняю индекс")
        write_index(index, "./embeddings/library.faiss")
        self.path_to_index = "./embeddings/library.faiss"

    def _find_index(self) -> bool:
        """Проверяем, есть ли индекс и актуален ли он"""
        print("Ищу индекс")
        index_path = Path("./embeddings/library.faiss")
        embeddings_path = Path("./embeddings/combined.npy")
        
        # Если нет папки embeddings
        if not index_path.exists():
            print("Индекс не найден")
            return False

        # Проверяем, не изменились ли эмбеддинги после создания индекса
        if embeddings_path.exists() and index_path.stat().st_mtime < embeddings_path.stat().st_mtime:
            print("Индекс устарел — обновляю")
            return False

        # Нашли индекс
        print("Нашел актуальный индекс")
        self.path_to_index = str(index_path)
        return True
        
    def _read_index(self):
        """Читает индекс из папочки эмбеддингов"""
        print("Читаю индекс")
        self.index = read_index(self.path_to_index)
        return self.index 
    
    def _rerank(self, query:str, found_chunks:list) -> list:
        """Загрузка ранкера, ранжирование, выгрузка ранкера"""
        self.loaded = self._load_model()
        print("Ранжирую")
        ready_for_prompt = self.loaded.rerank(query, found_chunks)
        ready_for_prompt = sorted(ready_for_prompt, key=lambda x: x[1], reverse=True)
        ready_for_prompt = [f"{found_chunks[idx]}" for idx, score in enumerate(ready_for_prompt)]
        return ready_for_prompt

    def search(self, query:str, index) -> str:
        """Поиск похожих векторов в индексе FAISS, ранжирование"""
        query_vector = self.embedder.embed_query(query)
        query_vector = query_vector.astype('float32')
        faiss.normalize_L2(query_vector)

        print("Сравниваю запрос")
        similarities, indices = index.search(query_vector, self.k)
        
        self.results = []
        for score, idx in zip(similarities[0], indices[0]):
            if idx != -1 and idx < len(self.chunks):
                row = self.chunks.iloc[idx]
                self.results.append({
                    'chunk': row['Chunk'],       
                    'author': row['Author'],    
                    'book': row['Name'],         
                    'score': float(score),
                    'chunk_id': int(idx)
                })
        texts = [i["chunk"] for i in self.results[:self.cosine_batch]]
        prompt_part = self._rerank(query, texts)
        self.loaded = self._unload_model()
        return prompt_part
    
    def run(self):
        """Запускает ретривер, проверяет есть ли индекс, если есть читает его, если он устарел, то обновляет его"""
        index_check = self._find_index()
        if  index_check == False:
            self.embedings = np.load("./embeddings/combined.npy")
            self.chunks = pd.read_csv("./embeddings/chunks.csv")
            self._create_index()
            self.index = self._read_index()
        elif index_check == True: 
            self.chunks = pd.read_csv("./embeddings/chunks.csv")
            self.index = self._read_index()
        del index_check
        gc.collect()