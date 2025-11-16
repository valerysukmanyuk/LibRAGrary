import re
import os
import gc
import numpy as np
import pandas as pd 
import chardet

class Embedder:
    """    Инициализация ретривера.
    Args:
        emb_model (OVEmbeddingModel): Инстанс модели эмбеддера.
        chunk_size (int): Максимальное количество токенов в одном чанке.
        chunk_overlap (int): Максимальное количество токенов для оверлапа между чанками.
        batch_size (int): Количество чанков в батче, который будет передан эмбеддеру.
    Returns:
        Embedder: Инстанс эмбеддера."""
    
    def __init__(self, emb_model, chunk_size, chunk_overlap, batch_size = 50):
        self.model = emb_model
        self.path_to_library="./books"
        self.output_dir="./embeddings"
        self.loaded = None
        self.loaded_tokenizer = None
        self.chunk_size=chunk_size
        self.chunk_overlap=chunk_overlap
        self.batch_size = batch_size

    def _load_model(self) -> tuple:
        """Загрузка модели эмбеддера в память"""
        print("Загружаю эмбеддер")
        self.loaded, self.loaded_tokenizer = self.model.load_model()
        return self.loaded, self.loaded_tokenizer

    def _unload_model(self):
        """Выгрузка модели эмбеддера из памяти"""
        print("Выгружаю эмбеддер")
        if self.model.pipe is not None:
            del self.model.pipe
            self.model.pipe = None
        gc.collect()
        self.loaded = None
        self.loaded_tokenizer = None

    def _check_ready_embeddings(self) -> list:
        """Проверяем для каких книг эмбеддинги уде есть в папке"""
        ready = []
        if os.path.exists(self.output_dir):
            for file in os.listdir(self.output_dir):
                if file.endswith(".npy") and file != "combined.npy":
                    ready.append(os.path.splitext(file)[0])  
        self.ready = ready
    
    def flatten_tokens(self, chunks: list) -> list:
        """Убираем вложенные списки после работы токенайзера"""
        flat = []
        for c in chunks:
            if isinstance(c, list):
                flat.extend(self.flatten_tokens(c))  
            elif isinstance(c, int):
                s = c
                if s:
                    flat.append(s)
        return flat

    def _clean_text(self, text: str) -> str:
        """Чистим тексты регулярками"""
        # Убираем HTML-теги, лишние строки, двойные пробелы, 
        # заменяем множественные пробелы и табы на один пробел
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'(?:\r\n|\r|\n){3,}', '\n\n', text)
        text = re.sub(r'[«»"“”‘’]', '', text)
        text = re.sub(r'[!?]', '.', text)
        text = re.sub(r'[\xa0–]', "", text) 
        text = re.sub(r'[\t]+', ' ', text)
        text = re.sub(r' {2,}', ' ', text)
        return text.strip()

    def _split_paragraphs(self, text: str) -> list:
        """Делим на параграфы по знаку абзаца"""
        paragraphs = text.split("\n\n")

        # Очищаем пробелы и фильтруем: оставляем только абзацы с >= 8 слов
        paragraphs = [
            p.strip() for p in paragraphs 
            if len(p.strip().split()) > 8
        ]
        return paragraphs

    def _chunk(self, text: str) -> list:
        """Чанкуем текст с учётом лимита в токенах и overlap."""
        clean = self._clean_text(text)
        paragraphs = self._split_paragraphs(clean)
        max_tokens = self.chunk_size - 5 # Резервируем 5 токенов на служебные токены
        overlap = self.chunk_overlap
        chunks = []
        buffer_tokens = []

        print("Чанкую")
        for paragraph in paragraphs:
            # Токенизируем параграф
            par_tokens = self.loaded_tokenizer.encode(paragraph, add_special_tokens=False)

            # Объединяем с остатком из предыдущего 
            current = buffer_tokens + par_tokens.input_ids.data.tolist()
            current = self.flatten_tokens(current)
            current_len = len(current)

            # Пока текущих токенов хватает на чанк — отрезаем и сохраняем
            while current_len >= max_tokens:
                chunk_tokens = current[:max_tokens]
                chunk_text = self.loaded_tokenizer.decode(chunk_tokens)
                chunks.append(f"passage: {chunk_text}")

                # Готовим current для следующей итерации:
                # Оставляем последние `overlap` токенов 
                current = current[max_tokens - overlap:]
                current_len = len(current) 

            # Отрезанный кусок становится буффером следующей итерации
            buffer_tokens = current 

        # После всех параграфов, если что-то осталось в буфере, добавляем как последний чанк
        if buffer_tokens:
            chunk_text = self.loaded_tokenizer.decode(buffer_tokens)
            if chunk_text:
                chunks.append(chunk_text)
        
        # Убираем пустые чанки, если такие есть
        chunks = [c for c in chunks if len(c) > 0]

        print(f"Чанки для книги готовы ({len(chunks)} шт.)")
        return chunks

    def embed_query(self, query: str):
        """Векторизация запроса пользователя"""
        if self.loaded == None:
            self.loaded, _ = self._load_model()
        query = f"query: {self._clean_text(query)}"
        print("Векторизую запрос")
        embedding = self.loaded.embed_query(query)

        return np.array([embedding])

    def _prepare_embedings(self):
        """Батчами создаем эмбеддинги для чанков, сохраняем их файлы по книгам в папку embeddings, 
        собираем все чанки и сохраняем их в csv в папку embeddings"""
        os.makedirs(self.output_dir, exist_ok=True)
        self.changed = False 

        chunks_path = os.path.join(self.output_dir, "chunks.csv") # Проверяем наличие уже существующих чанков
        
        if os.path.exists(chunks_path):
            chunk_df = pd.read_csv(chunks_path)
        else:
            chunk_df = pd.DataFrame(columns=["Author", "Name", "Chunk"])

        for file in os.listdir(self.path_to_library):
            base_name = os.path.splitext(file)[0]
            if base_name in self.ready:
                continue

            if self.loaded == None:
                self.loaded, _ = self._load_model()

            with open(f"{self.path_to_library}/{file}", 'rb') as f:
                raw = f.read()
            enc = chardet.detect(raw)["encoding"]

            book = raw.decode(enc, errors="ignore")

            chunks = self._chunk(book)

            # Датафрейм для чанков
            if "_" in base_name:
                author_name, book_name = base_name.split("_", 1)
            else:
                author_name = "Unknown"
                book_name = base_name

            tmp_df = pd.DataFrame({
                "Author": [author_name] * len(chunks),
                "Name": [book_name] * len(chunks),
                "Chunk": chunks
            })
            chunk_df = pd.concat([chunk_df, tmp_df], ignore_index=True)
            del tmp_df

            # Делаем батчами эмбеддинги
            book_embeddings = []
         
            print(f"Приступаю к эмбеддингам для книги, размер батча: {self.batch_size}")
            for i in range(0, len(chunks), self.batch_size):
                batch = chunks[i:i + self.batch_size]
                emb = self.loaded.embed_documents(batch)
                book_embeddings.extend(emb)

            self.changed = True # Отмечаем изменения в векторной библиотеке 
            np.save(f"{self.output_dir}/{base_name}.npy", np.array(book_embeddings, dtype=np.float32))
            del book_embeddings, chunks, book
            gc.collect()

        if self.loaded != None:
            self._unload_model()

        chunk_df.to_csv(os.path.join(self.output_dir, "chunks.csv"), index=False)
        del chunk_df
        gc.collect()

    def _load_all_embeddings(self):
        """Склеиваем все эмбеддинги каждой книжки и сохраняем в отдельный файл"""
        all_embeddings = []
        if self.changed == True: # Если были изменения, то собираю новый общий файл
            print("Добавляю новые эмбеддинги в общий файл")
            for file in os.listdir(self.output_dir):
                if file.endswith(".npy") and file != "combined.npy":
                    file_path = os.path.join(self.output_dir, file)
                    emb = np.load(file_path)
                    all_embeddings.append(emb)
                    print(f"Загружено {file}: {emb.shape}")

            combined = np.vstack(all_embeddings)
            np.save("embeddings/combined.npy", combined)
            del combined, all_embeddings, self.ready
            gc.collect()
        else:
            print("Новых эмбеддингов нет, загружаю faiss индекс")

    def run(self):
        """Запуск всего эмбеддера, который проверяет обновления библиотеки, делает эмбеддинги при надобности и 
        сохраняет файлы combined.npy, chunks.csv"""
        self._check_ready_embeddings()
        self._prepare_embedings()
        self._load_all_embeddings()