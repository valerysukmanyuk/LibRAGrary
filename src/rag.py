import gc
class RAG:
    """    Инициализация раг системы.

    Args:
        llm (OVLLM): Инстанс LLM.
        embedder (OVEmbeddingModel): Инстанс модели эмбеддера.
        retriever (OVRanker): Инстанс модели ранкера.

    Returns:
        RAG: Инстанс всей RAG системы"""
    def __init__(self, llm, embedder, retriever):
        self.llm = llm
        self.embedder = embedder
        self.retriever = retriever
        self.loaded_llm = None

    def _unload_LLM(self):
        """Выгрузка модели эмбеддера из памяти"""
        print("Выгружаю LLM")
        if self.llm.pipe is not None:
            del self.llm.pipe
            self.llm.pipe = None
        gc.collect()
        self.loaded_llm = None

    def run(self):
        """Запуск всей RAG-системы, начиная от проверки библиотеки, заканчивая ответом на вопрос пользователя"""
        self.embedder.run()
        self.retriever.run()
        self.loaded_llm = self.llm

        while True:
            self.query = input("\nПро какой книжный факт хотим вспомнить? (если хочешь выйти, отправь 'q')\n")

            if self.query == "q":
                self._unload_LLM()
                break

            self.retrieved = self.retriever.search(self.query, self.retriever.index)

            self.prompt = f"""
Проанализируй данные тебе фрагменты из книг в контексте, чтобы ответить на вопрос пользователя. Включи режим размышлений.
Отвечай только на русском языке. Будь краток, но информативен. 
Используй ТОЛЬКО те данные из контекста при подготовке ответа. Запрещенно использование любых других данных при подготовке ответа. 
Не упоминай фрагменты, юзер не должен о них знать.

Контекст: {self.retrieved}
Вопрос: {self.query}
"""

            self.loaded_llm.run(self.prompt)