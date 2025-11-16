from src.embedder import Embedder
from src.model import OVEmbeddingModel, OVRanker, OVLLM
from src.retriever import Retriever
from src.rag import RAG

# Запуск RAG системы
def rag_initialize(
    llm_device="NPU",
    llm_temp=0.2,
    llm_sampling=False,
    llm_top_p=0.9,
    llm_top_k=20,
    llm_max_new_tokens=200,
    embedder_device="GPU",
    ranker_device="CPU",
    ranker_top=5,
    chunk_size=256,
    chunk_overlap=25,
    batch_size=50,
    cosine_batch = 30):    
    
    """ Сбор всей раг системы вместе и ее запуск.
    Args:
        llm_device (str): Девайс для развертки: 'GPU', 'CPU', 'NPU'.
        llm_temp (int): Максимальное количество токенов в одном чанке.
        llm_sampling (int): Максимальное количество токенов для оверлапа между чанками.
        llm_top_p (int): Количество чанков в батче, который будет передан эмбеддеру.
        llm_max_new_tokens (int): Максимальное количество новых токенов в ответе
        embedder_device (str): Девайс для развертки: 'GPU', 'CPU'.
        ranker_device (str): Девайс для развертки: 'GPU', 'CPU'.
        ranker_top (int): Максимальное количество возвращаемых топ подходящих ответов
        chunk_size (int): Максимальное количество токенов в одном чанке.
        chunk_overlap (int): Максимальное количество токенов для оверлапа между чанками.
        batch_size (int): Количество чанков в батче, который будет передан эмбеддеру.
        cosine_batch (int): Размер батча индекса, который будет передан ранкеру после расчета косинусного сходства."""
    
    llm = OVLLM(device=llm_device, temp=llm_temp,
                sampling=llm_sampling, top_p=llm_top_p,
                top_k=llm_top_k, max_new_tokens=llm_max_new_tokens)
    
    embedder_model = OVEmbeddingModel(device=embedder_device)

    ranker_model = OVRanker(device=ranker_device, top_n=ranker_top)

    embedder = Embedder(embedder_model, chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap, batch_size=batch_size)
    
    retriever = Retriever(ranker_model=ranker_model, embedder=embedder, 
                          ranker_n_items=ranker_top, cosine_batch=cosine_batch)
    
    rag = RAG(llm=llm, embedder=embedder, retriever=retriever)

    rag.run()

if __name__ == "__main__":
    rag_initialize()