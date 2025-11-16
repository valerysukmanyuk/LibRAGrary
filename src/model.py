from openvino_genai import LLMPipeline, TextEmbeddingPipeline, StreamingStatus, TextRerankPipeline, GenerationConfig, Tokenizer
from openvino import Core

class OVLLM:
    """Инициализация LLM.
    Args:
        model_path (str): Путь до директории модели.
        device (str): Девайс для развертки 'GPU', 'CPU', 'NPU'.
        temp (float): Температура модели.
        sampling (bool): Семплинг включен или выключен.
        top_p (float): top-p модели при семплинге
        top_k (int): top-k модели при семплинге
        max_new_tokens (int): Максимальное количество новых токенов в ответе
    Returns:
        OVLLM: Инициализированная LLM."""
    
    def __init__(self, model_path="./models/LLMs/OpenVino_Mistral-7B-Instruct-v0.3-int4-cw-ov", device="NPU", temp=0.1, sampling=False, top_p=0.9, top_k=20, max_new_tokens=100):
        self.model_path = model_path
        self.device = device
        self.pipe = None    
        self.eos_id = None
        self.generation_config = GenerationConfig(
        temperature=temp,        
        top_p=top_p,             
        top_k=top_k,             
        do_sample=sampling,      
        max_new_tokens=max_new_tokens,     
        ignore_eos=False
    )

    def load_model(self):
        """Загрузка модели в память"""
        print("Загружаю LLM")
        self.pipe_config = {"MAX_PROMPT_LEN": 4000, "GENERATE_HINT": "BEST_PERF", "MIN_RESPONSE_LEN": 10}
        self.pipe = LLMPipeline(self.model_path, self.device, config=self.pipe_config)
        return self.pipe

    def streamer(self, subword):
        """Создание стримера"""
        print(subword, end='', flush=True)
        return StreamingStatus.RUNNING

    def run(self, prompt):
        """Запуск модели для генерации текста"""
        if not self.pipe:
            self.pipe = self.load_model()
        self.prompt = prompt
        if not self.eos_id:
            tokenizer = self.pipe.get_tokenizer()
            self.eos_id = tokenizer.get_eos_token_id()  
            del tokenizer
            if  self.eos_id is not None:
                self.generation_config.set_eos_token_id( self.eos_id)
        self.pipe.generate(self.prompt, generation_config=self.generation_config, streamer=self.streamer, max_new_tokens=512)


class OVEmbeddingModel:
    """Инициализация эмбеддера.
    Args:
        model_path (str): Путь до директории модели.
        device (str): Девайс для развертки 'GPU', 'CPU'.
    Returns:
        OVEmbeddingModel: Инициализированный эмбеддер."""
    
    def __init__(self, model_dir="./models/Embedders/intfloat_multilingual-e5-large", device="GPU"):
        self.model_dir = model_dir
        self.pipe = None
        self.device = device
        self.tokenizer = None

    def load_model(self):
        """Загрузка модели в память"""
        self.pipe = TextEmbeddingPipeline(self.model_dir, self.device)
        self.tokenizer = Tokenizer(self.model_dir)
        return self.pipe, self.tokenizer
    
class OVRanker:
    """Инициализация эмбеддера.
    Args:
        model_path (str): Путь до директории модели.
        device (str): Девайс для развертки 'GPU', 'CPU'.
        top_n (int): Максимальное количество возвращаемых топ подходящих ответов
    Returns:
        OVRanker: Инициализированный ранкер."""
    
    def __init__(self, model_dir = "./models/Rankers/BAAI_bge-reranker-v2-m3", device="GPU", top_n=25):
        self.model_dir = model_dir
        self.pipe = None
        self.device = device
        self.top_n = top_n

    
    def load_model(self):
        """Загрузка модели в память"""
        self.pipe = TextRerankPipeline(self.model_dir, device=self.device, top_n=self.top_n)
        return self.pipe
