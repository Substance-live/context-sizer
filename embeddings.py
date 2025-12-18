"""
Клиент для получения embeddings через локальный embedding-сервис.
"""
import logging
from typing import List
import numpy as np
import requests
from openai import OpenAI

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """
    Клиент для получения embeddings через OpenAI-совместимый API.
    Поддерживает пакетную отправку запросов.
    """
    
    def __init__(self, base_url: str, model: str, batch_size: int = 64):
        """
        Инициализирует клиент для получения embeddings.
        
        Args:
            base_url: URL эндпоинта (например, http://localhost:11434)
            model: Название модели для embeddings
            batch_size: Размер батча для пакетной обработки
        """
        self.base_url = base_url
        self.model = model
        self.batch_size = batch_size
        
        # Инициализация OpenAI клиента
        # Для Ollama может потребоваться настройка base_url
        # Если ваш API имеет другой формат, измените метод _get_embeddings_batch
        self.client = OpenAI(
            base_url=base_url,
            api_key="not-needed"  # Для локальных сервисов ключ не требуется
        )
        
        logger.info(f"Инициализирован EmbeddingClient: base_url={base_url}, model={model}")
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Получает embeddings для списка текстов с пакетной обработкой.
        
        Args:
            texts: Список текстов для получения embeddings
            
        Returns:
            Массив embeddings формы (n_texts, embedding_dim)
            
        Raises:
            Exception: При ошибках подключения или получения embeddings
        """
        if not texts:
            return np.array([])
        
        all_embeddings = []
        
        # Обрабатываем тексты батчами
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            logger.debug(f"Обработка батча {i//self.batch_size + 1}: {len(batch)} текстов")
            
            batch_embeddings = self._get_embeddings_batch(batch)
            all_embeddings.append(batch_embeddings)
        
        result = np.vstack(all_embeddings)
        logger.info(f"Получено embeddings для {len(texts)} текстов, размерность: {result.shape}")
        
        return result
    
    def _get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        Получает embeddings для батча текстов.
        
        Поддерживает два формата API:
        1. OpenAI-совместимый API (через библиотеку openai)
        2. Прямой HTTP запрос к Ollama API (если модель начинается с "ollama/")
        
        ВАЖНО: Если ваш embedding API имеет другой формат, измените этот метод.
        
        Args:
            texts: Список текстов для обработки
            
        Returns:
            Массив embeddings формы (len(texts), embedding_dim)
        """
        # Определяем, используется ли Ollama по префиксу модели
        use_ollama_direct = self.model.startswith('ollama/')
        ollama_model_name = self.model.replace('ollama/', '') if use_ollama_direct else None
        
        embeddings = []
        
        try:
            if use_ollama_direct:
                # Прямой HTTP запрос к Ollama API
                # Формат: POST /api/embeddings с телом {"model": "model-name", "prompt": "text"}
                for text in texts:
                    try:
                        response = requests.post(
                            f"{self.base_url}/api/embeddings",
                            json={
                                "model": ollama_model_name,
                                "prompt": text
                            },
                            timeout=60
                        )
                        response.raise_for_status()
                        result = response.json()
                        # Ollama возвращает embedding в поле 'embedding'
                        embedding = result.get('embedding', result.get('data', [{}])[0].get('embedding'))
                        if embedding is None:
                            raise ValueError(f"Не удалось извлечь embedding из ответа Ollama: {result}")
                        embeddings.append(embedding)
                    except requests.exceptions.RequestException as e:
                        logger.error(f"Ошибка HTTP запроса к Ollama для текста: {text[:50]}...")
                        logger.error(f"Детали ошибки: {e}")
                        raise
            else:
                # OpenAI-совместимый API
                for text in texts:
                    try:
                        response = self.client.embeddings.create(
                            model=self.model,
                            input=text
                        )
                        # Извлекаем embedding из ответа
                        embedding = response.data[0].embedding
                        embeddings.append(embedding)
                    except Exception as e:
                        logger.error(f"Ошибка при получении embedding для текста: {text[:50]}...")
                        logger.error(f"Детали ошибки: {e}")
                        # Если формат ответа отличается, проверьте структуру response
                        raise
            
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"Ошибка при получении embeddings для батча: {e}")
            raise RuntimeError(
                f"Не удалось получить embeddings. Проверьте подключение к {self.base_url} "
                f"и корректность модели {self.model}. "
                f"Для Ollama используйте формат модели 'ollama/model-name' или измените метод _get_embeddings_batch."
            ) from e

