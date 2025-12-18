"""
Утилиты для работы с текстом, конфигурацией и выводом.
"""
import re
import os
import logging
from typing import List, Tuple, Optional
from pathlib import Path
from dotenv import load_dotenv
from colorama import init, Fore, Style

# Инициализация colorama для Windows
init(autoreset=True)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def split_into_sentences(text: str) -> List[str]:
    """
    Разбивает текст на предложения используя регулярные выражения.
    Поддерживает русский и английский языки.
    
    Args:
        text: Входной текст
        
    Returns:
        Список предложений
    """
    # Удаляем лишние пробелы и переносы строк
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Паттерн для разделения предложений:
    # - точка, вопросительный или восклицательный знак
    # - за которыми следует пробел или конец строки
    # - или заглавная буква (начало нового предложения)
    # Учитываем сокращения типа "т.е.", "др.", "и т.д."
    pattern = r'(?<=[.!?])\s+(?=[А-ЯЁA-Z])|(?<=[.!?])\s*$'
    
    # Разбиваем по паттерну
    sentences = re.split(pattern, text)
    
    # Очищаем предложения от пробелов и фильтруем пустые
    sentences = [s.strip() for s in sentences if s.strip()]
    
    logger.debug(f"Разбито на {len(sentences)} предложений")
    return sentences


def load_config() -> dict:
    """
    Загружает конфигурацию из .env файла.
    
    Returns:
        Словарь с параметрами конфигурации
        
    Raises:
        ValueError: Если отсутствуют обязательные параметры или они некорректны
    """
    env_path = Path('.env')
    if not env_path.exists():
        raise FileNotFoundError(
            "Файл .env не найден. Скопируйте .env.example в .env и настройте параметры."
        )
    
    # Используем override=True для перезагрузки переменных из файла
    load_dotenv(override=True)
    
    config = {}
    
    # E1 - порог для удаления похожих предложений
    e1_str = os.getenv('E1')
    if e1_str is None:
        raise ValueError("Параметр E1 не найден в .env")
    try:
        e1 = float(e1_str)
        if e1 < 0:
            raise ValueError(f"E1 должен быть >= 0, получено: {e1}")
        config['e1'] = e1
    except ValueError as e:
        raise ValueError(f"Некорректное значение E1: {e1_str}. Ожидается число >= 0") from e
    
    # E2 - количество предложений, которые должны остаться после этапа 2
    e2_str = os.getenv('E2')
    if e2_str is None:
        raise ValueError("Параметр E2 не найден в .env")
    try:
        e2 = int(e2_str)
        if e2 < 0:
            raise ValueError(f"E2 должен быть >= 0, получено: {e2}")
        config['e2'] = e2
    except ValueError as e:
        raise ValueError(f"Некорректное значение E2: {e2_str}. Ожидается целое число >= 0") from e
    
    # EMB_URL - URL эндпоинта для embeddings
    config['emb_url'] = os.getenv('EMB_URL', 'http://localhost:11434')
    
    # EMB_MODEL - название модели
    emb_model = os.getenv('EMB_MODEL')
    if emb_model is None:
        raise ValueError("Параметр EMB_MODEL не найден в .env")
    config['emb_model'] = emb_model
    
    # BATCH_SIZE - размер батча
    batch_size_str = os.getenv('BATCH_SIZE', '64')
    try:
        batch_size = int(batch_size_str)
        if batch_size <= 0:
            raise ValueError(f"BATCH_SIZE должен быть > 0, получено: {batch_size}")
        config['batch_size'] = batch_size
    except ValueError as e:
        raise ValueError(f"Некорректное значение BATCH_SIZE: {batch_size_str}") from e
    
    logger.info(f"Загружена конфигурация: E1={e1}, E2={e2}, EMB_URL={config['emb_url']}, "
                f"EMB_MODEL={config['emb_model']}, BATCH_SIZE={config['batch_size']}")
    
    return config


def read_file(filepath: str) -> str:
    """
    Читает содержимое текстового файла.
    
    Args:
        filepath: Путь к файлу
        
    Returns:
        Содержимое файла
        
    Raises:
        FileNotFoundError: Если файл не найден
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {filepath}")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        logger.debug(f"Прочитан файл {filepath}, размер: {len(content)} символов")
        return content
    except UnicodeDecodeError as e:
        raise ValueError(f"Ошибка декодирования файла {filepath}: {e}") from e


def print_colored_output(
    sentences: List[str],
    kept_indices: set,
    removed_stage1: dict,  # {removed_idx: representative_idx}
    removed_stage2: set,
    query: str
) -> None:
    """
    Выводит весь исходный контекст с цветовой подсветкой в терминал.
    Каждое предложение выводится цветом в зависимости от его статуса.
    
    Args:
        sentences: Все предложения исходного контекста
        kept_indices: Индексы предложений, оставшихся в финальном контексте
        removed_stage1: Словарь {удаленный_индекс: индекс_представителя} для этапа 1
        removed_stage2: Множество индексов предложений, удаленных на этапе 2
        query: Текст запроса
    """
    print("\n" + "="*80)
    print(f"{Fore.CYAN}Запрос:{Style.RESET_ALL} {query}")
    print("="*80)
    print(f"\n{Fore.CYAN}Исходный контекст с цветовой разметкой:{Style.RESET_ALL}")
    print("-"*80)
    
    # Выводим весь исходный контекст с цветами
    for idx, sentence in enumerate(sentences):
        if idx in kept_indices:
            # Зеленый - оставлено в финальном контексте
            color = Fore.GREEN
            status = "[KEPT]"
            prefix = f"{color}[{idx}]{Style.RESET_ALL} {color}{status}{Style.RESET_ALL}"
            print(f"{prefix} {color}{sentence}{Style.RESET_ALL}")
        elif idx in removed_stage1:
            # Красный - удалено на этапе 1 (дубликат)
            color = Fore.RED
            rep_idx = removed_stage1[idx]
            status = f"[REMOVED_STAGE_1 → дубликат [{rep_idx}]]"
            prefix = f"{color}[{idx}]{Style.RESET_ALL} {color}{status}{Style.RESET_ALL}"
            print(f"{prefix} {color}{sentence}{Style.RESET_ALL}")
        elif idx in removed_stage2:
            # Желтый - удалено на этапе 2 (нерелевантное)
            color = Fore.YELLOW
            status = "[REMOVED_STAGE_2]"
            prefix = f"{color}[{idx}]{Style.RESET_ALL} {color}{status}{Style.RESET_ALL}"
            print(f"{prefix} {color}{sentence}{Style.RESET_ALL}")
        else:
            # Не должно происходить, но на всякий случай
            print(f"[{idx}] {sentence}")
    
    print("-"*80)
    print(f"\n{Fore.GREEN}Зеленый{Style.RESET_ALL} - предложения, оставшиеся в финальном контексте")
    print(f"{Fore.RED}Красный{Style.RESET_ALL} - предложения, удаленные на этапе 1 (дубликаты)")
    print(f"{Fore.YELLOW}Желтый{Style.RESET_ALL} - предложения, удаленные на этапе 2 (нерелевантные)")
    print("="*80 + "\n")


def print_statistics(
    total: int,
    removed_stage1_count: int,
    removed_stage2_count: int,
    kept_count: int
) -> None:
    """
    Выводит статистику обработки.
    
    Args:
        total: Общее количество предложений
        removed_stage1_count: Количество удаленных на этапе 1
        removed_stage2_count: Количество удаленных на этапе 2
        kept_count: Количество оставшихся предложений
    """
    print("\n" + "="*80)
    print(f"{Fore.CYAN}Статистика:{Style.RESET_ALL}")
    print(f"  Исходный контекст: {total} предложений")
    print(f"  {Fore.RED}Удалено на этапе 1 (дубликаты):{Style.RESET_ALL} {removed_stage1_count} предложений")
    print(f"  {Fore.YELLOW}Удалено на этапе 2 (нерелевантные):{Style.RESET_ALL} {removed_stage2_count} предложений")
    print(f"  {Fore.GREEN}Осталось в финальном контексте:{Style.RESET_ALL} {kept_count} предложений")
    print("="*80 + "\n")


def format_reduced_context(sentences: List[str], indices: List[int]) -> str:
    """
    Форматирует уменьшенный контекст для вывода в файл.
    
    Args:
        sentences: Все предложения
        indices: Индексы предложений для включения в результат
        
    Returns:
        Отформатированный текст
    """
    result_sentences = [sentences[i] for i in sorted(indices)]
    return "\n".join(result_sentences)

