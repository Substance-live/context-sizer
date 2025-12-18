"""
Реализация алгоритмов фильтрации контекста на основе евклидова расстояния.
"""
import logging
from typing import List, Tuple, Set, Dict
import numpy as np
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


def remove_similar(
    sentences: List[str],
    embeddings: np.ndarray,
    e1: float,
    query_emb: np.ndarray,
    verbose: bool = False
) -> Tuple[List[int], Dict[int, int]]:
    """
    Удаляет похожие предложения (дубликаты по смыслу) на этапе 1.
    
    Алгоритм:
    1. Вычисляет матрицу евклидовых расстояний между всеми парами предложений (O(n^2))
    2. Группирует предложения с расстоянием < e1 как похожие
    3. В каждой группе оставляет только одно предложение - то, которое ближе всего к запросу
    4. Если несколько предложений равноудалены от запроса, выбирается первое по порядку
    
    Args:
        sentences: Список предложений
        embeddings: Массив embeddings предложений формы (n_sentences, embedding_dim)
        e1: Порог для определения похожих предложений
        query_emb: Embedding запроса формы (1, embedding_dim) или (embedding_dim,)
        
    Returns:
        Кортеж (kept_indices, removed_mapping), где:
        - kept_indices: Список индексов предложений, оставшихся после фильтрации
        - removed_mapping: Словарь {удаленный_индекс: индекс_представителя}
    """
    n = len(sentences)
    if n == 0:
        return [], {}
    
    # Нормализуем query_emb к форме (1, embedding_dim)
    if query_emb.ndim == 1:
        query_emb = query_emb.reshape(1, -1)
    
    logger.info(f"Этап 1: Удаление похожих предложений (e1={e1})")
    logger.debug(f"Обработка {n} предложений")
    
    # Шаг 1: Вычисляем матрицу евклидовых расстояний между всеми парами предложений
    # Используем cdist для эффективного вычисления (O(n^2))
    distance_matrix = cdist(embeddings, embeddings, metric='euclidean')
    
    # Шаг 2: Находим группы похожих предложений
    # Предложения i и j похожи, если distance_matrix[i, j] < e1
    # Используем жадный алгоритм: помечаем предложения как обработанные
    # и для каждой группы выбираем представителя
    
    kept_indices = []  # Индексы предложений, которые останутся
    removed_mapping = {}  # {удаленный_индекс: индекс_представителя}
    processed = set()  # Множество обработанных индексов
    
    # Вычисляем расстояния от каждого предложения до запроса
    query_distances = cdist(embeddings, query_emb, metric='euclidean').flatten()
    
    if verbose:
        print("\n" + "="*80)
        print(f"{'='*80}")
        print("ЭТАП 1: РАСЧЕТЫ И АНАЛИЗ")
        print("="*80)
        print(f"\nРасстояния от предложений до запроса:")
        for idx in range(n):
            print(f"  [{idx}] расстояние = {query_distances[idx]:.4f}")
        print(f"\nМатрица расстояний между предложениями (первые 10x10):")
        print("  " + " ".join([f"{i:>6}" for i in range(min(10, n))]))
        for i in range(min(10, n)):
            row = " ".join([f"{distance_matrix[i, j]:>6.3f}" for j in range(min(10, n))])
            print(f"  {i:>2} {row}")
        print(f"\nПоиск групп похожих предложений (порог e1={e1}):")
        print("-"*80)
    
    for i in range(n):
        if i in processed:
            continue
        
        # Находим все предложения, похожие на i (включая само i)
        similar_indices = [i]
        for j in range(i + 1, n):
            if j not in processed and distance_matrix[i, j] < e1:
                similar_indices.append(j)
        
        if len(similar_indices) == 1:
            # Нет похожих предложений, оставляем как есть
            kept_indices.append(i)
            processed.add(i)
            if verbose:
                print(f"  Предложение [{i}]: нет похожих (расстояние до запроса: {query_distances[i]:.4f})")
        else:
            # Есть группа похожих предложений
            # Выбираем представителя - предложение с минимальным расстоянием до запроса
            # Если несколько равны, выбираем первое по порядку (min индекс)
            representative_idx = min(similar_indices, key=lambda idx: (query_distances[idx], idx))
            
            kept_indices.append(representative_idx)
            processed.add(representative_idx)
            
            # Помечаем остальные как удаленные
            for idx in similar_indices:
                if idx != representative_idx:
                    removed_mapping[idx] = representative_idx
                    processed.add(idx)
            
            if verbose:
                print(f"  Группа похожих предложений: {similar_indices}")
                for idx in similar_indices:
                    dist_to_query = query_distances[idx]
                    dist_to_rep = distance_matrix[idx, representative_idx] if idx != representative_idx else 0
                    status = "→ ПРЕДСТАВИТЕЛЬ" if idx == representative_idx else f"→ дубликат [{representative_idx}]"
                    print(f"    [{idx}] расстояние до запроса: {dist_to_query:.4f}, "
                          f"расстояние до представителя: {dist_to_rep:.4f} {status}")
            
            logger.debug(
                f"Группа похожих предложений: {similar_indices}, "
                f"представитель: [{representative_idx}]"
            )
    
    if verbose:
        print("-"*80)
        print(f"Итого на этапе 1: удалено {len(removed_mapping)} похожих предложений, "
              f"осталось {len(kept_indices)} предложений")
        print("="*80 + "\n")
    
    removed_count = len(removed_mapping)
    logger.info(
        f"Этап 1 завершен: удалено {removed_count} похожих предложений, "
        f"осталось {len(kept_indices)} предложений"
    )
    
    return kept_indices, removed_mapping


def filter_by_query(
    sentences: List[str],
    embeddings: np.ndarray,
    kept_indices: List[int],
    e2: int,
    query_emb: np.ndarray,
    verbose: bool = False
) -> Tuple[List[int], Set[int]]:
    """
    Фильтрует предложения по релевантности запросу на этапе 2.
    
    Алгоритм:
    1. Для каждого оставшегося после этапа 1 предложения вычисляет евклидово расстояние до запроса
    2. Оставляет только e2 предложений с наименьшим расстоянием до запроса
    3. Если после этапа 1 осталось меньше или равно e2 предложений, все остаются
    
    Args:
        sentences: Список всех предложений
        embeddings: Массив embeddings предложений формы (n_sentences, embedding_dim)
        kept_indices: Индексы предложений, оставшихся после этапа 1
        e2: Количество предложений, которые должны остаться в финальном контексте
        query_emb: Embedding запроса формы (1, embedding_dim) или (embedding_dim,)
        
    Returns:
        Кортеж (final_indices, removed_indices), где:
        - final_indices: Список индексов предложений в финальном контексте
        - removed_indices: Множество индексов предложений, удаленных на этапе 2
    """
    if len(kept_indices) == 0:
        return [], set()
    
    # Нормализуем query_emb к форме (1, embedding_dim)
    if query_emb.ndim == 1:
        query_emb = query_emb.reshape(1, -1)
    
    logger.info(f"Этап 2: Фильтрация по релевантности запросу (оставить {e2} предложений)")
    logger.debug(f"Обработка {len(kept_indices)} предложений после этапа 1")
    
    # Вычисляем расстояния от оставшихся предложений до запроса
    kept_embeddings = embeddings[kept_indices]
    distances = cdist(kept_embeddings, query_emb, metric='euclidean').flatten()
    
    if verbose:
        print("\n" + "="*80)
        print("ЭТАП 2: РАСЧЕТЫ И АНАЛИЗ")
        print("="*80)
        print(f"\nПосле этапа 1 осталось {len(kept_indices)} предложений")
        print(f"Требуется оставить {e2} предложений с наименьшим расстоянием до запроса")
        print(f"\nРасстояния от оставшихся предложений до запроса:")
        for idx, dist in zip(kept_indices, distances):
            print(f"  [{idx}] расстояние = {dist:.4f}")
    
    # Если после этапа 1 осталось меньше или равно e2 предложений, все остаются
    if len(kept_indices) <= e2:
        logger.info(
            f"После этапа 1 осталось {len(kept_indices)} предложений, "
            f"требуется оставить {e2}. Все предложения остаются."
        )
        if verbose:
            print(f"\nВсе {len(kept_indices)} предложений остаются (меньше или равно {e2})")
            print("="*80 + "\n")
        return kept_indices, set()
    
    # Создаем список пар (индекс, расстояние) и сортируем по расстоянию
    indexed_distances = [(idx, dist) for idx, dist in zip(kept_indices, distances)]
    indexed_distances.sort(key=lambda x: (x[1], x[0]))  # Сортируем по расстоянию, затем по индексу
    
    if verbose:
        print(f"\nСортировка предложений по расстоянию до запроса:")
        for rank, (idx, dist) in enumerate(indexed_distances, 1):
            status = "✓ ОСТАЕТСЯ" if rank <= e2 else "✗ УДАЛЯЕТСЯ"
            print(f"  {rank:>2}. [{idx}] расстояние = {dist:.4f} {status}")
    
    # Оставляем e2 предложений с наименьшим расстоянием
    final_indices = [idx for idx, _ in indexed_distances[:e2]]
    removed_indices = {idx for idx, _ in indexed_distances[e2:]}
    
    logger.info(
        f"Этап 2 завершен: удалено {len(removed_indices)} предложений с наибольшим расстоянием, "
        f"осталось {len(final_indices)} предложений"
    )
    
    if verbose:
        print(f"\nИтого на этапе 2: удалено {len(removed_indices)} предложений, "
              f"осталось {len(final_indices)} предложений")
        print("="*80 + "\n")
    
    # Логируем удаленные предложения
    for idx in removed_indices:
        distance = distances[kept_indices.index(idx)]
        logger.debug(
            f"Предложение [{idx}] удалено: расстояние до запроса = {distance:.4f}"
        )
    
    return final_indices, removed_indices

