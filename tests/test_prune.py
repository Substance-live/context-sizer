"""
Тесты для модуля prune.
"""
import pytest
import numpy as np
from prune import remove_similar, filter_by_query


class TestRemoveSimilar:
    """Тесты для функции удаления похожих предложений."""
    
    def test_no_similar_sentences(self):
        """Тест без похожих предложений."""
        sentences = ["Первое предложение.", "Второе предложение.", "Третье предложение."]
        # Создаем embeddings, которые далеко друг от друга
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        query_emb = np.array([0.5, 0.5, 0.5])
        e1 = 0.5  # Маленький порог
        
        kept_indices, removed_mapping = remove_similar(sentences, embeddings, e1, query_emb)
        
        # Все предложения должны остаться
        assert len(kept_indices) == 3
        assert len(removed_mapping) == 0
    
    def test_all_similar_sentences(self):
        """Тест, когда все предложения похожи."""
        sentences = ["Похожее предложение.", "Похожее предложение.", "Похожее предложение."]
        # Создаем одинаковые embeddings
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ])
        query_emb = np.array([0.5, 0.0, 0.0])  # Ближе к первому
        
        e1 = 0.1  # Маленький порог, но достаточный для одинаковых векторов
        
        kept_indices, removed_mapping = remove_similar(sentences, embeddings, e1, query_emb)
        
        # Должно остаться только одно предложение (первое, так как оно ближе к запросу)
        assert len(kept_indices) == 1
        assert kept_indices[0] == 0
        assert len(removed_mapping) == 2
        assert 1 in removed_mapping
        assert 2 in removed_mapping
    
    def test_group_of_similar(self):
        """Тест группы похожих предложений."""
        sentences = [
            "Первое предложение.",
            "Похожее на первое.",
            "Совсем другое.",
            "Еще одно похожее."
        ]
        # Первое, второе и четвертое похожи, третье отличается
        embeddings = np.array([
            [1.0, 0.0, 0.0],  # 0
            [1.0, 0.1, 0.0],  # 1 - похоже на 0
            [0.0, 0.0, 1.0],  # 2 - отличается
            [1.0, 0.05, 0.0]  # 3 - похоже на 0 и 1
        ])
        query_emb = np.array([0.9, 0.0, 0.0])  # Ближе к группе 0,1,3
        
        e1 = 0.2  # Порог для определения похожих
        
        kept_indices, removed_mapping = remove_similar(sentences, embeddings, e1, query_emb)
        
        # Должны остаться: одно из группы (0,1,3) и предложение 2
        assert len(kept_indices) == 2
        # Предложение 0 должно быть представителем (ближе всего к запросу)
        assert 0 in kept_indices
        assert 2 in kept_indices
        # Предложения 1 и 3 должны быть удалены
        assert 1 in removed_mapping
        assert 3 in removed_mapping
    
    def test_empty_sentences(self):
        """Тест с пустым списком предложений."""
        sentences = []
        embeddings = np.array([]).reshape(0, 3)
        query_emb = np.array([1.0, 0.0, 0.0])
        e1 = 0.5
        
        kept_indices, removed_mapping = remove_similar(sentences, embeddings, e1, query_emb)
        
        assert len(kept_indices) == 0
        assert len(removed_mapping) == 0


class TestFilterByQuery:
    """Тесты для функции фильтрации по релевантности запросу."""
    
    def test_keep_all_when_less_than_e2(self):
        """Тест, когда после этапа 1 осталось меньше предложений, чем E2."""
        sentences = ["Первое.", "Второе."]
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.9, 0.0, 0.0]
        ])
        kept_indices = [0, 1]
        query_emb = np.array([1.0, 0.0, 0.0])
        e2 = 5  # Требуется оставить 5, но есть только 2
        
        final_indices, removed_indices = filter_by_query(
            sentences, embeddings, kept_indices, e2, query_emb
        )
        
        # Все предложения должны остаться
        assert len(final_indices) == 2
        assert len(removed_indices) == 0
        assert 0 in final_indices
        assert 1 in final_indices
    
    def test_keep_exact_e2(self):
        """Тест, когда нужно оставить ровно E2 предложений."""
        sentences = ["Первое.", "Второе.", "Третье.", "Четвертое.", "Пятое."]
        embeddings = np.array([
            [1.0, 0.0, 0.0],  # 0 - ближе всего к запросу
            [0.9, 0.0, 0.0],  # 1 - второе по близости
            [0.8, 0.0, 0.0],  # 2 - третье по близости
            [0.0, 0.0, 1.0],  # 3 - далеко
            [0.0, 1.0, 0.0]   # 4 - далеко
        ])
        kept_indices = [0, 1, 2, 3, 4]
        query_emb = np.array([1.0, 0.0, 0.0])
        e2 = 3  # Оставить 3 предложения
        
        final_indices, removed_indices = filter_by_query(
            sentences, embeddings, kept_indices, e2, query_emb
        )
        
        # Должны остаться 3 предложения с наименьшим расстоянием (0, 1, 2)
        assert len(final_indices) == 3
        assert len(removed_indices) == 2
        assert 0 in final_indices
        assert 1 in final_indices
        assert 2 in final_indices
        assert 3 in removed_indices
        assert 4 in removed_indices
    
    def test_keep_top_by_distance(self):
        """Тест выбора предложений с наименьшим расстоянием."""
        sentences = ["Близкое1.", "Близкое2.", "Далёкое1.", "Далёкое2."]
        embeddings = np.array([
            [1.0, 0.0, 0.0],  # 0 - расстояние ~0
            [0.95, 0.0, 0.0], # 1 - расстояние ~0.05
            [0.0, 0.0, 1.0],  # 2 - расстояние ~1.41
            [0.0, 1.0, 0.0]   # 3 - расстояние ~1.41
        ])
        kept_indices = [0, 1, 2, 3]
        query_emb = np.array([1.0, 0.0, 0.0])
        e2 = 2  # Оставить 2 предложения
        
        final_indices, removed_indices = filter_by_query(
            sentences, embeddings, kept_indices, e2, query_emb
        )
        
        # Должны остаться 0 и 1 (ближайшие к запросу)
        assert len(final_indices) == 2
        assert len(removed_indices) == 2
        assert 0 in final_indices
        assert 1 in final_indices
        assert 2 in removed_indices
        assert 3 in removed_indices
    
    def test_empty_kept_indices(self):
        """Тест с пустым списком kept_indices."""
        sentences = ["Первое.", "Второе."]
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
        kept_indices = []
        query_emb = np.array([1.0, 0.0])
        e2 = 5
        
        final_indices, removed_indices = filter_by_query(
            sentences, embeddings, kept_indices, e2, query_emb
        )
        
        assert len(final_indices) == 0
        assert len(removed_indices) == 0
    
    def test_keep_equal_to_kept(self):
        """Тест, когда E2 равно количеству оставшихся предложений."""
        sentences = ["Первое.", "Второе.", "Третье."]
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [0.8, 0.0, 0.0]
        ])
        kept_indices = [0, 1, 2]
        query_emb = np.array([1.0, 0.0, 0.0])
        e2 = 3  # Оставить 3, есть 3
        
        final_indices, removed_indices = filter_by_query(
            sentences, embeddings, kept_indices, e2, query_emb
        )
        
        # Все предложения должны остаться
        assert len(final_indices) == 3
        assert len(removed_indices) == 0

