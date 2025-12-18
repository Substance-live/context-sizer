"""
Тесты для модуля utils.
"""
import pytest
import tempfile
import os
from pathlib import Path
from utils import split_into_sentences, load_config, read_file


class TestSplitIntoSentences:
    """Тесты для функции разбиения на предложения."""
    
    def test_simple_sentences(self):
        """Тест простых предложений."""
        text = "Первое предложение. Второе предложение. Третье предложение."
        result = split_into_sentences(text)
        assert len(result) == 3
        assert result[0] == "Первое предложение."
        assert result[1] == "Второе предложение."
        assert result[2] == "Третье предложение."
    
    def test_question_and_exclamation(self):
        """Тест с вопросительными и восклицательными знаками."""
        text = "Как дела? Отлично! Все хорошо."
        result = split_into_sentences(text)
        assert len(result) == 3
    
    def test_english_sentences(self):
        """Тест английских предложений."""
        text = "This is the first sentence. This is the second sentence. And this is the third."
        result = split_into_sentences(text)
        assert len(result) == 3
    
    def test_mixed_languages(self):
        """Тест смешанных языков."""
        text = "Это русское предложение. This is an English sentence. Еще одно русское."
        result = split_into_sentences(text)
        assert len(result) == 3
    
    def test_multiple_spaces(self):
        """Тест с множественными пробелами."""
        text = "Первое.    Второе.   Третье."
        result = split_into_sentences(text)
        assert len(result) == 3
    
    def test_newlines(self):
        """Тест с переносами строк."""
        text = "Первое предложение.\nВторое предложение.\nТретье предложение."
        result = split_into_sentences(text)
        assert len(result) == 3
    
    def test_empty_text(self):
        """Тест пустого текста."""
        text = ""
        result = split_into_sentences(text)
        assert len(result) == 0
    
    def test_single_sentence(self):
        """Тест одного предложения."""
        text = "Одно предложение."
        result = split_into_sentences(text)
        assert len(result) == 1
        assert result[0] == "Одно предложение."


class TestLoadConfig:
    """Тесты для функции загрузки конфигурации."""
    
    def test_valid_config(self):
        """Тест валидной конфигурации."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("E1=0.75\n")
            f.write("E2=15\n")
            f.write("EMB_URL=http://localhost:11434\n")
            f.write("EMB_MODEL=test-model\n")
            f.write("BATCH_SIZE=64\n")
            env_path = f.name
        
        # Сохраняем оригинальный .env если существует
        original_env = Path('.env')
        original_content = None
        had_original = original_env.exists()
        if had_original:
            original_content = original_env.read_text()
            original_env.unlink()
        
        try:
            import shutil
            shutil.copy(env_path, '.env')
            
            # Очищаем кэш переменных окружения для корректной загрузки
            import os
            for key in ['E1', 'E2', 'EMB_URL', 'EMB_MODEL', 'BATCH_SIZE']:
                os.environ.pop(key, None)
            
            config = load_config()
            assert config['e1'] == 0.75
            assert config['e2'] == 15
            assert isinstance(config['e2'], int)
            assert config['emb_url'] == 'http://localhost:11434'
            assert config['emb_model'] == 'test-model'
            assert config['batch_size'] == 64
        finally:
            # Восстанавливаем оригинальный .env если был
            if Path('.env').exists():
                Path('.env').unlink()
            if had_original and original_content:
                original_env.write_text(original_content)
            if Path(env_path).exists():
                Path(env_path).unlink()
    
    def test_missing_e1(self):
        """Тест отсутствия E1."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("E2=15\n")
            f.write("EMB_MODEL=test-model\n")
            env_path = f.name
        
        # Сохраняем оригинальный .env если существует
        original_env = Path('.env')
        original_content = None
        had_original = original_env.exists()
        if had_original:
            original_content = original_env.read_text()
            original_env.unlink()
        
        try:
            import shutil
            import os
            shutil.copy(env_path, '.env')
            
            # Очищаем кэш переменных окружения
            for key in ['E1', 'E2', 'EMB_URL', 'EMB_MODEL', 'BATCH_SIZE']:
                os.environ.pop(key, None)
            
            with pytest.raises(ValueError, match="E1"):
                load_config()
        finally:
            # Восстанавливаем оригинальный .env если был
            if Path('.env').exists():
                Path('.env').unlink()
            if had_original and original_content:
                original_env.write_text(original_content)
            if Path(env_path).exists():
                Path(env_path).unlink()
    
    def test_invalid_e1(self):
        """Тест некорректного значения E1."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("E1=invalid\n")
            f.write("E2=15\n")
            f.write("EMB_MODEL=test-model\n")
            env_path = f.name
        
        # Сохраняем оригинальный .env если существует
        original_env = Path('.env')
        original_content = None
        had_original = original_env.exists()
        if had_original:
            original_content = original_env.read_text()
            original_env.unlink()
        
        try:
            import shutil
            import os
            shutil.copy(env_path, '.env')
            
            # Очищаем кэш переменных окружения
            for key in ['E1', 'E2', 'EMB_URL', 'EMB_MODEL', 'BATCH_SIZE']:
                os.environ.pop(key, None)
            
            with pytest.raises(ValueError):
                load_config()
        finally:
            # Восстанавливаем оригинальный .env если был
            if Path('.env').exists():
                Path('.env').unlink()
            if had_original and original_content:
                original_env.write_text(original_content)
            if Path(env_path).exists():
                Path(env_path).unlink()


class TestReadFile:
    """Тесты для функции чтения файла."""
    
    def test_read_existing_file(self):
        """Тест чтения существующего файла."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            content = "Тестовое содержимое файла."
            f.write(content)
            filepath = f.name
        
        try:
            result = read_file(filepath)
            assert result == content
        finally:
            if Path(filepath).exists():
                Path(filepath).unlink()
    
    def test_read_nonexistent_file(self):
        """Тест чтения несуществующего файла."""
        with pytest.raises(FileNotFoundError):
            read_file("nonexistent_file.txt")

