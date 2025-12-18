#!/usr/bin/env python3
"""
Главный модуль приложения для уменьшения контекста.
"""
import sys
import argparse
import logging

from utils import (
    split_into_sentences,
    load_config,
    read_file,
    print_colored_output,
    print_statistics,
    format_reduced_context
)
from embeddings import EmbeddingClient
from prune import remove_similar, filter_by_query

logger = logging.getLogger(__name__)


def main():
    """Главная функция приложения."""
    parser = argparse.ArgumentParser(
        description='Уменьшение контекста для запросов нейросети путем удаления дубликатов и нерелевантных предложений'
    )
    parser.add_argument(
        'context_file',
        type=str,
        help='Путь к файлу с контекстом (context.txt)'
    )
    parser.add_argument(
        'query_file',
        type=str,
        help='Путь к файлу с запросом (query.txt)'
    )
    parser.add_argument(
        '--keep-original-if-empty',
        action='store_true',
        help='Сохранить оригинальный контекст, если результат фильтрации пуст'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Путь к файлу для сохранения уменьшенного контекста (по умолчанию stdout)'
    )
    parser.add_argument(
        '--no-color',
        action='store_true',
        help='Отключить цветной вывод в терминал'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Уровень логирования'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Подробный вывод расчетов для этапов 1 и 2'
    )
    
    args = parser.parse_args()
    
    # Настройка уровня логирования
    log_level = logging.DEBUG if args.verbose else getattr(logging, args.log_level)
    logging.getLogger().setLevel(log_level)
    
    try:
        # Загрузка конфигурации
        logger.info("Загрузка конфигурации...")
        config = load_config()
        
        # Чтение входных файлов
        logger.info(f"Чтение файла контекста: {args.context_file}")
        context_text = read_file(args.context_file)
        
        logger.info(f"Чтение файла запроса: {args.query_file}")
        query_text = read_file(args.query_file).strip()
        
        if not query_text:
            raise ValueError("Файл запроса пуст")
        
        # Разбиение контекста на предложения
        logger.info("Разбиение контекста на предложения...")
        sentences = split_into_sentences(context_text)
        
        if not sentences:
            logger.warning("Контекст не содержит предложений")
            if args.keep_original_if_empty:
                print(context_text)
            return
        
        logger.info(f"Получено {len(sentences)} предложений")
        
        # Инициализация клиента для embeddings
        logger.info("Инициализация клиента для embeddings...")
        embedding_client = EmbeddingClient(
            base_url=config['emb_url'],
            model=config['emb_model'],
            batch_size=config['batch_size']
        )
        
        # Получение embedding для запроса
        logger.info("Получение embedding для запроса...")
        query_emb = embedding_client.embed([query_text])[0]
        
        # Получение embeddings для предложений
        logger.info("Получение embeddings для предложений...")
        sentence_embeddings = embedding_client.embed(sentences)
        
        # Этап 1: Удаление похожих предложений
        logger.info("Запуск этапа 1: удаление похожих предложений...")
        kept_after_stage1, removed_stage1_mapping = remove_similar(
            sentences=sentences,
            embeddings=sentence_embeddings,
            e1=config['e1'],
            query_emb=query_emb,
            verbose=args.verbose
        )
        
        # Этап 2: Фильтрация по релевантности запросу
        logger.info("Запуск этапа 2: фильтрация по релевантности...")
        final_indices, removed_stage2_indices = filter_by_query(
            sentences=sentences,
            embeddings=sentence_embeddings,
            kept_indices=kept_after_stage1,
            e2=config['e2'],
            query_emb=query_emb,
            verbose=args.verbose
        )
        
        # Проверка на пустой результат
        if not final_indices:
            logger.warning("Результат фильтрации пуст!")
            if args.keep_original_if_empty:
                logger.info("Сохраняется оригинальный контекст (--keep-original-if-empty)")
                reduced_context = context_text
            else:
                reduced_context = ""
                print("ВНИМАНИЕ: После фильтрации контекст пуст. Используйте --keep-original-if-empty для сохранения оригинала.")
        else:
            # Форматирование уменьшенного контекста
            reduced_context = format_reduced_context(sentences, final_indices)
        
        # Визуальный вывод в терминал (если не отключен)
        if not args.no_color:
            # Определяем множества для визуализации
            kept_set = set(final_indices)
            removed_stage1_set = set(removed_stage1_mapping.keys())
            removed_stage2_set = removed_stage2_indices
            
            print_colored_output(
                sentences=sentences,
                kept_indices=kept_set,
                removed_stage1=removed_stage1_mapping,
                removed_stage2=removed_stage2_set,
                query=query_text
            )
        
        # Вывод статистики
        print_statistics(
            total=len(sentences),
            removed_stage1_count=len(removed_stage1_mapping),
            removed_stage2_count=len(removed_stage2_indices),
            kept_count=len(final_indices)
        )
        
        # Вывод уменьшенного контекста
        if args.output:
            logger.info(f"Сохранение результата в файл: {args.output}")
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(reduced_context)
            print(f"Результат сохранен в {args.output}")
        else:
            # Вывод в stdout
            print("\n" + "="*80)
            print("Уменьшенный контекст:")
            print("="*80 + "\n")
            print(reduced_context)
        
        logger.info("Обработка завершена успешно")
        
    except FileNotFoundError as e:
        logger.error(f"Ошибка: файл не найден - {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Ошибка валидации: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

