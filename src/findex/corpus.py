import logging
from collections.abc import Iterator
from pathlib import Path

log = logging.getLogger(__name__)


def iter_documents(root: Path) -> Iterator[tuple[int, str, str]]:
    # нумерація документів із 1
    doc_id = 1

    # Знайти всі файли з кінцівкою .txt у папці та її підпапках
    for file_path in sorted(root.rglob("*.txt")):
        try:
            # Прочитати текст із цього файлу
            text = file_path.read_text(encoding="utf-8", errors="replace")
            if "\ufffd" in text:
                log.warning("replacement characters in UTF-8 document %s", file_path)

            # Віддати цей документ назовні і зупинитися на паузу (yield)
            yield (doc_id, str(file_path), text)

            # Збільшити номер для наступного документа
            doc_id += 1
        except OSError as e:
            log.warning("cannot read %s: %s", file_path, e)
            # OSError, ValueError можливо
