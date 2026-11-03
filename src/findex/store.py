import pickle
from pathlib import Path

from findex.index import InvertedIndex


def save_index(index: InvertedIndex, path: Path | str) -> None:
    """Серіалізує та зберігає об'єкт InvertedIndex у бінарний файл."""
    file_path = Path(path)
    # Створюємо директорію, якщо вона ще не існує
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_index(path: Path | str) -> InvertedIndex:
    """Завантажує та десеріалізує InvertedIndex з бінарного файлу."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Файл індексу не знайдено за шляхом: {file_path}")

    with open(file_path, "rb") as f:
        index = pickle.load(f)

    if not isinstance(index, InvertedIndex):
        raise TypeError("Завантажений об'єкт не є екземпляром InvertedIndex")

    return index
