import pandas as pd
import os


class InventoryManager:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.load_data()

    @staticmethod
    def _required_columns():
        return ['id', 'name', 'category', 'color', 'material', 'style', 'image_path']

    def _normalize_data(self, frame):
        for col in self._required_columns():
            if col not in frame.columns:
                frame[col] = ""

        frame = frame[self._required_columns()].copy()
        for col in ['name', 'category', 'color', 'material', 'style', 'image_path']:
            frame[col] = frame[col].astype(str).str.strip()

        frame = frame[frame['name'] != ""]
        frame = frame.drop_duplicates(subset=['name', 'category', 'color', 'material', 'style'], keep='first')
        frame = frame.reset_index(drop=True)
        return frame

    def load_data(self):
        # Проверяем, существует ли файл, чтобы программа не вылетала
        if os.path.exists(self.file_path):
            return self._normalize_data(pd.read_csv(self.file_path))
        else:
            # Если файла нет, создаем пустой DataFrame
            return pd.DataFrame(columns=self._required_columns())

    def get_by_category(self, category):
        """Фильтрация вещей по категории (Top/Bottom)"""
        return self.data[self.data['category'] == category]

    def add_item(self, new_item_dict):
        """Добавление новой вещи в базу (сохранение в CSV)"""
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

        key_cols = ['name', 'category', 'color', 'material', 'style']
        normalized = {k: str(new_item_dict.get(k, "")).strip() for k in key_cols}
        if not normalized['name']:
            return False

        if not self.data.empty:
            existing = self.data.copy()
            for col in key_cols:
                existing[col] = existing[col].astype(str).str.strip()

            duplicate_mask = (existing[key_cols] == pd.Series(normalized)).all(axis=1)
            if duplicate_mask.any():
                return False

        if 'id' not in new_item_dict or pd.isna(new_item_dict['id']):
            new_item_dict['id'] = (self.data['id'].max() + 1) if not self.data.empty else 1
        self.data = pd.concat([self.data, pd.DataFrame([new_item_dict])], ignore_index=True)
        self.data = self._normalize_data(self.data)
        self.data.to_csv(self.file_path, index=False)
        return True

    def delete_item_by_id(self, item_id):
        """Удаление вещи по id (с сохранением в CSV)."""
        if self.data.empty:
            return False

        before = len(self.data)
        self.data = self.data[self.data['id'].astype(str) != str(item_id)].reset_index(drop=True)
        removed = len(self.data) < before
        if removed:
            self.data.to_csv(self.file_path, index=False)
        return removed