import pandas as pd
import os

class InventoryManager:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.load_data()

    def load_data(self):
        # Проверяем, существует ли файл, чтобы программа не вылетала
        if os.path.exists(self.file_path):
            return pd.read_csv(self.file_path)
        else:
            # Если файла нет, создаем пустой DataFrame
            return pd.DataFrame(columns=['id', 'name', 'category', 'color', 'material', 'style'])

    def get_by_category(self, category):
        """Фильтрация вещей по категории (Top/Bottom)"""
        return self.data[self.data['category'] == category]

    def add_item(self, new_item_dict):
        """Добавление новой вещи в базу (сохранение в CSV)"""
        self.data = pd.concat([self.data, pd.DataFrame([new_item_dict])], ignore_index=True)
        self.data.to_csv(self.file_path, index=False)