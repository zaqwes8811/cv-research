import hailort as hailo
import numpy as np
from PIL import Image

# Загружаем HEF файл
hef_path = "yolov8n.hef" # Укажите ваш путь
vdevice = hailo.VDevice()
network_group = vdevice.configure(hef_path)[0]
network_group.activate()

# Получаем информацию о входном и выходном слоях
input_vstream_info = network_group.input_vstreams_info()[0]
output_vstream_info = network_group.output_vstreams_info()[0]

print(f"Форма входа (ожидаемая): {input_vstream_info.shape}")
print(f"Форма выхода: {output_vstream_info.shape}")
print(f"Тип данных выхода: {output_vstream_info.format.type}")

# Подготовка данных (пример: создаем случайное изображение 640x640)
input_data = np.random.randint(0, 255, size=input_vstream_info.shape, dtype=np.uint8)

# Создаем буферы для данных
input_data = [input_data]
output_data = [np.zeros(output_vstream_info.shape, dtype=np.float32)] # Тип данных может быть float32

# Запуск инференса
network_group.run(input_data, output_data)

# Результат в output_data[0]
print(f"Реальная форма выходных данных: {output_data[0].shape}")

network_group.deactivate()