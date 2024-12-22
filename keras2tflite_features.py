import sys
import tensorflow as tf
from utils.dataloaders import LoadImages
from utils.general import check_dataset, check_yaml

imgsz=(288,512)
out_name='features_out.tflite'
ncalib=500

data = sys.argv[2]
# Загружаем модель через Keras API
model = tf.keras.models.load_model(sys.argv[1])

# Теперь можно работать с моделью, например, получить слои
for layer in model.layers:
    print(layer.name)

tfc3_7_layer = model.get_layer('tfc3_7')

# Создаем модель с двумя выходами: основным и для слоя tfc3_7
inputs = model.input
outputs = [model.output, tfc3_7_layer.output]
model_with_extra_output = tf.keras.models.Model(inputs=inputs, outputs=outputs)

# Конвертируем модель в TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model_with_extra_output)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter.target_spec.supported_types = [tf.float16]
converter.optimizations = [tf.lite.Optimize.DEFAULT]

from models.tf import representative_dataset_gen
dataset = LoadImages(check_dataset(check_yaml(data))["train"], img_size=imgsz, auto=False)
converter.representative_dataset = lambda: representative_dataset_gen(dataset, ncalib=ncalib)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.target_spec.supported_types = []
converter.inference_input_type = tf.uint8  # or tf.int8
converter.inference_output_type = tf.uint8  # or tf.int8
converter.experimental_new_quantizer = True
converter._experimental_disable_per_channel = True

tflite_model = converter.convert()
# Сохраняем TFLite модель
with open(out_name, 'wb') as f:
    f.write(tflite_model)
