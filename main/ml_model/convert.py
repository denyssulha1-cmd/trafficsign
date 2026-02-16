import tensorflow as tf
import tf2onnx

model = tf.keras.models.load_model("sign_model.h5")
spec = (tf.TensorSpec((None, 128, 128, 3), tf.float32, name="input"),)
output_path = "sign_model.onnx"
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)
