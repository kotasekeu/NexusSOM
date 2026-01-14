import tensorflow as tf
devices = tf.config.list_physical_devices()
print(f"Dostupná zařízení: {devices}")

gpu = tf.config.list_physical_devices('GPU')
if len(gpu) > 0:
    print("STATUS: GPU akcelerace (Metal) je aktivní.")
else:
    print("STATUS: GPU nenalezeno. Trénování bude pomalé (pouze CPU).")