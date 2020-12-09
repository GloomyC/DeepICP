import tensorflow as tf
def init_tf():
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(gpu_devices ) > 0
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
    