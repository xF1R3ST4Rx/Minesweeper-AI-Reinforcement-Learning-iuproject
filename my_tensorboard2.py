import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

class ModifiedTensorBoard(TensorBoard):
    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(log_dir)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        with self.writer.as_default():
            for key, value in logs.items():
                tf.summary.scalar(key, value, step=self.step)
        self.writer.flush()
        self.step += 1

    def update_stats(self, **stats):
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step=self.step)
        self.writer.flush()
        self.step += 1