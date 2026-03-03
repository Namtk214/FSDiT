"""Pickle-based checkpointing, compatible with local and GCS paths."""

import concurrent.futures
import pickle
import time
import os
import shutil


class Checkpoint:
    def __init__(self, filename, parallel=True):
        self._filename = filename
        self._values = {}
        self._parallel = parallel
        if parallel:
            self._worker = concurrent.futures.ThreadPoolExecutor(1, 'ckpt')
            self._promise = None

    def __setattr__(self, name, value):
        if name.startswith('_') or name in ('save', 'load', 'exists'):
            return super().__setattr__(name, value)
        self._values[name] = value

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return self._values.get(name)

    def set_model(self, model):
        for key in model.__dict__:
            data = getattr(model, key)
            if hasattr(data, 'save') or key == 'config':
                self._values[key] = data

    def save(self, filename=None):
        filename = filename or self._filename
        print(f'Saving checkpoint: {filename}')
        if self._parallel:
            if hasattr(self, '_promise') and self._promise:
                self._promise.result()
            self._promise = self._worker.submit(self._save, filename)
        else:
            self._save(filename)

    def _save(self, filename):
        data = {k: (v.save() if k != 'config' else v) for k, v in self._values.items()}
        data['_timestamp'] = time.time()
        content = pickle.dumps(data)
        if 'gs://' in filename:
            import tensorflow as tf
            tf.io.gfile.makedirs(os.path.dirname(filename))
            with tf.io.gfile.GFile(filename, 'wb') as f:
                f.write(content)
        else:
            dirname = os.path.dirname(filename)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            tmp = filename + '.tmp'
            with open(tmp, 'wb') as f:
                f.write(content)
            os.replace(tmp, filename)
        print('Checkpoint saved.')

    def load_as_dict(self, filename=None):
        filename = filename or self._filename
        if 'gs://' in filename:
            import tensorflow as tf
            with tf.io.gfile.GFile(filename, 'rb') as f:
                data = pickle.loads(f.read())
        else:
            with open(filename, 'rb') as f:
                data = pickle.loads(f.read())
        age = time.time() - data.get('_timestamp', time.time())
        print(f'Loaded checkpoint ({age:.0f}s ago)')
        return data

    def load_model(self, model, filename=None):
        d = self.load_as_dict(filename)
        replace = {}
        for key in model.__dict__:
            if key in d and key != 'config':
                replace[key] = getattr(model, key).load(d[key])
        return model.replace(**replace)
