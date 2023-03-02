"""Writers for Jaxline experiments."""

import abc
import os
from typing import Any, Mapping

from absl import logging
import clearml
import tensorflow as tf
import neptune.new as neptune
from neptune.new.integrations.python_logger import NeptuneHandler
from neptune.new.types import File
import numpy as np


class Writer(abc.ABC):
  """Logger interface."""
  def __init__(self, config, mode: str):
    self._config = config
    self._mode = mode

  @property
  def mode(self) -> str:
    return self._mode

  @mode.setter
  def mode(self, mode: str):
    self._mode = mode

  @abc.abstractmethod
  def write_scalars(self, global_step: int, scalars: Mapping[str, Any]):
    """Write scalars to logger."""

  @abc.abstractmethod
  def write_images(self, global_step: int, images: Mapping[str, Any]):
    """Write images to logger."""

  def _post_init(self):
    """Log initial items after logger initialization."""
    self._write_config()

  def _write_config(self):
    """Write config to logger. By default ignore it."""
    return


class ClearMlLogger(Writer):
  def __init__(self, config, mode):
    super().__init__(config, mode)
    task = clearml.Task.current_task()
    task.connect_configuration(name="config",
                              configuration=config.to_dict())

    self._writer = task.get_logger()

  def write_scalars(self, global_step: int, scalars: Mapping[str, Any]):
    global_step = int(global_step)
    self._writer.report_scalar(title="global_step",
                              series=self._mode,
                              value=global_step,
                              iteration=global_step)

    for k, v in scalars.items():
      self._writer.report_scalar(title=k,
                                series=self._mode,
                                value=float(v),
                                iteration=global_step)

  def write_images(self, global_step: int, images: Mapping[str, Any]):
    global_step = int(global_step)

    for k, v in images.items():
      if v.ndim == 2:
        self._writer.report_image(title=self.mode,
                                  series=k,
                                  iteration=global_step,
                                  image=v)
      elif v.ndim == 4:
        for i in range(v.shape[0]):
          self._writer.report_image(title=self.mode,
                                    series=f"{k}_{i}",
                                    iteration=global_step,
                                    image=v[i])


class NeptuneAiLogger(Writer):
  """Neptune AI logger."""

  def __init__(self, config, mode):
    super().__init__(config, mode)
    run = self._config.get("logger.kwargs.run", None)
    custom_run = self._config.get("logger.kwargs.custom_run_id", None)
    tags = self._config.get("logger.kwargs.tags", [])
    source_files = self._config.get("logger.kwargs.source_files", None)
    capture_stdout = self._config.get("logger.kwargs.capture_stdout", False)
    capture_stderr = self._config.get("logger.kwargs.capture_stderr", False)
    hardware_metrics = self._config.get("logger.kwargs.capture_hardware_mertics", False)
    self._writer = neptune.init_run(project=self._config.logger.kwargs.project,
                                name=self._config.logger.kwargs.name,
                                api_token=self._config.logger.kwargs.api_token,
                                with_id=run, tags=tags, source_files=source_files,
                                capture_hardware_metrics=hardware_metrics,
                                capture_stdout=capture_stdout,
                                capture_stderr=capture_stderr,
                                custom_run_id=custom_run)

    self._post_init()

  def write_scalars(self, global_step: int, scalars: Mapping[str, Any]):
    global_step = int(global_step)
    self._writer[self._artifact_tag("global_step")].log(global_step)
    for k, v in scalars.items():
      self._writer[self._artifact_tag(k)].log(float(v), step=global_step)

  def write_images(self, global_step: int, images: Mapping[str, Any]):
    global_step = int(global_step)
    for k, v in images.items():
      if v.ndim == 2:
        v = v[None, ..., None]
        self._writer[self._artifact_tag(k)].log(File.as_image(v))
      elif v.ndim == 4:
        for i in range(v.shape[0]):
          self._writer[self._artifact_tag(k)].log(File.as_image(v[i]))

  def _artifact_tag(self, scalar_key):
    return "{}/{}".format(self._mode, scalar_key)

  def _post_init(self):
    super()._post_init()
    self._config = self._config.unlock()
    self._config.log_id = self._writer["sys/id"].fetch()
    self._config = self._config.lock()

    if self._config.get("logger.kwargs.capture_stdout", False) is True:
      logging.get_absl_logger().addHandler(NeptuneHandler(run=self._writer))

  def _write_config(self):
    if self._config.logger.kwargs.get('log_config', True) is False:
      return

    self._writer['config'] = self._config.to_dict()

class TensorBoardLogger(Writer):
  """Writer to write experiment data to stdout."""

  def __init__(self, config, mode: str):
    """Initializes the writer."""
    super().__init__(config, mode)
    log_dir = os.path.join(config.checkpoint_dir, mode)
    self._writer = tf.summary.create_file_writer(log_dir)
    self._post_init()

  def write_scalars(self, global_step: int, scalars: Mapping[str, Any]):
    """Writes scalars to stdout."""
    global_step = int(global_step)
    with self._writer.as_default():
      for k, v in scalars.items():
        tf.summary.scalar(k, v, step=global_step)
    self._writer.flush()

  def write_images(self, global_step: int, images: Mapping[str, np.ndarray]):
    """Writes images to writers that support it."""
    global_step = int(global_step)
    with self._writer.as_default():
      for k, v in images.items():
        # Tensorboard only accepts [B, H, W, C] but we support [H, W] also.
        if v.ndim == 2:
          v = v[None, ..., None]
        tf.summary.image(k, v, step=global_step)
    self._writer.flush()