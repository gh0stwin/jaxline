"""Checkpointers for Jaxline experiments."""

import abc
import os
import pickle
from typing import Any, Optional

from absl import logging
import jax
from ml_collections import config_dict


class Checkpointer(abc.ABC):
  """An interface for checkpointer objects."""

  @abc.abstractmethod
  def save(self, ckpt_series: str, state: Any) -> None:
    """Saves the checkpoint."""

  @abc.abstractmethod
  def restore(self, ckpt_series: str ) -> Any:
    """Restores the checkpoint."""

  @abc.abstractmethod
  def restore_path(self, ckpt_series: str) -> Optional[str]:
    """Returns the restore path for the checkpoint, or None."""

  @abc.abstractmethod
  def can_be_restored(self, ckpt_series: str) -> bool:
    """Returns whether or not a given checkpoint series can be restored."""

  def wait_for_checkpointing_to_finish(self) -> None:
    """Waits for any async checkpointing to complete."""
    return

  def _override_or_insert(self, current_state, snapshot):
    """Update the current state based on a snapshot."""
    for sk, sv in snapshot.items():
      # Duck-typing for "is this a Jaxline Experiment class?".
      if (sk in current_state
          and hasattr(current_state[sk], "CHECKPOINT_ATTRS")
          and hasattr(current_state[sk], "NON_BROADCAST_CHECKPOINT_ATTRS")):
        for kk in sv.CHECKPOINT_ATTRS:
          setattr(current_state[sk], kk, getattr(sv, kk))
        for kk in sv.NON_BROADCAST_CHECKPOINT_ATTRS:
          setattr(
              current_state[sk], kk,
              jax.tree_map(copy.copy, getattr(sv, kk)))
      else:
        current_state[sk] = sv

class NoneCheckpointer(Checkpointer):
  def __init__(self, *args, **kwargs) -> None:
    super().__init__()

  def save(self, ckpt_series: str, state: Any) -> None:
    return

  def can_be_restored(self, ckpt_series: str) -> bool:
    return False

  def restore(self, ckpt_series: str) -> Any:
    raise FileNotFoundError(f"Cannot restore checkpoint for '{ckpt_series}' series")
    
  def restore_path(self, ckpt_series: str) -> Optional[str]:
    return ""


class LocalCheckpointer(Checkpointer):
  """A Checkpointer reliant on an in-memory global dictionary."""

  def __init__(self, checkpoint_dir: str):
    self._ckpt_dir = checkpoint_dir
    self._suffix = ".pkl"

  def save(self, ckpt_series: str, state: config_dict.ConfigDict) -> None:
    """Saves the checkpoint."""
    file_path = self._ckpt_series_file(ckpt_series)
    tmp_path = self._ckpt_series_file(ckpt_series + "_tmp")
    old_path = self._ckpt_series_file(ckpt_series + "_old")

    # Creates a rolling ckpt.
    with open(tmp_path, "wb") as checkpoint_file:
      pickle.dump(state, checkpoint_file, protocol=2)

    try:
      os.rename(file_path, old_path)
      remove_old = True
    except FileNotFoundError:
      remove_old = False  # No previous checkpoint to remove

    if remove_old:
      os.remove(old_path)

    os.rename(tmp_path, file_path)
    logging.info("Saved new checkpoint for '%s' series.", ckpt_series)

  def can_be_restored(self, ckpt_series: str) -> bool:
    """Returns whether or not a given checkpoint series can be restored."""
    return os.path.exists(self._ckpt_series_file(ckpt_series))

  def restore(self, ckpt_series: str) -> None:
    """Restores the checkpoint."""
    ckpt_path = self._ckpt_series_file(ckpt_series)
    ckpt_data = None

    try:
      with open(ckpt_path, "rb") as checkpoint_file:
        ckpt_data = pickle.load(checkpoint_file)
        logging.info("Returned checkpoint for '%s' series.", ckpt_series)
    except FileNotFoundError:
      logging.info("No existing checkpoint found at %s", ckpt_path)

    return ckpt_data

  def restore_path(self, ckpt_series: str) -> Optional[str]:
    """Returns the restore path for the checkpoint, or None."""
    if not self.can_be_restored(ckpt_series):
      return None

    return self._ckpt_series_file(ckpt_series)

  def _ckpt_series_file(self, ckpt_series: str) -> str:
    return os.path.join(self._ckpt_dir, ckpt_series + self._suffix)


class NeptuneAiCheckpointer(Checkpointer):
  import neptune.new as neptune
  
  def __init__(
    self,
    checkpoint_dir: str,
    base_dir: str,
    project: Optional[str] = None,
    name: Optional[str] = None,
    api_token: Optional[str] = None,
  ) -> None:
    super().__init__()
    self._tmp_local_dir = checkpoint_dir
    self._base_dir = base_dir
    self._last_model_path = f"{self._base_dir}/last_model_path"
    self._suffix = ".pkl"
    self._i = 0

    try:
      self._run = neptune.get_last_run()
    except neptune.NeptuneUninitializedException:
      self._run = None

    if self._run:
      return

    self._run = neptune.init(
      project=project,
      name=name,
      api_token=api_token,
      source_files=[],
      capture_hardware_metrics=False,
      capture_stdout=False,
      capture_stderr=False,
    )

  def can_be_restored(self, ckpt_series: str) -> bool:
    return self._run.exists(f"{self._base_dir}/{ckpt_series}__last_path")

  def save(self, ckpt_series: str, state: config_dict.ConfigDict) -> None:
    ckpt_file = os.path.join(self._tmp_local_dir, f"checkpoint{self._suffix}")

    with open(ckpt_file, "wb") as checkpoint_file:
      pickle.dump(state, checkpoint_file, protocol=2)

    path = f"{self._base_dir}/{ckpt_series}__{self._i}"
    self._run[path].upload(ckpt_file)
    self._run[f"{self._base_dir}/{ckpt_series}__last_path"] = path
    self._i += 1

  def restore(self, ckpt_series: str) -> config_dict.ConfigDict:
    checkpoint_path = self.restore_path(ckpt_series)
    self._run[checkpoint_path].download(self._tmp_local_dir)
    checkpoint_path = os.path.join(
      self._tmp_local_dir, checkpoint_path.rsplit("/", 1)[-1] + self._suffix
    )

    with open(checkpoint_path, "rb") as checkpoint_file:
      checkpoint = pickle.load(checkpoint_file)

    return checkpoint

  def restore_path(self, ckpt_series: str) -> Optional[str]:
    return self._run[f"{self._base_dir}/{ckpt_series}__last_path"].fetch()
