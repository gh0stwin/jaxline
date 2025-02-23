# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A Deepmind-specific platform for running Experiments with Jaxline."""

from concurrent import futures

from absl import flags
from absl import logging

import chex
import jax

from jaxline import base_config
from jaxline import checkpointers
from jaxline import train
from jaxline import utils
from jaxline import writers
from ml_collections import config_dict, config_flags


# TODO(tomhennigan) Add support for ipdb and pudb.
_CONFIG = config_flags.DEFINE_config_file(
    name="config",
    help_string="Training configuration file.",
)
# This flag is expected to be used only internally by jaxline.
# It is prefixed by "jaxline" to prevent a conflict with a "mode" flag defined
# by Monarch.
_JAXLINE_MODE = flags.DEFINE_string(
    name="jaxline_mode",
    default="train",
    help=("Execution mode. "
          " `train` will run training, `eval` will run evaluation."),
)
_JAXLINE_TPU_DRIVER = flags.DEFINE_string(
    name="jaxline_tpu_driver",
    default="",
    help="Whether to use tpu_driver.",
)
_JAXLINE_ENSURE_TPU = flags.DEFINE_bool(
    name="jaxline_ensure_tpu",
    default=False,
    help="Whether to ensure we have a TPU connected.",
)


def create_checkpointer(
    config: config_dict.ConfigDict,
    mode: str,
) -> checkpointers.Checkpointer:
  """Creates an object to be used as a checkpointer."""
  if config.type == "neptune_ai":
    return checkpointers.NeptuneAiCheckpointer(**config.kwargs)
  if config.type == "local":
    return checkpointers.LocalCheckpointer(**config.kwargs)

  return checkpointers.NoneCheckpointer(**config.kwargs)



def create_writer(config: config_dict.ConfigDict, mode: str) -> writers.Writer:
  """Creates an object to be used as a writer."""
  if config.logger.type == "clear_ml":
    return writers.ClearMlLogger(config, mode)
  elif config.logger.type == "neptune_ai":
    return writers.NeptuneAiLogger(config, mode)

  return writers.TensorBoardLogger(config, mode)


@utils.debugger_fallback
def main(experiment_class, argv, checkpointer_factory=create_checkpointer):
  """Main potentially under a debugger."""
  del argv  # Unused.

  # Make sure the required fields are available in the config.
  config = _CONFIG.value
  base_config.validate_config(config)

  if _JAXLINE_TPU_DRIVER.value:
    jax.config.update("jax_xla_backend", "tpu_driver")
    jax.config.update("jax_backend_target", _JAXLINE_TPU_DRIVER.value)
    logging.info("Backend: %s %r", _JAXLINE_TPU_DRIVER.value, jax.devices())

  if _JAXLINE_ENSURE_TPU.value:
    # JAX currently falls back to CPU if it cannot register the TPU platform.
    # In multi-host setups this can happen if we timeout waiting for hosts to
    # come back up at startup or after pre-emption. This test will crash the
    # task if TPU devices are not available. We have increased the number of
    # acceptable failures per-task to allow for this.
    # TODO(tomhennigan) This test will eventually be part of JAX itself.
    chex.assert_tpu_available()

  jaxline_mode = _JAXLINE_MODE.value
  if jaxline_mode == "train":
    # Run training.
    writer = create_writer(config, jaxline_mode)
    checkpointer = checkpointer_factory(config.checkpointer, jaxline_mode)
    train.train(experiment_class, config, checkpointer, writer)
  elif jaxline_mode.startswith("eval"):
    # Run evaluation.
    writer = create_writer(config, jaxline_mode)
    checkpointer = checkpointer_factory(config.checkpointer, jaxline_mode)
    train.evaluate(experiment_class, config, checkpointer, writer,
                   jaxline_mode)
  elif jaxline_mode == "train_eval_multithreaded":
    pool = futures.ThreadPoolExecutor(1)
    writer_train = create_writer(config, "train")
    writer_eval = create_writer(config, "eval")

    # Run training in a background thread!
    pool.submit(train.train, experiment_class, config,
                checkpointer_factory(config.checkpointer, "train"),
                writer_train)

    # Run eval!
    train.evaluate(experiment_class, config,
                   checkpointer_factory(config.checkpointer, "eval"),
                   writer_eval)

    # If we're here, eval has finished. Wait for train to finish!
    pool.shutdown()
  else:
    raise ValueError(f"Mode {jaxline_mode} not recognized.")
