# HyperState


Experimental and highly opinionated library for managing configs and mutable program state of machine learning training systems.

Features:
- load yaml/ron config files as dataclasses
- command line flag to override any config values
- checkpoint and restore full program state
- human readable and editable checkpoints
- (planned) versioning and schema evolution
- large binary objects are stored in separate files and only loaded when accessed
- (planned) edit hyperparameters of running experiment on the fly without restarts

## Config

HyperState requires your config to be a `@dataclass` with fields of type `int`, `float`, `str`, `List`, `Map`, or HyperState compatible `@dataclass`.

Load config:
```python
# Not currently implemented
@dataclass
class Config:
    lr: float
    batch_size: int

config: Config = hyperstate.load_config(Config, "config.yaml")
```

Load config and override specific values:
```python
# Not currently implemented
@dataclass
class OptimizerConfig:
    lr: float
    batch_size: int

@dataclass
class Config:
    optimzer: OptimizerConfig
    steps: int

overrides = ["optimizer.lr=0.1", "steps=100"]
config: Config = hyperstate.load_config(Config, "config.yaml", overrides=overrides)
```

## State

State objects must also be `@dataclass`es, and can additonally include opaque `hyperstate.Blob[T]` types with custom (de)serialization logic.
Both `Config` and `State` are managed by a `HyperState[Config, State]` object with `config` and `state` fields.
The `HyperState` object is created/loaded with `HyperState.load`:

```python
def load(
    config_clz: Type[C],
    state_clz: Type[S],
    initial_state: Callable[[C], S],
    path: str,
    checkpoint_dir: Optional[str] = None,
    checkpoint_key: Optional[str] = None,
    overrides: Optional[List[str]] = None,
) -> "HyperState[C, S]":
    """
    Loads a HyperState from a checkpoint (if exists) or initializes a new one.

    :param config_clz: The type of the config object.
    :param state_clz: The type of the state object.
    :param initial_state: A function that takes a config object and returns an initial state object.
    :param path: The path to the config file or full checkpoint directory.
    :param checkpoint_dir: The directory to store checkpoints.
    :param checkpoint_key: The key to use for the checkpoint. This must be a field of the state object (e.g. a field holding current iteration).
    :param overrides: A list of overrides to apply to the config. (Example: ["optimizer.lr=0.1"])
    """
    pass
```

### Checkpointing 

Just call `step` on the `HyperState` object to checkpoint the current config/state to the configured directory.

### Schedules

All `int`/`float` fields in the config can also be set to a schedule that will be updated at each step.

### Limitations

Currently, can't easily use `HyperState` with `PyTorch`.
Problem: To initialize optimizer class, needs to have parameter state as well as config.
Therefore, `HyperState` currently only stores the optimizer state and policy state which has to be manually updated each step.

Sketch of solution:
- the `State` dataclasses have to inherit/add property which intercepts fields accesses and gives transparent lazy loading
- therefore, we can partially initialize the state object and pass `State` to all init calls as well
- as long as there are no loops, initializer can access any other (fully initialized) state objects
- this allows us to support any types that implement a some hyperstate serialization interface (load(config, state, state_dict) -> self, get_state_dict() -> state_dict)

Limitation 2: Syncing state in distributed training.