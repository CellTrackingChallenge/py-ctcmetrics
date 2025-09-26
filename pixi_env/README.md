# Python environment managed by the pixi

This is a Python-centered project, and naturaly comes with dependencies--additional packages.
Clearly, there are several packages managers for Python. Some keep their environments outside
the project tree (e.g. conda or mamba) and one has to remember which environment to activate;
some keep the environment(s) inside the project tree (e.g. uv or pixi) and make it clear which
environment to use.

This folder here offers to control the dependecies with [the pixi](https://pixi.sh/) manager.


# Using pixi with this project

The pixi is controlled with `pixi.toml` configuration file, which ideally is kept synchronized
with the requirements defined in the root folder.

To start the (pixi-governed) environment in this folder:

```bash
cd path_to_this_repo/pixi_env
pixi shell
```

or, to start it in the root folder:

```bash
cd path_to_this_repo
pixi shell --manifest-path pixi_env/pixi.toml
```

Both examples normally give the conda/mamba-like-env-activated prompt... :+1:

