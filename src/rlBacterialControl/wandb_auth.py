"""Credential-free wandb setup for the training scripts.

Keeps API keys and entity names out of version control: nothing in this repo needs to
contain a secret, so the tree is safe to push to a public GitHub remote.

Credentials are looked up in this order (first hit wins, per field):

    1. JSON file at $WANDB_CREDENTIALS, else ~/.config/wandb_rl/credentials.json
    2. environment variables WANDB_API_KEY / WANDB_ENTITY
    3. whatever wandb itself already has (~/.netrc, i.e. a previous `wandb login`)

Nothing is required. If no key and no entity turn up, `init()` just calls `wandb.init()`
and lets wandb fall back to ~/.netrc, which is how a machine that has run `wandb login`
already works -- and how the cluster compute nodes see the login node's credentials.

Credentials file format (all keys optional):

    {
        "api_key": "<40-char wandb key from https://wandb.ai/authorize>",
        "entity":  "<wandb username or team, e.g. j-doe-some-university>"
    }

First-time setup on a new machine:

    mkdir -p ~/.config/wandb_rl
    cp scripts/credentials.example.json ~/.config/wandb_rl/credentials.json
    chmod 600 ~/.config/wandb_rl/credentials.json
    # then edit in your own key/entity

Typical use from a training script:

    from rlBacterialControl import wandb_auth
    wandb_auth.init(project="my-project", dir=folder_name, name=trial_name,
                    config=wandb_config, settings=wandb.Settings(symlink=False))
"""

import json
import os
import stat
import warnings

import wandb


# Overridable with the WANDB_CREDENTIALS environment variable.
DEFAULT_CREDENTIALS_PATH = os.path.join("~", ".config", "wandb_rl", "credentials.json")


def credentials_path():
    """Absolute path of the credentials JSON that would be read (it need not exist)."""
    return os.path.expanduser(os.environ.get("WANDB_CREDENTIALS", DEFAULT_CREDENTIALS_PATH))


def load_credentials():
    """Return the credentials dict, or {} if the file is absent.

    Raises ValueError if the file exists but is not a readable JSON object -- a silent
    fallback there would send runs to the wrong entity instead of failing loudly.
    """
    path = credentials_path()
    if not os.path.exists(path):
        return {}

    # On a shared cluster the home directory is visible to more than just the owner,
    # so a group/world-readable key file is worth a warning (not an error).
    mode = os.stat(path).st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        warnings.warn(f"{path} is group/world-accessible; run: chmod 600 {path}")

    with open(path) as f:
        try:
            creds = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path} is not valid JSON: {e}") from e
    if not isinstance(creds, dict):
        raise ValueError(f"{path} must contain a JSON object, got {type(creds).__name__}")
    return creds


def get_api_key():
    """wandb API key from the credentials file, else $WANDB_API_KEY, else None."""
    return load_credentials().get("api_key") or os.environ.get("WANDB_API_KEY") or None


def get_entity():
    """wandb entity (user or team) from the credentials file, else $WANDB_ENTITY, else None."""
    return load_credentials().get("entity") or os.environ.get("WANDB_ENTITY") or None


def login():
    """Authenticate with wandb if a key was configured; no-op otherwise.

    Returns True if a key was found and used, False if authentication was left to
    wandb's own ~/.netrc lookup.
    """
    key = get_api_key()
    if key is None:
        return False
    wandb.login(key=key)
    return True


def init(project, **kwargs):
    """`wandb.init()` with credentials resolved as documented in this module.

    An explicit entity= kwarg still wins over the configured one, so a script can pin a
    run to a specific team when it needs to. All other kwargs pass straight through.
    """
    login()
    if kwargs.get("entity") is None:
        entity = get_entity()
        if entity is not None:
            kwargs["entity"] = entity
        else:
            kwargs.pop("entity", None)  # let wandb use the account default
    return wandb.init(project=project, **kwargs)
