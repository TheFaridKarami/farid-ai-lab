from pathlib import Path

import yaml


def load_config(path: str|Path) ->dict:
    
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)