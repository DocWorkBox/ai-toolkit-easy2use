import os

TOOLKIT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_ROOT = os.path.join(TOOLKIT_ROOT, 'config')
KEYMAPS_ROOT = os.path.join(TOOLKIT_ROOT, "toolkit", "keymaps")
ORIG_CONFIGS_ROOT = os.path.join(TOOLKIT_ROOT, "toolkit", "orig_configs")
DIFFUSERS_CONFIGS_ROOT = os.path.join(TOOLKIT_ROOT, "toolkit", "diffusers_configs")
COMFY_MODELS_PATH = None
CHENGYU_MODELS_PATH = "/poddata/ai-toolkit-models"

# check if ENV variable is set
if 'MODELS_PATH' in os.environ and os.environ['MODELS_PATH'].strip() != "":
    MODELS_PATH = os.environ['MODELS_PATH']
else:
    MODELS_PATH = CHENGYU_MODELS_PATH


def get_path(path):
    # we allow absolute paths, but if it is not absolute, we assume it is relative to the toolkit root
    if not os.path.isabs(path):
        path = os.path.join(TOOLKIT_ROOT, path)
    return path
