import importlib.util
import os.path as osp
import sys


def load(name):
    pathname = osp.join(osp.dirname(__file__), name)
    # Generate a unique module name based on the file path
    module_name = f"multiagent.scenarios.{osp.splitext(osp.basename(name))[0]}"
    
    # Check if module is already loaded
    if module_name in sys.modules:
        return sys.modules[module_name]
    
    spec = importlib.util.spec_from_file_location(module_name, pathname)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {name} from {pathname}")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
