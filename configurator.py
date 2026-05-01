"""
nanoXPP Configurator - Clean & Modern Version
Handles config file loading and command line overrides
"""

import os
import sys
import argparse
from ast import literal_eval

def parse_config():
    parser = argparse.ArgumentParser(description='nanoXPP Training Config')
    parser.add_argument('config_file', nargs='?', default='config/train_xpp.py',
                        help='Path to config file (default: config/train_xpp.py)')
    
    # Allow overriding any config parameter via --key=value
    # Example: --max_iters=5000 --learning_rate=6e-4
    known_args, unknown = parser.parse_known_args()
    
    # Load the base config file
    config_path = known_args.config_file
    print(f"📄 Loading config: {config_path}")
    
    try:
        exec(open(config_path).read(), globals())
    except FileNotFoundError:
        print(f"❌ Error: Config file '{config_path}' not found!")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        sys.exit(1)

    # Apply command line overrides
    for arg in unknown:
        if arg.startswith('--'):
            try:
                key, value = arg[2:].split('=', 1)
                # Try to evaluate the value (so 1e-4 becomes float, True becomes bool, etc.)
                try:
                    value = literal_eval(value)
                except:
                    pass  # keep as string if eval fails
                
                globals()[key] = value
                print(f"🔧 Overriding {key} = {value}")
            except Exception:
                print(f"⚠️  Warning: Could not parse argument: {arg}")

    print(f"✅ Config loaded successfully. max_iters = {globals().get('max_iters', 'N/A')}")
    return globals()


# Run the configurator when imported
if __name__ == "__main__":
    parse_config()
