# The following is necessary if you want to use the fast tokenizer for deberta v2 or v3
import shutil
from pathlib import Path


def load_tokenizer(tokenizer_path):
    transformers_path = Path(tokenizer_path['transformers'])

    input_dir = Path(tokenizer_path['dir'])

    convert_file = input_dir / "convert_slow_tokenizer.py"
    conversion_path = transformers_path/convert_file.name

    if conversion_path.exists():
        conversion_path.unlink()

    shutil.copy(convert_file, transformers_path)
    deberta_v2_path = transformers_path / "models" / "deberta_v2"

    for filename in ['tokenization_deberta_v2.py', 'tokenization_deberta_v2_fast.py', "deberta__init__.py"]:
        if str(filename).startswith("deberta"):
            filepath = deberta_v2_path/str(filename).replace("deberta", "")
        else:
            filepath = deberta_v2_path/filename
        if filepath.exists():
            filepath.unlink()

        shutil.copy(input_dir/filename, filepath)
    
    return None