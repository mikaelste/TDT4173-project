from pathlib import Path


def get_unique_filename(base_name, folder="submissions", move_up=True):
    end = ""
    if folder == "models":
        end = ".pkl"

    # Use '..' to move up one folder level if move_up is True
    parent_folder = ".." if move_up else ""

    file_path = Path(parent_folder) / folder / f"{base_name}.csv{end}"
    count = 1
    while file_path.exists():
        file_path = Path(parent_folder) / folder / \
            f"{base_name}_{count}.csv{end}"
        count += 1
    return str(file_path)
