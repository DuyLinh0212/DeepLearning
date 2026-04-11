import os
import shutil


def _move_dir(src: str, dst_root: str):
    if not os.path.exists(src):
        print(f"Skip (not found): {src}")
        return
    os.makedirs(dst_root, exist_ok=True)
    dst = os.path.join(dst_root, os.path.basename(src))
    if os.path.exists(dst):
        raise FileExistsError(f"Destination already exists: {dst}")
    print(f"Moving {src} -> {dst}")
    shutil.move(src, dst)


def main():
    # Source directories
    src_data = os.path.join(os.getcwd(), "data")
    src_labels = os.path.join(os.getcwd(), "labels")

    # Destination root
    dst_root = r"F:\NgDuyLinh\datasetDeep"

    _move_dir(src_data, dst_root)
    _move_dir(src_labels, dst_root)
    print("Done.")


if __name__ == "__main__":
    main()
