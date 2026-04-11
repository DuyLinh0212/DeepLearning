import os
import shutil


def main():
    src_root = os.getcwd()
    dst_root = r"F:\NgDuyLinh\Do_an"
    dst_project = os.path.join(dst_root, "N5_DeepLearning")

    os.makedirs(dst_root, exist_ok=True)
    if os.path.exists(dst_project):
        raise FileExistsError(f"Destination already exists: {dst_project}")
    os.makedirs(dst_project, exist_ok=False)

    for name in os.listdir(src_root):
        src = os.path.join(src_root, name)
        dst = os.path.join(dst_project, name)
        print(f"Moving {src} -> {dst}")
        shutil.move(src, dst)

    print("Done.")


if __name__ == "__main__":
    main()
