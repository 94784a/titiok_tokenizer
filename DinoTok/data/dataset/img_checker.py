import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Tuple, List
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = False


def check_one(path: str) -> Tuple[str, bool, str]:
    try:
        # 快速头校验
        with Image.open(path) as im:
            im.verify()
        # 再次实际解码一次，防止 verify 通过但 decode 失败
        with Image.open(path) as im:
            im.convert("RGB").resize((8, 8))
        return (path, True, "")
    except Exception as e:
        return (path, False, repr(e))


def iter_class_dirs(root_split: str):
    # 只产出真正的类目录
    with os.scandir(root_split) as it:
        for entry in it:
            if entry.is_dir():
                yield entry.path


def iter_files_in_dir(d: str):
    # 只产出文件路径（避免先收集成大列表）
    with os.scandir(d) as it:
        for e in it:
            if e.is_file():
                yield e.path


def check_split_streaming(
    root: str, split: str, num_workers: int = 64
) -> List[Tuple[str, str]]:
    """
    流式检查一个 split：
      - 按类目录逐个扫描
      - 扫到文件就提交线程池
      - 进度条 total 随着扫描到的文件数递增
    返回 bad 列表 [(path, err), ...]
    """
    split_dir = os.path.join(root, split)
    bad: List[Tuple[str, str]] = []
    bad_lock = Lock()

    with ThreadPoolExecutor(max_workers=num_workers) as ex, tqdm(
        total=0, desc=f"Checking {split}", ncols=120, unit="img", dynamic_ncols=True
    ) as pbar:
        futures = set()

        for cls_dir in iter_class_dirs(split_dir):
            # 这一类下的文件数量（边遍历边提交，同时把数量加到 total）
            cls_count = 0
            for fp in iter_files_in_dir(cls_dir):
                fut = ex.submit(check_one, fp)
                futures.add(fut)
                cls_count += 1
            if cls_count:
                pbar.total += cls_count
                pbar.refresh()

            # 为了不让 futures 集合无限增大，分批清理已完成任务
            done = {f for f in futures if f.done()}
            for f in done:
                path, ok, err = f.result()
                if not ok:
                    with bad_lock:
                        bad.append((path, err))
                pbar.update(1)
            futures -= done

        # 收尾：等剩余任务完成
        for f in as_completed(futures):
            path, ok, err = f.result()
            if not ok:
                with bad_lock:
                    bad.append((path, err))
            pbar.update(1)

    return bad


if __name__ == "__main__":
    ROOT = "/high_perf_store2/users/xiexuezhen/Imagenet/imagenet"  # 改成你的路径
    for split in ["train", "val"]:
        bad = check_split_streaming(ROOT, split, num_workers=64)
        print(f"\n[{split}] total bad files: {len(bad)}")
        if bad:
            out = os.path.join(ROOT, f"bad_{split}.txt")
            with open(out, "w") as f:
                for p, e in bad:
                    f.write(f"{p}\t{e}\n")
            print(f"Saved list to {out}")
