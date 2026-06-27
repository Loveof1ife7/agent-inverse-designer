import os
import sys
import subprocess
from pathlib import Path

# ================= 配置 =================

# 1) 7-Zip 路径
SEVEN_ZIP_PATH = Path(r"C:\Program Files\7-Zip\7z.exe")

# 2) 源文件夹路径（要打包的目录）
SOURCE_DIR = Path(r"C:\Users\admin\Desktop\3Dtruss\P222_1\Batch_Output_Files_422")

# 3) 输出目录（放压缩包）
OUTPUT_DIR_NAME = "7z_Output"
OUTPUT_CONTAINER = SOURCE_DIR.parent / OUTPUT_DIR_NAME

# 4) 每个压缩包包含多少个文件（建议 5000~20000）
FILES_PER_ARCHIVE = 10000

# 5) 压缩等级：-mx1 极速（CPU开销小）
COMPRESSION_LEVEL = "-mx1"

# 6) 是否分卷（一般不需要；分卷会生成 .001/.002 增加文件数）
#    如果你必须限制单文件大小，比如网盘单文件上限，可以打开：
ENABLE_VOLUME_SPLIT = False
VOLUME_SIZE = "1024m"  # 1GB

# =======================================

def check_paths():
    if not SEVEN_ZIP_PATH.exists():
        print(f"❌ 找不到 7z: {SEVEN_ZIP_PATH}")
        sys.exit(1)
    if not SOURCE_DIR.exists():
        print(f"❌ 找不到源目录: {SOURCE_DIR}")
        sys.exit(1)
    if not SOURCE_DIR.is_dir():
        print(f"❌ 源路径不是目录: {SOURCE_DIR}")
        sys.exit(1)

def iter_files_relative_to_base(base_dir: Path, target_dir: Path):
    """
    只遍历 target_dir 这一层的 .txt，并按“文件名数字”升序输出相对路径。
    例如：1.txt, 2.txt, 10.txt, 100.txt, ...
    """
    import os

    names = []
    with os.scandir(target_dir) as it:
        for e in it:
            if e.is_file() and e.name.lower().endswith(".txt"):
                stem = e.name[:-4]  # 去掉 .txt
                if stem.isdigit():  # 只要纯数字文件名
                    names.append(e.name)

    # 按数字排序
    names.sort(key=lambda n: int(n[:-4]))

    for name in names:
        yield (target_dir / name).relative_to(base_dir)


def run_7zip_sharded():
    check_paths()

    OUTPUT_CONTAINER.mkdir(parents=True, exist_ok=True)
    print(f"📂 输出目录: {OUTPUT_CONTAINER}")

    base_dir = SOURCE_DIR.parent  # 让相对路径带上 Batch_Output_Crystal_422 这一层
    shard_idx = 1
    file_count_in_shard = 0
    total_files = 0

    # 当前 shard 的 list 文件
    list_path = OUTPUT_CONTAINER / f"filelist_{shard_idx:04d}.txt"
    list_rel = list_path.relative_to(base_dir)  # 给 7z 用的相对路径

    # 打开第一个 listfile
    f = open(list_path, "w", encoding="utf-8", newline="\n")

    print("-" * 60)
    print(f"🚀 开始分包压缩：每包 {FILES_PER_ARCHIVE} 个文件")
    print(f"📄 源目录: {SOURCE_DIR}")
    print("-" * 60)

    def finalize_one_shard(idx: int, list_file_rel: Path, count_in_shard: int):
        """调用 7z 把 listfile 里的文件打成一个 archive"""
        archive_path = OUTPUT_CONTAINER / f"Upload_Data_{idx:04d}.7z"
        cmd = [
            str(SEVEN_ZIP_PATH),
            "a",
            "-t7z",
            COMPRESSION_LEVEL,
            "-mmt=on",         # 开线程（对压缩有帮助）
            "-bb0",            # 少输出
            "-bsp1",           # 进度到 stderr
            "-scsUTF-8",       # listfile 用 UTF-8
        ]

        if ENABLE_VOLUME_SPLIT:
            cmd.append(f"-v{VOLUME_SIZE}")

        # 输出压缩包路径（用相对 base_dir 的路径更干净，但绝对也行）
        cmd.append(str(archive_path))
        # 用 @listfile 指定文件列表（相对 base_dir 的路径）
        cmd.append(f"@{list_file_rel.as_posix()}")

        print(f"\n📦 正在生成: {archive_path.name}  (包含 {count_in_shard} 个文件)")
        subprocess.run(cmd, check=True, cwd=str(base_dir))

    try:
        for rel_path in iter_files_relative_to_base(base_dir, SOURCE_DIR):
            f.write(rel_path.as_posix() + "\n")
            file_count_in_shard += 1
            total_files += 1

            if file_count_in_shard >= FILES_PER_ARCHIVE:
                f.close()
                finalize_one_shard(shard_idx, list_rel, file_count_in_shard)

                # 准备下一个 shard
                shard_idx += 1
                file_count_in_shard = 0
                list_path = OUTPUT_CONTAINER / f"filelist_{shard_idx:04d}.txt"
                list_rel = list_path.relative_to(base_dir)
                f = open(list_path, "w", encoding="utf-8", newline="\n")

        # 最后一包（可能不足 FILES_PER_ARCHIVE）
        f.close()
        if file_count_in_shard > 0:
            finalize_one_shard(shard_idx, list_rel, file_count_in_shard)
        else:
            # 如果最后一个 listfile 是空的，删掉
            list_path.unlink(missing_ok=True)

    except subprocess.CalledProcessError as e:
        print(f"\n❌ 7z 执行失败: {e}")
        sys.exit(2)
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断。你已经生成的分包压缩仍然可用。")
        sys.exit(3)
    finally:
        try:
            f.close()
        except Exception:
            pass

    print("\n" + "=" * 60)
    print("✅ 完成！")
    print(f"📦 总文件数: {total_files}")
    print(f"📁 压缩包目录: {OUTPUT_CONTAINER}")
    print("提示：上传时只需要上传 Upload_Data_*.7z（以及分卷的话还有 .001/.002）")
    print("=" * 60)

if __name__ == "__main__":
    run_7zip_sharded()
    input("按回车键退出...")
