"""锁定图核验与刷新（figures/locked/）。

AGENTS.md 规则 3/5 的执行工具：论文重建必须读取 figures/locked/ 图源，
普通绘图脚本只能写入非 locked 路径；论文修改后须运行本脚本核验，
只有用户明确确认新图后，才允许用 --refresh 更新锁定图。

用法：
    python scripts/pin_approved_paper_figures.py             # 只核验（只读，不写任何文件）
    python scripts/pin_approved_paper_figures.py --refresh   # 用户确认新图后更新锁定图与 manifest

核验三项：
  1) locked 目录内每个已登记文件的 sha256 与 manifest.json 一致
     （防止被重绘、替换或从旧备份恢复）；
  2) 论文草稿.docx 内嵌图片中存在同 sha256 的对象
     （确认论文实际引用的就是锁定图，而非同名替换品）；
  3) locked 目录不含 manifest 未登记的其它图片
     （普通绘图脚本若写入 locked 会在此暴露）。

退出码：0 = 全部通过；1 = 存在不一致。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
from _project import find_project_root  # noqa: E402
ROOT = find_project_root(__file__)
LOCKED_DIR = ROOT / "figures" / "locked"
MANIFEST = LOCKED_DIR / "manifest.json"
DOCX = ROOT / "论文草稿.docx"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_docx_media(docx_path: Path) -> dict[str, bytes]:
    if not docx_path.exists():
        raise FileNotFoundError(f"缺少论文文件：{docx_path}")
    with zipfile.ZipFile(docx_path) as zf:
        return {
            name: zf.read(name)
            for name in zf.namelist()
            if name.startswith("word/media/")
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="核验/刷新 figures/locked 锁定图")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="用户确认新图后，用论文当前图片更新锁定图与 manifest（默认只核验）",
    )
    args = parser.parse_args()

    if not MANIFEST.exists():
        print(f"[FAIL] 缺少 manifest：{MANIFEST}")
        return 1

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    media = load_docx_media(DOCX)
    media_by_sha: dict[str, list[str]] = {}
    for name, blob in media.items():
        media_by_sha.setdefault(sha256_bytes(blob), []).append(name)

    problems: list[str] = []

    if args.refresh:
        changed = 0
        for key, meta in manifest.items():
            target = ROOT / meta["file"]
            name = meta.get("docx_media")
            if not name or name not in media:
                problems.append(
                    f"[FAIL] {key}: 无法定位论文中的图片（docx_media={name!r}）；"
                    f"请先在 manifest 登记 docx_media"
                )
                continue
            blob = media[name]
            if not target.exists() or sha256_file(target) != sha256_bytes(blob):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(blob)
                meta["sha256"] = sha256_bytes(blob)
                meta["bytes"] = len(blob)
                changed += 1
                print(f"[REFRESH] {key} -> {target.name} ({len(blob)} bytes)")
        MANIFEST.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[OK] manifest 已更新，刷新 {changed} 项")
        if problems:
            print()
            for p in problems:
                print(p)
            return 1
        return 0

    for key, meta in manifest.items():
        target = ROOT / meta["file"]
        if not target.exists():
            problems.append(f"[FAIL] {key}: 锁定图缺失 {meta['file']}")
            continue
        actual = sha256_file(target)
        if actual != meta["sha256"]:
            problems.append(
                f"[FAIL] {key}: 锁定图被改动 {meta['file']}\n"
                f"        期望 {meta['sha256']}\n"
                f"        实际 {actual}"
            )
            continue
        if actual not in media_by_sha:
            problems.append(
                f"[FAIL] {key}: 论文 {DOCX.name} 中未找到该锁定图（sha256 不匹配）"
            )
            continue
        print(f"[OK] {key}: {target.name} == {media_by_sha[actual][0]} (sha256 {actual[:16]}…)")

    registered = {Path(meta["file"]).name for meta in manifest.values()}
    extra = sorted(
        p.name for p in LOCKED_DIR.glob("*.png") if p.name not in registered
    )
    if extra:
        problems.append(
            f"[WARN] locked 目录存在未登记图片（普通绘图脚本不应写入 locked）：{extra}"
        )

    if problems:
        print()
        for p in problems:
            print(p)
        print(f"\n[FAIL] 核验未通过（{len(problems)} 项）")
        return 1

    print(f"\n[OK] 锁定图核验通过：{len(manifest)} 张，论文引用与 figures/locked 一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
