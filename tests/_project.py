"""随包路径引导：不依赖任何本机绝对路径。

源程序包可能被整体拷贝到任意位置（本机的 ``<工程根>/scripts/``、交付时的
``支撑材料/源程序/``，或评阅者自己的目录）。本模块从**调用脚本所在目录**出发
逐级上溯，把同时含有 ``code/`` 与 ``results/`` 的目录认定为工程根，并统一注册

* ``_style/``  —— 随包提供的绘图样式模块（``plot_style``）；
* 源程序包目录 —— 使包内同名模块优先于其它位置的旧副本被导入。

用法::

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _project import add_script_path, find_project_root

    ROOT = find_project_root(__file__)
"""
from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
STYLE_DIR = PACKAGE_DIR / "_style"
if not STYLE_DIR.is_dir():                 # 开发布局：tests/ 与 code/ 同级
    STYLE_DIR = PACKAGE_DIR.parent / "code" / "_style"

#: 工程根标志：同时存在代码与结果目录才认定为工程根。
ROOT_MARKERS = ("code", "results")


def find_project_root(start: str | Path | None = None) -> Path:
    """上溯查找同时含 ``code/`` 与 ``results/`` 的目录。

    ``start`` 可以是脚本路径或目录，默认为本模块所在目录。若上溯到文件系统
    根仍未命中标志目录，则退回 ``start`` 的上两级目录（与旧代码
    ``Path(__file__).resolve().parents[1]`` 的语义保持一致）。
    """
    here = Path(start).resolve() if start is not None else PACKAGE_DIR
    if here.is_file():
        here = here.parent
    for base in (here, *here.parents):
        if all((base / marker).is_dir() for marker in ROOT_MARKERS):
            return base
    parents = here.parents
    return parents[1] if len(parents) > 1 else here


def add_script_path() -> None:
    """把源程序包目录置于 ``sys.path`` 首位，优先导入包内同名模块。"""
    path = str(PACKAGE_DIR)
    while path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)


def add_style_path() -> None:
    """把随包绘图样式目录 ``_style/`` 置于 ``sys.path`` 首位。"""
    path = str(STYLE_DIR)
    while path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)


def add_source_path() -> None:
    """注册源程序模块目录：优先同级 ``源程序/``，其次工程根 ``scripts/``。

    供 ``测试/`` 下的验收脚本使用——交付布局（``测试/`` 与 ``源程序/`` 同级）与
    开发布局（``tests/`` 与 ``scripts/`` 同级）都能导入到同一批模块。
    """
    candidates = [PACKAGE_DIR.parent / "源程序", PACKAGE_DIR.parent / "code",
                  find_project_root(PACKAGE_DIR) / "scripts"]
    for cand in reversed(candidates):          # 先注册的优先级更低
        if not cand.is_dir():
            continue
        path = str(cand)
        while path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
