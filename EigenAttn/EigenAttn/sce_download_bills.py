# -*- coding: utf-8 -*-
import re
import sys
import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError


SCE_URL = "https://www.sce.com/mysce/myaccount"


def log(msg: str):
    print(msg, flush=True)


def safe_filename(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r'[\\/:*?"<>|]+', "_", s)
    s = re.sub(r"\s+", " ", s)
    return s


def clean_address(text: str) -> str:
    """
    span[data-pii-encrypt] 里可能包含:
      '225 Oceano, Irvine, CA 92602Edit'
    去掉 Edit 及其后续内容。
    """
    t = " ".join((text or "").split())
    t = re.sub(r"\bEdit\b.*$", "", t).strip()
    return t


def dump_debug(page, outdir: Path, tag: str):
    outdir.mkdir(parents=True, exist_ok=True)
    html_path = outdir / f"debug_{tag}.html"
    png_path = outdir / f"debug_{tag}.png"
    try:
        html_path.write_text(page.content(), encoding="utf-8")
    except Exception:
        pass
    try:
        page.screenshot(path=str(png_path), full_page=True)
    except Exception:
        pass
    log(f"[DEBUG] dumped: {html_path} , {png_path}")


def wait_loaded(page, timeout_ms: int):
    # 等页面主要内容渲染（多账户页面一般会出现 Account: 或 Download Bill PDF）
    try:
        page.wait_for_timeout(800)
        page.wait_for_selector("text=Account:", timeout=timeout_ms)
        return
    except Exception:
        pass
    try:
        page.wait_for_selector("text=Download Bill PDF", timeout=timeout_ms)
        return
    except Exception:
        pass


def save_unique(path: Path) -> Path:
    """
    如果同名文件已存在，自动加 _2/_3... 避免覆盖
    """
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for i in range(2, 999):
        p2 = path.with_name(f"{stem}_{i}{suffix}")
        if not p2.exists():
            return p2
    return path.with_name(f"{stem}_999{suffix}")


def looks_like_address(raw_text: str) -> bool:
    """
    只保留“像地址”的文本，过滤掉账号/姓名/账户类型等：
    - 纯数字（>=6位）一般是账号 -> 排除
    - 含 Account/Residential/Commercial -> 排除
    - 地址常见特征：含逗号 / 含州+ZIP / 含街道后缀(Dr/Ave/Rd/Blvd/Ln/Ct/Way/Pl...)
    """
    s = " ".join((raw_text or "").split())
    s = re.sub(r"\bEdit\b.*$", "", s).strip()  # 去 Edit

    if not s or len(s) < 10:
        return False

    if re.fullmatch(r"\d{6,}", s):  # 纯账号
        return False

    if re.search(r"\b(Account|Residential|Commercial)\b", s, re.I):
        return False

    has_comma = ("," in s)
    has_state_zip = bool(re.search(r"\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b", s))
    has_street_suffix = bool(
        re.search(
            r"\b(St|Street|Dr|Drive|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane|Ct|Court|Way|Pl|Place|Pkwy|Parkway)\b",
            s,
            re.I,
        )
    )
    return has_comma or has_state_zip or has_street_suffix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="./bills_sce", help="下载保存目录")
    ap.add_argument("--state_path", default="./bills_sce/sce_state.json", help="登录态 storage_state 路径")
    ap.add_argument("--headless", action="store_true", help="无头模式")
    ap.add_argument("--timeout_ms", type=int, default=90000, help="超时毫秒")

    # 第一次用：手动登录后保存 state
    ap.add_argument(
        "--save_state_only",
        action="store_true",
        help="仅用于第一次：打开页面后你手动登录/MFA，回车后保存 state 并退出",
    )

    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)

        if args.save_state_only:
            context = browser.new_context(accept_downloads=True)
        else:
            if not state_path.exists():
                log(f"ERROR: 找不到 state：{state_path}。请先运行一次 --save_state_only 保存登录态。")
                sys.exit(2)
            context = browser.new_context(accept_downloads=True, storage_state=str(state_path))
            log(f"[STATE] loaded: {state_path}")

        page = context.new_page()
        page.set_default_timeout(args.timeout_ms)

        log(f"[1] 打开: {SCE_URL}")
        page.goto(SCE_URL, wait_until="domcontentloaded")
        wait_loaded(page, args.timeout_ms)

        if args.save_state_only:
            log("[STATE] 请在浏览器里手动登录（含 MFA）。完成后回到终端按回车保存登录态。")
            input()
            try:
                context.storage_state(path=str(state_path))
                log(f"[STATE] saved: {state_path}")
            except Exception as e:
                log(f"ERROR: 保存 state 失败: {e}")
                dump_debug(page, outdir, "save_state_failed")
                sys.exit(3)
            context.close()
            browser.close()
            log("[DONE] 保存登录态完成。后续直接运行不加 --save_state_only 即可批量下载。")
            return

        # 2) 找到所有 data-pii-encrypt span（里面混了账号/姓名/地址）
        all_pii = page.locator('span[data-pii-encrypt="true"]')
        total = all_pii.count()
        if total == 0:
            log("ERROR: 没找到 span[data-pii-encrypt=\"true\"]。可能没登录成功或页面结构变了。")
            dump_debug(page, outdir, "no_pii_span")
            sys.exit(4)

        log(f"[2] 原始 PII span 数量: {total}")

        # 2.1 过滤出真正地址
        address_items = []
        for i in range(total):
            sp = all_pii.nth(i)
            try:
                txt = sp.inner_text(timeout=3000)
            except Exception:
                try:
                    txt = sp.text_content(timeout=3000) or ""
                except Exception:
                    txt = ""

            txt = " ".join(txt.split())
            if looks_like_address(txt):
                addr = clean_address(txt)
                if addr:
                    address_items.append((sp, addr))

        if len(address_items) == 0:
            log("ERROR: 过滤后地址数量为 0。说明规则太严或页面结构不一致。")
            dump_debug(page, outdir, "no_address_after_filter")
            sys.exit(5)

        log(f"[2] 过滤后地址数量: {len(address_items)}")
        for k, (_, a) in enumerate(address_items, 1):
            log(f"    - [{k}] {a}")

        # 3) 对每个地址：定位其所属卡片容器 -> 点击该容器内的 Download Bill PDF -> 下载并重命名
        for idx, (addr_item, address) in enumerate(address_items, 1):
            log(f"\n[3.{idx}] 地址: {address}")

            # 3.1 用 JS 从当前地址节点往上找“包含 Download Bill PDF 的容器”
            try:
                addr_handle = addr_item.element_handle()
                container_handle = addr_handle.evaluate_handle(
                    """
                    (el) => {
                      let n = el;
                      for (let k=0; k<25 && n; k++) {
                        if (n.querySelector) {
                          const hit = n.querySelector("span.customAnchorTag__sceAnchorBold__3e-jA")
                                  || n.querySelector("span[class*='customAnchorTag__sceAnchorBold']");
                          if (hit && /Download\\s+Bill\\s+PDF/i.test(hit.innerText || "")) return n;
                          if ((n.innerText || "").match(/Download\\s+Bill\\s+PDF/i)) return n;
                        }
                        n = n.parentElement;
                      }
                      return el.ownerDocument.body;
                    }
                    """
                )
                container_el = container_handle.as_element()
            except Exception:
                container_el = None

            # 3.2 点击容器内的 Download Bill PDF
            last_err = None

            def click_download_in_container():
                nonlocal last_err
                # 优先按 class 精确/模糊匹配
                try:
                    cand = container_el.query_selector("span.customAnchorTag__sceAnchorBold__3e-jA")
                    if cand is None:
                        cand = container_el.query_selector("span[class*='customAnchorTag__sceAnchorBold']")
                    if cand is not None:
                        txt = (cand.inner_text() or "").strip()
                        if re.search(r"Download\s+Bill\s+PDF", txt, re.I):
                            cand.click()
                            return
                except Exception as e:
                    last_err = e

                # 再兜底：在 container 内扫描可点击元素并点击匹配文本
                try:
                    container_el.evaluate(
                        """
                        (root) => {
                          const all = root.querySelectorAll("span, a, button");
                          for (const el of all) {
                            const t = (el.innerText || "").trim();
                            if (/Download\\s+Bill\\s+PDF/i.test(t)) { el.click(); return true; }
                          }
                          return false;
                        }
                        """
                    )
                    return
                except Exception as e:
                    last_err = e

            # 如果找不到容器，则兜底：按“过滤后的地址序号”去点页面上的 Download Bill PDF
            if container_el is None:
                log("[WARN] 找不到卡片容器，将尝试全页按顺序点击 Download Bill PDF（可能不准确）。")
                try:
                    dl_all = page.locator("text=Download Bill PDF")
                    if dl_all.count() >= idx:
                        out_path = save_unique(outdir / f"{safe_filename(address)}.pdf")
                        with page.expect_download(timeout=args.timeout_ms) as dl_info:
                            dl_all.nth(idx - 1).click()
                        download = dl_info.value
                        download.save_as(str(out_path))
                        log(f"[OK] 已保存: {out_path}")
                        continue
                    else:
                        log("[ERROR] 页面上 Download Bill PDF 数量不足，无法全页兜底。")
                        dump_debug(page, outdir, f"not_enough_download_btn_{idx}")
                        continue
                except Exception as e:
                    log(f"[ERROR] 全页兜底下载失败: {e}")
                    dump_debug(page, outdir, f"download_fail_global_{idx}")
                    continue

            # 3.3 正常：expect_download + click
            out_path = save_unique(outdir / f"{safe_filename(address)}.pdf")
            try:
                with page.expect_download(timeout=args.timeout_ms) as dl_info:
                    click_download_in_container()
                download = dl_info.value
                download.save_as(str(out_path))
                log(f"[OK] 已保存: {out_path}")
            except PWTimeoutError:
                log("[ERROR] 未捕获到下载事件（可能打开新标签页或站点改为内嵌 PDF）。")
                dump_debug(page, outdir, f"download_timeout_{idx}")
            except Exception as e:
                log(f"[ERROR] 下载失败: {e}")
                if last_err:
                    log(f"  [DETAIL] click err: {last_err}")
                dump_debug(page, outdir, f"download_fail_{idx}")

        context.close()
        browser.close()


if __name__ == "__main__":
    main()