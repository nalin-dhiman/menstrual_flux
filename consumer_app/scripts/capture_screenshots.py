from __future__ import annotations

import os
from pathlib import Path

from playwright.sync_api import Page, sync_playwright


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "consumer_app" / "screenshots"
BASE_URL = os.environ.get(
    "PLANNER_BASE_URL", "http://127.0.0.1:8765/planner/"
)


def prepare_example(page: Page) -> None:
    page.goto(BASE_URL, wait_until="networkidle")
    page.locator("#adult-confirm").check()
    page.locator("#scope-confirm").check()
    page.locator("#continue-button").click()
    page.locator("#load-example").click()
    page.locator("#prediction-card").wait_for(state="visible")
    page.wait_for_timeout(3800)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)

        desktop = browser.new_context(viewport={"width": 1440, "height": 1000})
        page = desktop.new_page()
        prepare_example(page)
        page.screenshot(
            path=OUTPUT / "01_planning_window.png",
            full_page=True,
        )
        page.locator('.side-nav [data-view="insights"]').click()
        page.wait_for_timeout(450)
        page.screenshot(
            path=OUTPUT / "02_pattern.png",
            full_page=True,
        )
        page.locator('.side-nav [data-view="privacy"]').click()
        page.wait_for_timeout(450)
        page.screenshot(
            path=OUTPUT / "03_privacy.png",
            full_page=True,
        )
        desktop.close()

        mobile = browser.new_context(
            viewport={"width": 390, "height": 844},
            device_scale_factor=1,
            is_mobile=True,
            has_touch=True,
        )
        page = mobile.new_page()
        prepare_example(page)
        page.screenshot(
            path=OUTPUT / "04_mobile.png",
            full_page=False,
        )
        mobile.close()
        browser.close()

    print(f"planner screenshots written to {OUTPUT}")


if __name__ == "__main__":
    main()
