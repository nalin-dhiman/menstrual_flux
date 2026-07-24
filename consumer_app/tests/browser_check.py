from __future__ import annotations

import json
import os
from pathlib import Path
import re
from urllib.parse import urlsplit

from playwright.sync_api import Page, sync_playwright


BASE_URL = os.environ.get(
    "PLANNER_BASE_URL", "http://127.0.0.1:8765/planner/"
)


def acknowledge(page: Page) -> None:
    page.locator("#adult-confirm").check()
    page.locator("#scope-confirm").check()
    assert page.locator("#continue-button").is_enabled()
    page.locator("#continue-button").click()


def check_complete_flow(browser) -> None:
    context = browser.new_context(accept_downloads=True)
    page = context.new_page()
    page_errors: list[str] = []
    requested_origins: set[tuple[str, str]] = set()
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    page.on(
        "request",
        lambda request: requested_origins.add(
            (urlsplit(request.url).scheme, urlsplit(request.url).netloc)
        ),
    )

    page.goto(BASE_URL, wait_until="networkidle")
    assert page.title() == "Menstrual Flux · Cycle Planner"
    assert page.locator("#consent-layer").is_visible()
    assert page.locator("#continue-button").is_disabled()

    acknowledge(page)
    assert page.locator("#empty-state").is_visible()
    page.locator("#load-example").click()
    page.locator("#prediction-card").wait_for(state="visible")
    page.get_by_text("Example estimate").wait_for()
    assert "Most likely" in page.locator("#prediction-content").inner_text()

    page.locator('.side-nav [data-view="history"]').click()
    assert page.locator(".history-item").count() == 6
    page.locator(".history-item .delete-date").first.click()
    assert page.locator(".history-item").count() == 5

    page.locator('.side-nav [data-view="insights"]').click()
    assert page.locator(".interval-bar").count() == 4
    assert "Mean absolute error" in page.locator("#backtest-content").inner_text()

    page.locator('.side-nav [data-view="privacy"]').click()
    with page.expect_download() as download_info:
        page.locator("#export-data").click()
    download = download_info.value
    assert re.match(
        r"^menstrual-flux-planner-\d{4}-\d{2}-\d{2}\.json$",
        download.suggested_filename,
    )
    downloaded_path = Path(download.path())
    parsed = json.loads(downloaded_path.read_text())
    assert parsed["localOnlyRelease"] is True
    assert len(parsed["periodStarts"]) == 5

    page.locator("#request-erase").click()
    assert page.locator("#erase-dialog").get_attribute("open") == ""
    page.locator('#erase-dialog button[value="erase"]').click()
    page.locator("#consent-layer").wait_for(state="visible")
    assert page.locator("#consent-layer").is_visible()
    assert (
        page.evaluate("localStorage.getItem('menstrualFluxPlanner.v1')") is None
    )

    expected = urlsplit(BASE_URL)
    assert requested_origins == {(expected.scheme, expected.netloc)}
    assert page_errors == []

    privacy_page = context.new_page()
    privacy_page.goto(f"{BASE_URL}privacy.html", wait_until="networkidle")
    assert privacy_page.title() == "Privacy · Menstrual Flux Cycle Planner"
    assert "local-only application" in privacy_page.locator("main").inner_text()
    context.close()


def check_validation_persistence_and_offline(browser) -> None:
    context = browser.new_context()
    page = context.new_page()
    page_errors: list[str] = []
    page.on("pageerror", lambda error: page_errors.append(str(error)))

    page.goto(BASE_URL, wait_until="networkidle")
    acknowledge(page)

    page.locator("#quick-date").fill("2026-05-01")
    page.locator('#quick-add-form button[type="submit"]').click()
    page.locator('.side-nav [data-view="history"]').click()
    page.locator("#history-date").fill("2026-05-29")
    page.locator('#history-add-form button[type="submit"]').click()
    assert page.locator(".history-item").count() == 2

    page.locator("#history-date").fill("2026-05-29")
    page.locator('#history-add-form button[type="submit"]').click()
    assert "already recorded" in page.locator("#history-message").inner_text()

    page.locator("#history-date").fill("2026-06-05")
    page.locator('#history-add-form button[type="submit"]').click()
    assert page.locator("#date-warning-dialog").get_attribute("open") == ""
    page.locator('#date-warning-dialog button[value="cancel"]').click()
    assert page.locator(".history-item").count() == 2

    page.locator("#history-date").fill("2026-06-05")
    page.locator('#history-add-form button[type="submit"]').click()
    page.locator('#date-warning-dialog button[value="confirm"]').click()
    page.locator(".history-item").nth(2).wait_for()
    assert page.locator(".history-item").count() == 3

    page.reload(wait_until="networkidle")
    assert page.locator("#consent-layer").is_hidden()
    page.locator('.side-nav [data-view="history"]').click()
    assert page.locator(".history-item").count() == 3

    page.evaluate("navigator.serviceWorker.ready.then(() => true)")
    context.set_offline(True)
    page.reload(wait_until="domcontentloaded")
    page.locator(".view.active h1").wait_for()
    assert page.title() == "Menstrual Flux · Cycle Planner"

    assert page_errors == []
    context.close()


def main() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        check_complete_flow(browser)
        check_validation_persistence_and_offline(browser)
        browser.close()
    print("planner browser checks: passed")


if __name__ == "__main__":
    main()
