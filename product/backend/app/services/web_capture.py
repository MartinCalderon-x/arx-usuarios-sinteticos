"""Web capture service using Playwright for URL screenshots."""

import asyncio
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class WebCaptureService:
    """Service for capturing screenshots from URLs using Playwright."""

    def __init__(self):
        self._browser = None
        self._playwright = None

    async def _ensure_browser(self):
        """Ensure browser is initialized."""
        if self._browser is None:
            from playwright.async_api import async_playwright

            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-dev-shm-usage",  # Cloud Run has limited /dev/shm
                    "--disable-gpu",
                    "--disable-software-rasterizer",
                    "--single-process",  # Reduce memory usage
                    "--no-zygote",  # Required for single-process
                ],
            )
            logger.info("Playwright browser initialized")

    async def capture_screenshot(
        self,
        url: str,
        viewport_width: int = 1440,
        viewport_height: int = 900,
        full_page: bool = False,
        wait_for_load: bool = True,
        wait_timeout_ms: int = 10000,
    ) -> bytes:
        """
        Capture a screenshot from a URL.

        Args:
            url: The URL to capture
            viewport_width: Browser viewport width
            viewport_height: Browser viewport height
            full_page: Whether to capture full scrollable page
            wait_for_load: Wait for network idle before capture
            wait_timeout_ms: Timeout for page load

        Returns:
            Screenshot as PNG bytes
        """
        await self._ensure_browser()

        context = await self._browser.new_context(
            viewport={"width": viewport_width, "height": viewport_height},
            device_scale_factor=2,  # Retina quality
        )

        try:
            page = await context.new_page()

            # Navigate to URL
            await page.goto(
                url,
                wait_until="networkidle" if wait_for_load else "domcontentloaded",
                timeout=wait_timeout_ms,
            )

            # Small delay for any animations to settle
            await asyncio.sleep(0.5)

            # Capture screenshot
            screenshot = await page.screenshot(
                type="png",
                full_page=full_page,
            )

            logger.info(f"Captured screenshot from {url} ({len(screenshot)} bytes)")
            return screenshot

        finally:
            await context.close()

    async def capture_multiple_screenshots(
        self,
        url: str,
        scroll_positions: Optional[list[float]] = None,
        viewport_width: int = 1440,
        viewport_height: int = 900,
    ) -> list[bytes]:
        """
        Capture multiple screenshots at different scroll positions.

        Args:
            url: The URL to capture
            scroll_positions: List of scroll percentages (0.0 to 1.0)
            viewport_width: Browser viewport width
            viewport_height: Browser viewport height

        Returns:
            List of screenshots as PNG bytes
        """
        if scroll_positions is None:
            scroll_positions = [0.0]  # Just top of page

        await self._ensure_browser()

        context = await self._browser.new_context(
            viewport={"width": viewport_width, "height": viewport_height},
            device_scale_factor=2,
        )

        screenshots = []

        try:
            page = await context.new_page()

            # Navigate to URL
            await page.goto(url, wait_until="networkidle", timeout=15000)

            # Get total scroll height
            scroll_height = await page.evaluate("document.body.scrollHeight")
            viewport_h = await page.evaluate("window.innerHeight")
            max_scroll = max(0, scroll_height - viewport_h)

            for position in scroll_positions:
                # Scroll to position
                scroll_y = int(max_scroll * position)
                await page.evaluate(f"window.scrollTo(0, {scroll_y})")
                await asyncio.sleep(0.3)  # Wait for scroll

                # Capture
                screenshot = await page.screenshot(type="png")
                screenshots.append(screenshot)

            logger.info(f"Captured {len(screenshots)} screenshots from {url}")
            return screenshots

        finally:
            await context.close()

    async def close(self):
        """Close the browser instance."""
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
            logger.info("Playwright browser closed")


# Singleton instance
_web_capture_service: Optional[WebCaptureService] = None


def get_web_capture_service() -> WebCaptureService:
    """Get or create the web capture service singleton."""
    global _web_capture_service
    if _web_capture_service is None:
        _web_capture_service = WebCaptureService()
    return _web_capture_service
