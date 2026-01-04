from playwright.sync_api import sync_playwright
import os

def verify_presentation():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Load the local HTML file
        file_path = os.path.abspath("presentation/investor_presentation.html")
        page.goto(f"file://{file_path}")

        # Take a screenshot of the first slide
        page.screenshot(path="verification/slide_1.png")
        print("Screenshot of Slide 1 taken.")

        # Click next to go to slide 2
        page.get_by_role("button", name="Next").click()
        page.wait_for_timeout(600) # Wait for transition
        page.screenshot(path="verification/slide_2.png")
        print("Screenshot of Slide 2 taken.")

        browser.close()

if __name__ == "__main__":
    verify_presentation()
