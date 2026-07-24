const path = require('path');
const { pathToFileURL } = require('url');
const { chromium } = require('C:/Users/Dell/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

(async () => {
  const root = __dirname;
  const output = path.join(root, 'output');
  const browser = await chromium.launch({
    headless: true,
    executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
    args: ['--allow-file-access-from-files']
  });
  const page = await browser.newPage({ viewport: { width: 1200, height: 1600 }, deviceScaleFactor: 1 });
  await page.goto(pathToFileURL(path.join(root, 'index.html')).href, { waitUntil: 'load' });
  await page.evaluate(() => Promise.race([
    document.fonts.ready,
    new Promise(resolve => setTimeout(resolve, 2500))
  ]));

  const targets = [
    ['#card-01', 'xhs-01-cover.png'],
    ['#card-02', 'xhs-02-dataset.png'],
    ['#card-03', 'xhs-03-mcem-error.png'],
    ['#card-04', 'xhs-04-no-leakage.png'],
    ['#card-05', 'xhs-05-training.png'],
    ['#card-06', 'xhs-06-model-failed.png'],
    ['#card-07', 'xhs-07-takeaway.png']
  ];

  for (const [selector, filename] of targets) {
    const node = page.locator(selector);
    await node.screenshot({ path: path.join(output, filename) });
    const metrics = await node.evaluate(el => ({
      id: el.id,
      width: el.getBoundingClientRect().width,
      height: el.getBoundingClientRect().height,
      scrollWidth: el.scrollWidth,
      scrollHeight: el.scrollHeight
    }));
    console.log(JSON.stringify(metrics));
  }

  await browser.close();
})();
