// Record a fly-around video of a scan from the viewer, frame by frame (exact, no dropped frames).
//
//   cd out/<scan>/viewer && python3 -m http.server 8000 &
//   node tools/record_video.js http://localhost:8000/index.html /tmp/frames 240 0.5
//   ffmpeg -framerate 24 -i /tmp/frames/f_%04d.png -c:v libx264 -pix_fmt yuv420p scan.mp4
//
// Args: <viewer url> <frame dir> <frames> [orbit turns=0.5] [width=960] [height=720]
//       [dist=0.85] [elevation=0.7]  (orbit radius / camera height, x scene size; lower = closer).
// Needs Node + Playwright (npm i playwright). Uses the viewer's window.__renderFrame(u),
// which moves the camera frustum along the capture path while the view orbits the scene.
const { chromium } = require('playwright');
const fs = require('fs');
(async () => {
  const [url, out, nStr, orbit = '0.5', w = '960', h = '720', dist = '0.85', elev = '0.7'] = process.argv.slice(2);
  const n = +nStr;
  fs.mkdirSync(out, { recursive: true });
  const b = await chromium.launch({ args: ['--use-gl=swiftshader', '--enable-webgl', '--ignore-gpu-blocklist'] });
  const p = await b.newPage({ viewport: { width: +w, height: +h }, deviceScaleFactor: 1 });
  p.setDefaultTimeout(600000);
  await p.goto(url);
  await p.waitForFunction(() => window.__ready === true && typeof window.__renderFrame === 'function', null, { timeout: 600000 });
  const t0 = Date.now();
  for (let k = 0; k < n; k++) {
    await p.evaluate(([u, o, d, e]) => window.__renderFrame(u, { orbit: o, dist: d, height: e }),
                     [k / (n - 1), +orbit, +dist, +elev]);
    await p.screenshot({ path: `${out}/f_${String(k).padStart(4, '0')}.png` });
    if (k % 40 === 0) console.log(`frame ${k + 1}/${n} ${((Date.now() - t0) / 1000).toFixed(0)}s`);
  }
  await b.close();
})();
