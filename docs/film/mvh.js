// mvh.js: the motion-video kit, drawn on a 2D canvas. Every function takes the time it draws at, so a frame is a
// pure function of t: render any t in any order and get the same picture (that is what lets the harness render in
// parallel and check any moment). Load it with <script src="mvh.js"></script>; it defines window.MVH.
//
// Taken from the Prometheus release film. The rules it encodes (see skills/motion-video/TASTE.md): text arrives by
// word with a short blur, never all at once; captions sit top left with a copper rule; cards are dark glass with a
// soft rim; a cursor labelled "you" is the viewer doing something; a change of pace is a pace map, not an edit of
// every scene.
(function () {
  const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
  const ease = x => { x = clamp(x); return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2; };
  const easeOut = x => 1 - Math.pow(1 - clamp(x), 3);
  const span = (t, a, b) => clamp((t - a) / (b - a));
  const lerp = (a, b, k) => a + (b - a) * k;
  const fadeWindow = (t, a, b, fi = 0.5, fo = 0.5) => Math.min(span(t, a, a + fi), 1 - span(t, b - fo, b));

  // --- pacing ---------------------------------------------------------------------------------------------------
  // pace([[film_s, scene_s], ...]): film seconds -> scene seconds, piecewise linear. Retime a cut (faster here,
  // slower there) by editing the table, never the scenes.
  function pace(table) {
    return t => {
      for (let i = 1; i < table.length; i++) if (t <= table[i][0]) {
        const [a, A] = table[i - 1], [b, B] = table[i];
        return A + (t - a) / (b - a) * (B - A);
      }
      return table[table.length - 1][1];
    };
  }
  // holds([[at, length], ...]): an extended clock with pauses. heldT(x) freezes the scene at `at` for `length`
  // seconds (x keeps running, so a beat inserted there, such as a chat, animates on x while the scene waits).
  function holds(list) {
    return x => {
      let off = 0;
      for (const [at, len] of list) {
        if (x < at + off) return x - off;
        if (x < at + off + len) return at;
        off += len;
      }
      return x - off;
    };
  }

  // --- the kit ---------------------------------------------------------------------------------------------------
  function make(g, theme = {}) {
    const C = Object.assign({ bg: "#0a110e", ink: "#eef3ee", dim: "#8fa39a", accent: "#d98b4e", ok: "#7fd6a2",
                              bad: "#ff5a4e", card: "rgba(12,22,18,0.97)", line: "rgba(200,220,210,0.16)" }, theme.colors || {});
    const SANS = theme.sans || '"IBM Plex Sans", "DejaVu Sans", sans-serif';
    const MONO = theme.mono || '"IBM Plex Mono", "DejaVu Sans Mono", monospace';

    function text(str, x, y, size, o = {}) {
      g.save();
      g.globalAlpha *= o.alpha ?? 1;
      g.fillStyle = o.color || C.ink;
      g.font = `${o.weight || 400} ${size}px ${o.mono ? MONO : SANS}`;
      g.textAlign = o.align || "left"; g.textBaseline = o.base || "alphabetic";
      if (o.track) g.letterSpacing = o.track + "px";
      g.fillText(str, x, y);
      g.restore();
    }
    function measure(str, size, o = {}) {
      g.save(); g.font = `${o.weight || 400} ${size}px ${o.mono ? MONO : SANS}`; if (o.track) g.letterSpacing = o.track + "px";
      const w = g.measureText(str).width; g.restore(); return w;
    }
    // words arriving one by one with a short blur and rise: the film's headline move
    function blurWords(str, x, y, size, t, a, o = {}) {
      let wx = x;
      str.split(" ").forEach((wd, i) => {
        const k = easeOut(span(t, a + i * (o.stagger ?? 0.06), a + 0.3 + i * (o.stagger ?? 0.06)));
        g.save(); g.filter = `blur(${((1 - k) * 14).toFixed(1)}px)`;
        text(wd, wx, y + 14 * (1 - k), size, { ...o, alpha: (o.alpha ?? 1) * k });
        g.restore();
        wx += measure(wd + " ", size, o);
      });
    }
    // a caption: kicker, headline (blur-in), copper rule, a line under it; on screen from a to b
    function caption(t, a, b, kicker, title, sub, o = {}) {
      const k = fadeWindow(t, a, b, 0.25, 0.4), x = o.x ?? 140, dx = 60 * (1 - easeOut(span(t, a, a + 0.5)));
      if (k <= 0) return;
      g.save(); g.globalAlpha *= k;
      text(kicker, x + dx * 0.5, 96, 20, { mono: true, color: C.accent, track: 4 });
      blurWords(title, x - 6 + dx * 0.4, 166, o.size || 80, t, a, { weight: 600, track: -2 });
      g.fillStyle = C.accent; g.fillRect(x, 188, 360 * easeOut(span(t, a + 0.1, a + 0.7)), 3);
      if (sub) text(sub, x, 228, 24, { color: C.dim, weight: 300, alpha: easeOut(span(t, a + 0.2, a + 0.7)) });
      g.restore();
    }
    // a dark glass card with a soft rim glow
    function card(x, y, w, h, k = 1, o = {}) {
      g.save(); g.globalAlpha *= k;
      g.shadowColor = "rgba(0,0,0,0.6)"; g.shadowBlur = 40; g.fillStyle = o.fill || C.card;
      g.beginPath(); g.roundRect(x, y, w, h, o.r ?? 18); g.fill();
      if (o.rim !== false) { g.shadowColor = o.glow || "rgba(240,150,70,0.8)"; g.shadowBlur = 22; g.strokeStyle = o.stroke || C.accent; g.lineWidth = 1.5; g.stroke(); }
      g.restore();
    }
    // a line typed out from t0 to t1, with a blinking cursor until it is done
    function typed(str, x, y, size, t, t0, t1, o = {}) {
      const n = Math.floor(clamp((t - t0) / (t1 - t0)) * str.length), shown = str.slice(0, n);
      text(shown, x, y, size, o);
      if (n < str.length && n > 0 && Math.floor(t * 6) % 2 === 0) {
        g.save(); g.globalAlpha *= o.alpha ?? 1; g.fillStyle = C.accent;
        g.fillRect(x + measure(shown, size, o) + 4, y - size * 0.85, Math.max(3, size / 10), size * 1.05); g.restore();
      }
      return n >= str.length;
    }
    // a rounded chip (a check that passed, a finding, a label)
    function chip(label, x, y, k = 1, o = {}) {
      const size = o.size || 18, w = measure(label, size, { mono: true, weight: 500 }) + 28, h = size * 2.2;
      g.save(); g.globalAlpha *= k; g.strokeStyle = o.color || C.ok; g.lineWidth = 1.5;
      g.beginPath(); g.roundRect(x, y, w, h, h / 2); g.stroke(); g.restore();
      text(label, x + 14, y + h / 2 + size * 0.36, size, { mono: true, weight: 500, color: o.color || C.ok, alpha: k });
      return w;
    }
    // the viewer's cursor, labelled "you"; press 0..1 squeezes it for a click
    function cursor(x, y, k = 1, press = 0, o = {}) {
      const s = (o.scale || 1.5) * (1 - 0.15 * press);
      g.save(); g.globalAlpha *= k; g.translate(x, y); g.scale(s, s);
      g.fillStyle = "#f4f7f2"; g.strokeStyle = "#0a120e"; g.lineWidth = 2;
      g.beginPath(); g.moveTo(0, 0); g.lineTo(0, 34); g.lineTo(9, 26); g.lineTo(16, 41); g.lineTo(22, 38); g.lineTo(15, 23); g.lineTo(27, 23); g.closePath();
      g.fill(); g.stroke(); g.restore();
      if (o.label === false) return;
      g.save(); g.globalAlpha *= k; g.fillStyle = C.accent; g.beginPath(); g.roundRect(x + 40, y + 56, 64, 32, 8); g.fill(); g.restore();
      text(o.label || "you", x + 72, y + 79, 20, { mono: true, weight: 500, color: "#0a120e", align: "center", alpha: k });
    }
    // a carousel: items slide in from the right, one every `gap` seconds from t0; draw(j, local_t, alpha) draws item j
    // at the origin of its slot (w x h); items moving are blurred and smaller, the one in the middle is sharp
    function carousel(t, t0, gap, n, w, h, draw, o = {}) {
      let p = -1;
      for (let j = 0; j <= n; j++) p += ease(span(t, t0 + j * gap, t0 + j * gap + (o.move ?? 0.32)));
      for (let j = 0; j < n; j++) {
        const d = j - p, ad = Math.abs(d);
        if (ad > 1.3) continue;
        const a = clamp(1.3 - ad), x = (o.cx ?? g.canvas.width / 2) + d * (w + 120) - w / 2, y = (o.cy ?? g.canvas.height / 2) - h / 2;
        g.save(); g.filter = `blur(${Math.min(14, ad * 18).toFixed(1)}px)`;
        g.translate(x + w / 2, y + h / 2); g.scale(1 - 0.08 * ad, 1 - 0.08 * ad); g.translate(-w / 2, -h / 2);
        draw(j, Math.max(0, t - (t0 + j * gap)), a);
        g.restore();
      }
    }
    // a wide soft colour field: the background's mood for a scene
    function glow(x, y, rx, ry, rgb, a) {
      g.save(); g.translate(x, y); g.scale(rx / ry, 1);
      const r = g.createRadialGradient(0, 0, 0, 0, 0, ry);
      r.addColorStop(0, `rgba(${rgb},${a})`); r.addColorStop(1, `rgba(${rgb},0)`);
      g.fillStyle = r; g.fillRect(-ry, -ry, 2 * ry, 2 * ry); g.restore();
    }
    return { C, SANS, MONO, text, measure, blurWords, caption, card, typed, chip, cursor, carousel, glow };
  }

  window.MVH = { clamp, ease, easeOut, span, lerp, fadeWindow, pace, holds, make };
})();
