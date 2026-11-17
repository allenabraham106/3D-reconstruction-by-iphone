"""synth.py: a film's soundtrack, synthesised from nothing (no samples, no licences), on the film's own event times.

    from synth import Score
    s = Score(length=8.0, bpm=128)
    s.groove(levels=[(0, 0), (1.9, 2), (6.5, 0)])      # bass / kick / hats by beat energy (0 silent .. 3 everything)
    s.put(s.hit(0.6), 1.9)                              # a hit on the logo
    s.put(s.whoosh(0.5), 1.4)                           # a rise into it
    s.write("score.wav")

Times are film seconds. When the film is paced (MVH.pace) or held (MVH.holds), give the same tables to Score
(pace=..., holds=...) and write events in scene seconds with s.at(scene_s): a retime then moves the sound with the
picture. The lessons from the Prometheus film: one hit per hard cut, a rise into a reveal, soft clicks for UI, and no
repeated bright chimes (they turn into a "ding ding" fast).
"""
import wave

import numpy as np
from scipy.signal import butter, sosfilt

SR = 48000


def hz(note):
    return 440.0 * 2 ** ((note - 69) / 12)


class Score:
    def __init__(self, length, bpm=128, pace=None, holds=None, seed=7):
        self.length, self.bpm, self.beat = length, bpm, 60 / bpm
        self.n = int(SR * length)
        self.L, self.R = np.zeros(self.n), np.zeros(self.n)
        self.rng = np.random.default_rng(seed)
        self.pace, self.holds = pace, holds or []

    # --- time ---------------------------------------------------------------------------------------------------
    def at(self, scene_s, held=False):
        """scene seconds -> film seconds, through the holds and the pace table (their inverse)"""
        x = scene_s
        if not held:
            for a, ln in sorted(self.holds):
                if scene_s > a:
                    x += ln
        if not self.pace:
            return x
        for (a, A), (b, B) in zip(self.pace, self.pace[1:]):
            if x <= B:
                return a + (x - A) / (B - A) * (b - a)
        return self.pace[-1][0]

    # --- placing and filters ------------------------------------------------------------------------------------
    def put(self, sig, t, pan=0.0, gain=1.0):
        i = int(t * SR)
        if i >= self.n or i < 0:
            return
        sig = sig[: self.n - i] * gain
        self.L[i:i + len(sig)] += sig * np.sqrt(0.5 * (1 - pan))
        self.R[i:i + len(sig)] += sig * np.sqrt(0.5 * (1 + pan))

    @staticmethod
    def lp(x, f, order=2):
        return sosfilt(butter(order, f, "low", fs=SR, output="sos"), x)

    @staticmethod
    def hp(x, f, order=2):
        return sosfilt(butter(order, f, "high", fs=SR, output="sos"), x)

    @staticmethod
    def env(n, a=0.005, r=0.1, hold=0.0):
        e = np.ones(n)
        na = max(1, int(a * SR))
        e[:na] = np.linspace(0, 1, na)
        start = min(n, na + int(hold * SR))
        e[start:] = np.exp(-np.arange(n - start) / max(1, int(r * SR)))
        return e

    @staticmethod
    def saw(f, n, detune=0.0):
        ph = np.cumsum(np.full(n, f * (1 + detune)) / SR)
        return 2 * (ph % 1) - 1

    # --- sounds -------------------------------------------------------------------------------------------------
    def whoosh(self, dur=1.0, lo=300, hi=6000, rise=True):
        n = int(dur * SR)
        noise = self.rng.standard_normal(n)
        k = np.linspace(0, 1, n) if rise else np.linspace(1, 0, n)
        out = np.zeros(n)
        for i in range(8):                                  # a band sweeping up (or down)
            a, b = i * n // 8, (i + 1) * n // 8
            f = lo + (hi - lo) * k[(a + b) // 2]
            out[a:b] = self.lp(noise, min(f, SR / 2 - 100))[a:b]
        return out * np.sin(np.pi * np.linspace(0, 1, n)) ** 2 * 0.5

    def hit(self, gain=1.0):
        n = int(0.9 * SR)
        tt = np.arange(n) / SR
        body = np.sin(2 * np.pi * np.cumsum(40 + 90 * np.exp(-tt * 18)) / SR) * np.exp(-tt * 4.5)
        snap = self.hp(self.rng.standard_normal(n), 2000) * np.exp(-tt * 40) * 0.3
        return (body + snap) * gain

    def tick(self, f=3200, dur=0.03):
        n = int(dur * SR)
        return np.sin(2 * np.pi * f * np.arange(n) / SR) * self.env(n, 0.001, r=dur / 3)

    def pluck(self, note, dur=0.22, bright=2400):
        n = int(dur * SR)
        s = self.saw(hz(note), n) * 0.6 + self.saw(hz(note + 12), n, 0.003) * 0.25
        return self.lp(s, bright) * self.env(n, 0.002, r=dur / 3)

    def chime(self, note, dur=1.2):
        n = int(dur * SR)
        tt = np.arange(n) / SR
        f = hz(note)
        return (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(2 * np.pi * 2.76 * f * tt)) * np.exp(-tt * 3.0) * self.env(n, 0.002, r=dur)

    def bass(self, note, dur):
        n = int(dur * SR)
        tt = np.arange(n) / SR
        s = np.sin(2 * np.pi * hz(note) * tt) + 0.3 * np.tanh(3 * np.sin(2 * np.pi * hz(note) * tt))
        return s * self.env(n, 0.004, r=dur / 2.5)

    def kick(self):
        n = int(0.45 * SR)
        tt = np.arange(n) / SR
        return np.sin(2 * np.pi * np.cumsum(46 + 120 * np.exp(-tt * 28)) / SR) * np.exp(-tt * 7.5)

    def hat(self, open_=False):
        n = int((0.18 if open_ else 0.05) * SR)
        return self.hp(self.rng.standard_normal(n), 7000) * self.env(n, 0.001, r=0.05 if open_ else 0.012)

    # --- the groove ---------------------------------------------------------------------------------------------
    def groove(self, levels, root=38, chords=((50, 57, 60, 64, 65), (46, 53, 57, 62, 65), (41, 48, 53, 57, 64), (48, 55, 60, 64, 67))):
        """levels: [(film_s, 0..3), ...]: 1 bass pulse, 2 + kick and hats, 3 + arpeggio; 0 silent"""
        step = self.beat / 4
        bar = 4 * self.beat
        lv_at = lambda t: max([lv for s, lv in levels if s <= t] or [0])
        k = 0
        while k * step < self.length:
            t = k * step
            lv = lv_at(t)
            ch = chords[int(t // bar) % len(chords)]
            if lv >= 1 and k % 2 == 0:
                self.put(self.lp(self.bass(ch[0] - 12, step * 1.8), 900), t, 0, 0.14)
            if lv >= 2 and k % 4 == 0:
                self.put(self.kick(), t, 0, 0.42 if k % 16 in (0, 8) else 0.3)
            if lv >= 2 and k % 4 == 2:
                self.put(self.hat(), t, 0.3, 0.17)
            if lv >= 3:
                self.put(self.hat(open_=(k % 8 == 6)), t, -0.25, 0.08)
                arp = [ch[1], ch[2], ch[3], ch[4], ch[3] + 12, ch[2], ch[4], ch[1] + 12]
                self.put(self.pluck(arp[k % 8] + 12, 0.2, 1800 + 1400 * ((k % 8) / 7)), t, 0.5 * np.sin(k * 0.7), 0.08)
            k += 1

    def write(self, path):
        mix = np.stack([self.L, self.R], axis=1)
        peak = float(np.max(np.abs(mix))) or 1.0
        mix = mix / max(peak, 1.0) * 0.89
        with wave.open(path, "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes((mix * 32767).astype("<i2").tobytes())
        print(f"{path}: {self.length:.1f} s, peak {min(peak, 1.0) * 0.89:.2f}")
