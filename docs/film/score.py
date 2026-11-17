"""Pioneer Home film score: airy and light. A rise as the walk begins, soft ticks while the room fills,
a pluck as the map settles, one bright note when Pioneer is found, a lift into the app and the end."""
from synth import Score

s = Score(length=24.0, bpm=112)
s.groove(levels=[(0.0, 1), (3.4, 2), (6.8, 1), (11.6, 2), (16.4, 3), (20.6, 1), (23.4, 0)])
s.put(s.whoosh(0.9, 200, 5000), 0.0, 0, 0.08)
for j in range(14):                                    # the room filling in
    s.put(s.tick(2400 + 180 * (j % 4), 0.018), 0.6 + j * 0.4, 0.3 * ((j % 2) * 2 - 1), 0.035)
s.put(s.pluck(57, 0.6, 1200), 6.8, 0, 0.11)            # the map settles
s.put(s.whoosh(0.5, 400, 6000), 11.2, 0, 0.07)
s.put(s.pluck(69, 0.5, 1700), 11.7, 0, 0.12)           # Pioneer is found
s.put(s.whoosh(0.6, 300, 7000), 16.0, 0, 0.1)          # the app rises
s.put(s.hit(0.55), 16.9, 0)
s.put(s.whoosh(0.6, 300, 6000), 20.3, 0, 0.08)
s.put(s.hit(0.6), 20.8, 0)                             # the end card
s.write("score.wav")
