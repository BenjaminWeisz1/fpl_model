// One heatmap colour map for every table (website_plan.md): the images' map, min-max scaled.
// #F8696B (worst) -> #FFEB84 -> #63BE7B (best); invert for columns where lower is better.
"use strict";

const Heatmap = (() => {
  const WORST = [0xF8, 0x69, 0x6B], MID = [0xFF, 0xEB, 0x84], BEST = [0x63, 0xBE, 0x7B];

  const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));

  // A scale over the values currently shown (nulls ignored).
  function scale(values, invert = false) {
    let min = Infinity, max = -Infinity;
    for (const v of values) {
      if (v === null || v === undefined || Number.isNaN(v)) continue;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    return { min, max, invert };
  }

  function colour(value, s) {
    if (value === null || value === undefined || !Number.isFinite(s.min)) return "";
    let t = s.max > s.min ? (value - s.min) / (s.max - s.min) : 0.5;
    if (s.invert) t = 1 - t;
    const rgb = t < 0.5 ? mix(WORST, MID, t * 2) : mix(MID, BEST, (t - 0.5) * 2);
    return `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
  }

  return { scale, colour };
})();
