// BLISS project page: 3D dataset explorer, intervention test P3, single- and two-defect shells, leaderboard,
// results across the protocols and the same-shell comparison. Data comes from static/data/*.json, written by
// scripts/build_explorer.py and scripts/build_data.py. Plot style: the base of the FLOATBench page.

// ------------------------------------------------------------------ theme --
const PAPER = {
  red: "#b02c27", lightRed: "#d69b99", middleRed: "#d06662", darkRed: "#932421", dark2Red: "#802421",
  blue: "#294366", lightBlue: "#ade1f4", middleBlue: "#7cc0cd", darkBlue: "#1c2b4a", grey: "#b8b8b8",
  greyDark: "#8f8f8f", lightGrey: "#f2f2f2", darkGrey: "#555555", lightBrown: "#8f7a6e", brown: "#66574e",
  deepRed: "#6f1b17",
};
const T = { ink: "#000000", ink2: PAPER.darkGrey, ink3: PAPER.darkGrey, line: PAPER.lightGrey };
// light grey (low) to dark red (high), as the damage scale of the paper figures
const RED_SEQ = [[0, PAPER.lightGrey], [0.2, PAPER.lightRed], [0.4, PAPER.middleRed], [0.6, PAPER.red],
                 [0.8, PAPER.darkRed], [1, PAPER.dark2Red]];
// input w: deep blue (deep defect) to light grey (sphere), as the defect maps of the paper
const BLUE_SEQ = [[0, "#0a1fb0"], [0.45, "#2a62e0"], [0.8, "#8fb3ec"], [1, PAPER.lightGrey]];
const MODEL_COL = { "FigConvNet": PAPER.darkBlue, "MeshGraphNet": PAPER.blue, "SFNO": PAPER.red,
  "Transformer": PAPER.middleRed, "Transolver": PAPER.middleBlue, "DeepONet": PAPER.brown, "MLP-20": PAPER.greyDark,
  "MLP-80": PAPER.grey, "SFNO-64x128": PAPER.lightRed };

function baseLayout(extra = {}) {
  return Object.assign({
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Helvetica, Arial, sans-serif", size: 13, color: T.ink2 },
    margin: { l: 58, r: 16, t: 36, b: 48 },
    hoverlabel: { font: { family: "Helvetica, Arial, sans-serif", size: 12 } },
    showlegend: false,
  }, extra);
}
function axis(title, extra = {}) {
  return Object.assign({ title: { text: title, font: { size: 12, color: T.ink } }, gridcolor: T.line,
    zerolinecolor: T.line, linecolor: T.line, tickfont: { color: T.ink3 } }, extra);
}
const PLOT_CFG = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d"] };
function render(id, traces, layout) {
  const el = document.getElementById(id);
  if (el && window.Plotly) Plotly.react(el, traces, layout, PLOT_CFG);
}
const $ = (id) => document.getElementById(id);
const getJSON = (f) => fetch("static/data/" + f).then(r => r.json());
function selectButton(groupId, btn) {
  document.querySelectorAll("#" + groupId + " .button").forEach(b => b.classList.remove("is-selected"));
  btn.classList.add("is-selected");
}

// ----------------------------------------------------------- explorer --
// Families of shells; every shell is drawn from its list of defects with the generator of the dataset,
// w = sum_i -delta_i t exp(-(beta_i / beta0_i)^2), beta0_i = lambda_i (12(1 - nu^2))^(-1/4) (R/t)^(-1/2).
const CATS = [
  { key: "sparse", label: "Multi-defect, sparse", fields: true },
  { key: "dense", label: "Multi-defect, dense", fields: true },
  { key: "p3", label: "Two defects moving closer (P3)", fields: true },
  { key: "double", label: "Two-defect", fields: false },
  { key: "single", label: "Single-defect", fields: false },
];
const state = { grid: null, geo: null, cat: null, shells: {}, catKey: "dense", sel: null, mode: "geom", exag: 10 };

function loadShell(id) {
  if (state.shells[id]) return Promise.resolve(state.shells[id]);
  return getJSON("shells/" + id + ".json").then(d => (state.shells[id] = d));
}
function items(key) {
  const C = state.cat;
  if (key === "sparse" || key === "dense") return C.core.filter(c => c.regime === key).sort((a, b) => a.n - b.n || a.dmax - b.dmax);
  if (key === "p3") return C.p3.slice().sort((a, b) => a.source - b.source || +b.spacing - +a.spacing);
  return C[key].slice();
}
function shellGeom(rec) {
  // R and slenderness of the block: core and P3 R = 25.4, R/t = 110; two-defect R/t = 100; single-defect R = 24.85, R/t = 108
  const R = rec.R || state.cat.R_core, eta = rec.eta || state.cat.eta_core;
  return { R: R, t: R / eta, nu: 0.5 };
}
function generateW(rec) {
  const g = state.grid, n = g.x.length, w = new Float64Array(n), sg = shellGeom(rec);
  const pref = Math.pow(12 * (1 - sg.nu * sg.nu), -0.25) * Math.pow(sg.R / sg.t, -0.5);
  for (const [d, l, th, ph] of rec.defects) {
    const cx = Math.sin(ph) * Math.cos(th), cy = Math.sin(ph) * Math.sin(th), cz = Math.cos(ph), b0 = l * pref;
    for (let i = 0; i < n; i++) {
      const c = Math.max(-1, Math.min(1, g.x[i] * cx + g.y[i] * cy + g.z[i] * cz)), b = Math.acos(c) / b0;
      if (b < 4) w[i] += -d * sg.t * Math.exp(-b * b);
    }
  }
  return w;
}
function deviationNodes(s, key = "u") {
  const g = state.geo, n = g.x.length, ux = s[key + "x"], uy = s[key + "y"], uz = s[key + "z"]; let ub = 0;
  for (let i = 0; i < n; i++) ub += ux[i] * g.x[i] + uy[i] * g.y[i] + uz[i] * g.z[i];
  ub /= n;
  const D = { x: new Float64Array(n), y: new Float64Array(n), z: new Float64Array(n), m: new Float64Array(n), site: 0 };
  for (let i = 0; i < n; i++) {
    D.x[i] = ux[i] - ub * g.x[i]; D.y[i] = uy[i] - ub * g.y[i]; D.z[i] = uz[i] - ub * g.z[i];
    D.m[i] = Math.hypot(D.x[i], D.y[i], D.z[i]); if (D.m[i] > D.m[D.site]) D.site = i;
  }
  return D;
}
function scene3d() {
  const ax = { visible: false, showbackground: false };
  return { xaxis: ax, yaxis: ax, zaxis: ax, aspectmode: "data", camera: { eye: { x: 1.2, y: -1.2, z: 1.05 } } };
}
function drawShell() {
  const rec = state.sel; if (!rec) return;
  const cat = CATS.find(c => c.key === state.catKey), collapse = (state.mode === "collapse" || state.mode === "after") && cat.fields;
  const g = state.grid, n = g.x.length, sg = shellGeom(rec), w = generateW(rec);
  const finish = (color, cs, cbar, disp, site) => {
    const X = new Float64Array(n), Y = new Float64Array(n), Z = new Float64Array(n), e = state.exag;
    for (let i = 0; i < n; i++) {
      const r = sg.R + (collapse ? w[i] : e * w[i]);
      X[i] = r * g.x[i] + (disp ? e * disp.x[i] : 0); Y[i] = r * g.y[i] + (disp ? e * disp.y[i] : 0); Z[i] = r * g.z[i] + (disp ? e * disp.z[i] : 0);
    }
    const tr = [{ type: "mesh3d", x: X, y: Y, z: Z, i: g.i, j: g.j, k: g.k, intensity: color, colorscale: cs,
      lighting: { ambient: 0.7, diffuse: 0.55, specular: 0.1, roughness: 0.7 }, flatshading: false,
      colorbar: { title: { text: cbar, font: { size: 11, color: T.ink2 } }, thickness: 10, len: 0.6, outlinewidth: 0, tickfont: { size: 10, color: T.ink3 } },
      hoverinfo: "skip" }];
    if (site !== null) tr.push({ type: "scatter3d", mode: "markers", x: [X[site] * 1.02], y: [Y[site] * 1.02], z: [Z[site] * 1.02],
      marker: { size: 7, color: PAPER.red, line: { color: "#fff", width: 1.5 } }, hoverinfo: "skip" });
    render("plot-3d", tr, baseLayout({ margin: { l: 0, r: 0, t: 6, b: 0 }, scene: scene3d() }));
  };
  if (!cat.fields) render("plot-lpf", [], baseLayout({ xaxis: axis("", { visible: false }), yaxis: axis("", { visible: false }),
    annotations: [{ text: "table rows only: the defect list and \u03ba, no load path", showarrow: false, font: { color: T.ink2, size: 12 } }],
    margin: { l: 10, r: 10, t: 10, b: 10 } }));
  else loadShell(rec.id).then(s => {
    const k = rec.kappa / s.lpf[s.peak], y = s.lpf.map(v => v * k), x = y.map((v, i) => i);
    const at = state.mode === "geom" ? 0 : state.mode === "collapse" ? s.peak : Math.min(s.peak + 3, y.length - 1);
    render("plot-lpf", [
      { type: "scatter", mode: "lines", x: x, y: y, line: { color: PAPER.blue, width: 2.5 }, hovertemplate: "increment %{x}: %{y:.3f}<extra></extra>" },
      { type: "scatter", mode: "markers", x: [s.peak], y: [y[s.peak]], marker: { size: 11, color: PAPER.red, symbol: "star" }, hovertemplate: "collapse: \u03ba = %{y:.3f}<extra></extra>" },
      { type: "scatter", mode: "markers", x: [at], y: [y[at]], marker: { size: 13, color: "rgba(0,0,0,0)", line: { color: "#000", width: 2 } }, hoverinfo: "skip" }],
      baseLayout({ xaxis: axis("solver increment"), yaxis: axis("load / classical load"), margin: { l: 58, r: 16, t: 10, b: 44 } }));
  });
  if (!collapse) {
    finish(Array.from(w), BLUE_SEQ, "w [mm]", null, null);
    $("view-how").innerHTML = "The geometry, drawn from the list of defects with the depth exaggerated &times;" + state.exag +
      ". Colour: the surface deviation w. Drag to rotate.";
  } else {
    loadShell(rec.id).then(s => {
      const D = deviationNodes(s, state.mode === "after" ? "m" : "u"), near = g.near, m = new Array(n), disp = { x: new Float64Array(n), y: new Float64Array(n), z: new Float64Array(n) };
      let site = 0;
      for (let i = 0; i < n; i++) { const j = near[i]; m[i] = D.m[j]; disp.x[i] = D.x[j]; disp.y[i] = D.y[j]; disp.z[i] = D.z[j]; if (j === D.site) site = i; }
      finish(m, RED_SEQ, "|D| [mm]", disp, site);
      $("view-how").innerHTML = (state.mode === "after" ? "After collapse, three solver increments past the peak: the buckle grows at the failure site"
        : "At collapse, the first limit point of the load path: the buckling deviation") + " (displacement exaggerated &times;" +
        state.exag + "), from the finite-element solution at the 8,192 scoring nodes. The red marker is the failure site.";
    });
  }
}
function drawCard() {
  const r = state.sel, k = state.catKey;
  const catlab = { sparse: "multi-defect, sparse (centres &ge; 25&deg; apart)", dense: "multi-defect, dense (centres &ge; 10&deg; apart)",
    p3: "P3 record: one pair brought to " + r.spacing + "&deg;", double: "two-defect shell", single: "single-defect shell" }[k];
  const rows = [["shell", r.id], ["family", catlab], ["&kappa;", r.kappa.toFixed(3)]];
  if (k === "sparse" || k === "dense") rows.push(["defects", r.n], ["deepest defect", r.dmax.toFixed(2) + " t"],
    ["site ratio q", r.q.toFixed(2) + (r.q > 0.9 ? " near-tie" : r.q < 0.7 ? " clear" : "")],
    ["splits", ["B1", "B2", "B3"].map(b => b + " " + (r.parts[b] || "&ndash;")).join(" &middot; ")]);
  if (k === "p3") { const fam = state.cat.p3.filter(q => q.source === r.source).sort((a, b) => +b.spacing - +a.spacing);
    rows.push(["source shell", r.source], ["spacing of the pair", r.spacing + "&deg;"],
      ["&kappa; at 25 / 20 / 15 / 12.5 / 10&deg;", fam.map(q => q.kappa.toFixed(3)).join(" / ")]); }
  if (k === "double") rows.push(["spacing", r.sep + "&deg;"], ["pair", r.same ? "identical" : "different depths"], ["R, R/t", r.R + " mm, " + r.eta]);
  if (k === "single") rows.push(["depth &delta;", r.delta + " t"], ["width &lambda;", r.lam], ["R, R/t", r.R + " mm, " + r.eta]);
  $("sim-card").innerHTML = rows.map(([a, b]) => "<div><dt>" + a + "</dt><dd>" + b + "</dd></div>").join("");
  const ds = r.defects.slice().sort((a, b) => b[0] - a[0]), show = ds.slice(0, 5);
  $("defect-table").innerHTML = "<thead><tr><th class='l'>defect</th><th>depth &delta; [t]</th><th>width &lambda;</th><th>&theta; [deg]</th><th>&phi; [deg]</th></tr></thead><tbody>" +
    show.map((d, i) => "<tr><td class='l'>" + (i + 1) + "</td><td>" + d[0].toFixed(3) + "</td><td>" + d[1] + "</td><td>" +
      (d[2] * 180 / Math.PI).toFixed(1) + "</td><td>" + (d[3] * 180 / Math.PI).toFixed(1) + "</td></tr>").join("") +
    (ds.length > 5 ? "<tr><td class='l' colspan='5'>&hellip; and " + (ds.length - 5) + " more defects</td></tr>" : "") + "</tbody>";
}
function drawCatPlot() {
  const k = state.catKey, L = items(k), sel = state.sel, big = (r) => r.id === sel.id ? 15 : 9;
  let tr, lay;
  if (k === "sparse" || k === "dense") {
    tr = [{ type: "scatter", mode: "markers", x: L.map(r => r.n), y: L.map(r => r.dmax), customdata: L.map(r => r.id),
      text: L.map(r => "shell " + r.id + ", κ = " + r.kappa.toFixed(3)), hovertemplate: "%{text}<extra></extra>",
      marker: { size: L.map(big), color: L.map(r => r.kappa), colorscale: RED_SEQ, reversescale: true, cmin: 0.35, cmax: 0.62,
        symbol: L.map(r => r.q > 0.9 ? "diamond" : "circle"), line: { color: L.map(r => r.id === sel.id ? "#000" : "#fff"), width: 1.2 },
        colorbar: { title: { text: "κ", font: { size: 11 } }, thickness: 8, len: 0.9, outlinewidth: 0 } } }];
    lay = { xaxis: axis("number of defects"), yaxis: axis("deepest defect / t") };
  } else if (k === "p3") {
    const srcs = [...new Set(L.map(r => r.source))], pal = [PAPER.darkBlue, PAPER.blue, PAPER.middleBlue, PAPER.middleRed, PAPER.red];
    tr = srcs.map((s, i) => { const r = L.filter(q => q.source === s);
      return { type: "scatter", mode: "lines+markers", name: "source " + s, x: r.map(q => +q.spacing), y: r.map(q => q.kappa), customdata: r.map(q => q.id),
        line: { color: pal[i % 5], width: 2 }, marker: { size: r.map(big), color: pal[i % 5], line: { color: r.map(q => q.id === sel.id ? "#000" : "#fff"), width: 1.2 } },
        hovertemplate: "source " + s + ", %{x}°: κ = %{y:.3f}<extra></extra>" }; });
    lay = { xaxis: axis("spacing of the pair [deg]", { autorange: "reversed" }), yaxis: axis("κ") };
  } else if (k === "double") {
    tr = [{ type: "scatter", mode: "markers", x: L.map(r => r.sep), y: L.map(r => r.kappa), customdata: L.map(r => r.id),
      marker: { size: L.map(big), color: L.map(r => r.same ? PAPER.blue : PAPER.red), symbol: L.map(r => r.same ? "circle" : "diamond"),
        line: { color: L.map(r => r.id === sel.id ? "#000" : "#fff"), width: 1.2 } }, hovertemplate: "shell %{customdata}: %{x}°, κ = %{y:.3f}<extra></extra>" }];
    lay = { xaxis: axis("spacing of the two centres [deg]"), yaxis: axis("κ") };
  } else {
    const lams = [...new Set(L.map(r => r.lam))].sort((a, b) => a - b), pal = [PAPER.darkBlue, PAPER.blue, PAPER.middleBlue, PAPER.middleRed, PAPER.red];
    tr = lams.map((l, i) => { const r = L.filter(q => q.lam === l).sort((a, b) => a.delta - b.delta);
      return { type: "scatter", mode: "lines+markers", name: "λ = " + l, x: r.map(q => q.delta), y: r.map(q => q.kappa), customdata: r.map(q => q.id),
        line: { color: pal[i % 5], width: 2 }, marker: { size: r.map(big), color: pal[i % 5], line: { color: r.map(q => q.id === sel.id ? "#000" : "#fff"), width: 1.2 } },
        hovertemplate: "λ = " + l + ", δ = %{x}: κ = %{y:.3f}<extra></extra>" }; });
    lay = { xaxis: axis("depth δ / t"), yaxis: axis("κ") };
  }
  render("plot-cat", tr, baseLayout(Object.assign({ margin: { l: 52, r: 8, t: 8, b: 42 } }, lay)));
  const el = $("plot-cat");
  if (!el._bound) { el._bound = true; el.on("plotly_click", ev => { const id = ev.points[0].customdata; selectShell(items(state.catKey).find(r => r.id === id)); }); }
}
function drawCatTable() {
  const k = state.catKey, L = items(k);
  const head = { sparse: ["shell", "defects", "deepest [t]", "κ", "q"], dense: ["shell", "defects", "deepest [t]", "κ", "q"],
    p3: ["record", "source", "spacing", "κ"], double: ["shell", "spacing", "pair", "κ"], single: ["shell", "δ [t]", "λ", "κ"] }[k];
  const cells = (r) => ({ sparse: [r.id, r.n, r.dmax.toFixed(2), r.kappa.toFixed(3), r.q.toFixed(2)],
    dense: [r.id, r.n, r.dmax.toFixed(2), r.kappa.toFixed(3), r.q.toFixed(2)], p3: [r.id, r.source, r.spacing + "°", r.kappa.toFixed(3)],
    double: [r.id, r.sep + "°", r.same ? "identical" : "different", r.kappa.toFixed(3)], single: [r.id, r.delta, r.lam, r.kappa.toFixed(3)] }[k]);
  $("cat-table").innerHTML = "<thead><tr>" + head.map((h, i) => "<th" + (i ? "" : " class='l'") + ">" + h + "</th>").join("") + "</tr></thead><tbody>" +
    L.map(r => "<tr data-id='" + r.id + "'" + (r.id === state.sel.id ? " class='sel'" : "") + ">" + cells(r).map((c, i) => "<td" + (i ? "" : " class='l'") + ">" + c + "</td>").join("") + "</tr>").join("") + "</tbody>";
  $("cat-table").querySelectorAll("tbody tr").forEach(tr => tr.addEventListener("click", () => selectShell(L.find(r => r.id === +tr.dataset.id))));
}
function selectShell(rec) {
  state.sel = rec; drawCatPlot(); drawCatTable(); drawCard(); drawShell();
}
function selectCat(key) {
  state.catKey = key;
  document.querySelectorAll("#cat-tabs .button").forEach(b => b.classList.toggle("is-selected", b.dataset.k === key));
  const cat = CATS.find(c => c.key === key), cb = document.querySelector("#mode3d .button[data-v='collapse']");
  cb.disabled = !cat.fields; if (!cat.fields && state.mode === "collapse") { state.mode = "geom"; selectButton("mode3d", document.querySelector("#mode3d .button[data-v='geom']")); }
  const L = items(key), pick = key === "dense" ? L.find(r => r.id === 6337) : key === "p3" ? L.find(r => r.source === 3890 && +r.spacing === 10) : null;
  selectShell(pick || L[0]);
}
function initExplorer() {
  const tabs = $("cat-tabs");
  CATS.forEach(c => { const b = document.createElement("button"); b.className = "button is-small"; b.dataset.k = c.key;
    b.textContent = c.label + " (" + items(c.key).length + ")"; b.addEventListener("click", () => selectCat(c.key)); tabs.appendChild(b); });
  document.querySelectorAll("#mode3d .button").forEach(b => b.addEventListener("click", () => {
    if (b.disabled) return; selectButton("mode3d", b); state.mode = b.dataset.v; drawShell(); }));
  $("exag").addEventListener("input", e => { state.exag = +e.target.value; $("exag-lab").textContent = state.exag; drawShell(); });
  $("exag-lab").textContent = state.exag;
  const find = () => { const id = +$("shell-find").value.trim();
    for (const c of CATS) { const r = items(c.key).find(q => q.id === id); if (r) { selectCat(c.key); selectShell(r); return; } }
    $("shell-find").classList.add("is-danger"); setTimeout(() => $("shell-find").classList.remove("is-danger"), 1200); };
  $("shell-go").addEventListener("click", find); $("shell-find").addEventListener("keydown", e => { if (e.key === "Enter") find(); });
  selectCat("dense");
}

// -------------------------------------------------------- leaderboard --
// Table 2 of the paper. Per protocol: MRE kappa [%], buckling R2, RMSE U [mm], rel. L2 U [%], clear-site [%].
// A trailing "*" marks the best learned model on 8,192 nodes (bold in the paper).
const LB = [
  ["references", null],
  ["training mean", [["5.13", "0.237", "0.0339", "88", "0"], ["8.33", "0.221", "0.0443", "85", "0"], ["5.11", "0.340", "0.0285", "82", "0"]]],
  ["weakest-link rule", [["2.49", "--", "--", "--", "--"], ["2.72", "--", "--", "--", "--"], ["5.75", "--", "--", "--", "--"]]],
  ["depth rules ‡", [["--", "--", "--", "--", "64"], ["--", "--", "--", "--", "24"], ["--", "--", "--", "--", "43"]]],
  ["nearest neighbour on w", [["4.79", "0.773", "0.0185", "50", "57"], ["3.88", "0.806", "0.0221", "44", "59"], ["4.86", "−0.270", "0.0396", "104", "21"]]],
  ["solver, one step before", [["--", "0.993", "--", "--", "100"], ["--", "0.994", "--", "--", "98"], ["--", "0.990", "--", "--", "99"]]],
  ["solver, one step after", [["--", "0.966", "--", "--", "100"], ["--", "0.966", "--", "--", "100"], ["--", "0.931", "--", "--", "100"]]],
  ["8,192 nodes", null],
  ["FigConvNet", [["1.04", "0.985*", "0.0048*", "13*", "96"], ["1.47", "0.979*", "0.0073*", "15*", "78"], ["6.68", "−1.308", "0.0534", "129", "31"]]],
  ["MeshGraphNet", [["0.70", "0.968", "0.0069", "20", "89"], ["1.25", "0.937", "0.0126", "25", "59"], ["5.99", "−1.105", "0.0510", "119", "32*"]]],
  ["SFNO", [["0.58", "0.934", "0.0100", "26", "98*"], ["1.29", "0.915", "0.0146", "28", "75"], ["9.37", "−0.587", "0.0442", "120", "30"]]],
  ["SFNO-64×128 ∗", [["0.53*", "0.895", "0.0126", "28", "98*"], ["1.75", "0.812", "0.0218", "38", "82*"], ["6.16", "−0.713", "0.0460", "125", "30"]]],
  ["Transformer", [["0.65", "0.969", "0.0068", "17", "88"], ["1.08*", "0.957", "0.0104", "21", "60"], ["6.45", "−0.589", "0.0442", "119", "32*"]]],
  ["Transolver", [["0.91", "0.922", "0.0108", "27", "72"], ["1.62", "0.921", "0.0141", "30", "31"], ["6.80", "−1.251", "0.0525", "138", "31"]]],
  ["DeepONet", [["1.32", "0.764", "0.0189", "51", "62"], ["2.38", "0.752", "0.0250", "53", "53"], ["4.72*", "0.295*", "0.0295*", "82*", "25"]]],
  ["full mesh, 76,805 nodes (scored on the 8,192)", null],
  ["FigConvNet", [["0.91", "0.987", "0.0045", "12", "97"], ["", "", "", "", ""], ["", "", "", "", ""]]],
  ["MeshGraphNet", [["0.82", "0.923", "0.0107", "29", "69"], ["", "", "", "", ""], ["", "", "", "", ""]]],
  ["SFNO", [["0.58", "0.890", "0.0129", "30", "99"], ["1.83", "0.847", "0.0196", "35", "80"], ["7.89", "−0.539", "0.0434", "119", "28"]]],
  ["Transolver", [["0.67", "0.928", "0.0104", "25", "76"], ["1.36", "0.925", "0.0137", "29", "32"], ["6.46", "−0.202", "0.0383", "104", "38"]]],
  ["list of defects (parametric input)", null],
  ["MLP-20 †", [["1.19", "0.816", "0.0166", "43", "71"], ["3.14", "0.345", "0.0394", "67", "39"], ["div.", "div.", "div.", "div.", "5"]]],
  ["MLP-80 †", [["1.20", "0.817", "0.0166", "43", "72"], ["2.36", "0.720", "0.0265", "55", "46"], ["div.", "div.", "div.", "div.", "2"]]],
  ["XGBoost †", [["0.72", "--", "--", "--", "--"], ["1.69", "--", "--", "--", "--"], ["6.50", "--", "--", "--", "--"]]],
];
const LB_HEAD = ["MRE κ [%]", "buckling R²", "RMSE U [mm]", "rel. L² U [%]", "clear site [%]"];
let LB_SORT = { col: -1, dir: 1 }, LB_P = 0, FULL = null;
const numOf = (c) => { const v = parseFloat(String(c).replace("−", "-").replace("*", "")); return isNaN(v) ? null : v; };

function drawLeaderboard(p) {
  LB_P = p;
  let h = "<thead><tr><th class='l'>model</th>" + LB_HEAD.map((x, i) => "<th class='sortable' data-c='" + i + "'>" + x +
    (LB_SORT.col === i ? (LB_SORT.dir > 0 ? " ▲" : " ▼") : "") + "</th>").join("") + "</tr></thead><tbody>";
  let rows = [];
  const flush = () => {
    if (LB_SORT.col >= 0) rows.sort((a, b) => { const x = numOf(a[1][LB_SORT.col]), y = numOf(b[1][LB_SORT.col]);
      if (x === null) return 1; if (y === null) return -1; return LB_SORT.dir * (x - y); });
    for (const [name, r] of rows) h += "<tr data-m='" + name + "'" + (RANK.model && modelOf(name) === RANK.model ? " class='sel'" : "") + "><td class='l'>" + name + "</td>" + r.map(c => { const best = c.endsWith("*");
      return "<td" + (best ? " class='best'" : "") + ">" + (best ? c.slice(0, -1) : c) + "</td>"; }).join("") + "</tr>";
    rows = [];
  };
  for (const [name, rs] of LB) {
    if (rs === null) { flush(); h += "<tr class='group'><td class='l' colspan='6'>" + name + "</td></tr>"; continue; }
    if (!rs[p].every(c => c === "")) rows.push([name, rs[p]]);
  }
  flush();
  $("lb").innerHTML = h + "</tbody>";
  $("lb").querySelectorAll("th.sortable").forEach(th => th.addEventListener("click", () => {
    const c = +th.dataset.c; LB_SORT = { col: c, dir: LB_SORT.col === c ? -LB_SORT.dir : -1 }; drawLeaderboard(LB_P); }));
  $("lb").querySelectorAll("tbody tr[data-m]").forEach(tr => tr.addEventListener("click", () => selectModel(tr.dataset.m)));
}

const FULL_HEAD = { kappa_mre: "MRE κ [%]", kappa_r2: "κ R²", unconservative: "unconserv.", buckling_r2: "buckling R²",
  field_r2: "field R²", defect_r2: "defect-region R²", rmse: "RMSE [mm]", rmse_noshrink: "RMSE w/o shrink",
  rel_l2: "rel. L² mean / median", rel_l2_noshrink: "rel. L² w/o shrink", clear_site: "clear site", near_tie: "near-tie winner",
  sparse_kappa: "sparse κ err", sparse_r2: "sparse bkl. R²", dense_kappa: "dense κ err", dense_r2: "dense bkl. R²" };
const fmtCell = (c) => typeof c === "string" ? c : c.length === 3 ? c[0] + " <span class='ci'>[" + c[1] + ", " + c[2] + "]</span>" : String(c[0]);
function drawFull(p) {
  if (!FULL) return;
  const rows = FULL.protocols[["B1", "B2", "B3"][p]], cols = FULL.columns.filter(c => rows.some(r => r.cells[c] !== undefined));
  let h = "<thead><tr><th class='l'>model</th>" + cols.map(c => "<th>" + FULL_HEAD[c] + "</th>").join("") + "</tr></thead><tbody>", g = null;
  for (const r of rows) {
    if (r.group !== g) { g = r.group; h += "<tr class='group'><td class='l' colspan='" + (cols.length + 1) + "'>" + g + "</td></tr>"; }
    h += "<tr><td class='l'>" + r.name + "</td>" + cols.map(c => "<td>" + (r.cells[c] === undefined ? "" : fmtCell(r.cells[c])) + "</td>").join("") + "</tr>";
  }
  $("lb-full").innerHTML = h + "</tbody>";
}

// ------------------------------------------------------------ ranking --
const RANK = { metric: "buckling_r2", model: null };
const HIGHER = { buckling_r2: true, kappa_mre: false, clear_site: true, near_tie: true };
function rankColor(r) {
  if (r.group === "references") return PAPER.grey;
  if (r.group === "solver") return PAPER.greyDark;
  if (r.group === "models on the list of defects") return PAPER.lightBrown;
  return MODEL_COL[r.name.split(" (")[0]] || PAPER.blue;
}
function drawRank() {
  if (!FULL) return;
  const m = RANK.metric, pct = (m === "clear_site" || m === "near_tie") ? 100 : 1;
  const rows = FULL.protocols[["B1", "B2", "B3"][LB_P]].filter(r => r.cells[m] && typeof r.cells[m] !== "string");
  rows.sort((a, b) => (HIGHER[m] ? 1 : -1) * (a.cells[m][0] - b.cells[m][0]));
  const name = (r) => r.name.split(" (")[0].replace("\n", " ");
  const tr = [{ type: "bar", orientation: "h", y: rows.map(name), x: rows.map(r => r.cells[m][0] * pct), customdata: rows.map(r => r.name),
    marker: { color: rows.map(rankColor), line: { color: rows.map(r => r.name === RANK.model ? "#000" : "rgba(0,0,0,0)"), width: 2.5 } },
    error_x: { type: "data", symmetric: false, array: rows.map(r => r.cells[m].length === 3 ? (r.cells[m][2] - r.cells[m][0]) * pct : 0),
      arrayminus: rows.map(r => r.cells[m].length === 3 ? (r.cells[m][0] - r.cells[m][1]) * pct : 0), color: PAPER.darkGrey, thickness: 1 },
    hovertemplate: "%{y}: %{x:.3f}<extra></extra>" }];
  const xl = { buckling_r2: "buckling R\u00b2", kappa_mre: "median relative error of \u03ba [%]", clear_site: "clear-site accuracy [%]", near_tie: "near-tie winner rate [%]" }[m];
  render("plot-rank", tr, baseLayout({ xaxis: axis(xl, { type: m === "kappa_mre" ? "log" : "linear" }),
    yaxis: axis("", { automargin: true, tickfont: { size: 11, color: T.ink3 } }), margin: { l: 10, r: 16, t: 10, b: 48 } }));
  const el = $("plot-rank");
  if (!el._bound) { el._bound = true; el.on("plotly_click", ev => { RANK.model = ev.points[0].customdata; drawRank(); drawModelCard(); drawLeaderboard(LB_P); drawLadder(); }); }
}
function drawModelCard() {
  const n = RANK.model; if (!n || !FULL) { $("model-card").innerHTML = "<p class='how'>Click a model in the ranking.</p>"; return; }
  const P = ["B1", "B2", "B3"], get = (p, c) => { const r = FULL.protocols[p].find(x => x.name === n); const v = r && r.cells[c];
    return (v && typeof v !== "string") ? v[0] : null; };
  const f = (v, c) => v === null ? "&ndash;" : (c === "clear_site" || c === "near_tie") ? (100 * v).toFixed(0) + "%" : v.toFixed(c === "kappa_mre" ? 2 : 3);
  const cols = [["kappa_mre", "MRE \u03ba [%]"], ["buckling_r2", "buckling R\u00b2"], ["clear_site", "clear site"], ["near_tie", "near-tie"]];
  let h = "<p class='result-tag'>model</p><h4 class='title is-5'>" + n.split(" (")[0] + "</h4><table class='table is-narrow is-fullwidth data'><thead><tr><th class='l'></th>" +
    cols.map(c => "<th>" + c[1] + "</th>").join("") + "</tr></thead><tbody>" +
    P.map(p => "<tr><td class='l'><b>" + p + "</b></td>" + cols.map(c => "<td>" + f(get(p, c[0]), c[0]) + "</td>").join("") + "</tr>").join("") + "</tbody></table>";
  const sp = P.map(p => [get(p, "sparse_r2"), get(p, "dense_r2"), get(p, "sparse_kappa"), get(p, "dense_kappa")]);
  h += "<p class='is-size-7 has-text-weight-semibold'>by regime of the test shells</p><table class='table is-narrow is-fullwidth data'><thead><tr><th class='l'></th><th>sparse bkl. R\u00b2</th><th>dense bkl. R\u00b2</th><th>sparse \u03ba err</th><th>dense \u03ba err</th></tr></thead><tbody>" +
    P.map((p, i) => "<tr><td class='l'><b>" + p + "</b></td>" + sp[i].map((v, j) => "<td>" + (v === null ? "&ndash;" : v.toFixed(j < 2 ? 3 : 2)) + "</td>").join("") + "</tr>").join("") + "</tbody></table>";
  const b1 = get("B1", "buckling_r2"), b3 = get("B3", "buckling_r2");
  if (b1 !== null && b3 !== null) h += "<p class='takeaway-live'>Buckling R\u00b2 goes from " + b1.toFixed(3) + " on B1 to " + b3.toFixed(3) +
    " on B3" + (b3 < 0.34 ? ", below the training mean (0.340)." : ".") + "</p>";
  $("model-card").innerHTML = h;
}
function modelOf(shortName) {
  // the full-table row of a Table 2 row name
  if (!FULL) return null;
  const rows = FULL.protocols.B1, base = shortName.replace(/ [\u2020\u2217\u2021]$/, "").trim();
  if (base.startsWith("SFNO-64")) { const r = rows.find(x => x.name.startsWith("SFNO-64")); return r ? r.name : null; }
  const r = rows.find(x => x.name === base) || rows.find(x => x.name.split(" (")[0] === base) ||
    rows.find(x => x.name.toLowerCase().startsWith(base.toLowerCase()));
  return r ? r.name : null;
}
function selectModel(shortName) {
  const m = modelOf(shortName); if (!m) return;
  RANK.model = m; drawRank(); drawModelCard(); drawLeaderboard(LB_P); drawLadder();
}
function initRank() {
  document.querySelectorAll("#rank-metric .button").forEach(b => b.addEventListener("click", () => { selectButton("rank-metric", b); RANK.metric = b.dataset.m; drawRank(); }));
  RANK.model = FULL.protocols.B1[0].name; drawRank(); drawModelCard();
}

// ------------------------------------------------------------- ladder --
const LADDER = { metric: "buckling_r2", hidden: new Set() };
function ladderNames() {
  return FULL.protocols.B1.filter(r => r.group === "learned field models" || r.name.startsWith("MLP")).map(r => r.name.split(" (")[0]);
}
function drawLadder() {
  if (!FULL) return;
  const m = LADDER.metric, P = ["B1", "B2", "B3"], X = ["B1 within distribution", "B2 deeper defects", "B3 sparse to dense"];
  const pct = (m === "clear_site" || m === "near_tie") ? 100 : 1;
  const val = (p, name) => { const r = FULL.protocols[p].find(x => x.name.split(" (")[0] === name); const c = r && r.cells[m];
    return (c && typeof c !== "string") ? c[0] * pct : null; };
  const selShort = RANK.model ? RANK.model.split(" (")[0] : null;
  const tr = ladderNames().filter(n => !LADDER.hidden.has(n)).map(n => ({ type: "scatter", mode: "lines+markers", name: n, x: X,
    y: P.map(p => val(p, n)), opacity: (!selShort || selShort === n || !ladderNames().includes(selShort)) ? 1 : 0.25,
    line: { width: selShort === n ? 6 : 3, color: MODEL_COL[n.replace("SFNO-64×128", "SFNO-64x128").replace("$", "")] || PAPER.greyDark },
    marker: { size: selShort === n ? 13 : 9 }, hovertemplate: n + ": %{y:.3f}<extra></extra>" }));
  tr.push({ type: "scatter", mode: "lines", name: "training mean", x: X, y: P.map(p => val(p, "training mean")),
    line: { dash: "dash", color: PAPER.greyDark, width: 2 }, hovertemplate: "training mean: %{y:.3f}<extra></extra>" });
  const ylab = { buckling_r2: "buckling R²", kappa_mre: "median relative error of κ [%]",
    clear_site: "clear-site accuracy [%]", near_tie: "near-tie winner rate [%]" }[m];
  render("plot-ladder", tr, baseLayout({ showlegend: true, legend: { orientation: "h", x: 0, y: -0.15 },
    xaxis: axis(""), yaxis: axis(ylab, { type: m === "kappa_mre" ? "log" : "linear" }), margin: { l: 58, r: 16, t: 14, b: 78 } }));
}
function initLadder() {
  const chips = $("ladder-chips");
  ladderNames().forEach(n => {
    const b = document.createElement("button"); b.className = "chip"; b.setAttribute("aria-pressed", "true");
    b.innerHTML = "<span class='dot' style='background:" + (MODEL_COL[n] || PAPER.greyDark) + "'></span>" + n;
    b.addEventListener("click", () => { const on = !LADDER.hidden.has(n); on ? LADDER.hidden.add(n) : LADDER.hidden.delete(n);
      b.setAttribute("aria-pressed", String(!on)); drawLadder(); });
    chips.appendChild(b);
  });
  document.querySelectorAll("#ladder-tabs .button").forEach(b => b.addEventListener("click", () => {
    selectButton("ladder-tabs", b); LADDER.metric = b.dataset.m; drawLadder(); }));
  drawLadder();
}

// ------------------------------------------------------ same-shell shift --
let PR = null;
function ramp(t) {
  t = Math.max(0, Math.min(1, t));
  const stops = [[242, 242, 242], [214, 155, 153], [208, 102, 98], [176, 44, 39], [147, 36, 33], [128, 36, 33]];
  const x = t * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(x)), u = x - i;
  return "rgb(" + stops[i].map((v, j) => Math.round(v + (stops[i + 1][j] - v) * u)).join(",") + ")";
}
function paint(canvas, d, vmax, site, trueSite) {
  const W = 220, ctx = canvas.getContext("2d"); canvas.width = W; canvas.height = W;
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, W, W);
  const X = PR.x, Y = PR.y, s = W / 2 - 6, order = d.map((v, i) => i).sort((i, j) => d[i] - d[j]);
  for (const i of order) { ctx.fillStyle = ramp(d[i] / vmax); ctx.fillRect(W / 2 + X[i] * s - 1.3, W / 2 - Y[i] * s - 1.3, 2.6, 2.6); }
  const mark = (i, kind) => { const x = W / 2 + X[i] * s, y = W / 2 - Y[i] * s; ctx.lineWidth = 2.5; ctx.strokeStyle = PAPER.darkBlue;
    ctx.beginPath(); if (kind === "o") ctx.arc(x, y, 9, 0, 2 * Math.PI); else { ctx.moveTo(x - 7, y - 7); ctx.lineTo(x + 7, y + 7); ctx.moveTo(x + 7, y - 7); ctx.lineTo(x - 7, y + 7); }
    ctx.stroke(); };
  mark(trueSite, "x"); if (site !== null) mark(site, "o");
}
function angDeg(i, j) {
  const zi = Math.sqrt(Math.max(0, 1 - PR.x[i] ** 2 - PR.y[i] ** 2)), zj = Math.sqrt(Math.max(0, 1 - PR.x[j] ** 2 - PR.y[j] ** 2));
  return Math.acos(Math.max(-1, Math.min(1, PR.x[i] * PR.x[j] + PR.y[i] * PR.y[j] + zi * zj))) * 180 / Math.PI;
}
function showShift(k) {
  const sh = PR.shells[k], vmax = Math.max(...sh.truth.d);
  document.querySelectorAll("#shift-shells .button").forEach((b, i) => b.classList.toggle("is-selected", i === k));
  $("shift-truth").innerHTML = "<div class='rowlab'>solver</div><div class='panel-c'><canvas></canvas><div class='cap'>&kappa; = " + sh.truth.kappa.toFixed(3) + "</div></div>";
  paint($("shift-truth").querySelector("canvas"), sh.truth.d, vmax, null, sh.truth.site);
  const g = $("shift-grid"); g.innerHTML = "<div></div>" + PR.models.map(m => "<div class='collab'>" + m + "</div>").join("");
  for (const prot of ["B1", "B3"]) {
    g.insertAdjacentHTML("beforeend", "<div class='rowlab'>trained on " + prot + "<br><span style='font-weight:400'>" +
      (prot === "B1" ? "dense shells seen" : "sparse shells only") + "</span></div>");
    for (const m of PR.models) {
      const r = sh.models[m][prot], a = angDeg(r.site, sh.truth.site), ok = a < 10;
      const err = 100 * (r.kappa - sh.truth.kappa) / sh.truth.kappa, div = !isFinite(err) || Math.abs(err) > 1000;
      g.insertAdjacentHTML("beforeend", "<div class='panel-c'><canvas></canvas><div class='cap'>" + (div ? "<span class='bad'>diverged</span><br>&nbsp;" :
        "<span class='" + (ok ? "ok" : "bad") + "'>site " + (ok ? "✓" : "off by " + a.toFixed(0) + "°") + "</span><br>&kappa; " +
        (err >= 0 ? "+" : "") + err.toFixed(1) + "%") + "</div></div>");
      paint(g.lastElementChild.querySelector("canvas"), r.d, vmax, r.site, sh.truth.site);
    }
  }
}

// --------------------------------------------------------------- init --
document.addEventListener("DOMContentLoaded", () => {
  drawLeaderboard(0);
  document.querySelectorAll("#lb-tabs .button").forEach(b => b.addEventListener("click", () => {
    selectButton("lb-tabs", b); drawLeaderboard(+b.dataset.p); drawFull(+b.dataset.p); drawRank(); }));
  getJSON("leaderboard.json").then(d => { FULL = d; drawFull(0); initLadder(); initRank(); });
  Promise.all([getJSON("grid.json"), getJSON("geometry.json"), getJSON("catalog.json")]).then(([gr, g, c]) => {
    state.grid = gr; state.geo = g; state.cat = c; initExplorer();
  });
  getJSON("predictions.json").then(d => {
    PR = d;
    d.shells.forEach((s, i) => { const b = document.createElement("button"); b.className = "button is-small"; b.textContent = "dense shell " + s.id;
      b.addEventListener("click", () => showShift(i)); $("shift-shells").appendChild(b); });
    showShift(0);
  });
});
