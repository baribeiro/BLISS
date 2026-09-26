// BLISS project page: dataset explorer and leaderboard. Numbers of the leaderboard are Table 2 of the paper.

// ----------------------------------------------------------------------------------------------------------------
// Leaderboard (Table 2). Per protocol: MRE kappa [%], buckling R2, RMSE U [mm], rel. L2 U [%], clear-site [%].
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

function drawLeaderboard(p) {
  const t = document.getElementById("lb");
  let h = "<thead><tr><th>model</th>" + LB_HEAD.map(x => "<th class='has-text-right'>" + x + "</th>").join("") + "</tr></thead><tbody>";
  for (const [name, rows] of LB) {
    if (rows === null) { h += "<tr class='group'><td colspan='6'>" + name + "</td></tr>"; continue; }
    const r = rows[p];
    if (r.every(c => c === "")) continue;
    h += "<tr><td>" + name + "</td>" + r.map(c => {
      const best = c.endsWith("*");
      return "<td class='has-text-right" + (best ? " best" : "") + "'>" + (best ? c.slice(0, -1) : c) + "</td>";
    }).join("") + "</tr>";
  }
  t.innerHTML = h + "</tbody>";
}

document.querySelectorAll("#lb-tabs button").forEach(b => b.addEventListener("click", () => {
  document.querySelectorAll("#lb-tabs button").forEach(x => x.classList.remove("is-dark"));
  b.classList.add("is-dark");
}));


// ----------------------------------------------------------------------------------------------------------------
// Dataset explorer: shells of the review subset seen from the pole, w and the buckling deviation |D|.
let EX = null, MODE = "p3", SRC = null;
const LAYOUT = (title) => ({
  title: { text: title, font: { size: 13 } }, margin: { l: 10, r: 10, t: 36, b: 10 },
  xaxis: { visible: false, scaleanchor: "y" }, yaxis: { visible: false },
  plot_bgcolor: "rgba(0,0,0,0)", paper_bgcolor: "rgba(0,0,0,0)", showlegend: false,
});

function draw(rec) {
  const site = rec.site;
  const tw = { x: EX.x, y: EX.y, mode: "markers", type: "scatter",
    marker: { size: 3, color: rec.w, colorscale: [[0, "#0a1fb0"], [0.5, "#8fb3ec"], [1, "#e9ecf1"]],
              colorbar: { title: { text: "w [mm]", side: "right" }, thickness: 10 } }, hoverinfo: "skip" };
  const td = { x: EX.x, y: EX.y, mode: "markers", type: "scatter",
    marker: { size: 3, color: rec.d, colorscale: [[0, "#f7f3f2"], [0.5, "#e07a5f"], [1, "#7a1f1b"]],
              colorbar: { title: { text: "|D| [mm]", side: "right" }, thickness: 10 } }, hoverinfo: "skip" };
  const ts = { x: [EX.x[site]], y: [EX.y[site]], mode: "markers", type: "scatter",
    marker: { size: 16, symbol: "circle-open", color: "#1c2b4a", line: { width: 3 } }, hoverinfo: "skip" };
  Plotly.react("plot-w", [tw], LAYOUT("input w"), { displayModeBar: false, responsive: true });
  Plotly.react("plot-d", [td, ts], LAYOUT("buckling deviation at the peak, failure site circled"), { displayModeBar: false, responsive: true });
}

function angle(i, j) {
  // angle between two nodes from their projected coordinates on the unit hemisphere
  const zi = Math.sqrt(Math.max(0, 1 - EX.x[i] ** 2 - EX.y[i] ** 2)), zj = Math.sqrt(Math.max(0, 1 - EX.x[j] ** 2 - EX.y[j] ** 2));
  const c = EX.x[i] * EX.x[j] + EX.y[i] * EX.y[j] + zi * zj;
  return Math.acos(Math.max(-1, Math.min(1, c))) * 180 / Math.PI;
}

function showP3(k) {
  const r = EX.p3[SRC][k], r0 = EX.p3[SRC][0];
  document.getElementById("p3-label").textContent = r.spacing + "°";
  const drop = 100 * (r.kappa - r0.kappa) / r0.kappa;
  document.getElementById("readout").textContent =
    "κ = " + r.kappa.toFixed(3) + "  (" + (drop >= 0 ? "+" : "") + drop.toFixed(1) + "% from 25°)   ·   failure site moved " +
    angle(r.site, r0.site).toFixed(0) + "° from its position at 25°";
  draw(r);
}

function showShell(i) {
  const r = EX.shells[i];
  document.getElementById("readout").textContent = "shell " + r.id + " · κ = " + r.kappa.toFixed(3) + " · q = " + r.q.toFixed(2);
  draw(r);
}

function showTables() {
  // the single- and two-defect shells are released as table rows: defect parameters and kappa
  const lams = [...new Set(EX.single.map(r => r.lam))].sort((a, b) => a - b);
  const pal = ["#0a1fb0", "#2a62e0", "#5b8def", "#b02c27", "#7a1f1b"];
  const ts = lams.map((l, i) => {
    const r = EX.single.filter(x => x.lam === l).sort((a, b) => a.delta - b.delta);
    return { x: r.map(x => x.delta), y: r.map(x => x.kappa), mode: "lines+markers", name: "\u03bb = " + l,
             line: { color: pal[i % pal.length] }, marker: { size: 7 } };
  });
  Plotly.react("plot-single", ts, { title: { text: "single-defect shells", font: { size: 13 } }, margin: { l: 50, r: 10, t: 36, b: 45 },
    xaxis: { title: "depth \u03b4 / t" }, yaxis: { title: "\u03ba" }, legend: { orientation: "h", y: -0.25 },
    plot_bgcolor: "rgba(0,0,0,0)", paper_bgcolor: "rgba(0,0,0,0)" }, { displayModeBar: false, responsive: true });
  const same = EX.double.filter(x => x.same), diff = EX.double.filter(x => !x.same);
  const td = [
    { x: same.map(x => x.sep), y: same.map(x => x.kappa), mode: "markers", name: "identical pair", marker: { size: 9, color: "#1c2b4a" },
      text: same.map(x => "shell " + x.id + ", \u03b4 = " + x.d1), hoverinfo: "text+x+y" },
    { x: diff.map(x => x.sep), y: diff.map(x => x.kappa), mode: "markers", name: "different depths", marker: { size: 9, color: "#b02c27", symbol: "diamond" },
      text: diff.map(x => "shell " + x.id + ", \u03b4 = " + x.d1 + " / " + x.d2), hoverinfo: "text+x+y" },
  ];
  Plotly.react("plot-double", td, { title: { text: "two-defect shells", font: { size: 13 } }, margin: { l: 50, r: 10, t: 36, b: 45 },
    xaxis: { title: "spacing of the two centres [deg]" }, yaxis: { title: "\u03ba" }, legend: { orientation: "h", y: -0.25 },
    plot_bgcolor: "rgba(0,0,0,0)", paper_bgcolor: "rgba(0,0,0,0)" }, { displayModeBar: false, responsive: true });
  document.getElementById("readout").textContent = EX.single.length + " single- and " + EX.double.length +
    " two-defect shells of the review subset (defect parameters and \u03ba, no fields)";
}

fetch("static/data/explorer.json").then(r => r.json()).then(d => {
  EX = d;
  document.getElementById("p3-help").textContent = "One pair of defects of a core shell is brought from 25\u00b0 to 10\u00b0 apart and every other defect is kept (intervention test P3).";
  const ps = document.getElementById("p3-select");
  Object.keys(d.p3).forEach(k => { const o = document.createElement("option"); o.value = k; o.textContent = "source shell " + k; ps.appendChild(o); });
  SRC = "3890"; ps.value = SRC;
  ps.addEventListener("change", () => { SRC = ps.value; showP3(+document.getElementById("p3-slider").value); });
  const sel = document.getElementById("shell-select");
  d.shells.forEach((s, i) => { const o = document.createElement("option"); o.value = i; o.textContent = "shell " + s.id + " – " + s.label; sel.appendChild(o); });
  sel.addEventListener("change", () => showShell(+sel.value));
  document.getElementById("p3-slider").addEventListener("input", e => showP3(+e.target.value));
  document.querySelectorAll("#explorer-tabs li").forEach(li => li.addEventListener("click", () => {
    document.querySelectorAll("#explorer-tabs li").forEach(x => x.classList.remove("is-active"));
    li.classList.add("is-active"); MODE = li.dataset.mode;
    document.getElementById("p3-controls").classList.toggle("is-hidden", MODE !== "p3");
    document.getElementById("shell-controls").classList.toggle("is-hidden", MODE !== "shells");
    document.getElementById("shell-plots").classList.toggle("is-hidden", MODE === "tables");
    document.getElementById("table-plots").classList.toggle("is-hidden", MODE !== "tables");
    if (MODE === "p3") showP3(+document.getElementById("p3-slider").value);
    else if (MODE === "shells") showShell(+sel.value);
    else showTables();
  }));
  showP3(0);
  const want = new URLSearchParams(location.search).get("tab");       // ?tab=shells or ?tab=tables opens that view
  if (want) { const li = document.querySelector('#explorer-tabs li[data-mode="' + want + '"]'); if (li) li.click(); }
});

// ----------------------------------------------------------------------------------------------------------------
// Sortable Table 2: clicking a header sorts the rows of each group by that column.
let LB_SORT = { col: -1, dir: 1 }, LB_P = 0;
function numOf(c) { const v = parseFloat(String(c).replace("−", "-").replace("*", "")); return isNaN(v) ? null : v; }
function drawLeaderboardSorted(p) {
  LB_P = p;
  const t = document.getElementById("lb");
  let h = "<thead><tr><th>model</th>" + LB_HEAD.map((x, i) => "<th class='has-text-right sortable' data-c='" + i + "'>" + x +
    (LB_SORT.col === i ? (LB_SORT.dir > 0 ? " ▲" : " ▼") : "") + "</th>").join("") + "</tr></thead><tbody>";
  let group = null, rows = [];
  const flush = () => {
    if (LB_SORT.col >= 0) rows.sort((a, b) => {
      const x = numOf(a[1][LB_SORT.col]), y = numOf(b[1][LB_SORT.col]);
      if (x === null) return 1; if (y === null) return -1; return LB_SORT.dir * (x - y);
    });
    for (const [name, r] of rows) {
      h += "<tr><td>" + name + "</td>" + r.map(c => { const best = c.endsWith("*");
        return "<td class='has-text-right" + (best ? " best" : "") + "'>" + (best ? c.slice(0, -1) : c) + "</td>"; }).join("") + "</tr>";
    }
    rows = [];
  };
  for (const [name, rs] of LB) {
    if (rs === null) { flush(); h += "<tr class='group'><td colspan='6'>" + name + "</td></tr>"; continue; }
    if (rs[p].every(c => c === "")) continue;
    rows.push([name, rs[p]]);
  }
  flush();
  t.innerHTML = h + "</tbody>";
  t.querySelectorAll("th.sortable").forEach(th => th.addEventListener("click", () => {
    const c = +th.dataset.c; LB_SORT = { col: c, dir: LB_SORT.col === c ? -LB_SORT.dir : -1 }; drawLeaderboardSorted(LB_P); drawFull(LB_P);
  }));
}
document.querySelectorAll("#lb-tabs button").forEach(b => b.addEventListener("click", () => { drawLeaderboardSorted(+b.dataset.p); drawFull(+b.dataset.p); }));
drawLeaderboardSorted(0);

// ----------------------------------------------------------------------------------------------------------------
// Full results tables and the interactive ladder, from leaderboard.json.
let FULL = null;
const FULL_HEAD = { kappa_mre: "MRE κ [%]", kappa_r2: "κ R²", unconservative: "unconserv.", buckling_r2: "buckling R²",
  field_r2: "field R²", defect_r2: "defect-region R²", rmse: "RMSE [mm]", rmse_noshrink: "RMSE w/o shrink",
  rel_l2: "rel. L² mean/median", rel_l2_noshrink: "rel. L² w/o shrink", clear_site: "clear site", near_tie: "near-tie winner",
  sparse_kappa: "sparse κ err", sparse_r2: "sparse bkl. R²", dense_kappa: "dense κ err", dense_r2: "dense bkl. R²" };
function fmtCell(c) {
  if (typeof c === "string") return c;
  if (c.length === 3) return c[0] + " <span class='ci'>[" + c[1] + ", " + c[2] + "]</span>";
  return String(c[0]);
}
function drawFull(p) {
  if (!FULL) return;
  const prot = ["B1", "B2", "B3"][p], rows = FULL.protocols[prot];
  const cols = FULL.columns.filter(c => rows.some(r => r.cells[c] !== undefined));
  let h = "<thead><tr><th>model</th>" + cols.map(c => "<th class='has-text-right'>" + FULL_HEAD[c] + "</th>").join("") + "</tr></thead><tbody>";
  let g = null;
  for (const r of rows) {
    if (r.group !== g) { g = r.group; h += "<tr class='group'><td colspan='" + (cols.length + 1) + "'>" + g + "</td></tr>"; }
    h += "<tr><td>" + r.name + "</td>" + cols.map(c => "<td class='has-text-right'>" + (r.cells[c] === undefined ? "" : fmtCell(r.cells[c])) + "</td>").join("") + "</tr>";
  }
  document.getElementById("lb-full").innerHTML = h + "</tbody>";
}
const LADDER_COL = { "FigConvNet": "#1c2b4a", "MeshGraphNet": "#294366", "SFNO": "#b02c27", "Transformer": "#e07a5f",
  "Transolver": "#5b8def", "DeepONet": "#7a1f1b", "MLP-20": "#8a8f98", "MLP-80": "#b7bcc5" };
function drawLadder(metric) {
  if (!FULL) return;
  const P = ["B1", "B2", "B3"], traces = [];
  const names = FULL.protocols.B1.filter(r => r.group === "learned field models" || r.name.startsWith("MLP")).map(r => r.name);
  for (const n of names) {
    const ys = P.map(p => { const r = FULL.protocols[p].find(x => x.name === n); const c = r && r.cells[metric];
      return (c && typeof c !== "string") ? c[0] * ((metric === "clear_site" || metric === "near_tie") ? 100 : 1) : null; });
    if (ys.every(v => v === null)) continue;
    const short = n.split(" (")[0].replace("\n", " ");
    traces.push({ x: ["B1 within distribution", "B2 deeper defects", "B3 sparse to dense"], y: ys, mode: "lines+markers", name: short,
      line: { width: 3, color: LADDER_COL[short] || "#999" }, marker: { size: 9 } });
  }
  const ref = FULL.protocols.B1.find(r => r.name === "training mean");
  const refy = P.map(p => { const c = FULL.protocols[p].find(r => r.name === "training mean").cells[metric]; return (c && typeof c !== "string") ? c[0] * ((metric === "clear_site" || metric === "near_tie") ? 100 : 1) : null; });
  traces.push({ x: ["B1 within distribution", "B2 deeper defects", "B3 sparse to dense"], y: refy, mode: "lines", name: "training mean",
    line: { dash: "dash", color: "#555", width: 2 } });
  const ylab = { buckling_r2: "buckling R²", kappa_mre: "median relative error of κ [%]", clear_site: "clear-site accuracy [%]", near_tie: "near-tie winner rate [%]" }[metric];
  Plotly.react("plot-ladder", traces, { margin: { l: 60, r: 10, t: 10, b: 40 }, yaxis: { title: ylab, type: metric === "kappa_mre" ? "log" : "linear" },
    legend: { orientation: "h", y: -0.18 }, plot_bgcolor: "rgba(0,0,0,0)", paper_bgcolor: "rgba(0,0,0,0)" }, { displayModeBar: false, responsive: true });
}
document.querySelectorAll("#ladder-tabs button").forEach(b => b.addEventListener("click", () => {
  document.querySelectorAll("#ladder-tabs button").forEach(x => x.classList.remove("is-dark")); b.classList.add("is-dark"); drawLadder(b.dataset.m);
}));
fetch("static/data/leaderboard.json").then(r => r.json()).then(d => { FULL = d; drawFull(0); drawLadder("buckling_r2"); });

// ----------------------------------------------------------------------------------------------------------------
// See the shift: the same dense shell predicted by each model trained on B1 and on B3, drawn on canvases.
let PR = null;
function ramp(t) {   // white -> salmon -> dark red
  t = Math.max(0, Math.min(1, t));
  const a = [247, 243, 242], b = [224, 122, 95], c = [122, 31, 27];
  const [p, q, u] = t < 0.5 ? [a, b, t / 0.5] : [b, c, (t - 0.5) / 0.5];
  return "rgb(" + p.map((v, i) => Math.round(v + (q[i] - v) * u)).join(",") + ")";
}
function paint(canvas, d, vmax, site, trueSite) {
  const W = 220, ctx = canvas.getContext("2d"); canvas.width = W; canvas.height = W;
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, W, W);
  const X = PR.x, Y = PR.y, s = W / 2 - 6;
  const order = d.map((v, i) => i).sort((i, j) => d[i] - d[j]);
  for (const i of order) { ctx.fillStyle = ramp(d[i] / vmax); ctx.fillRect(W / 2 + X[i] * s - 1.2, W / 2 - Y[i] * s - 1.2, 2.4, 2.4); }
  const mark = (i, kind) => {
    const x = W / 2 + X[i] * s, y = W / 2 - Y[i] * s; ctx.lineWidth = 2.5; ctx.strokeStyle = "#1c2b4a";
    if (kind === "o") { ctx.beginPath(); ctx.arc(x, y, 9, 0, 2 * Math.PI); ctx.stroke(); }
    else { ctx.beginPath(); ctx.moveTo(x - 7, y - 7); ctx.lineTo(x + 7, y + 7); ctx.moveTo(x + 7, y - 7); ctx.lineTo(x - 7, y + 7); ctx.stroke(); }
  };
  mark(trueSite, "x"); if (site !== null) mark(site, "o");
}
function angDeg(i, j) {
  const zi = Math.sqrt(Math.max(0, 1 - PR.x[i] ** 2 - PR.y[i] ** 2)), zj = Math.sqrt(Math.max(0, 1 - PR.x[j] ** 2 - PR.y[j] ** 2));
  return Math.acos(Math.max(-1, Math.min(1, PR.x[i] * PR.x[j] + PR.y[i] * PR.y[j] + zi * zj))) * 180 / Math.PI;
}
function showShift(k) {
  const sh = PR.shells[k], vmax = Math.max(...sh.truth.d);
  document.querySelectorAll("#shift-shells .button").forEach((b, i) => b.classList.toggle("is-selected", i === k));
  const truth = document.getElementById("shift-truth");
  truth.innerHTML = "<div class='rowlab'>solver</div><div class='panel-c'><canvas></canvas><div class='cap'>κ = " + sh.truth.kappa.toFixed(3) + "</div></div>";
  paint(truth.querySelector("canvas"), sh.truth.d, vmax, null, sh.truth.site);
  const g = document.getElementById("shift-grid"); g.innerHTML = "<div></div>" + PR.models.map(m => "<div class='collab'>" + m + "</div>").join("");
  for (const prot of ["B1", "B3"]) {
    g.insertAdjacentHTML("beforeend", "<div class='rowlab'>trained on " + prot + "<br><span style='font-weight:400'>" +
      (prot === "B1" ? "dense shells seen" : "sparse shells only") + "</span></div>");
    for (const m of PR.models) {
      const r = sh.models[m][prot], a = angDeg(r.site, sh.truth.site), ok = a < 10;
      const err = 100 * (r.kappa - sh.truth.kappa) / sh.truth.kappa, div = !isFinite(err) || Math.abs(err) > 1000;
      g.insertAdjacentHTML("beforeend", "<div class='panel-c'><canvas></canvas><div class='cap'>" + (div ? "<span class='bad'>diverged</span><br>&nbsp;" :
        "<span class='" + (ok ? "ok" : "bad") + "'>site " + (ok ? "✓" : "off by " + a.toFixed(0) + "°") + "</span><br>κ " +
        (err >= 0 ? "+" : "") + err.toFixed(1) + "%") + "</div></div>");
      paint(g.lastElementChild.querySelector("canvas"), r.d, vmax, r.site, sh.truth.site);
    }
  }
}
fetch("static/data/predictions.json").then(r => r.json()).then(d => {
  PR = d; const bs = document.getElementById("shift-shells");
  d.shells.forEach((s, i) => { const b = document.createElement("button"); b.className = "button is-small"; b.textContent = "dense shell " + s.id;
    b.addEventListener("click", () => showShift(i)); bs.appendChild(b); });
  showShift(0);
});
