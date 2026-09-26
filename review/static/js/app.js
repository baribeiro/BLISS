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
  b.classList.add("is-dark"); drawLeaderboard(+b.dataset.p);
}));
drawLeaderboard(0);

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
