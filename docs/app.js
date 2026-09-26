/* WannaCry site — all charts are drawn from assets/eda/eda.json and assets/results/results.json */
const DEMO_URL = "https://huggingface.co/spaces/REPLACE_ME/wannacry-demo";   // set when the Space exists
const VIDEO_DIR = "assets/video/";                                           // <stem>_annotated.mp4
const VIDEOS = ["C3896.mp4", "C3897.mp4", "C3902.mp4", "C3905.mp4"];
const CLASSES = ["accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn", "stopped_vehicle",
  "jaywalking", "failure_to_yield", "illegal_turn", "solid_line_crossing", "stop_line", "congestion",
  "road_obstacle", "fire_smoke"];
const pretty = s => s.replace(/_/g, " ");

/* ---------- theme */
const root = document.documentElement;
try { const t = localStorage.getItem("wc-theme"); if (t) root.dataset.theme = t; } catch (e) {}
document.getElementById("themeBtn").addEventListener("click", () => {
  const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("wc-theme", root.dataset.theme); } catch (e) {}
  redrawAll();
});
const css = v => getComputedStyle(root).getPropertyValue(v).trim();

/* ---------- demo */
const demoEmbed = DEMO_URL.includes("REPLACE_ME") ? "" :
  DEMO_URL.replace("https://huggingface.co/spaces/", "https://").replace(/\/([^/]+)$/, "-$1") + ".hf.space";
if (demoEmbed) document.getElementById("demoFrame").src = demoEmbed;
else document.getElementById("demoFrame").style.height = "120px", document.getElementById("demoFrame").srcdoc = "<p style='font:16px system-ui;padding:24px;color:#56626f'>The live demo will appear here once the Hugging Face Space is online.</p>";
for (const id of ["demoLink", "demoLink2"]) document.getElementById(id).href = DEMO_URL;

/* ---------- data */
let EDA, RES, charts = [], current = VIDEOS[0], countVideo = VIDEOS[0];
Promise.all([fetch("assets/eda/eda.json").then(r => r.json()), fetch("assets/results/results.json").then(r => r.json())])
  .then(([e, r]) => { EDA = e; RES = r; buildStatic(); redrawAll(); selectVideo(VIDEOS[0]); })
  .catch(err => console.error("data load failed", err));

function buildStatic() {
  // video table + thumbnails
  const tb = document.querySelector("#vidTable tbody");
  EDA.videos.forEach(v => {
    const t = v.tracks, two = (t.motorcycle || 0) + (t.bicycle || 0);
    tb.insertAdjacentHTML("beforeend", `<tr><td>${v.name}</td><td>${v.res}</td><td class="num">${v.fps}</td>
      <td class="num">${Math.floor(v.duration / 60)}:${String(Math.round(v.duration % 60)).padStart(2, "0")}</td>
      <td class="num">${v.frames.toLocaleString("en")}</td><td class="num">${v.mean_brightness}</td>
      <td class="num">${t.person || 0}</td><td class="num">${t.car || 0}</td><td class="num">${t.bus || 0}</td>
      <td class="num">${t.truck || 0}</td><td class="num">${two}</td></tr>`);
  });
  const cap = document.getElementById("vidTable").createCaption();
  cap.className = "note"; cap.style.cssText = "caption-side:bottom;text-align:left;padding-top:8px";
  cap.textContent = "Object columns count distinct tracks; brightness is the mean grey level (0–255).";
  document.getElementById("thumbs").innerHTML = EDA.videos.map(v =>
    `<figure><img src="assets/eda/thumb_${v.name.slice(0, 5)}.jpg" alt="Frame from ${v.name}" loading="lazy" width="640" height="360"><figcaption>${v.name} · brightness ${v.mean_brightness}</figcaption></figure>`).join("");

  // segmented controls
  const seg = (id, onPick, getCur, role) => {
    const el = document.getElementById(id);
    el.innerHTML = VIDEOS.map(v => `<button type="button" data-v="${v}" ${role === "tab" ? 'role="tab"' : ""}>${v.slice(0, 5)}</button>`).join("");
    el.addEventListener("click", e => { const b = e.target.closest("button"); if (b) onPick(b.dataset.v); });
    return () => el.querySelectorAll("button").forEach(b =>
      b.setAttribute(role === "tab" ? "aria-selected" : "aria-pressed", String(b.dataset.v === getCur())));
  };
  window.syncCountSeg = seg("countSeg", v => { countVideo = v; syncCountSeg(); drawCounts(); }, () => countVideo);
  window.syncVidSeg = seg("vidSeg", v => selectVideo(v), () => current, "tab");
  syncCountSeg();

  // score table
  const sb = document.querySelector("#scoreTable tbody");
  Object.entries(RES.per_class).sort((a, b) => b[1].mean - a[1].mean).forEach(([c, s]) => {
    sb.insertAdjacentHTML("beforeend", `<tr><td>${pretty(c)}</td><td class="num">${s.gt}</td><td class="num">${s.pred}</td>
      <td class="num">${s.f1_03.toFixed(2)}</td><td class="num">${s.f1_05.toFixed(2)}</td><td class="num">${s.f1_07.toFixed(2)}</td>
      <td class="num"><span class="bar" style="width:${Math.round(s.mean * 60)}px"></span>${s.mean.toFixed(2)}</td></tr>`);
  });
  sb.insertAdjacentHTML("beforeend", `<tr><td><b>Score A</b></td><td></td><td></td><td></td><td></td><td></td><td class="num"><b>${RES.score_a.toFixed(3)}</b></td></tr>`);

  // galleries
  const card = (x, fail) => `<button type="button" class="card${fail ? " fail" : ""}" data-v="${x.video}" data-t="${x.t}">
      <img src="${x.img}" alt="${x.caption}" loading="lazy" width="960" height="540">
      <span><b>${fail ? x.kind : pretty(x.label)}</b>${x.caption} · ${x.video.slice(0, 5)}</span></button>`;
  document.getElementById("examples").innerHTML = RES.examples.map(x => card(x, false)).join("");
  document.getElementById("failures").innerHTML = RES.failures.slice(0, 8).map(x => card(x, true)).join("");
  document.getElementById("failNote").textContent =
    `Across the four videos we miss ${RES.fail_counts.missed} labelled events and raise ${RES.fail_counts["false alarm"]} events with no matching label. Typical failures: events whose timing is off by more than the matching tolerance, crowded scenes where tracks break, and classes we labelled but no rule covers (U-turn, near miss). Click a card to see it in the video.`;
  document.querySelectorAll(".gallery").forEach(g => g.addEventListener("click", e => {
    const b = e.target.closest(".card"); if (!b) return;
    selectVideo(b.dataset.v, +b.dataset.t - 2);
    document.querySelector(".player").scrollIntoView({ behavior: "smooth", block: "start" });
  }));

  // signal bars
  const sig = document.getElementById("signalBars");
  sig.innerHTML = VIDEOS.map(v => {
    const dur = EDA.videos.find(x => x.name === v).duration;
    const bars = EDA.signal[v].filter(r => r.state !== "unknown").map(r =>
      `<i style="left:${r.start / dur * 100}%;width:${(r.end - r.start) / dur * 100}%;background:var(--${r.state === "red" ? "red" : "green"})" title="${r.state} ${r.start}–${r.end} s"></i>`).join("");
    return `<div class="sigrow"><span>${v.slice(0, 5)}</span><div class="sigbar" role="img" aria-label="Signal phases of ${v}">${bars}</div></div>`;
  }).join("");
}

/* ---------- charts (Chart.js) */
function baseOpts(extra = {}) {
  Chart.defaults.font.family = "'IBM Plex Sans', 'Segoe UI', sans-serif";
  const ink = css("--muted"), grid = css("--line");
  return Object.assign({
    responsive: true, maintainAspectRatio: false, animation: false,
    plugins: { legend: { labels: { color: ink, boxWidth: 12, font: { family: "IBM Plex Sans" } } } },
    scales: {
      x: { ticks: { color: ink }, grid: { color: grid }, title: { display: true, text: "time (s)", color: ink } },
      y: { ticks: { color: ink }, grid: { color: grid }, beginAtZero: true }
    }
  }, extra);
}
const palette = () => [css("--accent"), css("--blue"), css("--green"), css("--red"), css("--gt"), "#9b6bd3"];
function mk(id, cfg) { const c = new Chart(document.getElementById(id), cfg); charts.push(c); return c; }

function redrawAll() {
  if (!EDA) return;
  charts.forEach(c => c.destroy()); charts = [];
  const P = palette();
  mk("cBright", { type: "line", data: { datasets: VIDEOS.map((v, i) => ({ label: v.slice(0, 5),
      data: EDA.brightness[v].t.map((t, k) => ({ x: t, y: EDA.brightness[v].v[k] })), borderColor: P[i], backgroundColor: P[i],
      pointRadius: 0, borderWidth: 2, tension: .3 })) },
    options: baseOpts({ scales: { x: { type: "linear", ticks: { color: css("--muted") }, grid: { color: css("--line") }, title: { display: true, text: "time (s)", color: css("--muted") } },
      y: { min: 0, max: 140, ticks: { color: css("--muted") }, grid: { color: css("--line") }, title: { display: true, text: "grey level", color: css("--muted") } } } }) });
  const maxMin = Math.max(...VIDEOS.map(v => EDA.density[v].minute.length));
  mk("cDensity", { type: "bar", data: { labels: Array.from({ length: maxMin }, (_, i) => `min ${i + 1}`),
      datasets: VIDEOS.map((v, i) => ({ label: v.slice(0, 5), data: EDA.density[v].vehicles, backgroundColor: P[i] })) },
    options: baseOpts({ scales: { x: { ticks: { color: css("--muted") }, grid: { display: false } },
      y: { ticks: { color: css("--muted") }, grid: { color: css("--line") }, title: { display: true, text: "new vehicles", color: css("--muted") } } } }) });
  drawCounts(); drawDash(); drawTimeline();
}
let countChart;
function drawCounts() {
  if (countChart) { countChart.destroy(); charts = charts.filter(c => c !== countChart); }
  const d = EDA.counts[countVideo], P = palette();
  const keys = ["person", "car", "bus", "truck", "motorcycle", "bicycle"].filter(k => d[k]);
  countChart = mk("cCounts", { type: "line", data: { labels: d.t,
      datasets: keys.map((k, i) => ({ label: k, data: d[k], borderColor: P[i], backgroundColor: P[i] + "55", fill: true, pointRadius: 0, borderWidth: 1.5, tension: .25 })) },
    options: baseOpts({ scales: { x: { ticks: { color: css("--muted"), maxTicksLimit: 12 }, grid: { color: css("--line") }, title: { display: true, text: "time (s)", color: css("--muted") } },
      y: { stacked: true, ticks: { color: css("--muted") }, grid: { color: css("--line") }, title: { display: true, text: "objects per frame", color: css("--muted") } } } }) });
}
function drawDash() {
  const counts = {};
  VIDEOS.forEach(v => RES.videos[v].pred.forEach(([s, , l]) => {
    const m = Math.floor(s / 60); counts[l] = counts[l] || []; counts[l][m] = (counts[l][m] || 0) + 1;
  }));
  const n = 6, P = palette();
  mk("cDash", { type: "bar", data: { labels: Array.from({ length: n }, (_, i) => `minute ${i + 1}`),
      datasets: Object.keys(counts).map((l, i) => ({ label: pretty(l), data: Array.from({ length: n }, (_, m) => counts[l][m] || 0),
        backgroundColor: i < P.length ? P[i] : `hsl(${i * 47},55%,55%)` })) },
    options: baseOpts({ scales: { x: { stacked: true, ticks: { color: css("--muted") }, grid: { display: false } },
      y: { stacked: true, ticks: { color: css("--muted") }, grid: { color: css("--line") }, title: { display: true, text: "events (4 videos)", color: css("--muted") } } } }) });
}

/* ---------- player + timeline */
const vid = document.getElementById("vid"), tl = document.getElementById("timeline");
let rows = [], TL = { left: 150, top: 8, row: 22, riskH: 44 };
function selectVideo(v, t = 0) {
  current = v; syncVidSeg();
  const src = VIDEO_DIR + v.replace(".mp4", "_annotated.mp4");
  if (!vid.src.endsWith(src)) {
    vid.poster = `assets/eda/thumb_${v.slice(0, 5)}.jpg`; vid.src = src;
    vid.onerror = () => { document.getElementById("novid").hidden = false; };
    vid.onloadeddata = () => { document.getElementById("novid").hidden = true; };
  }
  const seek = () => { try { vid.currentTime = Math.max(0, t); } catch (e) {} };
  if (vid.readyState >= 1) seek(); else vid.addEventListener("loadedmetadata", seek, { once: true });
  drawTimeline(Math.max(0, t));
}
function drawTimeline(tNow) {
  if (!RES) return;
  const d = RES.videos[current], dpr = window.devicePixelRatio || 1;
  const used = new Set([...d.pred, ...d.gt].map(e => e[2]));
  rows = CLASSES.filter(c => used.has(c));
  const W = Math.max(tl.parentElement.clientWidth, 640), H = TL.top + rows.length * TL.row + TL.riskH + 26;
  tl.width = W * dpr; tl.height = H * dpr; tl.style.height = H + "px"; tl.style.width = W + "px";
  const g = tl.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
  const x = t => TL.left + (W - TL.left - 10) * t / d.duration;
  g.font = "12px 'IBM Plex Mono', monospace"; g.textBaseline = "middle";
  rows.forEach((c, i) => {
    const y = TL.top + i * TL.row;
    g.fillStyle = i % 2 ? "transparent" : css("--bg"); g.fillRect(TL.left, y, W - TL.left - 10, TL.row);
    g.fillStyle = css("--muted"); g.fillText(pretty(c), 6, y + TL.row / 2);
    d.pred.filter(e => e[2] === c).forEach(([s, e]) => { g.fillStyle = css("--pred"); g.fillRect(x(s), y + 3, Math.max(2, x(e) - x(s)), 7); });
    d.gt.filter(e => e[2] === c).forEach(([s, e]) => { g.fillStyle = css("--gt"); g.fillRect(x(s), y + 12, Math.max(2, x(e) - x(s)), 7); });
  });
  // risk strip
  const ry = TL.top + rows.length * TL.row + 8, rh = TL.riskH - 12;
  g.fillStyle = css("--muted"); g.fillText("risk", 6, ry + rh / 2);
  g.strokeStyle = css("--line"); g.setLineDash([4, 4]); g.beginPath(); g.moveTo(TL.left, ry + rh / 2); g.lineTo(W - 10, ry + rh / 2); g.stroke(); g.setLineDash([]);
  g.strokeStyle = css("--risk"); g.lineWidth = 1.2; g.beginPath();
  (d.risk || []).forEach(([t, r], k) => { const px = x(t), py = ry + rh - r * rh; k ? g.lineTo(px, py) : g.moveTo(px, py); }); g.stroke();
  // axis
  const ay = ry + rh + 14; g.fillStyle = css("--muted");
  for (let s = 0; s <= d.duration; s += 30) { g.fillRect(x(s), ay - 10, 1, 4); g.fillText(`${s}s`, x(s) - 8, ay); }
  // playhead
  const tp = tNow ?? vid.currentTime ?? 0;
  g.fillStyle = css("--ink"); g.fillRect(x(tp) - 1, TL.top, 2, ay - TL.top - 10);
  document.getElementById("nowLbl").textContent = `t = ${tp.toFixed(1)} s`;
}
tl.addEventListener("click", e => {
  const d = RES.videos[current], r = tl.getBoundingClientRect();
  const t = (e.clientX - r.left - TL.left) / (r.width - TL.left - 10) * d.duration;
  if (t < 0) return;
  // snap to the start of a clicked event if the click is on a bar
  const i = Math.floor((e.clientY - r.top - TL.top) / TL.row), c = rows[i];
  const hit = c && [...d.pred, ...d.gt].find(ev => ev[2] === c && t >= ev[0] - 1 && t <= ev[1] + 1);
  selectVideo(current, hit ? hit[0] : t);
});
vid.addEventListener("timeupdate", () => drawTimeline());
addEventListener("resize", () => drawTimeline());
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redrawAll);

/* ---------- team details (filled in once roles are confirmed) */
const TEAM = {
  // doniyor: { role: "…", did: "…" },
};
for (const [k, v] of Object.entries(TEAM)) {
  const r = document.querySelector(`[data-role="${k}"]`), d = document.querySelector(`[data-did="${k}"]`);
  if (r && v.role) r.textContent = v.role; if (d && v.did) d.textContent = v.did;
}
