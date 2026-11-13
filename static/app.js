"use strict";
const $ = (id) => document.getElementById(id);
const pretty = (s) => s.replace(/_/g, " ");
const pct = (x) => (x * 100).toFixed(1) + "%";
const f2 = (x) => (x == null ? "n/a" : x.toFixed(3));

function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v; else if (k === "text") n.textContent = v; else n.setAttribute(k, v);
  }
  for (const k of kids) n.append(k instanceof Node ? k : document.createTextNode(k));
  return n;
}
async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(url + " " + r.status);
  return r.json();
}

let file = null;
let statusInfo = { ready: false, uncertain_below: 0.5 };

/* ---------------------------------------------------------------- status */
async function loadStatus() {
  statusInfo = await getJSON("/api/status");
  const banner = $("banner");
  const sel = $("modelSel");
  sel.replaceChildren();
  if (!statusInfo.ready) {
    banner.hidden = false;
    banner.replaceChildren(
      el("strong", { text: "No trained model is installed, so this app cannot classify photos yet." }),
      el("div", { text: "Train one on the real dataset, then reload this page:" }),
      el("pre", { text: "python inspect_dataset.py\npython train.py --arch mobilenetv3_large_100 --mode linear_probe" })
    );
    if (statusInfo.rejected.length) {
      banner.append(el("div", { text: "Rejected entries: " + statusInfo.rejected.map((r) => `${r.id} (${r.reason})`).join("; ") }));
    }
    sel.append(el("option", { text: "No models available" }));
    sel.disabled = true;
  } else {
    banner.hidden = true;
    const { models } = await getJSON("/api/models");
    for (const m of models) sel.append(el("option", { value: m.id, text: `${m.arch} (${m.mode.replace("_", " ")})` }));
    if (models.length > 1) sel.append(el("option", { value: "all", text: "Compare all models" }));
    sel.disabled = false;
  }
  updateGo();
}
function updateGo() { $("go").disabled = !(statusInfo.ready && file); }

/* ---------------------------------------------------------------- upload */
function setFile(f) {
  if (!f) return;
  if (!["image/jpeg", "image/png", "image/webp"].includes(f.type)) { showError("Use a JPEG, PNG or WebP image."); return; }
  if (f.size > 8 * 1024 * 1024) { showError("Images must be 8 MB or smaller."); return; }
  file = f;
  const img = $("preview");
  img.src = URL.createObjectURL(f);
  img.hidden = false;
  $("plateHint").hidden = true;
  resetRing();
  updateGo();
}
$("file").addEventListener("change", (e) => setFile(e.target.files[0]));
const plate = $("plate");
plate.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("file").click(); } });
["dragenter", "dragover"].forEach((t) => plate.addEventListener(t, (e) => { e.preventDefault(); plate.classList.add("drag"); }));
["dragleave", "drop"].forEach((t) => plate.addEventListener(t, (e) => { e.preventDefault(); plate.classList.remove("drag"); }));
plate.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));
document.addEventListener("paste", (e) => { const f = [...(e.clipboardData?.files || [])][0]; if (f) setFile(f); });

function resetRing() { const r = $("ringFill"); r.style.strokeDashoffset = 100; r.classList.remove("uncertain"); }
function showError(msg) {
  $("empty").hidden = true;
  const box = $("result");
  box.hidden = false;
  box.replaceChildren(el("div", { class: "warn", role: "alert", text: msg }));
}

/* ---------------------------------------------------------------- predict */
function evidenceText(ev) {
  const where = ev.basis === "official_test_split"
    ? `${ev.n_images} official test images`
    : `an internal holdout of ${ev.n_images} training-set images (this dish has no images in the official validation or test splits)`;
  const low = ev.low_support ? " Very few images, so treat this figure as rough." : "";
  return `Measured for this dish: F1 ${f2(ev.f1)} (precision ${f2(ev.precision)}, recall ${f2(ev.recall)}) on ${where}.${low}`;
}

function renderResult(r, primary) {
  const wrap = el("div");
  const top = r.predictions[0];
  wrap.append(el("h3", { class: "dish", text: pretty(top.class) }));
  wrap.append(el("p", { class: "conf", text: `${pct(top.probability)} confidence · ${r.arch}` }));
  if (r.uncertain) {
    wrap.append(el("div", { class: "warn", text: `Low confidence (under ${Math.round(statusInfo.uncertain_below * 100)}%). This may not be one of the 43 dishes the model knows, or the photo may be unclear.` }));
  }
  wrap.append(el("div", { class: "ev", text: evidenceText(top.evidence) }));
  const ul = el("ul", { class: "bars" });
  for (const p of r.predictions) {
    const bar = el("span", { class: "bar" }, el("i"));
    bar.firstChild.style.width = (p.probability * 100).toFixed(1) + "%";
    ul.append(el("li", {}, el("span", { text: pretty(p.class) }), bar, el("span", { class: "pct", text: pct(p.probability) })));
  }
  wrap.append(ul);
  if (primary) {
    const ring = $("ringFill");
    ring.classList.toggle("uncertain", r.uncertain);
    requestAnimationFrame(() => { ring.style.strokeDashoffset = 100 - top.probability * 100; });
  }
  return wrap;
}

$("go").addEventListener("click", async () => {
  if (!file) return;
  const go = $("go");
  go.disabled = true; go.textContent = "Identifying…";
  try {
    const fd = new FormData();
    fd.append("image", file);
    fd.append("model", $("modelSel").value);
    const r = await fetch("/api/predict", { method: "POST", body: fd });
    const data = await r.json();
    if (!r.ok) { showError(data.message || "The request failed."); return; }
    $("empty").hidden = true;
    const box = $("result");
    box.hidden = false;
    box.replaceChildren(renderResult(data.results[0], true));
    if (data.results.length > 1) {
      const c = el("div", { class: "compare" }, el("h3", { text: "Other models" }));
      for (const x of data.results.slice(1)) c.append(renderResult(x, false));
      box.append(c);
    }
    box.append(el("p", { class: "fine", text: data.note }));
  } catch (e) {
    showError("Could not reach the server. Check that app.py is still running.");
  } finally {
    go.textContent = "Identify dish"; updateGo();
  }
});

/* ---------------------------------------------------------------- models panel */
async function loadModels() {
  const body = $("modelsBody");
  body.replaceChildren();
  if (!statusInfo.ready) {
    body.append(el("p", { class: "note", text: "No trained models yet. After training, this section lists each model with its measured accuracy and per-dish results." }));
    return;
  }
  const { models } = await getJSON("/api/models");
  const rows = models.map((m) => el("tr", {},
    el("td", { text: m.arch }), el("td", { text: m.mode.replace("_", " ") }),
    el("td", { class: "n", text: f2(m.headline.internal_holdout_macro_f1_43) }),
    el("td", { class: "n", text: pct(m.headline.test_accuracy_open_set) }),
    el("td", { class: "n", text: f2(m.headline.test_macro_f1_open_set_22) }),
    el("td", { class: "n", text: pct(m.headline.test_accuracy_closed_set_22) }),
    el("td", { class: "n", text: String(m.training.best_epoch) })));
  const head = ["Architecture", "Training", "Holdout macro-F1 (43 dishes)", "Test accuracy", "Test macro-F1 (22 dishes)", "Test accuracy, 22-way only", "Best epoch"];
  body.append(el("div", { class: "tablewrap" }, el("table", {},
    el("thead", {}, el("tr", {}, ...head.map((h, i) => el("th", i > 1 ? { class: "n", text: h } : { text: h })))),
    el("tbody", {}, ...rows))));
  body.append(el("p", { class: "note", text: "The official test split contains only 22 of the 43 dishes. “Test accuracy” lets the model choose among all 43 dishes, so a guess outside those 22 counts as wrong. “22-way only” restricts the choice to the 22 dishes in the test split, which is easier and always at least as high. The holdout comes from the training set and covers all 43 dishes; it can be optimistic if the training set contains near-duplicate photos." }));

  for (const m of models) {
    const full = await getJSON("/api/metrics/" + encodeURIComponent(m.id));
    const pcs = full.test.open_set_all_classes.per_class;
    const hold = full.internal_holdout.open_set_all_classes.per_class;
    const trs = Object.keys(pcs).sort((a, b) => pcs[a].f1 - pcs[b].f1).map((c) => el("tr", {},
      el("td", { text: pretty(c) }), el("td", { class: "n", text: f2(pcs[c].f1) }),
      el("td", { class: "n", text: f2(pcs[c].recall) }), el("td", { class: "n", text: String(pcs[c].support) }),
      el("td", {}, pcs[c].low_support ? el("span", { class: "tag low", text: "few images" }) : "")));
    const un = full.class_coverage.unevaluated_on_official_splits.sort().map((c) => el("tr", {},
      el("td", { text: pretty(c) }), el("td", { class: "n", text: f2(hold[c].f1) }), el("td", { class: "n", text: String(hold[c].support) })));
    body.append(el("details", {},
      el("summary", { text: `Per-dish results: ${m.arch} (${m.mode.replace("_", " ")})` }),
      el("h3", { text: "Dishes in the official test split (22)" }),
      el("div", { class: "tablewrap" }, el("table", {}, el("thead", {}, el("tr", {}, el("th", { text: "Dish" }), el("th", { class: "n", text: "F1" }), el("th", { class: "n", text: "Recall" }), el("th", { class: "n", text: "Test images" }), el("th", { text: "" }))), el("tbody", {}, ...trs))),
      el("h3", { text: "Dishes with no official validation or test images (21)" }),
      el("p", { class: "note", text: "Only the internal holdout from the training set could be used for these." }),
      el("div", { class: "tablewrap" }, el("table", {}, el("thead", {}, el("tr", {}, el("th", { text: "Dish" }), el("th", { class: "n", text: "Holdout F1" }), el("th", { class: "n", text: "Holdout images" }))), el("tbody", {}, ...un)))));
  }
}

/* ---------------------------------------------------------------- dataset panel */
async function loadDataset() {
  const d = await getJSON("/api/dataset");
  const body = $("datasetBody");
  const rep = d.report || {};
  const counts = rep.label_counts || (rep.coverage && rep.coverage.counts) || {};
  const sizes = rep.split_sizes || (rep.coverage && rep.coverage.split_sizes) || {};
  const tr = counts.train || {}, va = counts.validation || {}, te = counts.test || {};
  body.append(el("p", { class: "lede" }, el("a", { href: d.url, text: d.id }), ` · licence ${d.license} · ${Object.values(sizes).reduce((a, b) => a + b, 0)} images (train ${sizes.train}, validation ${sizes.validation}, test ${sizes.test})`));
  body.append(el("p", { class: "note", text: `Training has ${Object.keys(tr).length} dishes. Validation and test share the same ${Object.keys(va).length}, so the other ${Object.keys(tr).length - Object.keys(va).length} dishes cannot be scored on the official splits. Reported metrics are labelled accordingly. Source of these counts: ${d.source.replace(/_/g, " ")}.` }));
  const names = Object.keys(tr).sort((a, b) => (va[b] ? 1 : 0) - (va[a] ? 1 : 0) || a.localeCompare(b));
  const trs = names.map((c) => el("tr", {}, el("td", { text: pretty(c) }), el("td", { class: "n", text: String(tr[c]) }), el("td", { class: "n", text: va[c] != null ? String(va[c]) : "none" }), el("td", { class: "n", text: te[c] != null ? String(te[c]) : "none" })));
  body.append(el("details", {}, el("summary", { text: "Images per dish and split" }),
    el("div", { class: "tablewrap" }, el("table", {}, el("thead", {}, el("tr", {}, el("th", { text: "Dish" }), el("th", { class: "n", text: "Train" }), el("th", { class: "n", text: "Validation" }), el("th", { class: "n", text: "Test" }))), el("tbody", {}, ...trs)))));
  $("attrib").textContent = d.attribution;
}

(async () => {
  try { await loadStatus(); } catch (e) { $("banner").hidden = false; $("banner").textContent = "Could not reach the server API."; }
  loadModels().catch(() => {});
  loadDataset().catch(() => {});
})();
