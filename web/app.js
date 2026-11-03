// MapVision in-browser demo: runs the exported U-Net with ONNX Runtime Web.
// Preprocessing mirrors mapvision.inference.predict_image: RGB, resize to
// 256x256, scale to [0, 1], CHW float32. Output is argmax over 5 classes.

const ORT_VERSION = "1.20.1";
ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
ort.env.wasm.numThreads = 1; // multi-threading needs cross-origin isolation

const els = {
  status: document.getElementById("status"),
  samples: document.getElementById("samples"),
  file: document.getElementById("file"),
  input: document.getElementById("input"),
  pred: document.getElementById("pred"),
  overlay: document.getElementById("overlay"),
  gt: document.getElementById("gt"),
  gtFigure: document.getElementById("gt-figure"),
  panels: document.querySelector(".panels"),
  alpha: document.getElementById("alpha"),
  legend: document.getElementById("legend"),
  timing: document.getElementById("timing"),
};

let data, session, size, lastPrediction = null;

function setStatus(text, kind = "") {
  els.status.textContent = text;
  els.status.className = `status ${kind}`;
}

const fmt = (v) => (v === null || v === undefined ? "N/A" : v.toFixed(3));
const rgb = (c) => `rgb(${c[0]}, ${c[1]}, ${c[2]})`;

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Could not read that image."));
    img.src = src;
  });
}

function renderLegend(fractions) {
  els.legend.innerHTML = "";
  data.classes.forEach((cls, i) => {
    const li = document.createElement("li");
    const pct = fractions ? ` <span class="pct">${(fractions[i] * 100).toFixed(1)}%</span>` : "";
    li.innerHTML = `<span class="swatch" style="background:${rgb(cls.color)}"></span>${cls.name}${pct}`;
    els.legend.appendChild(li);
  });
}

function renderResults() {
  const t = data.test, m = data.model;
  document.getElementById("results-note").textContent =
    `Held-out test split (${t.num_samples} tiles, never used for model selection). ` +
    `${m.parameters.toLocaleString()}-parameter U-Net, best of ${m.epochs_trained} epochs ` +
    `by validation mIoU (epoch ${m.best_epoch}), trained on ${m.train_samples} tiles.`;

  const stats = [
    ["Mean IoU", t.mean_iou],
    ["Mean Dice", t.mean_dice],
    ["Pixel accuracy", t.pixel_accuracy],
    ["Best val mIoU", m.best_val_miou],
  ];
  document.getElementById("stats").innerHTML = stats
    .map(([name, v]) => `<div class="stat"><div class="value">${fmt(v)}</div><div class="name">${name}</div></div>`)
    .join("");

  document.getElementById("per-class").innerHTML = data.classes
    .map(({ name, color }) => {
      const iou = t.per_class_iou[name];
      const width = Math.max(0, (iou || 0) * 100);
      return `<tr>
        <td><span class="swatch" style="display:inline-block;vertical-align:-2px;margin-right:6px;background:${rgb(color)}"></span>${name}</td>
        <td>${fmt(iou)}</td><td>${fmt(t.per_class_dice[name])}</td>
        <td><div class="bar-track"><div class="bar" style="width:${width}%"></div></div></td>
      </tr>`;
    })
    .join("");
}

function renderSamples() {
  data.samples.forEach((sample, index) => {
    const button = document.createElement("button");
    button.className = "sample";
    button.type = "button";
    button.setAttribute("aria-pressed", "false");
    button.setAttribute("aria-label", `Test tile ${index + 1}`);
    button.innerHTML = `<img src="${sample.image}" alt="">`;
    button.addEventListener("click", () => runSample(sample, button));
    els.samples.appendChild(button);
  });
}

function setPressed(button) {
  els.samples.querySelectorAll(".sample").forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
}

function toTensor(ctx) {
  const { data: px } = ctx.getImageData(0, 0, size, size);
  const plane = size * size;
  const chw = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    chw[i] = px[i * 4] / 255;
    chw[plane + i] = px[i * 4 + 1] / 255;
    chw[2 * plane + i] = px[i * 4 + 2] / 255;
  }
  return new ort.Tensor("float32", chw, [1, 3, size, size]);
}

function argmax(logits, numClasses) {
  const plane = size * size;
  const classes = new Uint8Array(plane);
  const counts = new Array(numClasses).fill(0);
  for (let i = 0; i < plane; i++) {
    let best = 0, bestVal = logits[i];
    for (let c = 1; c < numClasses; c++) {
      const v = logits[c * plane + i];
      if (v > bestVal) { bestVal = v; best = c; }
    }
    classes[i] = best;
    counts[best]++;
  }
  return { classes, fractions: counts.map((n) => n / plane) };
}

function drawPrediction(classes) {
  const predCtx = els.pred.getContext("2d");
  const out = predCtx.createImageData(size, size);
  classes.forEach((cls, i) => {
    const c = data.classes[cls].color;
    out.data.set([c[0], c[1], c[2], 255], i * 4);
  });
  predCtx.putImageData(out, 0, 0);
  drawOverlay();
}

function drawOverlay() {
  if (!lastPrediction) return;
  const alpha = els.alpha.value / 100;
  const src = els.input.getContext("2d").getImageData(0, 0, size, size).data;
  const ctx = els.overlay.getContext("2d");
  const out = ctx.createImageData(size, size);
  lastPrediction.forEach((cls, i) => {
    const c = data.classes[cls].color;
    for (let k = 0; k < 3; k++) {
      out.data[i * 4 + k] = src[i * 4 + k] * (1 - alpha) + c[k] * alpha;
    }
    out.data[i * 4 + 3] = 255;
  });
  ctx.putImageData(out, 0, 0);
}

async function predict(img) {
  const ctx = els.input.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(img, 0, 0, size, size); // stretch like PIL resize((256, 256))

  setStatus("Running…");
  const start = performance.now();
  const { logits } = await session.run({ image: toTensor(ctx) });
  const elapsed = performance.now() - start;

  const { classes, fractions } = argmax(logits.data, data.classes.length);
  lastPrediction = classes;
  drawPrediction(classes);
  renderLegend(fractions);
  els.timing.textContent = `Inference ${elapsed.toFixed(0)} ms in your browser (WebAssembly, 1 thread)`;
  setStatus("Ready", "ready");
}

function showGroundTruth(src) {
  els.gtFigure.classList.toggle("hidden", !src);
  els.panels.classList.toggle("no-gt", !src);
  if (src) els.gt.src = src;
}

async function runSample(sample, button) {
  setPressed(button);
  showGroundTruth(sample.gt);
  await predict(await loadImage(sample.image));
}

els.file.addEventListener("change", async () => {
  const file = els.file.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  try {
    setPressed(null);
    showGroundTruth(null);
    await predict(await loadImage(url));
  } catch (err) {
    setStatus(err.message, "error");
  } finally {
    URL.revokeObjectURL(url);
    els.file.value = "";
  }
});

els.alpha.addEventListener("input", drawOverlay);

async function init() {
  try {
    data = await (await fetch("data.json")).json();
    size = data.image_size;
    renderLegend(null);
    renderResults();
    renderSamples();
    session = await ort.InferenceSession.create("model/mapvision.onnx", { executionProviders: ["wasm"] });
    setStatus("Ready", "ready");
    els.samples.querySelector(".sample").click();
  } catch (err) {
    console.error(err);
    setStatus("Model failed to load. Try reloading.", "error");
  }
}

init();
