const ROOT = '..';

const state = {
  originalImages: [],
  interpImages: [],
  metrics: null,
  originalIndex: 0,
  interpIndex: 0,
  originalTimer: null,
  interpTimer: null,
};

const els = {
  message: document.getElementById('message'),
  originalCanvas: document.getElementById('originalCanvas'),
  interpCanvas: document.getElementById('interpCanvas'),
  originalSlider: document.getElementById('originalSlider'),
  interpSlider: document.getElementById('interpSlider'),
  originalLabel: document.getElementById('originalLabel'),
  interpLabel: document.getElementById('interpLabel'),
  metricChips: document.getElementById('metricChips'),
  tableWrap: document.getElementById('tableWrap'),
  psnrChart: document.getElementById('psnrChart'),
  ssimChart: document.getElementById('ssimChart'),
  mseChart: document.getElementById('mseChart'),
  fsimChart: document.getElementById('fsimChart'),
};

function setStatus(text, isError = false) {
  els.message.textContent = text;
  els.message.className = isError ? 'error' : 'status';
}

async function loadImage(src) {
  const img = new Image();
  img.src = src;
  await img.decode().catch(() => {
    throw new Error(`Could not load image: ${src}`);
  });
  return img;
}

async function loadImages(prefix, count) {
  const out = [];
  for (let i = 0; i < count; i++) {
    const src = `${ROOT}/demo_output/images/${prefix}/frame_${String(i).padStart(4, '0')}.png`;
    out.push(await loadImage(src));
  }
  return out;
}

async function loadData() {
  const res = await fetch(`${ROOT}/demo_output/metrics.json`);
  if (!res.ok) throw new Error(`metrics.json returned HTTP ${res.status}`);
  state.metrics = await res.json();

  const originalCount = state.metrics.num_input_frames || 1;
  const interpCount = state.metrics.num_output_frames || originalCount;

  state.originalImages = await loadImages('original', originalCount);
  state.interpImages = await loadImages('interpolated', interpCount);

  els.originalSlider.max = Math.max(0, originalCount - 1);
  els.interpSlider.max = Math.max(0, interpCount - 1);
}

function drawImage(canvas, img) {
  const ctx = canvas.getContext('2d');
  const scale = Math.min(canvas.width / img.width, canvas.height / img.height);
  const w = img.width * scale;
  const h = img.height * scale;
  const x = (canvas.width - w) / 2;
  const y = (canvas.height - h) / 2;
  ctx.fillStyle = '#020617';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, x, y, w, h);
}

function updateOriginal() {
  const idx = Number(els.originalSlider.value);
  state.originalIndex = idx;
  drawImage(els.originalCanvas, state.originalImages[idx]);
  els.originalLabel.textContent = `Frame ${idx}`;
}

function updateInterp() {
  const idx = Number(els.interpSlider.value);
  state.interpIndex = idx;
  drawImage(els.interpCanvas, state.interpImages[idx]);
  els.interpLabel.textContent = `Frame ${idx}`;
}

function renderMetrics() {
  const metrics = state.metrics?.metrics || {};
  els.metricChips.innerHTML = '';
  for (const [key, value] of Object.entries(metrics)) {
    const chip = document.createElement('span');
    chip.className = 'metric-chip';
    chip.textContent = `${key}: ${formatValue(value)}`;
    els.metricChips.appendChild(chip);
  }

  const rows = state.metrics?.frame_metrics || [];
  if (!rows.length) {
    els.tableWrap.innerHTML = '<p class="small">No interpolated frames had aligned ground truth for metrics.</p>';
    return;
  }

  const cols = ['frame_index', 'pair', 'alpha', 'mse', 'rmse', 'psnr', 'ssim', 'gradient_difference', 'fsim_lite'];
  const header = `<tr>${cols.map(c => `<th>${c}</th>`).join('')}</tr>`;
  const body = rows.map(row => `<tr>${cols.map(c => `<td>${formatValue(row[c])}</td>`).join('')}</tr>`).join('');
  els.tableWrap.innerHTML = `<table><thead>${header}</thead><tbody>${body}</tbody></table>`;
}

function formatValue(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return 'nan';
  if (typeof value === 'number') return Number.isFinite(value) ? value.toFixed(4) : String(value);
  if (Array.isArray(value)) return value.join('–');
  return String(value);
}

function setupSliders() {
  els.originalSlider.addEventListener('input', updateOriginal);
  els.interpSlider.addEventListener('input', updateInterp);

  document.getElementById('playOriginal').addEventListener('click', () => play('original'));
  document.getElementById('pauseOriginal').addEventListener('click', () => pause('original'));
  document.getElementById('playInterp').addEventListener('click', () => play('interp'));
  document.getElementById('pauseInterp').addEventListener('click', () => pause('interp'));
}

function play(kind) {
  const key = kind === 'original' ? 'original' : 'interp';
  const images = key === 'original' ? state.originalImages : state.interpImages;
  const slider = key === 'original' ? els.originalSlider : els.interpSlider;
  const update = key === 'original' ? updateOriginal : updateInterp;
  pause(kind);

  state[`${key}Timer`] = setInterval(() => {
    const next = (Number(slider.value) + 1) % images.length;
    slider.value = String(next);
    update();
  }, 220);
}

function pause(kind) {
  const key = kind === 'original' ? 'original' : 'interp';
  if (state[`${key}Timer`]) clearInterval(state[`${key}Timer`]);
  state[`${key}Timer`] = null;
}

function drawLineChart(canvas, rows, key, color = '#2563eb') {
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, w, h);

  const points = rows.map((r, i) => ({ i, v: Number(r[key]) })).filter(p => Number.isFinite(p.v));
  if (!points.length) {
    ctx.fillStyle = '#64748b';
    ctx.fillText('No finite data', 20, 40);
    return;
  }

  const vals = points.map(p => p.v);
  const min = Math.min(...vals);
  const max = Math.max(...vals);
  const pad = 34;
  const range = max === min ? 1 : max - min;

  ctx.strokeStyle = '#e2e8f0';
  ctx.lineWidth = 1;
  for (let g = 0; g <= 4; g++) {
    const y = pad + (h - 2 * pad) * g / 4;
    ctx.beginPath();
    ctx.moveTo(pad, y);
    ctx.lineTo(w - pad, y);
    ctx.stroke();
  }

  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  points.forEach((p, idx) => {
    const x = pad + (w - 2 * pad) * p.i / Math.max(1, rows.length - 1);
    const y = pad + (h - 2 * pad) * (1 - (p.v - min) / range);
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  points.forEach(p => {
    const x = pad + (w - 2 * pad) * p.i / Math.max(1, rows.length - 1);
    const y = pad + (h - 2 * pad) * (1 - (p.v - min) / range);
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fill();
  });

  ctx.fillStyle = '#334155';
  ctx.font = '12px system-ui';
  ctx.fillText(`${key} min ${min.toFixed(2)}`, pad, h - 10);
  ctx.fillText(`max ${max.toFixed(2)}`, w - 110, h - 10);
}

function renderCharts() {
  const rows = state.metrics?.frame_metrics || [];
  drawLineChart(els.psnrChart, rows, 'psnr', '#2563eb');
  drawLineChart(els.ssimChart, rows, 'ssim', '#047857');
  drawLineChart(els.mseChart, rows, 'mse', '#dc2626');
  drawLineChart(els.fsimChart, rows, 'fsim_lite', '#7c3aed');
}

async function init() {
  try {
    await loadData();
    setupSliders();
    updateOriginal();
    updateInterp();
    renderMetrics();
    renderCharts();
    setStatus(`Loaded ${state.originalImages.length} original frames and ${state.interpImages.length} interpolated frames.`);
  } catch (err) {
    console.error(err);
    setStatus(`Dashboard load failed. Run the pipeline first, then serve this folder with: python -m http.server 8000. Error: ${err.message}`, true);
  }
}

init();
