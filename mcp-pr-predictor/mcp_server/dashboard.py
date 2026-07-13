"""
Dashboard HTML para el MCP server de PR Merge Predictor.
Servido en GET /dashboard — los datos vienen de GET /api/predictions.
"""
from __future__ import annotations


def generate_html() -> str:
    return """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PR Merge Predictor</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dayjs@1.11.10/dayjs.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dayjs@1.11.10/plugin/relativeTime.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dayjs@1.11.10/locale/es.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
* { font-family: 'Inter', system-ui, sans-serif; }
body { background: #f8fafc; }

/* Score bar */
.score-track { background:#e2e8f0; border-radius:99px; height:5px; overflow:hidden; }
.score-fill  { height:100%; border-radius:99px; transition:width .5s ease; }

/* SHAP row */
.shap-row { display:none; }
.shap-row.open { display:table-row; }

/* SHAP bar — zero-centered layout */
.shap-container {
    display:flex; align-items:center; width:100%;
}
.shap-half {
    flex:1; height:8px; border-radius:4px 0 0 4px;
    background:#f1f5f9; display:flex; justify-content:flex-end; overflow:hidden;
}
.shap-half.right {
    border-radius:0 4px 4px 0;
    justify-content:flex-start;
}
.shap-neg-fill { background:linear-gradient(to left,#f87171,#ef4444); height:100%; border-radius:4px 0 0 4px; }
.shap-pos-fill { background:linear-gradient(to right,#34d399,#10b981); height:100%; border-radius:0 4px 4px 0; }
.shap-center-line { width:2px; background:#cbd5e1; height:16px; flex-shrink:0; }

/* Badges */
.badge-merge  { background:#dcfce7; color:#166534; }
.badge-reject { background:#fee2e2; color:#991b1b; }
.badge-high   { background:#dbeafe; color:#1e40af; }
.badge-medium { background:#fef9c3; color:#854d0e; }
.badge-low    { background:#f1f5f9; color:#475569; }

/* SHAP button */
.btn-shap {
    border:1px solid #e2e8f0; background:#f8fafc; color:#64748b;
    border-radius:6px; padding:3px 10px; font-size:11px; font-weight:500;
    cursor:pointer; transition:all .15s; white-space:nowrap;
}
.btn-shap:hover { background:#f1f5f9; border-color:#94a3b8; }
.btn-shap.open  { background:#ede9fe; border-color:#a78bfa; color:#5b21b6; }

/* Table hover */
tr.pr-row:hover td { background:#f8fafc; }
tr.pr-row td { transition:background .1s; }

/* Skeleton pulse */
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
.skeleton { animation:pulse 1.5s infinite; background:#e2e8f0; border-radius:6px; }

/* Scrollbar */
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:#f8fafc; }
::-webkit-scrollbar-thumb { background:#cbd5e1; border-radius:99px; }
</style>
</head>
<body>

<!-- ── NAVBAR ─────────────────────────────────────────────────────────── -->
<nav class="bg-white border-b border-slate-200 sticky top-0 z-20">
  <div class="max-w-7xl mx-auto px-6 h-14 flex items-center justify-between">
    <div class="flex items-center gap-3">
      <div class="w-7 h-7 rounded-lg bg-indigo-600 flex items-center justify-center shadow-sm">
        <svg class="w-3.5 h-3.5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.2"
            d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0
               002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2
               a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>
        </svg>
      </div>
      <div>
        <p class="text-[13px] font-semibold text-slate-800 leading-tight">PR Merge Predictor</p>
        <p class="text-[11px] text-slate-400 leading-tight">Analytics Dashboard</p>
      </div>
    </div>
    <div class="flex items-center gap-3">
      <div id="status-dot" class="flex items-center gap-1.5">
        <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
        <span class="text-xs text-slate-500">Live</span>
      </div>
      <span id="last-updated" class="text-[11px] text-slate-400 hidden"></span>
      <button onclick="loadData(true)"
        class="flex items-center gap-1.5 text-xs font-medium text-slate-600 bg-slate-100
               hover:bg-slate-200 px-3 py-1.5 rounded-lg transition-colors">
        <svg id="refresh-icon" class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
            d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0
               0a8.003 8.003 0 01-15.357-2m15.357 2H15"/>
        </svg>
        Actualizar
      </button>
    </div>
  </div>
</nav>

<!-- ── MAIN ───────────────────────────────────────────────────────────── -->
<main class="max-w-7xl mx-auto px-6 py-8 space-y-6">

  <!-- MODEL SELECTOR -->
  <section class="bg-white rounded-xl border border-slate-200 p-5">
    <div class="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
      <div class="min-w-0">
        <p class="text-[11px] font-medium text-slate-400 uppercase tracking-widest">Modelo activo</p>
        <h1 id="model-name" class="text-lg font-semibold text-slate-900 mt-1 truncate">Cargando modelo...</h1>
        <p id="model-description" class="text-xs text-slate-500 mt-1 max-w-3xl truncate">-</p>
      </div>
      <div class="flex flex-col sm:flex-row sm:items-end gap-3">
        <div>
          <label for="model-selector" class="block text-[11px] font-medium text-slate-400 uppercase tracking-widest mb-1">
            Modelo a servir
          </label>
          <select id="model-selector" onchange="changeModel(this.value)"
            class="min-w-[320px] max-w-full text-xs border border-slate-200 rounded-lg px-3 py-2 text-slate-700
                   focus:outline-none focus:border-indigo-400 focus:ring-1 focus:ring-indigo-200 bg-white">
            <option value="">Cargando...</option>
          </select>
        </div>
        <div id="model-metrics" class="flex flex-wrap gap-2 text-[11px] text-slate-500"></div>
      </div>
    </div>
    <p id="model-switch-status" class="text-[11px] text-slate-400 mt-3 hidden"></p>
    <div class="mt-4 pt-4 border-t border-slate-100 flex flex-col lg:flex-row lg:items-end gap-4">
      <div>
        <p class="text-[11px] font-medium text-slate-400 uppercase tracking-widest mb-2">Modo de decision</p>
        <div class="inline-flex rounded-lg border border-slate-200 bg-slate-50 p-1">
          <button id="mode-manual" onclick="setDecisionMode('manual')"
            class="px-3 py-1.5 rounded-md text-xs font-semibold text-slate-600">No automatico</button>
          <button id="mode-automatic" onclick="setDecisionMode('automatic')"
            class="px-3 py-1.5 rounded-md text-xs font-semibold text-slate-600">Automatico</button>
        </div>
      </div>
      <div id="threshold-box" class="hidden">
        <label for="threshold-input" class="block text-[11px] font-medium text-slate-400 uppercase tracking-widest mb-1">
          Threshold automatico
        </label>
        <div class="flex items-center gap-2">
          <input id="threshold-input" type="number" min="0" max="1" step="0.01"
            class="w-28 text-xs border border-slate-200 rounded-lg px-3 py-2 text-slate-700
                   focus:outline-none focus:border-indigo-400 focus:ring-1 focus:ring-indigo-200 bg-white">
          <button onclick="saveThreshold()"
            class="text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-700 px-3 py-2 rounded-lg">
            Guardar
          </button>
          <span id="threshold-suggested" class="text-[11px] text-slate-400"></span>
        </div>
      </div>
      <p id="settings-status" class="text-[11px] text-slate-400"></p>
    </div>
  </section>

  <!-- KPI CARDS -->
  <div class="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">

    <div class="bg-white rounded-xl border border-slate-200 p-5 hover:shadow-md transition-shadow">
      <div class="flex items-start justify-between">
        <div>
          <p class="text-[11px] font-medium text-slate-400 uppercase tracking-widest">Total PRs</p>
          <p id="kpi-total" class="text-3xl font-bold text-slate-900 mt-1 tabular-nums">—</p>
        </div>
        <div class="w-9 h-9 rounded-lg bg-indigo-50 flex items-center justify-center">
          <svg class="w-4 h-4 text-indigo-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
              d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0
                 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
          </svg>
        </div>
      </div>
      <p class="text-[11px] text-slate-400 mt-2">analizadas en total</p>
    </div>

    <div class="hidden bg-white rounded-xl border border-slate-200 p-5 hover:shadow-md transition-shadow">
      <div class="flex items-start justify-between">
        <div>
          <p class="text-[11px] font-medium text-slate-400 uppercase tracking-widest">Score Prom.</p>
          <p id="kpi-avg" class="text-3xl font-bold text-slate-900 mt-1 tabular-nums">—</p>
        </div>
        <div class="w-9 h-9 rounded-lg bg-violet-50 flex items-center justify-center">
          <svg class="w-4 h-4 text-violet-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
              d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"/>
          </svg>
        </div>
      </div>
      <p class="text-[11px] text-slate-400 mt-2">probabilidad de merge</p>
    </div>

    <div class="bg-white rounded-xl border border-slate-200 p-5 hover:shadow-md transition-shadow">
      <div class="flex items-start justify-between">
        <div>
          <p class="text-[11px] font-medium text-slate-400 uppercase tracking-widest">Tasa Merge</p>
          <p id="kpi-rate" class="text-3xl font-bold text-emerald-600 mt-1 tabular-nums">—</p>
        </div>
        <div class="w-9 h-9 rounded-lg bg-emerald-50 flex items-center justify-center">
          <svg class="w-4 h-4 text-emerald-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
              d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
          </svg>
        </div>
      </div>
      <p class="text-[11px] text-slate-400 mt-2">predicciones positivas</p>
    </div>

    <div class="hidden bg-white rounded-xl border border-slate-200 p-5 hover:shadow-md transition-shadow">
      <div class="flex items-start justify-between">
        <div>
          <p class="text-[11px] font-medium text-slate-400 uppercase tracking-widest">Última PR</p>
          <p id="kpi-last" class="text-2xl font-bold text-slate-900 mt-1 font-mono">—</p>
        </div>
        <div class="w-9 h-9 rounded-lg bg-amber-50 flex items-center justify-center">
          <svg class="w-4 h-4 text-amber-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
              d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"/>
          </svg>
        </div>
      </div>
      <p id="kpi-last-sub" class="text-[11px] text-slate-400 mt-2 truncate">—</p>
    </div>
    <div class="bg-white rounded-xl border border-slate-200 p-5 flex flex-col">
      <div class="mb-3">
        <h2 class="text-sm font-semibold text-slate-800">DistribuciÃ³n</h2>
        <p class="text-[11px] text-slate-400 mt-0.5">Merge vs Rechazo predicho</p>
      </div>
      <div class="flex-1 flex items-center justify-center min-h-[150px]">
        <canvas id="chart-donut"></canvas>
      </div>
      <div id="donut-legend" class="flex justify-center gap-4 mt-3"></div>
    </div>

    <div class="bg-white rounded-xl border border-slate-200 p-5">
      <h2 class="text-sm font-semibold text-slate-800 mb-1">Histograma de Scores</h2>
      <p class="text-[11px] text-slate-400 mb-3">Frecuencia de probabilidades de merge</p>
      <div class="h-[150px]">
        <canvas id="chart-histogram"></canvas>
      </div>
    </div>
  </div>

  <!-- CHARTS ROW -->
  <div hidden class="grid grid-cols-1 lg:grid-cols-3 gap-4">

    <!-- Timeline -->
    <div class="hidden lg:col-span-2 bg-white rounded-xl border border-slate-200 p-6">
      <div class="flex items-start justify-between mb-5">
        <div>
          <h2 class="text-sm font-semibold text-slate-800">Score por PR</h2>
          <p class="text-[11px] text-slate-400 mt-0.5">Probabilidad de merge a lo largo del tiempo</p>
        </div>
        <div class="flex items-center gap-3 text-[11px] text-slate-400">
          <span class="flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-full bg-emerald-400 inline-block"></span>Merge</span>
          <span class="flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-full bg-red-400 inline-block"></span>Rechazo</span>
        </div>
      </div>
      <div style="height:210px">
        <canvas id="chart-timeline"></canvas>
      </div>
    </div>

    <!-- Donut -->
    <div class="hidden bg-white rounded-xl border border-slate-200 p-6 flex flex-col">
      <div class="mb-4">
        <h2 class="text-sm font-semibold text-slate-800">Distribución</h2>
        <p class="text-[11px] text-slate-400 mt-0.5">Merge vs Rechazo predicho</p>
      </div>
      <div class="flex-1 flex items-center justify-center" style="min-height:150px">
        <canvas id="chart-donut-old"></canvas>
      </div>
      <div id="donut-legend-old" class="flex justify-center gap-5 mt-4"></div>
    </div>
  </div>

  <!-- CONFIDENCE DISTRIBUTION (mini bar chart) -->
  <div hidden class="grid grid-cols-1 lg:grid-cols-3 gap-4">
    <div class="hidden bg-white rounded-xl border border-slate-200 p-6">
      <h2 class="text-sm font-semibold text-slate-800 mb-4">Distribución de Confianza</h2>
      <div id="conf-bars" class="space-y-3"></div>
    </div>

    <!-- Score histogram -->
    <div class="hidden lg:col-span-2 bg-white rounded-xl border border-slate-200 p-6">
      <h2 class="text-sm font-semibold text-slate-800 mb-1">Histograma de Scores</h2>
      <p class="text-[11px] text-slate-400 mb-4">Frecuencia de probabilidades de merge</p>
      <div style="height:140px">
        <canvas id="chart-histogram-old"></canvas>
      </div>
    </div>
  </div>

  <!-- PR TABLE -->
  <div class="bg-white rounded-xl border border-slate-200 overflow-hidden">
    <div class="px-6 py-4 border-b border-slate-100 flex flex-wrap items-center justify-between gap-3">
      <div>
        <h2 class="text-sm font-semibold text-slate-800">PRs Procesadas</h2>
        <p class="text-[11px] text-slate-400 mt-0.5">Ordenadas por fecha de análisis · Click en SHAP para ver el desglose</p>
      </div>
      <div class="flex items-center gap-2">
        <div class="relative">
          <svg class="w-3.5 h-3.5 text-slate-400 absolute left-2.5 top-1/2 -translate-y-1/2 pointer-events-none"
            fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
              d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/>
          </svg>
          <input id="search-input" type="text" placeholder="Buscar repo, PR..."
            oninput="filterTable()"
            class="pl-8 pr-3 py-1.5 text-xs border border-slate-200 rounded-lg
                   text-slate-700 placeholder-slate-400 focus:outline-none focus:border-indigo-400
                   focus:ring-1 focus:ring-indigo-200 w-48">
        </div>
        <select id="filter-label" onchange="filterTable()"
          class="text-xs border border-slate-200 rounded-lg px-2.5 py-1.5 text-slate-600
                 focus:outline-none focus:border-indigo-400 bg-white">
          <option value="">Todos</option>
          <option value="likely_merged">Solo Merge</option>
          <option value="likely_rejected">Solo Rechazo</option>
        </select>
      </div>
    </div>

    <div class="overflow-x-auto">
      <table class="w-full">
        <thead>
          <tr class="bg-slate-50 border-b border-slate-100">
            <th class="text-left text-[11px] font-semibold text-slate-400 uppercase tracking-wider px-6 py-3">Repo</th>
            <th class="text-left text-[11px] font-semibold text-slate-400 uppercase tracking-wider px-4 py-3">PR</th>
            <th class="text-left text-[11px] font-semibold text-slate-400 uppercase tracking-wider px-4 py-3">Procesado</th>
            <th class="text-left text-[11px] font-semibold text-slate-400 uppercase tracking-wider px-4 py-3 w-44">Score</th>
            <th class="text-left text-[11px] font-semibold text-slate-400 uppercase tracking-wider px-4 py-3">Predicción</th>
            <th class="text-left text-[11px] font-semibold text-slate-400 uppercase tracking-wider px-4 py-3">Confianza</th>
            <th class="text-left text-[11px] font-semibold text-slate-400 uppercase tracking-wider px-4 py-3">Modelo</th>
            <th class="text-center text-[11px] font-semibold text-slate-400 uppercase tracking-wider px-4 py-3">Explicación</th>
          </tr>
        </thead>
        <tbody id="pr-table-body">
          <tr><td colspan="8" class="py-16 text-center">
            <div class="flex flex-col items-center gap-2">
              <div class="w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin"></div>
              <p class="text-sm text-slate-400">Cargando predicciones...</p>
            </div>
          </td></tr>
        </tbody>
      </table>
    </div>

    <div class="px-6 py-3 border-t border-slate-100 flex items-center justify-between">
      <p id="table-footer" class="text-[11px] text-slate-400">— resultados</p>
      <div class="flex items-center gap-2">
        <button id="page-prev" onclick="changePage(-1)"
          class="text-[11px] font-medium text-slate-500 bg-slate-100 hover:bg-slate-200 px-2.5 py-1 rounded-md disabled:opacity-40 disabled:cursor-not-allowed">
          Anterior
        </button>
        <span id="page-info" class="text-[11px] text-slate-400">Página 1</span>
        <button id="page-next" onclick="changePage(1)"
          class="text-[11px] font-medium text-slate-500 bg-slate-100 hover:bg-slate-200 px-2.5 py-1 rounded-md disabled:opacity-40 disabled:cursor-not-allowed">
          Siguiente
        </button>
      </div>
    </div>
  </div>

</main>

<!-- ── JAVASCRIPT ─────────────────────────────────────────────────────── -->
<script>
dayjs.extend(dayjs_plugin_relativeTime);
dayjs.locale('es');

let allData = [];
let modelState = null;
let settingsState = null;
let tableData = [];
let currentPage = 1;
const PAGE_SIZE = 100;
let tlChart = null, donutChart = null, histChart = null;

/* ── Color helpers ──────────────────────────────── */
function scoreColor(p) {
  if (p >= 0.70) return '#10b981';
  if (p >= 0.45) return '#f59e0b';
  return '#ef4444';
}

/* ── Badge renderers ────────────────────────────── */
function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function fmtMetric(value) {
  return typeof value === 'number' ? value.toFixed(3) : 'n/a';
}

function modelOptionLabel(model) {
  const metrics = model.metrics || {};
  const availability = model.available ? '' : ' | no disponible';
  return `${model.name} | AP ${fmtMetric(metrics.ap_not_merged)} | AUC ${fmtMetric(metrics.roc_auc)}${availability}`;
}

function renderModelState(state) {
  modelState = state;
  const active = state.active_model || {};
  const selector = document.getElementById('model-selector');
  const metrics = active.metrics || {};
  selector.innerHTML = (state.models || []).map(model => `
    <option value="${escapeHtml(model.id)}" ${model.available ? '' : 'disabled'}>
      ${escapeHtml(modelOptionLabel(model))}
    </option>
  `).join('');
  selector.value = active.id || '';
  selector.disabled = false;
  document.getElementById('model-name').textContent = active.name || 'Modelo no disponible';
  document.getElementById('model-description').textContent = active.description || '-';
  document.getElementById('model-metrics').innerHTML = `
    <span class="inline-flex items-center rounded-full bg-indigo-50 text-indigo-700 px-2.5 py-1 font-medium">
      AP not_merged ${fmtMetric(metrics.ap_not_merged)}
    </span>
    <span class="inline-flex items-center rounded-full bg-emerald-50 text-emerald-700 px-2.5 py-1 font-medium">
      ROC-AUC ${fmtMetric(metrics.roc_auc)}
    </span>
    <span class="inline-flex items-center rounded-full bg-slate-100 text-slate-600 px-2.5 py-1 font-medium">
      ${active.feature_count || '-'} features
    </span>`;
}

async function loadModels() {
  try {
    const res = await fetch('/api/models');
    if (!res.ok) throw new Error('HTTP ' + res.status);
    renderModelState(await res.json());
  } catch(e) {
    document.getElementById('model-name').textContent = 'No se pudo cargar el modelo';
    document.getElementById('model-description').textContent = String(e.message || e);
  }
}

async function changeModel(modelId) {
  if (!modelId || (modelState?.active_model?.id === modelId)) return;
  const selector = document.getElementById('model-selector');
  const status = document.getElementById('model-switch-status');
  selector.disabled = true;
  status.textContent = 'Cambiando modelo activo...';
  status.classList.remove('hidden', 'text-red-500');
  status.classList.add('text-slate-400');
  try {
    const res = await fetch('/api/models/active', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({model_id: modelId}),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'No se pudo cambiar el modelo');
    renderModelState(data);
    status.textContent = 'Modelo activo actualizado. Las nuevas predicciones usan este modelo.';
  } catch(e) {
    status.textContent = 'Error: ' + (e.message || e);
    status.classList.remove('text-slate-400');
    status.classList.add('text-red-500');
    selector.disabled = false;
    if (modelState) renderModelState(modelState);
  }
}

function renderSettings(state) {
  settingsState = state;
  const manual = document.getElementById('mode-manual');
  const automatic = document.getElementById('mode-automatic');
  const autoOn = state.mode === 'automatic';
  manual.className = autoOn
    ? 'px-3 py-1.5 rounded-md text-xs font-semibold text-slate-600'
    : 'px-3 py-1.5 rounded-md text-xs font-semibold bg-white text-indigo-700 shadow-sm';
  automatic.className = autoOn
    ? 'px-3 py-1.5 rounded-md text-xs font-semibold bg-white text-indigo-700 shadow-sm'
    : 'px-3 py-1.5 rounded-md text-xs font-semibold text-slate-600';
  document.getElementById('threshold-box').classList.toggle('hidden', !autoOn);
  document.getElementById('threshold-input').value = Number(state.threshold || 0.5).toFixed(2);
  document.getElementById('threshold-suggested').textContent =
    `sugerido ${Number(state.suggested_threshold || 0.5).toFixed(2)}`;
}

async function loadSettings() {
  try {
    const res = await fetch('/api/settings');
    if (!res.ok) throw new Error('HTTP ' + res.status);
    renderSettings(await res.json());
  } catch(e) {
    document.getElementById('settings-status').textContent = 'No se pudo cargar settings';
  }
}

async function saveSettings(payload) {
  const status = document.getElementById('settings-status');
  status.textContent = 'Guardando...';
  try {
    const res = await fetch('/api/settings', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'No se pudo guardar');
    renderSettings(data);
    status.textContent = data.mode === 'automatic'
      ? 'Automatico activo para nuevas predicciones.'
      : 'Modo no automatico activo.';
  } catch(e) {
    status.textContent = 'Error: ' + (e.message || e);
  }
}

function setDecisionMode(mode) {
  const threshold = settingsState?.threshold ?? 0.5;
  saveSettings({mode, threshold});
}

function saveThreshold() {
  const threshold = Number(document.getElementById('threshold-input').value);
  saveSettings({mode: settingsState?.mode || 'automatic', threshold});
}

function labelBadge(l) {
  const ok = l === 'likely_merged';
  return `<span class="inline-flex items-center gap-1 text-[11px] font-semibold px-2.5 py-0.5 rounded-full ${ok ? 'badge-merge' : 'badge-reject'}">
    ${ok ? '<svg class="w-3 h-3" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"/></svg>' : '<svg class="w-3 h-3" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"/></svg>'}
    ${ok ? 'Merge' : 'Rechazo'}
  </span>`;
}

function confBadge(c) {
  const map   = {high:'badge-high', medium:'badge-medium', low:'badge-low'};
  const label = {high:'Alta', medium:'Media', low:'Baja'};
  return `<span class="text-[11px] font-medium px-2 py-0.5 rounded-full ${map[c]||'badge-low'}">${label[c]||c}</span>`;
}

function decisionBadge(pr) {
  if (!pr.auto_decision) return '';
  const ok = pr.auto_decision === 'merge';
  return `<span class="ml-1 inline-flex text-[10px] font-semibold px-2 py-0.5 rounded-full ${ok ? 'badge-merge' : 'badge-reject'}">auto: ${ok ? 'merge' : 'no merge'}</span>`;
}

function shortRepo(repo) {
  if (!repo) return '<span class="text-slate-300">—</span>';
  const [owner, name] = repo.split('/');
  if (!name) return `<span class="text-xs font-medium text-slate-700">${repo}</span>`;
  return `<span class="text-[11px] text-slate-400">${owner}/</span><span class="text-[13px] font-semibold text-slate-800">${name}</span>`;
}

/* ── SHAP panel ─────────────────────────────────── */
function modelBadge(pr) {
  const name = pr.model_name || pr.model_id || 'legacy';
  return `<span class="inline-flex max-w-[180px] truncate text-[11px] font-medium px-2 py-0.5 rounded-full bg-slate-100 text-slate-600" title="${escapeHtml(name)}">${escapeHtml(name)}</span>`;
}

const _shapCache = {};  // { id: top_factors[] }

function renderShapBars(factors) {
  if (!factors || !factors.length) return '<p class="text-xs text-slate-400 py-2">Sin datos SHAP disponibles.</p>';
  const maxAbs = Math.max(...factors.map(f => Math.abs(f.impact)), 0.0001);
  return factors.map(f => {
    const pct = Math.round(Math.abs(f.impact) / maxAbs * 100);
    const pos = f.direction === 'hacia_merge';
    const dirLabel = pos
      ? `<span class="text-emerald-600 font-semibold">↑ merge</span>`
      : `<span class="text-red-500 font-semibold">↓ rechazo</span>`;
    const negBar = pos ? '' : `<div class="shap-neg-fill" style="width:${pct}%"></div>`;
    const posBar = pos ? `<div class="shap-pos-fill" style="width:${pct}%"></div>` : '';
    return `<div class="flex items-center gap-3 py-2 border-b border-slate-50 last:border-0">
      <div class="w-52 text-xs text-slate-700 font-medium truncate" title="${f.feature}">${f.feature}</div>
      <div class="flex-1 flex items-center gap-0.5">
        <div class="flex-1 shap-half">${negBar}</div>
        <div class="shap-center-line"></div>
        <div class="flex-1 shap-half right">${posBar}</div>
      </div>
      <div class="w-24 text-right text-[11px]">${dirLabel}</div>
      <div class="w-14 text-right font-mono text-[11px] text-slate-400">${f.impact > 0 ? '+' : ''}${f.impact.toFixed(3)}</div>
    </div>`;
  }).join('');
}

function shapPlaceholderRow(id, hasFeatures, explanation, needsExplanation) {
  return `<tr class="shap-row" id="shap-${id}">
    <td colspan="8" class="bg-gradient-to-b from-slate-50 to-white px-10 py-4 border-b border-slate-100">
      <div class="flex items-center gap-2 mb-3">
        <svg class="w-3.5 h-3.5 text-violet-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
            d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
        </svg>
        <p class="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">
          Impacto SHAP — factores que más influyeron en la predicción
        </p>
        <div class="ml-auto flex items-center gap-4 text-[10px] text-slate-400">
          <span class="flex items-center gap-1"><span class="inline-block w-3 h-2 rounded-sm bg-emerald-400"></span> hacia merge</span>
          <span class="flex items-center gap-1"><span class="inline-block w-3 h-2 rounded-sm bg-red-400"></span> contra merge</span>
        </div>
      </div>
      <div id="shap-content-${id}" class="space-y-0">
        ${hasFeatures
          ? `<div class="flex items-center gap-2 py-3 text-slate-400 text-xs">
               <div class="w-4 h-4 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin"></div>
               Calculando explicación SHAP...
             </div>`
          : `<p class="text-xs text-slate-400 py-2">Features no disponibles para esta predicción.</p>`}
      </div>
      ${explanation ? `<div class="mt-4 rounded-lg border border-amber-200 bg-amber-50 p-3">
        <p class="text-[11px] font-semibold text-amber-700 uppercase tracking-wider mb-2">Explicacion de Claude</p>
        <p class="whitespace-pre-wrap text-xs text-slate-700 leading-relaxed">${escapeHtml(explanation)}</p>
      </div>` : ''}
      ${!explanation && needsExplanation ? `<div class="mt-4 rounded-lg border border-slate-200 bg-slate-50 p-3">
        <p class="text-xs text-slate-500">Explicacion de Claude pendiente.</p>
      </div>` : ''}
    </td>
  </tr>`;
}

async function toggleShap(id, hasFeatures) {
  const row     = document.getElementById('shap-' + id);
  const btn     = document.getElementById('btn-' + id);
  const content = document.getElementById('shap-content-' + id);
  if (!row) return;

  const isOpen = row.classList.contains('open');
  if (isOpen) {
    row.classList.remove('open');
    btn.classList.remove('open');
    btn.textContent = 'Ver SHAP';
    return;
  }

  // Abrir panel
  row.classList.add('open');
  btn.classList.add('open');
  btn.textContent = 'Ocultar';

  // Si ya lo tenemos cacheado en JS, renderizar directo
  if (_shapCache[id]) {
    content.innerHTML = renderShapBars(_shapCache[id]);
    return;
  }

  if (!hasFeatures) return;

  // Spinner ya está en el DOM — fetchear del server
  try {
    const res  = await fetch('/api/shap/' + id);
    const data = await res.json();
    if (data.error) {
      content.innerHTML = `<p class="text-xs text-red-400 py-2">Error: ${data.error}</p>`;
      return;
    }
    _shapCache[id] = data.top_factors;
    content.innerHTML = renderShapBars(data.top_factors);
  } catch(e) {
    content.innerHTML = `<p class="text-xs text-red-400 py-2">Error al calcular SHAP.</p>`;
  }
}

/* ── Table render ───────────────────────────────── */
function renderTable(data) {
  tableData = data;
  currentPage = 1;
  renderTablePage();
}

function renderTablePage() {
  const tbody = document.getElementById('pr-table-body');
  const total = tableData.length;
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  currentPage = Math.min(Math.max(currentPage, 1), totalPages);
  const start = (currentPage - 1) * PAGE_SIZE;
  const end = Math.min(start + PAGE_SIZE, total);
  const data = tableData.slice(start, end);

  document.getElementById('table-footer').textContent =
    total
      ? `${start + 1}-${end} de ${total} resultado${total !== 1 ? 's' : ''}`
      : '0 resultados';
  document.getElementById('page-info').textContent = `Página ${currentPage} de ${totalPages}`;
  document.getElementById('page-prev').disabled = currentPage <= 1;
  document.getElementById('page-next').disabled = currentPage >= totalPages;

  if (!total) {
    tbody.innerHTML = `<tr><td colspan="8" class="py-16 text-center">
      <div class="flex flex-col items-center gap-2">
        <svg class="w-10 h-10 text-slate-200" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5"
            d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293
               l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
        </svg>
        <p class="text-sm text-slate-400 font-medium">Sin predicciones aún</p>
        <p class="text-xs text-slate-300">Analizá una PR con el tool predict_pr_merge</p>
      </div>
    </td></tr>`;
    return;
  }

  tbody.innerHTML = data.map((pr, i) => {
    const id  = pr.id ?? i;
    const pct = Math.round((pr.merge_probability || 0) * 100);
    const col = scoreColor(pr.merge_probability || 0);
    const time = pr.processed_at ? dayjs(pr.processed_at).fromNow() : '—';
    const num  = pr.pr_number || '?';
    const hasFeat = !!pr.has_features;

    // Si SHAP ya está cacheado en el server, pre-cargarlo en el cache JS
    // (shap_ready = 1 indica que ya existe en DB, se fetcheará al primer click)

    return `
    <tr class="pr-row border-b border-slate-50" data-repo="${(pr.repo||'').toLowerCase()}" data-num="${num}" data-label="${pr.label||''}" data-model="${(pr.model_name||pr.model_id||'').toLowerCase()}">
      <td class="px-6 py-3.5">${shortRepo(pr.repo)}</td>
      <td class="px-4 py-3.5">
        <a href="${pr.pr_url}" target="_blank"
           class="font-mono text-xs text-indigo-600 hover:text-indigo-800 hover:underline font-medium">
          #${num}
        </a>
      </td>
      <td class="px-4 py-3.5 text-xs text-slate-400 whitespace-nowrap">${time}</td>
      <td class="px-4 py-3.5">
        <div class="flex items-center gap-2">
          <div class="flex-1 score-track" style="min-width:80px">
            <div class="score-fill" style="width:${pct}%;background:${col}"></div>
          </div>
          <span class="text-xs font-bold tabular-nums w-8 text-right" style="color:${col}">${pct}%</span>
        </div>
      </td>
      <td class="px-4 py-3.5">${labelBadge(pr.label)}${decisionBadge(pr)}</td>
      <td class="px-4 py-3.5">${confBadge(pr.confidence)}</td>
      <td class="px-4 py-3.5">${modelBadge(pr)}</td>
      <td class="px-4 py-3.5 text-center">
        ${hasFeat
          ? `<button id="btn-${id}" class="btn-shap" onclick="toggleShap(${id}, true)">Ver SHAP</button>`
          : '<span class="text-slate-200 text-xs">—</span>'}
      </td>
    </tr>
    ${hasFeat ? shapPlaceholderRow(
      id,
      true,
      pr.negative_explanation,
      pr.auto_decision === 'no_merge'
    ) : ''}`;
  }).join('');
}

function changePage(delta) {
  currentPage += delta;
  renderTablePage();
}

function filterTable() {
  const q     = (document.getElementById('search-input').value || '').toLowerCase().trim();
  const label = document.getElementById('filter-label').value;
  const filtered = allData.filter(pr => {
    const repo = (pr.repo || '').toLowerCase();
    const num = String(pr.pr_number || '');
    const lbl = pr.label || '';
    const model = (pr.model_name || pr.model_id || '').toLowerCase();
    const matchQ = !q || repo.includes(q) || num.includes(q) || model.includes(q);
    const matchL = !label || lbl === label;
    return matchQ && matchL;
  });
  renderTable(filtered);
}

/* ── KPI render ─────────────────────────────────── */
function renderKpis(data) {
  document.getElementById('kpi-total').textContent = data.length;
  if (!data.length) return;
  const avg   = data.reduce((s,d) => s + (d.merge_probability||0), 0) / data.length;
  const mergeN = data.filter(d => d.label === 'likely_merged').length;
  document.getElementById('kpi-avg').textContent  = Math.round(avg*100) + '%';
  document.getElementById('kpi-rate').textContent = Math.round(mergeN/data.length*100) + '%';
  const last = data[0];
  document.getElementById('kpi-last').textContent    = '#' + (last.pr_number || '—');
  document.getElementById('kpi-last-sub').textContent = last.repo || '—';
}

/* ── Confidence bars ────────────────────────────── */
function renderConfBars(data) {
  const counts = {high:0, medium:0, low:0};
  data.forEach(d => { if (counts[d.confidence] !== undefined) counts[d.confidence]++; });
  const total = data.length || 1;
  const colors = {high:'bg-blue-400', medium:'bg-amber-400', low:'bg-slate-300'};
  const labels = {high:'Alta', medium:'Media', low:'Baja'};
  document.getElementById('conf-bars').innerHTML = Object.entries(counts).map(([k,v]) => `
    <div>
      <div class="flex justify-between text-xs text-slate-600 mb-1">
        <span class="font-medium">${labels[k]}</span>
        <span class="tabular-nums text-slate-400">${v} (${Math.round(v/total*100)}%)</span>
      </div>
      <div class="w-full bg-slate-100 rounded-full h-2">
        <div class="${colors[k]} h-2 rounded-full transition-all duration-700" style="width:${Math.round(v/total*100)}%"></div>
      </div>
    </div>`).join('');
}

/* ── Charts ─────────────────────────────────────── */
function renderTimeline(data) {
  const ordered = [...data].reverse().slice(-60);
  const labels  = ordered.map(d => dayjs(d.processed_at).format('DD/MM HH:mm'));
  const scores  = ordered.map(d => d.merge_probability || 0);
  const ptColors = ordered.map(d => scoreColor(d.merge_probability || 0));

  if (tlChart) tlChart.destroy();
  const ctx = document.getElementById('chart-timeline').getContext('2d');
  const grad = ctx.createLinearGradient(0, 0, 0, 210);
  grad.addColorStop(0, 'rgba(99,102,241,0.15)');
  grad.addColorStop(1, 'rgba(99,102,241,0)');
  tlChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        data: scores,
        borderColor: '#6366f1',
        borderWidth: 2,
        backgroundColor: grad,
        pointBackgroundColor: ptColors,
        pointBorderColor: '#fff',
        pointBorderWidth: 2,
        pointRadius: 5,
        pointHoverRadius: 7,
        fill: true,
        tension: 0.35,
      }]
    },
    options: {
      responsive:true, maintainAspectRatio:false,
      plugins: {
        legend: {display:false},
        tooltip: {
          callbacks: {
            label: ctx => ' ' + Math.round(ctx.raw*100) + '% merge',
          },
          backgroundColor:'#1e293b',
          titleFont:{size:11}, bodyFont:{size:12},
          padding:10, cornerRadius:8,
        }
      },
      scales: {
        y: {
          min:0, max:1,
          ticks: { callback: v => Math.round(v*100)+'%', font:{size:11}, color:'#94a3b8' },
          grid: { color:'#f1f5f9' },
          border: {dash:[4,4]},
        },
        x: {
          ticks: { font:{size:10}, color:'#94a3b8', maxRotation:30, maxTicksLimit:8 },
          grid: { display:false },
        }
      }
    }
  });
}

function renderDonut(data) {
  const mergeN  = data.filter(d => d.label === 'likely_merged').length;
  const rejectN = data.length - mergeN;
  if (donutChart) donutChart.destroy();
  const ctx = document.getElementById('chart-donut').getContext('2d');
  donutChart = new Chart(ctx, {
    type: 'pie',
    data: {
      datasets: [{
        data: [mergeN, rejectN],
        backgroundColor: ['#10b981','#f87171'],
        borderColor: '#fff', borderWidth: 4,
        hoverBorderWidth: 4,
      }]
    },
    options: {
      responsive:true, maintainAspectRatio:false,
      plugins: {
        legend: {display:false},
        tooltip: {
          callbacks: { label: c => ' ' + c.label + ': ' + c.raw },
          backgroundColor:'#1e293b', padding:10, cornerRadius:8,
        }
      }
    }
  });
  document.getElementById('donut-legend').innerHTML = `
    <span class="flex items-center gap-1.5 text-xs text-slate-600">
      <span class="w-3 h-3 rounded-full bg-emerald-400 inline-block"></span>
      <strong>${mergeN}</strong> Merge
    </span>
    <span class="flex items-center gap-1.5 text-xs text-slate-600">
      <span class="w-3 h-3 rounded-full bg-red-400 inline-block"></span>
      <strong>${rejectN}</strong> Rechazo
    </span>`;
}

function renderHistogram(data) {
  const bins = Array(10).fill(0);
  data.forEach(d => {
    const idx = Math.min(9, Math.floor((d.merge_probability||0) * 10));
    bins[idx]++;
  });
  const labels = ['0–10','10–20','20–30','30–40','40–50','50–60','60–70','70–80','80–90','90–100'];
  const colors = bins.map((_, i) => i >= 5 ? '#10b981' : '#f87171');

  if (histChart) histChart.destroy();
  const ctx = document.getElementById('chart-histogram').getContext('2d');
  histChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data: bins, backgroundColor: colors,
        borderRadius:4, borderSkipped:false,
      }]
    },
    options: {
      responsive:true, maintainAspectRatio:false,
      plugins: { legend:{display:false} },
      scales: {
        y: { ticks:{font:{size:10},color:'#94a3b8'}, grid:{color:'#f1f5f9'} },
        x: { ticks:{font:{size:10},color:'#94a3b8'}, grid:{display:false} }
      }
    }
  });
}

/* ── Main loader ────────────────────────────────── */
async function loadData(showSpin) {
  if (showSpin) {
    const icon = document.getElementById('refresh-icon');
    icon.classList.add('animate-spin');
    setTimeout(() => icon.classList.remove('animate-spin'), 800);
  }
  try {
    const res  = await fetch('/api/predictions');
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    allData = data;
    renderKpis(data);
    renderDonut(data);
    renderHistogram(data);
    renderTable(data);
    const now = dayjs().format('HH:mm:ss');
    const el  = document.getElementById('last-updated');
    el.textContent = 'Actualizado a las ' + now;
    el.classList.remove('hidden');
  } catch(e) {
    console.error('Error al cargar datos:', e);
  }
}

/* ── Auto-refresh cada 60s ──────────────────────── */
loadModels();
loadSettings();
loadData(false);
setInterval(() => loadData(false), 60000);
</script>
</body>
</html>"""
