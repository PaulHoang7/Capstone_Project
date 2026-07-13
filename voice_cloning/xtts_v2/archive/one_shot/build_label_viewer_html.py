"""Generate static HTML labeling tool — no server needed.

Opens directly in browser via file:// protocol. All images load from
relative paths (bubble_crops/*.png).

Features:
- Table with: crop image | prefill | editable text | skip checkbox
- Progress saved in localStorage (resume across sessions)
- Export button downloads updated CSV
- Pagination (50 rows/page) for performance
"""
import argparse
import csv
import json
from pathlib import Path


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>VN Manga OCR Labeler (Static)</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: system-ui, sans-serif; background: #1a1a1a; color: #eee; margin: 0; padding: 0; }
  header { background: #222; padding: 14px 24px; position: sticky; top: 0; z-index: 100; display: flex; align-items: center; gap: 24px; border-bottom: 2px solid #333; }
  header h1 { margin: 0; color: #4da6ff; font-size: 18px; }
  #stats { color: #aaa; font-size: 13px; }
  .progress { flex: 1; background: #333; height: 10px; border-radius: 5px; overflow: hidden; }
  .progress-fill { background: linear-gradient(90deg, #4caf50, #4da6ff); height: 100%; transition: width 0.3s; }
  button {
    padding: 8px 16px; border: none; border-radius: 4px;
    font-weight: 600; cursor: pointer; font-size: 13px;
  }
  .btn-export { background: #4caf50; color: white; }
  .btn-nav { background: #444; color: white; }
  .filter-bar { padding: 10px 24px; background: #252525; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
  select, input { padding: 6px 10px; background: #333; color: white; border: 1px solid #555; border-radius: 3px; }
  table { width: 100%; border-collapse: collapse; }
  tbody tr { border-bottom: 1px solid #333; }
  tbody tr:hover { background: #252525; }
  tbody tr.labeled { background: #1a3a1a; }
  tbody tr.skipped { background: #3a1a1a; opacity: 0.6; }
  td { padding: 12px 8px; vertical-align: middle; }
  td.crop-cell { width: 440px; text-align: center; }
  td.crop-cell img {
    max-width: 420px; max-height: 340px; border-radius: 4px; background: #000;
    cursor: zoom-in; transition: transform 0.15s;
  }
  td.crop-cell img:hover { transform: scale(1.02); box-shadow: 0 0 0 2px #4da6ff; }
  /* Fullscreen zoom modal */
  #zoom-overlay {
    display: none; position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
    background: rgba(0,0,0,0.92); z-index: 1000; cursor: zoom-out;
    justify-content: center; align-items: center; padding: 40px; box-sizing: border-box;
  }
  #zoom-overlay.active { display: flex; }
  #zoom-overlay img { max-width: 95vw; max-height: 95vh; box-shadow: 0 0 40px #000; }
  #zoom-hint {
    position: fixed; top: 20px; right: 24px; color: #aaa; font-size: 13px;
    background: rgba(0,0,0,0.6); padding: 6px 12px; border-radius: 3px;
  }
  td.id-cell { width: 80px; font-family: monospace; color: #888; font-size: 12px; }
  td.prefill-cell { color: #ffa726; font-family: monospace; font-size: 13px; width: 25%; padding: 12px; background: #2a2000; border-radius: 4px; margin: 4px; }
  td.text-cell { width: 30%; padding: 8px; }
  td.text-cell textarea {
    width: 100%; background: #111; color: #fff; border: 2px solid #555;
    padding: 8px; font-family: monospace; font-size: 14px;
    border-radius: 4px; resize: vertical; min-height: 56px;
  }
  td.text-cell textarea:focus { border-color: #4da6ff; outline: none; }
  td.text-cell textarea.edited { border-color: #ffc107; }
  td.skip-cell { width: 60px; text-align: center; }
  td.skip-cell input[type=checkbox] { width: 22px; height: 22px; cursor: pointer; }
  .pagination { display: flex; justify-content: center; gap: 6px; padding: 16px; background: #222; }
  .pagination button { padding: 6px 12px; }
  .pagination button.active { background: #4da6ff; }
  .hint { color: #888; font-size: 11px; padding: 6px 24px; background: #181818; }
</style>
</head>
<body>

<header>
  <h1>📝 VN Manga OCR Labeler</h1>
  <div id="stats">0/0 labeled</div>
  <div class="progress"><div class="progress-fill" id="progress-fill" style="width:0%"></div></div>
  <button class="btn-export" onclick="exportCSV()">💾 Export CSV</button>
</header>

<div class="filter-bar">
  <label>Series:</label>
  <select id="filter-series" onchange="applyFilter()">
    <option value="">all</option>
  </select>
  <label>Show:</label>
  <select id="filter-status" onchange="applyFilter()">
    <option value="">all</option>
    <option value="unlabeled">unlabeled only</option>
    <option value="labeled">labeled only</option>
    <option value="skipped">skipped only</option>
  </select>
  <label>Per page:</label>
  <select id="per-page" onchange="renderPage()">
    <option>25</option>
    <option selected>50</option>
    <option>100</option>
  </select>
  <span id="filter-count" style="color:#aaa"></span>
</div>

<div class="hint">Tip: click textarea → type corrected text → Tab to next row. Click ảnh bubble để xem full-size. Check "Skip" for non-bubble images.</div>

<table>
  <tbody id="rows"></tbody>
</table>

<div class="pagination" id="pagination"></div>

<div id="zoom-overlay" onclick="closeZoom()">
  <span id="zoom-hint">Click anywhere or press ESC to close</span>
  <img id="zoom-img" src="" alt="full-size bubble">
</div>

<script>
const ROWS_DATA = __DATA_PLACEHOLDER__;
const STORAGE_KEY = 'vn_manga_labels_v1';

let filtered = ROWS_DATA.slice();
let currentPage = 0;

function loadSaved() {
  const s = localStorage.getItem(STORAGE_KEY);
  if (!s) return;
  const saved = JSON.parse(s);
  for (const row of ROWS_DATA) {
    if (saved[row.id]) {
      row.corrected_text = saved[row.id].corrected_text || '';
      row.skip = saved[row.id].skip || '';
    }
  }
}

function saveToStorage() {
  const out = {};
  for (const row of ROWS_DATA) {
    if (row.corrected_text || row.skip) {
      out[row.id] = {corrected_text: row.corrected_text, skip: row.skip};
    }
  }
  localStorage.setItem(STORAGE_KEY, JSON.stringify(out));
}

function applyFilter() {
  const series = document.getElementById('filter-series').value;
  const status = document.getElementById('filter-status').value;
  filtered = ROWS_DATA.filter(r => {
    if (series && r.series !== series) return false;
    if (status === 'unlabeled' && (r.corrected_text || r.skip)) return false;
    if (status === 'labeled' && !r.corrected_text) return false;
    if (status === 'skipped' && r.skip !== 'x') return false;
    return true;
  });
  document.getElementById('filter-count').textContent = `(${filtered.length} rows)`;
  currentPage = 0;
  renderPage();
}

function renderPage() {
  const perPage = parseInt(document.getElementById('per-page').value);
  const start = currentPage * perPage;
  const end = Math.min(start + perPage, filtered.length);
  const tbody = document.getElementById('rows');
  tbody.innerHTML = '';
  for (let i = start; i < end; i++) {
    const r = filtered[i];
    const tr = document.createElement('tr');
    tr.dataset.id = r.id;
    if (r.skip === 'x') tr.classList.add('skipped');
    else if (r.corrected_text) tr.classList.add('labeled');
    tr.innerHTML = `
      <td class="id-cell">${r.id}<br><small>${r.series}</small></td>
      <td class="crop-cell"><img loading="lazy" src="${r.image}" alt="crop ${r.id}"></td>
      <td class="prefill-cell">${escapeHTML(r.prefill)}</td>
      <td class="text-cell"><textarea data-field="corrected_text" data-id="${r.id}" placeholder="Gõ text đúng ở đây (để trống = dùng prefill)">${escapeHTML(r.corrected_text || '')}</textarea></td>
      <td class="skip-cell"><input type="checkbox" data-field="skip" data-id="${r.id}" ${r.skip === 'x' ? 'checked' : ''}></td>
    `;
    tbody.appendChild(tr);
  }
  // Bind events
  tbody.querySelectorAll('textarea').forEach(el => {
    el.addEventListener('input', e => {
      const id = e.target.dataset.id;
      const row = ROWS_DATA.find(r => r.id === id);
      row.corrected_text = e.target.value;
      e.target.classList.add('edited');
      updateRowClass(e.target.closest('tr'), row);
      debouncedSave();
      updateStats();
    });
  });
  tbody.querySelectorAll('input[type=checkbox]').forEach(el => {
    el.addEventListener('change', e => {
      const id = e.target.dataset.id;
      const row = ROWS_DATA.find(r => r.id === id);
      row.skip = e.target.checked ? 'x' : '';
      updateRowClass(e.target.closest('tr'), row);
      saveToStorage();
      updateStats();
    });
  });
  // Click crop image → fullscreen zoom
  tbody.querySelectorAll('.crop-cell img').forEach(el => {
    el.addEventListener('click', e => {
      const overlay = document.getElementById('zoom-overlay');
      const zoomImg = document.getElementById('zoom-img');
      zoomImg.src = e.target.src;
      overlay.classList.add('active');
    });
  });
  renderPagination();
  updateStats();
}

function closeZoom() {
  document.getElementById('zoom-overlay').classList.remove('active');
}
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeZoom();
});

function updateRowClass(tr, row) {
  tr.classList.remove('labeled', 'skipped');
  if (row.skip === 'x') tr.classList.add('skipped');
  else if (row.corrected_text) tr.classList.add('labeled');
}

function renderPagination() {
  const perPage = parseInt(document.getElementById('per-page').value);
  const pages = Math.ceil(filtered.length / perPage);
  const pgEl = document.getElementById('pagination');
  pgEl.innerHTML = '';
  const makeBtn = (label, pg, active) => {
    const b = document.createElement('button');
    b.textContent = label;
    b.className = 'btn-nav' + (active ? ' active' : '');
    b.onclick = () => { currentPage = pg; renderPage(); window.scrollTo(0, 0); };
    return b;
  };
  if (currentPage > 0) pgEl.appendChild(makeBtn('◀ Prev', currentPage - 1));
  // Show first, last, current, current-1, current+1
  const shown = new Set([0, pages-1, currentPage, currentPage-1, currentPage+1]);
  let last = -2;
  for (let i = 0; i < pages; i++) {
    if (!shown.has(i) && i !== 0 && i !== pages-1) continue;
    if (i > last + 1) {
      const sp = document.createElement('span'); sp.textContent = '...'; sp.style.padding = '0 8px';
      pgEl.appendChild(sp);
    }
    pgEl.appendChild(makeBtn(String(i+1), i, i === currentPage));
    last = i;
  }
  if (currentPage < pages - 1) pgEl.appendChild(makeBtn('Next ▶', currentPage + 1));
}

function updateStats() {
  const labeled = ROWS_DATA.filter(r => r.corrected_text || r.skip === 'x').length;
  const total = ROWS_DATA.length;
  const pct = (labeled / total * 100).toFixed(1);
  document.getElementById('stats').textContent = `${labeled}/${total} labeled (${pct}%)`;
  document.getElementById('progress-fill').style.width = pct + '%';
}

let saveTimer = null;
function debouncedSave() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(saveToStorage, 500);
}

function exportCSV() {
  const headers = ['id','series','page','order','image','prefill','corrected_text','skip'];
  const escape = s => {
    if (s == null) return '';
    s = String(s);
    if (s.includes(',') || s.includes('"') || s.includes('\\n')) {
      return '"' + s.replace(/"/g, '""') + '"';
    }
    return s;
  };
  const lines = ['\\ufeff' + headers.join(',')];
  for (const r of ROWS_DATA) {
    lines.push(headers.map(h => escape(r[h] || '')).join(','));
  }
  const blob = new Blob([lines.join('\\n')], {type: 'text/csv;charset=utf-8'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'labels_edited.csv'; a.click();
  URL.revokeObjectURL(url);
}

function escapeHTML(s) {
  return (s||'').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

// Init
loadSaved();
// Populate series filter
const seriesSet = new Set(ROWS_DATA.map(r => r.series));
const sSel = document.getElementById('filter-series');
for (const s of Array.from(seriesSet).sort()) {
  const opt = document.createElement('option');
  opt.value = s; opt.textContent = s;
  sSel.appendChild(opt);
}
applyFilter();
</script>
</body>
</html>
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="/home/bes/Desktop/Tin/labeling_task/labels.csv")
    p.add_argument("--out", default="/home/bes/Desktop/Tin/labeling_task/labeler.html")
    args = p.parse_args()

    rows = []
    with open(args.csv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"Loaded {len(rows)} rows from {args.csv}")

    data_json = json.dumps(rows, ensure_ascii=False)
    html = HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", data_json)
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"Wrote labeler: {args.out}")
    print(f"\nOpen in browser: file://{Path(args.out).absolute()}")


if __name__ == "__main__":
    main()
