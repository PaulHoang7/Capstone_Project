"""XTTS casting server (CTC v3) — mirror of the VITS2 casting server, but voices
are addressed by an IN-DOMAIN VieNeu reference clip (one per trained speaker)
instead of an integer sid. On-demand synth, model + cond-latents cached.

Run (in xtts_env):
  PYTHONPATH=$PWD:$PWD/Capstone_project/voice_cloning/xtts_v2/coqui_tts \
  /home/bes/miniconda3/envs/xtts_env/bin/python -m uvicorn \
    Capstone_project.voice_cloning.xtts_v2.casting_server_xtts:app --host 0.0.0.0 --port 8777
"""
import json, os, sys, hashlib
from pathlib import Path
import numpy as np
from scipy.io.wavfile import write as wav_write
from flask import Flask, jsonify, send_file, request, Response

R = "/home/bes/Desktop/Tin"
XV = f"{R}/Capstone_project/voice_cloning/xtts_v2"
sys.path.insert(0, f"{XV}/coqui_tts"); sys.path.insert(0, XV)
CACHE = Path("/tmp/casting_xtts_cache"); CACHE.mkdir(exist_ok=True)
CKPT = "/mnt/nfs-data/tin_dataset/checkpoints/xtts_vieneu_ctc_v3_inference"
GALLERY_SENT = "Xin chào, đây là giọng đọc thử cho nhân vật trong truyện."

import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from xtts_gen_config import gen_config_for_text, clean_for_xtts

_cfg = XttsConfig(); _cfg.load_json(os.path.join(CKPT, "config.json"))
_xtts = Xtts.init_from_config(_cfg)
_xtts.load_checkpoint(_cfg, checkpoint_dir=CKPT, use_deepspeed=False, eval=True)
_xtts.cuda()

REFS = {int(k): v for k, v in json.load(open("/tmp/xtts_sid_refs.json")).items()}
# drop the one speaker VITS2 screening flagged as broken (proxy for bad data)
try:
    _bad = {r["sid"] for r in json.load(open("/tmp/screen193/cer.json")) if r["cer"] > 0.30}
except Exception:
    _bad = set()
VOICES = sorted(s for s in REFS if s not in _bad)


def _name(sid):
    # FULL speaker name = prefix_book (e.g. capybara1812_1003). The prefix alone
    # (capybara1812) is just a dataset source tag — different books under the same
    # prefix are genuinely DIFFERENT voices (ECAPA: same-speaker 0.72 vs
    # same-prefix-diff-book 0.29 ≈ cross-prefix 0.28). So show the full name.
    base = os.path.basename(REFS[sid])
    return "_".join(base.split("_")[:2])


_pages = json.load(open(f"{R}/zip/vits2_page_demo/demo.json"))
LINES = []
for pg in _pages:
    for ln in pg["lines"]:
        LINES.append({"idx": len(LINES), "speaker": ln["speaker"], "text": ln["text"]})
CHARACTERS = sorted({l["speaker"] for l in LINES}, key=lambda s: (len(s), s))

_cond = {}
def _get_cond(sid):
    if sid not in _cond:
        _cond[sid] = _xtts.get_conditioning_latents(audio_path=REFS[sid], gpt_cond_len=6, max_ref_length=30)
    return _cond[sid]


def _synth(text, sid):
    key = hashlib.sha1(f"{sid}|{text}".encode()).hexdigest()[:16]
    fp = CACHE / f"{key}.wav"
    if fp.exists():
        return fp
    gl, se = _get_cond(sid)
    cleaned = clean_for_xtts(text)
    out = _xtts.inference(cleaned, language="vi", gpt_cond_latent=gl, speaker_embedding=se,
                          **gen_config_for_text(cleaned))
    wav = np.asarray(out["wav"], dtype=np.float32)
    try:
        import librosa
        if len(wav) > 1000:
            t, _ = librosa.effects.trim(wav, top_db=25)
            if len(t) > 1000:
                wav = t
    except Exception:
        pass
    wav_write(fp, 24000, wav)
    return fp


app = Flask(__name__)


@app.get("/voices")
def voices():
    return jsonify({
        "voices": [{"sid": s, "name": _name(s)} for s in VOICES],
        "characters": CHARACTERS, "lines": LINES,
    })


@app.get("/audition")
def audition():
    sid = int(request.args["sid"])
    return send_file(_synth(GALLERY_SENT, sid), mimetype="audio/wav")


@app.get("/line")
def line():
    idx = int(request.args["idx"]); sid = int(request.args["sid"])
    return send_file(_synth(LINES[idx]["text"], sid), mimetype="audio/wav")


@app.get("/")
def index():
    return Response(_HTML, mimetype="text/html")


_HTML = r"""<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Casting giọng XTTS</title><style>
body{font-family:system-ui,Arial;margin:0;background:#f4f5f7;color:#1a1a1a}
header{background:#2a2a3a;color:#fff;padding:14px 20px}
h2{margin:0 0 4px} small{color:#bbb}
.wrap{padding:20px;max-width:980px}
.card{background:#fff;border:1px solid #e2e2e2;border-radius:10px;padding:14px 16px;margin:12px 0}
table{width:100%;border-collapse:collapse} td,th{padding:8px;border-bottom:1px solid #eee;text-align:left}
select{font-size:15px;padding:5px 8px;border-radius:6px;border:1px solid #bbb;min-width:240px}
button{font-size:14px;padding:6px 12px;border-radius:6px;border:1px solid #888;background:#fafafa;cursor:pointer}
button:hover{background:#eee}
.warn{color:#b00;font-size:13px}
.badge{display:inline-block;background:#7b3fb1;color:#fff;border-radius:10px;padding:1px 8px;font-size:12px;margin-right:6px}
.row{display:flex;gap:8px;align-items:center;margin:4px 0}
</style></head><body>
<header><h2>🎭 Casting giọng — XTTS (CTC v3)</h2>
<small>Giọng = clip tham chiếu của 1 speaker VieNeu in-domain. Chọn cho từng nhân vật, không trùng. Synth on-demand (XTTS chậm hơn VITS2).</small></header>
<div class="wrap">
<div class="card"><b>1. Nghe thử một giọng bất kỳ</b>
  <div class="row"><label>Giọng: </label><select id="aud"></select>
    <button onclick="auditionPlay()">▶ Nghe thử</button><span id="audnote" style="color:#888"></span></div></div>
<div class="card"><b>2. Phân vai (mỗi giọng chỉ 1 nhân vật)</b>
  <div id="warn" class="warn"></div>
  <table id="cast"><thead><tr><th>Nhân vật</th><th>Số câu</th><th>Giọng</th><th></th></tr></thead><tbody></tbody></table>
  <p><button onclick="playPage()">▶ Nghe cả trang theo casting</button>
     <button onclick="exportCasting()">⬇ Xuất casting JSON</button>
     <span id="status" style="color:#888;margin-left:8px"></span></p>
  <pre id="out" style="white-space:pre-wrap"></pre></div></div>
<script>
let D=null, casting={}, audio=new Audio();
fetch('voices').then(r=>r.json()).then(d=>{D=d; init();});
function lab(o){return `sid ${o.sid} · ${o.name}`;}
function opt(v,used){return D.voices.map(o=>`<option value="${o.sid}" ${o.sid==v?'selected':''} ${used&&used.has(o.sid)?'disabled':''}>${lab(o)}${used&&used.has(o.sid)?' — đã dùng':''}</option>`).join('');}
function lineCount(ch){return D.lines.filter(l=>l.speaker===ch).length;}
function init(){
  document.getElementById('aud').innerHTML=D.voices.map(o=>`<option value="${o.sid}">${lab(o)}</option>`).join('');
  D.characters.forEach((ch,i)=>{casting[ch]=D.voices[i % D.voices.length].sid;});
  document.querySelector('#cast tbody').innerHTML=D.characters.map(ch=>`<tr>
    <td><span class="badge">${ch}</span></td><td>${lineCount(ch)}</td>
    <td><select data-ch="${ch}" onchange="onPick(this)"></select></td>
    <td><button onclick="playChar('${ch}')">▶ ${ch}</button></td></tr>`).join('');
  refresh();
}
function refresh(){
  document.querySelectorAll('#cast select').forEach(sel=>{
    const ch=sel.dataset.ch, used=new Set(Object.entries(casting).filter(([k])=>k!==ch).map(([,v])=>v));
    sel.innerHTML=opt(casting[ch],used);
  });
  const vals=Object.values(casting),dup=vals.length!==new Set(vals).size;
  document.getElementById('warn').textContent=dup?'⚠ Có giọng trùng — hãy đổi.':'';
  document.getElementById('out').textContent='casting = '+JSON.stringify(casting,null,1);
}
function onPick(sel){casting[sel.dataset.ch]=parseInt(sel.value);refresh();}
function auditionPlay(){const sid=document.getElementById('aud').value;
  document.getElementById('audnote').textContent='đang tạo (XTTS ~vài giây)…';
  audio.src='audition?sid='+sid; audio.onplaying=()=>document.getElementById('audnote').textContent='sid '+sid; audio.play();}
function playSeq(items,i=0){if(i>=items.length){document.getElementById('status').textContent='xong';return;}
  document.getElementById('status').textContent=`đang phát ${i+1}/${items.length}…`;
  audio.src=items[i]; audio.onended=()=>playSeq(items,i+1); audio.play();}
function playChar(ch){const sid=casting[ch];
  playSeq(D.lines.filter(l=>l.speaker===ch).map(l=>`line?idx=${l.idx}&sid=${sid}`));}
function playPage(){playSeq(D.lines.map(l=>`line?idx=${l.idx}&sid=${casting[l.speaker]}`));}
function exportCasting(){const b=new Blob([JSON.stringify(casting,null,2)],{type:'application/json'});
  const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='casting_xtts.json';a.click();}
</script></body></html>"""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8777, threaded=True)
