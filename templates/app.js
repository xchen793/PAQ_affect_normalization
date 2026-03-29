/* ========= Shared Logic (app.js) ========= */
// Minimal “storage” for a single session
const STORE_KEY = "morph_task_v1";
const DEFAULT_CONFIG = {
  subjectId: null,
  startedAt: null,
  deviceCheck: null,
  pretest: { brightnessOk: null, distanceOk: null, colorBlind: null },
  sets: [],    // array of {setId, frames:Int, jndLeft:?, jndRight:?, confidence:?, notes:?}
  currentIndex: 0,
  finishedAt: null,
};

// Utility
const $ = (sel, root=document) => root.querySelector(sel);
const $$ = (sel, root=document) => Array.from(root.querySelectorAll(sel));
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

function loadState(){
  const raw = localStorage.getItem(STORE_KEY);
  if(!raw) return {...DEFAULT_CONFIG};
  try{ return JSON.parse(raw); }catch(e){ return {...DEFAULT_CONFIG}; }
}
function saveState(st){ localStorage.setItem(STORE_KEY, JSON.stringify(st)); }
function resetState(){ localStorage.removeItem(STORE_KEY); }

// Progress helpers
function setsCompleted(state){
  return state.sets.filter(s => s.jndLeft !== null && s.jndRight !== null).length;
}
function progressPct(state){
  if(state.sets.length===0) return 0;
  return Math.round(100 * setsCompleted(state) / state.sets.length);
}

// Export as downloadable JSON
function downloadJSON(filename, dataObj) {
  const blob = new Blob([JSON.stringify(dataObj, null, 2)], {type: "application/json"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// --- INTRO PAGE ---
function initIntro(){
  const state = loadState();
  if(!state.startedAt) state.startedAt = new Date().toISOString();
  saveState(state);

  $("#startBtn")?.addEventListener("click", () => {
    const sid = $("#subjectId")?.value.trim();
    if(!sid){ alert("Please enter a Subject ID."); return; }
    state.subjectId = sid;
    // Configure your reference sets here (edit to match your assets)
    state.sets = [
      { setId: "ref_0", frames: 100, jndLeft: null, jndRight: null, confidence: null, notes:"" },
      { setId: "ref_1", frames: 100, jndLeft: null, jndRight: null, confidence: null, notes:"" },
      { setId: "ref_2", frames: 100, jndLeft: null, jndRight: null, confidence: null, notes:"" },
      { setId: "ref_3", frames: 100, jndLeft: null, jndRight: null, confidence: null, notes:"" },
      { setId: "ref_4", frames: 100, jndLeft: null, jndRight: null, confidence: null, notes:"" },
    ];
    saveState(state);
    location.href = "pretest.html";
  });

  $("#clearAll")?.addEventListener("click", () => {
    if(confirm("Clear all local data for this task?")){
      resetState();
      location.reload();
    }
  });
}

// --- PRETEST PAGE ---
function initPretest(){
  const state = loadState();
  $("#toTask")?.addEventListener("click", () => {
    const brightness = $("#chkBright").checked;
    const distance   = $("#chkDistance").checked;
    const cb         = $("#selCB").value;

    if(!brightness || !distance){
      alert("Please confirm brightness and viewing distance before proceeding.");
      return;
    }
    state.pretest.brightnessOk = brightness;
    state.pretest.distanceOk   = distance;
    state.pretest.colorBlind   = cb;
    saveState(state);
    location.href = "task.html";
  });
}

// --- TASK PAGE ---
let sweepTimer = null;
function initTask(){
  const state = loadState();
  const set = state.sets[state.currentIndex];
  if(!set){ location.href="end.html"; return; }

  // UI bindings
  const slider = $("#frameSlider");
  const label  = $("#frameLabel");
  const playBtn= $("#playPause");
  const speed  = $("#speed");
  const jndL   = $("#markLeft");
  const jndR   = $("#markRight");
  const conf   = $("#confidence");
  const note   = $("#notes");
  const next   = $("#nextSet");
  const prev   = $("#prevSet");
  const bar    = $(".progress .bar");
  const refTag = $("#refTag");
  const jndLP  = $("#jndLeftPreview");
  const jndRP  = $("#jndRightPreview");

  // Load previously marked values if any
  if(set.jndLeft !== null) jndLP.textContent = set.jndLeft;
  if(set.jndRight !== null) jndRP.textContent = set.jndRight;
  if(set.confidence) conf.value = set.confidence;
  if(set.notes) note.value = set.notes;

  refTag.textContent = set.setId;
  bar.style.width = progressPct(state) + "%";
  $("#where").textContent = `${state.currentIndex+1} / ${state.sets.length}`;

  // Media (replace src with your assets)
  const media = $("#media");
  // Example: swap to <video> if you use video stimuli
  const img = document.createElement("img");
  img.alt = "Stimulus";
  img.src = `assets/${set.setId}/frame_0001.png`;
  media.innerHTML = "";
  media.appendChild(img);

  // Slider logic
  slider.max = (set.frames-1).toString();
  slider.value = "0";
  label.textContent = "0";

  function updateFrame(n){
    const frame = clamp(n, 0, set.frames-1);
    slider.value = frame;
    label.textContent = frame.toString();
    // Swap image to requested frame (zero-padded to 4 as example)
    const idx = String(frame+1).padStart(4,"0");
    img.src = `assets/${set.setId}/frame_${idx}.png`;
  }
  slider.addEventListener("input", e => updateFrame(parseInt(e.target.value,10)));

  // Sweep
  function toggleSweep(){
    if(sweepTimer){ clearInterval(sweepTimer); sweepTimer=null; playBtn.textContent="Play (Space)"; return; }
    playBtn.textContent="Pause (Space)";
    sweepTimer = setInterval(() => {
      const step = parseInt(speed.value,10);
      let f = parseInt(slider.value,10) + step;
      if(f >= set.frames){ f = 0; }
      updateFrame(f);
    }, 60);
  }
  playBtn.addEventListener("click", toggleSweep);

  // JND mark
  jndL.addEventListener("click", () => {
    set.jndLeft = parseInt(slider.value,10);
    jndLP.textContent = set.jndLeft;
    saveState(state);
    flash("Left JND saved");
  });
  jndR.addEventListener("click", () => {
    set.jndRight = parseInt(slider.value,10);
    jndRP.textContent = set.jndRight;
    saveState(state);
    flash("Right JND saved");
  });

  conf.addEventListener("input", ()=>{ set.confidence = parseInt(conf.value,10); saveState(state); });
  note.addEventListener("change", ()=>{ set.notes = note.value; saveState(state); });

  // Nav between sets
  prev.addEventListener("click", () => {
    state.currentIndex = Math.max(0, state.currentIndex-1);
    saveState(state); location.reload();
  });
  next.addEventListener("click", () => {
    if(set.jndLeft===null || set.jndRight===null){
      if(!confirm("You haven't marked both JNDs. Continue anyway?")) return;
    }
    state.currentIndex++;
    if(state.currentIndex >= state.sets.length){ location.href="end.html"; }
    else { saveState(state); location.reload(); }
  });

  // Keys
  document.addEventListener("keydown", (e) => {
    if(["INPUT","TEXTAREA"].includes(document.activeElement.tagName)) return;
    if(e.key===" "){ e.preventDefault(); toggleSweep(); }
    if(e.key==="a" || e.key==="ArrowLeft"){ updateFrame(parseInt(slider.value,10)-1); }
    if(e.key==="d" || e.key==="ArrowRight"){ updateFrame(parseInt(slider.value,10)+1); }
    if(e.key==="j"){ set.jndLeft = parseInt(slider.value,10); jndLP.textContent=set.jndLeft; saveState(state); flash("Left JND saved"); }
    if(e.key==="k"){ set.jndRight= parseInt(slider.value,10); jndRP.textContent=set.jndRight; saveState(state); flash("Right JND saved"); }
  });

  // Tiny toast
  function flash(msg){
    const t = document.createElement("div");
    t.textContent = msg;
    t.style.position="fixed"; t.style.bottom="24px"; t.style.left="50%"; t.style.transform="translateX(-50%)";
    t.style.background="#0f1a2a"; t.style.border="1px solid #2a3b57"; t.style.padding="10px 14px"; t.style.borderRadius="10px"; t.style.boxShadow="var(--shadow)";
    document.body.appendChild(t);
    setTimeout(()=>t.remove(), 900);
  }
}

// --- END PAGE ---
function initEnd(){
  const state = loadState();
  state.finishedAt = new Date().toISOString();
  saveState(state);
  $("#export")?.addEventListener("click", () => {
    const out = {
      subjectId: state.subjectId,
      startedAt: state.startedAt,
      finishedAt: state.finishedAt,
      deviceCheck: state.deviceCheck,
      pretest: state.pretest,
      answers: state.sets.map(s => ({
        setId: s.setId,
        jndLeft: s.jndLeft,
        jndRight: s.jndRight,
        confidence: s.confidence,
        notes: s.notes
      }))
    };
    downloadJSON(`${state.subjectId || "subject"}_morph_task.json`, out);
  });
  $("#restart")?.addEventListener("click", () => {
    if(confirm("Start over and clear all saved progress?")){
      resetState(); location.href="intro.html";
    }
  });
}

// Initialize per-page
document.addEventListener("DOMContentLoaded", () => {
  const page = document.body.dataset.page;
  if(page==="intro")   initIntro();
  if(page==="pretest") initPretest();
  if(page==="task")    initTask();
  if(page==="end")     initEnd();
});
