const imgEl   = document.getElementById("frame");
const slider  = document.getElementById("slider");
const valueEl = document.getElementById("value");
const fill    = document.getElementById("fill");
const backBtn = document.getElementById("backBtn");
const nextBtn = document.getElementById("nextBtn");
const status  = document.getElementById("status");

let frames = []; // [{name, url}, ...]

// load list of frames from the server
boot();
async function boot(){
  try{
    const res = await fetch("/api/frames");
    const data = await res.json();
    frames = data.frames || [];
    if (!frames.length){
      status.textContent = "No frames available.";
      return;
    }
    slider.min = 0;
    slider.max = frames.length - 1;
    slider.value = 0;
    backBtn.disabled = false;
    nextBtn.disabled = false;
    slider.disabled  = false;
    status.textContent = `Loaded ${frames.length} images`;
    setFrame(0);
  }catch(e){
    console.error(e);
    status.textContent = "Failed to load frames.";
  }
}

function setFrame(i){
  if (!frames.length) return;
  i = Math.max(0, Math.min(frames.length-1, i|0));
  const f = frames[i];
  if (imgEl.src !== f.url) imgEl.src = f.url;

  slider.value = i;
  valueEl.textContent = `Frame ${i+1} / ${frames.length} — ${Math.round(i/(frames.length-1)*100)}%`;
  fill.style.width = `${((i+1)/frames.length)*100}%`;
}

function step(d){ setFrame(Number(slider.value) + d); }

// events
slider.oninput = ()=> setFrame(Number(slider.value));
backBtn.onclick = ()=> step(-1);
nextBtn.onclick = ()=> step(+1);
document.addEventListener("keydown", (e)=>{
  if (e.key === "ArrowLeft") step(-1);
  if (e.key === "ArrowRight") step(+1);
});
