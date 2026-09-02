const MODELS = ["qwen3.5-4b", "qwen3.5-9b", "granite4.1-3b", "granite4.1-8b"];
const METHODS = ["summary", "patch", "delta", "summary_reason"];
const COLORS = ["#196b4b", "#d08a2f", "#3f6f91", "#b74a3c", "#7967a8", "#4f8b87", "#b36b93", "#68714c"];
const state = { overview: null, evaluations: null, selected: new Set(), comparison: null };

const $ = (selector) => document.querySelector(selector);
const esc = (value) => String(value ?? "—").replace(/[&<>'"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[c]));
const number = (value, digits=3) => value === null || value === undefined ? "—" : Number(value).toFixed(digits);
const lossNumber = (value) => {
  if (value === null || value === undefined) return "—";
  const numeric = Number(value);
  if (numeric !== 0 && Math.abs(numeric) < 0.001) return numeric.toExponential(3);
  return numeric.toFixed(5);
};
const percent = (value) => `${Math.round((Number(value)||0)*100)}%`;
const duration = (seconds) => {
  if (seconds === null || seconds === undefined) return "—";
  seconds = Math.max(0, Number(seconds));
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.floor(seconds/60)}m ${Math.round(seconds%60)}s`;
  return `${Math.floor(seconds/3600)}h ${Math.round((seconds%3600)/60)}m`;
};
const bytes = (value) => value ? `${(value/1024/1024).toFixed(1)} MB` : "—";
const badge = (run) => {
  const shown = run.stale ? "STALE" : (run.state === "VALIDATING" && run.validation_type === "VALIDATION_V2" ? "V2 VALIDATING" : run.state);
  return `<span class="badge ${run.state.toLowerCase()} ${run.stale?'stale':''}">${esc(shown)}</span>`;
};

async function api(path, options={}) {
  const response = await fetch(path, {cache: "no-store", ...options});
  if (!response.ok) {
    let detail = "";
    try { detail = (await response.json()).detail || ""; } catch (_) {}
    throw new Error(detail || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

async function refresh() {
  try {
    [state.overview, state.evaluations] = await Promise.all([
      api("/api/runs"),
      api("/api/evaluations")
    ]);
    renderAll();
    if (state.selected.size) await loadComparison();
    $("#last-updated").textContent = `Updated ${new Date(state.overview.generated_at).toLocaleTimeString()}`;
  } catch (error) { toast(`Refresh failed: ${error.message}`); }
}

function renderAll() {
  $("#runs-root").textContent = state.overview.runs_root;
  renderStats(); renderEvaluations(); renderMatrix(); renderRuns(); renderSelection();
}

function renderEvaluations() {
  const jobs = state.evaluations?.jobs || [];
  $("#empty-evaluations").classList.toggle("hidden", jobs.length>0);
  $("#evaluation-table").innerHTML = jobs.map(job=>{
    const activeScenarios = job.current_scenarios || [];
    const current = activeScenarios.length > 1 ? `S${activeScenarios[0]}–S${activeScenarios[activeScenarios.length-1]}` : (job.current_scenario ? `S${job.current_scenario}` : (job.epoch ? `Epoch ${job.epoch}` : "—"));
    const item = job.items ? `${job.item||0}/${job.items}` : "";
    const phase = String(job.phase||"waiting").replaceAll("_", " ");
    const completed = `${job.completed_scenarios?.length||0}/${job.scenario_count||"?"}`;
    return `<tr data-eval-run="${esc(job.run_id)}" data-eval-name="${esc(job.evaluation_name)}"><td><div class="run-primary">${esc(job.evaluation_name)}</div><div class="run-secondary">${esc(job.run_id)}</div></td><td>${esc(job.type)}</td><td>${badge(job)}</td><td><div>${esc(current)}</div><div class="run-secondary">${esc(phase)} ${esc(item)}</div></td><td><div class="table-progress"><div class="progress-label"><span>${completed}</span><span>${percent(job.progress)}</span></div><div class="progress"><i style="width:${percent(job.progress)}"></i></div></div></td><td class="metric-value">${number(job.esm)}</td><td class="metric-value">${number(job.update_f1)}</td><td>${job.gpu===null||job.gpu===undefined?"—":`GPU ${esc(job.gpu)}`}</td><td>${duration(job.eta_seconds)}</td></tr>`;
  }).join("");
  document.querySelectorAll("#evaluation-table tr").forEach(row=>row.addEventListener("click",()=>openEvaluation(row.dataset.evalRun,row.dataset.evalName)));
}

function renderStats() {
  const o = state.overview, completed = o.runs.filter(r=>r.state==="COMPLETED");
  const scored = completed.filter(r=>r.latest_validation.closed_loop_final_state_f1!==undefined);
  const avg = scored.length ? scored.reduce((a,r)=>a+r.latest_validation.closed_loop_final_state_f1,0)/scored.length : null;
  const cards = [
    ["Total runs", o.runs.length, "all recorded experiments"],
    ["Running", o.counts.RUNNING, `${o.counts.VALIDATING||0} validating · ${o.counts.PAUSED||0} paused`],
    ["Completed", o.counts.COMPLETED, "available for comparison"],
    ["Failed", o.counts.FAILED, "open details for traceback"],
    ["Matrix coverage", `${new Set(o.runs.filter(r=>r.model_key!=="unknown").map(r=>`${r.model_key}/${r.method}`)).size}/16`, "model × method cells"],
    ["Mean final F1", avg===null?"—":number(avg), "latest completed runs"]
  ];
  $("#stats").innerHTML = cards.map(c=>`<div class="stat"><span class="label">${c[0]}</span><strong>${c[1]}</strong><small>${c[2]}</small></div>`).join("");
}

function renderMatrix() {
  let html = `<div></div>${METHODS.map(m=>`<div class="matrix-head">${label(m)}</div>`).join("")}`;
  for (const model of MODELS) {
    const spec = state.overview.runs.find(r=>r.model_key===model);
    html += `<div class="matrix-label"><div>${esc(model)}<small>${spec?.parameters_b||model.match(/\d+b/)?.[0]||""} parameters</small></div></div>`;
    for (const method of METHODS) {
      const candidates = state.overview.runs.filter(r=>r.model_key===model && r.method===method);
      const run = candidates[0];
      if (!run) { html += `<div class="matrix-cell empty"><span class="muted">No run yet</span></div>`; continue; }
      html += `<div class="matrix-cell" data-run="${esc(run.run_id)}">${candidates.length>1?`<span class="repeat-count">×${candidates.length}</span>`:""}${badge(run)}<div class="run-name">${esc(run.run_id)}</div><div class="mini"><span>${percent(run.progress)}</span><span>${run.latest_validation.closed_loop_final_state_f1!==undefined?`F1 ${number(run.latest_validation.closed_loop_final_state_f1)}`:duration(run.eta_seconds)}</span></div><div class="progress"><i style="width:${percent(run.progress)}"></i></div></div>`;
    }
  }
  $("#matrix").innerHTML = html;
  document.querySelectorAll(".matrix-cell[data-run]").forEach(el=>el.addEventListener("click",()=>openDetail(el.dataset.run)));
}

function filteredRuns() {
  const query = $("#search").value.trim().toLowerCase(), status = $("#state-filter").value;
  return state.overview.runs.filter(run => (!status || run.state===status) && (!query || `${run.run_id} ${run.model_key} ${run.method}`.toLowerCase().includes(query)));
}

function renderRuns() {
  const runs = filteredRuns(); $("#empty-runs").classList.toggle("hidden", runs.length>0);
  $("#run-table").innerHTML = runs.map(run=>`<tr data-run="${esc(run.run_id)}"><td class="select-cell"><input class="select-run" type="checkbox" ${state.selected.has(run.run_id)?"checked":""} aria-label="Compare ${esc(run.run_id)}"></td><td><div class="run-primary">${esc(run.run_id)}</div><div class="run-secondary">seed ${run.seed??"—"} · ${duration(run.elapsed_seconds)}</div></td><td><div>${esc(run.model_key)}</div><div class="run-secondary">${label(run.method)}</div></td><td><div class="status-manage">${badge(run)}${canArchive(run)?`<button class="button danger compact archive-run" data-archive-run="${esc(run.run_id)}">Archive</button>`:""}</div></td><td>${run.gpu===null||run.gpu===undefined?"—":`GPU ${esc(run.gpu)}`}</td><td><div class="table-progress"><div class="progress-label"><span>${run.global_step}/${run.total_steps||"?"}</span><span>${percent(run.progress)}</span></div><div class="progress"><i style="width:${percent(run.progress)}"></i></div></div></td><td class="metric-value">${lossNumber(run.latest_train.loss)}</td><td class="metric-value">${number(run.latest_validation.closed_loop_final_state_f1)}</td><td class="metric-value">${run.latest_train.target_tokens_per_second?`${number(run.latest_train.target_tokens_per_second,1)} tok/s`:"—"}</td><td>${duration(run.eta_seconds)}</td></tr>`).join("");
  document.querySelectorAll("#run-table tr").forEach(row=>row.addEventListener("click",e=>{ if(!e.target.closest("input,button")) openDetail(row.dataset.run); }));
  document.querySelectorAll(".select-run").forEach(box=>box.addEventListener("change",e=>toggleSelection(e.target.closest("tr").dataset.run,e.target.checked)));
  document.querySelectorAll(".archive-run").forEach(button=>button.addEventListener("click",()=>archiveRun(button.dataset.archiveRun)));
}

const canArchive = run => ["COMPLETED", "FAILED"].includes(run.state);

async function archiveRun(runId) {
  if (!window.confirm(`Archive ${runId}?\n\nIt will disappear from the dashboard but remain recoverable in the workspace trash.`)) return;
  try {
    await api(`/api/runs/${encodeURIComponent(runId)}?confirm=${encodeURIComponent(runId)}`, {method:"DELETE"});
    state.selected.delete(runId);
    state.comparison = null;
    $("#detail-drawer").setAttribute("aria-hidden", "true");
    await refresh();
    toast(`Archived ${runId}`);
  } catch (error) { toast(`Archive failed: ${error.message}`); }
}

function toggleSelection(runId, selected) {
  if (selected) state.selected.add(runId); else state.selected.delete(runId);
  renderSelection(); renderRuns(); loadComparison();
}

function renderSelection() {
  const runs = [...state.selected];
  $("#selection-chips").innerHTML = runs.length ? runs.map((id,i)=>`<span class="chip"><i style="background:${COLORS[i%COLORS.length]}"></i>${esc(id)}<button data-remove="${esc(id)}">×</button></span>`).join("") : `<span class="muted">Select runs using the checkboxes above.</span>`;
  document.querySelectorAll("[data-remove]").forEach(button=>button.addEventListener("click",()=>toggleSelection(button.dataset.remove,false)));
  if (!runs.length) { $("#comparison-chart").innerHTML='<div class="chart-empty">Choose two or more runs to overlay learning curves.</div>'; $("#comparison-table").innerHTML=""; }
}

async function loadComparison() {
  const ids = [...state.selected]; if (!ids.length) return;
  try { state.comparison = await api(`/api/comparison?${ids.map(id=>`run_id=${encodeURIComponent(id)}`).join("&")}`); renderComparison(); }
  catch(error) { toast(`Comparison failed: ${error.message}`); }
}

function metricSeries(run, metricKey) {
  const [eventName,key] = metricKey.split(".");
  if (eventName === "train") return run.metrics.filter(m=>m.event==="train" && m[key]!==undefined).map(m=>({x:Number(m.global_step??0),y:Number(m[key])}));
  return run.metrics
    .filter(m=>["validation_teacher_forced","validation_epoch"].includes(m.event))
    .map(m=>({m,value:m[key] ?? (key==="teacher_forced_loss" && m.event==="validation_teacher_forced" ? m.loss : undefined)}))
    .filter(item=>item.value!==undefined)
    .map(item=>({x:Number(item.m.global_step??item.m.epoch??0),y:Number(item.value)}));
}

function renderComparison() {
  const metric = $("#metric-select").value;
  const series = state.comparison.runs.map((run,i)=>({name:run.summary.run_id,color:COLORS[i%COLORS.length],points:metricSeries(run,metric)}));
  drawChart($("#comparison-chart"),series,metric);
  $("#comparison-table").innerHTML = state.comparison.runs.map((run,i)=>{const s=run.summary,v=s.latest_validation,t=s.latest_train;return `<div class="compare-card" style="border-top:3px solid ${COLORS[i%COLORS.length]}"><strong>${esc(s.run_id)}</strong><span class="muted">${esc(s.model_key)} · ${label(s.method)}</span><dl><dt>Status</dt><dd>${s.state}</dd><dt>Train loss</dt><dd>${lossNumber(t.loss)}</dd><dt>Val TF loss</dt><dd>${lossNumber(v.teacher_forced_loss)}</dd><dt>Update F1</dt><dd>${number(v.one_step_update_f1)}</dd><dt>Final State F1</dt><dd>${number(v.closed_loop_final_state_f1)}</dd><dt>Gold Quiz ESM</dt><dd>${number(v.quiz_esm)}</dd><dt>Gold Quiz Tool F1</dt><dd>${number(v.quiz_tool_f1)}</dd><dt>Closed Quiz ESM</dt><dd>${number(v.closed_loop_quiz_esm)}</dd><dt>Closed Quiz Tool F1</dt><dd>${number(v.closed_loop_quiz_tool_f1)}</dd><dt>Closed Quiz Arg</dt><dd>${number(v.closed_loop_quiz_arg_exact)}</dd><dt>Token/s</dt><dd>${number(t.target_tokens_per_second,1)}</dd><dt>Peak GPU</dt><dd>${t.peak_cuda_memory_mib?`${number(t.peak_cuda_memory_mib,0)} MiB`:"—"}</dd></dl></div>`;}).join("");
}

function drawChart(container, series, labelText) {
  const points = series.flatMap(s=>s.points); if (!points.length) {container.innerHTML='<div class="chart-empty">This metric has not been recorded yet.</div>';return;}
  const W=1000,H=300,p={l:72,r:24,t:30,b:42};
  const xs=points.map(point=>point.x),ys=points.map(point=>point.y);
  let xmin=Math.min(...xs),xmax=Math.max(...xs); if(xmin===xmax)xmax=xmin+1;
  const logarithmic=labelText==="train.loss";
  const positiveLosses=ys.filter(value=>value>0);
  const minimumPositive=positiveLosses.length?Math.min(...positiveLosses):1e-12;
  const transformY=value=>logarithmic?Math.log10(Math.max(value,minimumPositive)):value;
  const transformed=ys.map(transformY);
  let ymin=Math.min(...transformed),ymax=Math.max(...transformed);
  if(logarithmic){ymin=Math.floor(ymin);ymax=Math.ceil(ymax);}
  if(ymin===ymax){ymin-=logarithmic?1:.5;ymax+=logarithmic?1:.5;}
  const x=value=>p.l+(value-xmin)/(xmax-xmin)*(W-p.l-p.r);
  const y=value=>H-p.b-(transformY(value)-ymin)/(ymax-ymin)*(H-p.t-p.b);
  const chartLabel=logarithmic?`${labelText} · log Y`:labelText;
  let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(chartLabel)} chart"><text x="${p.l}" y="18" fill="#6f7d75" font-size="11">${esc(chartLabel)}</text>`;
  for(let i=0;i<5;i++){const yy=p.t+i*(H-p.t-p.b)/4,transformedValue=ymax-i*(ymax-ymin)/4,displayValue=logarithmic?10**transformedValue:transformedValue;svg+=`<line x1="${p.l}" y1="${yy}" x2="${W-p.r}" y2="${yy}" stroke="#e1e3dc"/><text x="${p.l-8}" y="${yy+4}" text-anchor="end" fill="#7a857e" font-size="10">${logarithmic?lossNumber(displayValue):number(displayValue,3)}</text>`;}
  series.forEach(s=>{if(!s.points.length)return;const path=s.points.map((pt,i)=>`${i?'L':'M'}${x(pt.x).toFixed(1)},${y(pt.y).toFixed(1)}`).join(' ');svg+=`<path d="${path}" fill="none" stroke="${s.color}" stroke-width="2.5" stroke-linejoin="round"/>`;s.points.forEach(pt=>svg+=`<circle cx="${x(pt.x)}" cy="${y(pt.y)}" r="3" fill="${s.color}"/>`);});
  svg+=`<text x="${p.l}" y="${H-12}" fill="#7a857e" font-size="10">step ${xmin}</text><text x="${W-p.r}" y="${H-12}" text-anchor="end" fill="#7a857e" font-size="10">step ${xmax}</text></svg>`;container.innerHTML=svg;
}

async function openDetail(runId) {
  const drawer=$("#detail-drawer"),content=$("#detail-content"); drawer.setAttribute("aria-hidden","false");content.innerHTML='<div class="chart-empty">Loading run…</div>';
  try { const d=await api(`/api/runs/${encodeURIComponent(runId)}`),s=d.summary,t=s.latest_train,v=s.latest_validation,test=d.ollama_test||{},qm=(test.quiz||{}).all||{},mm=test.memory||{}; content.innerHTML=`<p class="eyebrow">RUN DETAIL</p><h2>${esc(runId)}</h2><p>${badge(s)} <span class="muted">${esc(s.model_key)} · ${label(s.method)} · seed ${s.seed??"—"}</span></p>${canArchive(s)?`<button class="button danger archive-detail" data-archive-run="${esc(runId)}">Archive run</button>`:""}${s.failure?`<div class="failure">${esc(s.failure)}</div>`:""}<div class="detail-grid">${detailCard("GPU",s.gpu===null||s.gpu===undefined?"—":`GPU ${esc(s.gpu)}`)}${detailCard("Progress",percent(s.progress))}${detailCard("Global step",`${s.global_step}/${s.total_steps||"?"}`)}${detailCard("Elapsed",duration(s.elapsed_seconds))}${detailCard("Train loss",lossNumber(t.loss))}${detailCard("Val TF loss",lossNumber(v.teacher_forced_loss))}${detailCard("Final State F1",number(v.closed_loop_final_state_f1))}${detailCard("Gold Quiz ESM",number(v.quiz_esm))}${detailCard("Gold Quiz Tool F1",number(v.quiz_tool_f1))}${detailCard("Closed Quiz ESM",number(v.closed_loop_quiz_esm))}${detailCard("Closed Quiz Tool F1",number(v.closed_loop_quiz_tool_f1))}${detailCard("Closed Quiz Arg Exact",number(v.closed_loop_quiz_arg_exact))}${detailCard("Target token/s",number(t.target_tokens_per_second,1))}</div>${d.ollama_test?`<div class="detail-section"><h3>Ollama Test · S91–S100</h3><div class="detail-grid">${detailCard("Memory final F1",number(mm.final_state_f1))}${detailCard("Quiz ESM",number(qm.exact_state_match))}${detailCard("Quiz Tool F1",number(qm.tool_f1))}${detailCard("Quiz Arg Exact",number(qm.argument_exact_match))}${detailCard("Memory prefill/decode",`${duration(mm.prefill_seconds)} / ${duration(mm.decode_seconds)}`)}${detailCard("Quiz prefill/decode",`${duration(qm.prefill_seconds)} / ${duration(qm.decode_seconds)}`)}${detailCard("Memory latency",duration(mm.latency_seconds))}${detailCard("Quiz latency",duration(qm.latency_seconds))}</div><div class="code-block">${esc(JSON.stringify(test.environment||{},null,2))}</div></div>`:""}<div class="detail-section"><h3>Learning curve</h3><div id="detail-chart" class="chart-shell"></div></div><div class="detail-section"><h3>Best checkpoint</h3>${d.best_checkpoint?`<div class="code-block">${esc(JSON.stringify(d.best_checkpoint,null,2))}</div>`:'<p class="muted">No checkpoint selected yet.</p>'}</div><div class="detail-section"><h3>Checkpoints</h3>${d.checkpoints.length?d.checkpoints.map(c=>`<div class="checkpoint-row"><code>${esc(c.name)}</code><span>${bytes(c.size_bytes)}</span></div>`).join(''):'<p class="muted">No checkpoints yet.</p>'}</div><div class="detail-section"><h3>Validation artifacts</h3><div class="code-block">${esc(JSON.stringify(d.validations,null,2))}</div></div><div class="detail-section"><h3>Run configuration</h3><div class="code-block">${esc(JSON.stringify(d.config,null,2))}</div></div>`; document.querySelector(".archive-detail")?.addEventListener("click",()=>archiveRun(runId)); drawChart($("#detail-chart"),[{name:runId,color:COLORS[0],points:metricSeries({metrics:d.metrics},"train.loss")}],"train.loss"); }
  catch(error){content.innerHTML=`<div class="failure">${esc(error.message)}</div>`;}
}

async function openEvaluation(runId, evaluationName) {
  const drawer=$("#detail-drawer"),content=$("#detail-content"); drawer.setAttribute("aria-hidden","false");content.innerHTML='<div class="chart-empty">Loading evaluation…</div>';
  try {
    const d=await api(`/api/evaluations/${encodeURIComponent(runId)}/${encodeURIComponent(evaluationName)}`);
    if(d.type==="VALIDATION"){
      const s=d.status||{},e=s.evaluation||{};
      content.innerHTML=`<p class="eyebrow">VALIDATION DETAIL</p><h2>${esc(runId)}</h2><div class="detail-grid">${detailCard("Epoch",s.epoch??"—")}${detailCard("Phase",esc(String(e.phase||"epoch validation").replaceAll("_"," ")))}${detailCard("Item",e.items?`${e.item||0}/${e.items}`:"—")}</div><div class="detail-section"><h3>Completed artifacts</h3><div class="code-block">${esc(JSON.stringify(d.artifacts,null,2))}</div></div>`;
      return;
    }
    const p=d.progress||{},s=d.summary||{},m=s.memory||{},q=s.closed_loop_quiz||{};
    content.innerHTML=`<p class="eyebrow">TEST DETAIL</p><h2>${esc(evaluationName)}</h2><p class="muted">${esc(runId)}</p><div class="detail-grid">${detailCard("Status",p.status|| (s.complete?"COMPLETED":"RUNNING"))}${detailCard("Current",p.current_scenario?`S${p.current_scenario}`:"—")}${detailCard("Progress",percent(p.progress??(s.completed_scenarios?.length/(s.expected_scenarios?.length||1))))}${detailCard("E2E ESM",number(q.esm))}${detailCard("Tool F1",number(q.tool_f1))}${detailCard("Update F1",number(m.update_f1))}${detailCard("Turns",m.turns??"—")}${detailCard("Quiz tasks",q.tasks??"—")}${detailCard("ETA",duration(p.eta_seconds))}</div><div class="detail-section"><h3>Scenario results</h3><div class="code-block">${esc(JSON.stringify(d.scenarios,null,2))}</div></div><div class="detail-section"><h3>Manifest</h3><div class="code-block">${esc(JSON.stringify(d.manifest,null,2))}</div></div>`;
  } catch(error){content.innerHTML=`<div class="failure">${esc(error.message)}</div>`;}
}

const detailCard=(name,value)=>`<div class="detail-card"><span>${name}</span><strong>${value}</strong></div>`;
const label=(value)=>({summary:"Summary",patch:"Patch",delta:"Appended Delta",summary_reason:"Summary + Reason"}[value]||esc(value));
function toast(message){const el=$("#toast");el.textContent=message;el.classList.remove("hidden");setTimeout(()=>el.classList.add("hidden"),3500);}

$("#refresh-button").addEventListener("click",refresh);
$("#search").addEventListener("input",renderRuns);
$("#state-filter").addEventListener("change",renderRuns);
$("#metric-select").addEventListener("change",()=>state.comparison&&renderComparison());
$("#clear-selection").addEventListener("click",()=>{state.selected.clear();state.comparison=null;renderRuns();renderSelection();});
document.querySelectorAll("[data-close]").forEach(el=>el.addEventListener("click",()=>$("#detail-drawer").setAttribute("aria-hidden","true")));
document.addEventListener("keydown",e=>{if(e.key==="Escape")$("#detail-drawer").setAttribute("aria-hidden","true")});
refresh(); setInterval(refresh,10000);
