// Editable research deck. Run with the bundled presentation runtime; no inference.
import fs from 'node:fs/promises';
import path from 'node:path';
import {createRequire} from 'node:module';
import {fileURLToPath,pathToFileURL} from 'node:url';
const repo=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const build=path.join(repo,'outputs/report-build',new Date().toISOString().replace(/[:.]/g,'-'));
const require=createRequire(path.join(build,'author.mjs'));
const {Presentation,PresentationFile}=await import(require.resolve('@oai/artifact-tool'));
const skill=process.env.PRESENTATIONS_SKILL_DIR;
if(!skill)throw Error('Set PRESENTATIONS_SKILL_DIR to the installed presentation skill');
const {resolvePresentationFont,applyPresentationChartFont,finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const font=resolvePresentationFont();
const original=JSON.parse(await fs.readFile(path.join(repo,'results/qwen15_confirmatory.json')));
const exploratory=JSON.parse(await fs.readFile(path.join(repo,'results/qwen15_exploratory_mixture.json')));
const benefit=JSON.parse(await fs.readFile(path.join(repo,'results/benefit_development.json')));
const benefitDiagnostics=JSON.parse(await fs.readFile(path.join(repo,'results/benefit_diagnostics.json')));
const presentation=Presentation.create({slideSize:{width:1280,height:720}});
const ink='#142a3a',blue='#176b91',rust='#b95732',grey='#51636f';
// Excel chart workbooks retain 15 significant digits; analytical JSON is untouched.
const chartNumber=v=>Number(v.toPrecision(12));
function text(slide,value,left,top,width,height,size=27,color=ink,bold=false){
 const shape=slide.shapes.add({geometry:'textbox',position:{left,top,width,height},fill:'none',line:{fill:'none',width:0}});
 shape.text=value;shape.text.insets=0;shape.text.style={fontSize:size,typeface:font,color,bold};return shape;
}
function slide(title,note=''){
 const s=presentation.slides.add();s.background.fill='#ffffff';
 text(s,title,64,42,1152,presentation.slides.items.length===1?160:80,presentation.slides.items.length===1?56:44,ink,true);
 text(s,String(presentation.slides.items.length),1175,676,45,24,18,grey);
 s.speakerNotes.text=note;return s;
}
function para(s,value,top,height=88,size=27){return text(s,value,72,top,1130,height,size);}
function foot(s,value){text(s,value,72,620,1110,48,18,grey);}
function table(s,values,top=160,widths=[400,240,240,240]){
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:72,top,width:1136,height:values.length*66,columnWidths:widths,values});
 t.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length}).assign({textStyle:{fontSize:25,typeface:font,color:ink},margins:{left:12,right:12,top:12,bottom:10}});
 t.cells.block({row:0,column:0,rowCount:1,columnCount:values[0].length}).assign({fill:'#e9f1f5',textStyle:{fontSize:25,typeface:font,color:ink,bold:true}});
 t.borders.assign({fill:'#d4dfe5',width:1,style:'solid'});return t;
}
function frontier(s,includeMixture){
 const palette={'small-only':'#55788b','strong-only':'#181f25','cheap-only':'#398455','always-ic':'#9c6a24','adaptive':'#7c559d','random-ic-budget-matched':'#b95732','cheap-cost-matched':'#717881'};
 const names={'small-only':'Small','strong-only':'Strong','cheap-only':'Cheap','always-ic':'Always IC','adaptive':'Adaptive IC','random-ic-budget-matched':'Random IC','cheap-cost-matched':'Cheap mixture'};
 const series=Object.keys(palette).map(method=>{
  const rows=original.results.filter(r=>r.method===method);
  return {name:names[method],xValues:rows.map(r=>chartNumber(r.mean_seconds)),values:rows.map(r=>chartNumber(r.accuracy)),marker:{symbol:method==='adaptive'?'diamond':'circle',size:9,fill:palette[method]},line:{fill:'none',width:0}};
 });
 if(includeMixture)series.unshift({name:'Random model preselection',xValues:exploratory.curve.map(r=>chartNumber(r.mean_seconds)),values:exploratory.curve.map(r=>chartNumber(r.accuracy)),marker:{symbol:'none'},line:{fill:ink,width:3}});
 const c=s.charts.add('scatter',{position:{left:70,top:140,width:1136,height:440},series,scatterOptions:{style:'lineWithMarkers'},hasLegend:true,
  legend:{position:'bottom',textStyle:{fontSize:20}},xAxis:{min:.2,max:2.5,title:{text:'Mean warm service seconds / query',textStyle:{fontSize:23}},textStyle:{fontSize:21},numberFormatCode:'0.0'},
  yAxis:{min:.2,max:.7,title:{text:'Expected exact match',textStyle:{fontSize:23}},textStyle:{fontSize:21},numberFormatCode:'0%',majorGridlines:{fill:'#e0e7eb',width:1}}});
 applyPresentationChartFont(c,{fontFamily:font});return c;
}

let s=slide('Confidence acquisition and\nincremental-benefit routing',
 'COMP4901B research draft. Original confirmatory and post-test exploratory analyses are complete. The bounded extension completed its frozen development screen and returned STOP. No fresh-test inference ran. Report source: deliverables/REPORT.md.');
text(s,'Qwen2.5-1.5B → Qwen2.5-14B · Closed-book TriviaQA',72,270,1100,90,31,blue);
text(s,'Original adaptive-IC study\nand completed benefit-aware extension',72,400,1100,120,32);
foot(s,'27 October 2026 · Group review draft · Editable figures and tables');

s=slide('Adaptive confidence acquisition',
 'The small answer is generated first. Cheap confidence comes from ordinary-QA prefill hidden states. Ambiguous cheap decisions acquire a separate IC prompt. Every branch pays full small QA cost. IC follows the authors’ released implementation with transferred Qwen settings. https://arxiv.org/html/2506.09669v4');
function box(label,x,y,w=300){const b=s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:100},fill:'#f1f5f7',line:{fill:'#bdcdd6',width:1}});b.text=label;b.text.style={fontSize:27,typeface:font,color:ink};return b;}
const a=box('Generate small answer',72,180),b=box('Cheap correctness probe',470,180),c=box('Clear: accept / escalate',870,180);
const d=box('Ambiguous: acquire IC',470,400),e=box('IC: accept / escalate',870,400);
for(const [from,to,side] of [[a,b,'right'],[b,c,'right'],[b,d,'bottom'],[d,e,'right']])s.shapes.connect(from,to,{kind:'straight',fromSide:side,toSide:side==='bottom'?'top':'left',line:{fill:grey,width:2},tail:{type:'arrow',width:'med',length:'med'}});
foot(s,'Baselines: small, strong, cheap-only, always-IC and 20-seed random-IC');

s=slide('Experimental separation and settings',
 'All original fitting and thresholds preceded final-test inference. Assessment was used for viability, not threshold optimization. The extension reuses development only and reserves a disjoint fresh test. Dataset revision and exclusion hashes are recorded. https://aclanthology.org/P17-1147/');
table(s,[['Partition','n','Use'],['Training','1,024','Fit predictors'],['Calibration','128','Select regularization / rules'],['Assessment','128','Development gate'],['Original test','512','Frozen study; then exploration'],['Fresh reservation','512','Unused; extension confirmation only']],145,[300,130,706]);
foot(s,'NF4 double 4-bit · FP32 compute · eager · batch 1 · greedy · 64-token cap\nPinned revisions; sequential GPU0 inference on Kaggle T4×2');

s=slide('Every required model call has a cost',
 'Cost is synchronized warm service time, replayed additively per query. It is not deployment latency, throughput, energy or dollars. Single-model policies pay only their QA cost; cold loads and offline collection are separate. Original final test means are taken directly from qwen15_confirmatory.json.');
para(s,'Cascade cost = small QA + router + acquired IC + escalated strong QA',165,110,33);
table(s,[['Component','Mean per call'],['Small QA','0.3336 s'],['Cheap probe','0.000176 s'],['IC','0.1514 s'],['Strong QA','1.8655 s']],270,[650,486]);
foot(s,'Original test: 1,536 collected QA/IC calls; 1,203.42 warm service seconds\nLoads and warmups are separate. Replay is not a live deployment benchmark.');

s=slide('Frozen confirmatory result: no meaningful IC gain',
 'Primary comparator mixes two frozen cheap-routing endpoints using cost-only weights. 10,000 paired percentile bootstrap resamples, seed4901; weights recomputed per resample, Bonferroni across eight operating points. Practical support required point gain≥3pp and familywise lower>0, exact cost match and IC use. This does not require lower>3pp. Protocol addendum was committed after development but before test.');
table(s,[['Penalties','Adaptive EM','Gain vs cheap, pp','Familywise 95%'],['0–.2 (5 points)','60.55%','+0.04','[−0.86, +0.82]'],['.4','33.59%','−1.34','[−4.14, +1.42]'],['.8, 1.6 (no IC)','26.56%','0','[0, 0]']],165,[260,250,300,326]);
para(s,'No positive statistical evidence; neither IC-using rule meets the precommitted 3-point practical threshold.',465,100,30);
foot(s,'n = 512 · Both eligible upper bounds are below +3 pp\nThis rules out the target gain under this bootstrap rule, not all possible IC benefits.');

s=slide('Exploratory model preselection',
 'Post-test exploratory analysis only. Independent strong choice probability p before generation: accuracy .265625+.3515625p; cost .333587+1.531904p seconds. One model runs. All40 original nonendpoint points are dominated in observed means, not simultaneous statistical dominance. At .7732s: random preselection36.65% vs adaptive33.59%, +3.06pp pointwise95[1.00,5.30]. Bootstrap10000seed4903; pointwise, no multiplicity correction.');
frontier(s,true);
foot(s,'POST-TEST EXPLORATORY · One model runs, selected before generation\nAt .7732 s/query: preselection 36.65% vs adaptive 33.59%; +3.06 pp [1.00, 5.30], pointwise 95%');

s=slide('Correctness prediction differs from escalation benefit',
 'Original-test diagnostics, descriptive AUROCs without uncertainty intervals. Cheap correctnessAUROC .7574898, IC .6758644 over512. Low confidence predicting strong rescue among376 small errors: cheap .3919022, IC .4396061. Different targets and conditioning sets. Rescue discrimination omits harms on initially correct answers, so it is not all-query signed-benefit discrimination.');
const dg=exploratory.diagnostics;
const chart=s.charts.add('bar',{position:{left:72,top:155,width:700,height:410},categories:['Cheap probe','IC'],series:[
 {name:'Small correctness',values:[dg.small_correctness_probe_auroc,dg.ic_correctness_auroc].map(chartNumber),fill:blue},
 {name:'Rescue among small errors',values:[dg.low_probe_rescue_auroc_among_small_errors,dg.low_ic_rescue_auroc_among_small_errors].map(chartNumber),fill:rust}],
 barOptions:{grouping:'clustered',direction:'column'},hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:21}},yAxis:{min:0,max:1,title:'AUROC',textStyle:{fontSize:23},numberFormatCode:'0.0'},xAxis:{textStyle:{fontSize:23}},dataLabels:{showValue:true,numberFormatCode:'0.000',position:'outEnd',textStyle:{fontSize:22}}});applyPresentationChartFont(chart,{fontFamily:font});
text(s,'Low-budget adaptive calls:\n\n0.360 net benefit / call\nvs 0.352 globally\n\n2.080 s / selected call\nvs 1.865 s overall',820,180,365,360,27);
foot(s,'Original-test descriptive diagnosis · Different target populations\nThe targeting advantage did not recover first-model and IC overhead.');

s=slide('Final extension: predict signed incremental benefit',
 'The final bounded extension targets strong_correct-small_correct in{-1,0,1} and predicts small/strong QA service cost. Query-only TFIDFmax8000,1–2grams,min_df2,sublinearTF,L2; cascade original1536prefillfeatures+StandardScaler. Multioutputridge alphas1,10,100,1000; alpha selected by calibration benefitMSE. No neural feature encoder, response-token features, model pair change or architecture search. Relative quality/cost routing is established prior work: RouteLLM https://arxiv.org/html/2406.18665v4 and Unified Routing/Cascading https://arxiv.org/html/2410.10347v4');
para(s,'Target Δ = strong correctness − small correctness\n−1: harm     0: no change     +1: rescue',155,120,32);
text(s,'Before generation',72,330,500,50,32,blue,true);
text(s,'Question TF–IDF → ridge → choose one model\n\nNo small call on the strong branch',72,405,510,160,27);
text(s,'After the small answer',700,330,510,50,32,rust,true);
text(s,'Ordinary prompt features → ridge → escalate\n\nFull small QA cost always charged',700,405,510,160,27);
foot(s,'One regularized predictor per feature family; four penalties considered\nThe contribution is the controlled evaluation and cost diagnosis, not a new routing principle.');

s=slide('Benefit-aware development result: STOP',
 'Protocol committed5f2149b before extension inference. Completed1024 missing strong training answers:1717.42 warm QAseconds plus308.57load and2.02warmup. Alpha10query and1000hidden selected only on128calibration. All16 fixed points retained. Gate required exact random-preselection cost match,≥8questions perbranch,calibration/assessment gains≥3pp,Bonferroni16 lower>0. Nonepassed. Max calibrationgain1.075pp, maxmatchedassessmentgain .128pp; no point had8perbranch. Intervals are descriptive development maxima over fixed grid, not a selected new policy. Newtest remains untouched. AUROC target is positive signed benefit (rescue) versus zero or negative benefit over all128 queries. Sources: results/benefit_development.json and benefit_diagnostics.json.');
const diag=benefitDiagnostics.assessment_diagnostics;
para(s,`Rescue AUROC over all queries: question ${diag.preselection.positive_benefit_auroc.toFixed(3)}, hidden ${diag.cascade.positive_benefit_auroc.toFixed(3)}`,155,55,29);
const bp=(kind,penalty)=>benefit.points.find(p=>p.kind===kind&&p.penalty===penalty).assessment;
const gainCI=a=>`${(100*a.random_preselection.gain).toFixed(3)} [${(100*a.familywise_95[0]).toFixed(3)}, ${(100*a.familywise_95[1]).toFixed(3)}]`;
table(s,[['Fixed method / penalties','EM','s/query','Gain vs random, pp [95%]'],
 ['Query, .025/.05/.1',`${(100*bp('preselection',.05).accuracy).toFixed(2)}%`,bp('preselection',.05).mean_seconds.toFixed(4),gainCI(bp('preselection',.05))],
 ['Cascade, .05/.1',`${(100*bp('cascade',.05).accuracy).toFixed(2)}%`,bp('cascade',.05).mean_seconds.toFixed(4),'Above strong-only budget'],
 ['Cascade, .4/.8/1.6',`${(100*bp('cascade',.4).accuracy).toFixed(2)}%`,bp('cascade',.4).mean_seconds.toFixed(4),gainCI(bp('cascade',.4))]],225,[350,150,160,476]);
para(s,'No point meets the fixed gate. No further search.\nThe reserved fresh 512-question test remains unused.',515,95,29);
foot(s,'DEVELOPMENT ONLY, n=128. All 16 points retained in the report.\nIntervals: paired Bonferroni-16. Largest matched gain falls well below the 3 pp target.');

s=slide('Cascade break-even condition',
 'LetS,Tbe componentcosts, a escalate, qpreselect,hrouter,Delta correctnessdifference. CascadeC=E[S]+E[h_c]+E[aT]. PreselectionC=E[S]+E[h_q]+E[q(T-S)]. Same decisions have equalaccuracy; cascadeextra=E[aS]+E[h_c-h_q]. Atconstantcosts, positiveglobalbenefitg andconditionalbenefitg_r, matchedrandompreselectionbreak-even g_r/g>t/(t-s)+h/[r(t-s)]. OriginalmeanQAillustration t/(t-s)=1.2178; actualevaluationusesquerycostsnotconstantassumption.');
para(s,'Same routing decisions → same accuracy\nCascade extra cost = E[escalate × small cost] + router-cost difference',155,130,32);
para(s,'At matched budget, constant costs and positive mean benefit:\nselected benefit / global benefit > t/(t − s) + h/[r(t − s)]',330,140,30);
para(s,'Original mean-cost illustration: targeting lift must exceed 1.218\nbefore adding router or IC overhead.',520,90,28);
foot(s,'General per-query break-even derivation is in the report.\nAbove the strong-only budget, compare with its endpoint; do not extrapolate p > 1.');

s=slide('Contribution and limits',
 'No universal negative claim, no deployment latency or monetary claim, no newconfirmatoryclaimfromalreadyinspectedtest. Onefinaldataset/modelpair;NF4;transferredIC;512test;coarseoriginalthresholdgrid;query-levelbootstrap;componentreplay. Extension development returned STOP; no new confirmatory test ran. Authors/courseformatnotyetprovided;retaineditableformatforrevision.');
text(s,'Established',72,165,500,50,32,blue,true);
text(s,'Frozen adaptive-IC result is negative\n\nPreselection improves the exploratory frontier\n\nSimple benefit predictors fail the fixed development screen',72,250,525,300,29);
text(s,'Limits',700,165,500,50,32,rust,true);
text(s,'One dataset and final model pair\n4-bit quantization and transferred IC\n512-question original test\nWarm component-time replay\nExploration on an inspected test\nReused development assessment',700,250,510,315,28);
foot(s,'Original confirmation, post-test exploration and development STOP remain separate.\nFinal author list and course submission format still need to be inserted.');

s=slide('References and reproducibility',
 'Verified primary sources. See deliverables/REPORT.md for links and full methodological detail; deliverables/REPRODUCIBILITY.md for artifact paths, hashes and commands. No claim that directbenefitrouting is novel.');
para(s,'Joshi et al. · TriviaQA · ACL 2017\naclanthology.org/P17-1147/',155,90,26);
para(s,'Ong et al. · RouteLLM · ICLR 2025\narxiv.org/abs/2406.18665',270,90,26);
para(s,'Dekoninck, Baader & Vechev · Unified Routing and Cascading\narxiv.org/abs/2410.10347',385,90,26);
para(s,'Chen et al. · Query-Level Uncertainty · arXiv v4, 2026\narxiv.org/abs/2506.09669 · authors’ implementation pinned',500,90,26);
foot(s,'Editable report and presentation · Frozen protocols and original results preserved\nReproduction uses cached components; no additional GPU inference is needed for those analyses.');

await fs.mkdir(build,{recursive:true});await fs.mkdir(path.join(build,'previews'),{recursive:true});
for(const [i,sl] of presentation.slides.items.entries()){
 const blob=await presentation.export({slide:sl,format:'png',scale:1});
 await fs.writeFile(path.join(build,'previews',`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await blob.arrayBuffer()));
}
const candidate=path.join(build,'candidate.pptx');await(await PresentationFile.exportPptx(presentation)).save(candidate);
const finalPath=process.argv[2]||path.join(repo,'deliverables/COMP4901B_research.pptx');
const result=await finalizePresentation({workspaceDir:repo,candidatePath:candidate,finalPath,
 pythonExecutable:process.env.RUNTIME_PYTHON,integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',...([3,4,5,9].flatMap(n=>['--require-native-table-slide',String(n)]))],
 requiredNativeTableOwnerSlides:[3,4,5,9],requiredNativeChartOwnerSlides:[6,7],
 materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:[font]},
 verifyArtifactToolImport:true,receiptPath:path.join(build,'validation.json')});
console.log(JSON.stringify({finalPath,slides:presentation.slides.items.length,font,result}));
