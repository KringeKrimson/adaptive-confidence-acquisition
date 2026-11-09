// Revise the existing twelve-slide deck from preserved numerical results. No experiments.
import fs from 'node:fs/promises';
import path from 'node:path';
import {createRequire} from 'node:module';
import {fileURLToPath,pathToFileURL} from 'node:url';
import crypto from 'node:crypto';
import {spawnSync} from 'node:child_process';
const repo=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const build=path.join(repo,'outputs/report-build',new Date().toISOString().replace(/[:.]/g,'-'));
await fs.mkdir(build,{recursive:true});
if(process.env.RUNTIME_NODE_MODULES)await fs.symlink(process.env.RUNTIME_NODE_MODULES,path.join(build,'node_modules'),'dir');
const require=createRequire(path.join(build,'author.mjs'));
const {FileBlob,PresentationFile}=await import(require.resolve('@oai/artifact-tool'));
const skill=process.env.PRESENTATIONS_SKILL_DIR;
if(!skill)throw Error('Set PRESENTATIONS_SKILL_DIR to the installed presentation skill');
const {resolvePresentationFont,applyPresentationChartFont,finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const source=process.argv[3]||path.join(repo,'deliverables/COMP4901B_research.pptx');
const sourceCopy=path.join(build,'source.pptx');await fs.copyFile(source,sourceCopy);
const sourceSha256=crypto.createHash('sha256').update(await fs.readFile(sourceCopy)).digest('hex');
const p=await PresentationFile.importPptx(await FileBlob.load(sourceCopy));
if(p.slides.items.length!==12)throw Error('Requires the existing12-slide research deck');
const font=resolvePresentationFont({fontFamily:'Helvetica Neue'});
const ink='#142a3a',blue='#176b91',rust='#b95732',grey='#51636f';
const read=async name=>JSON.parse(await fs.readFile(path.join(repo,'results',name+'.json')));
const benefit=await read('benefit_development'),diag=await read('benefit_diagnostics');
const audit=await read('pregeneration_audit'),auditDiag=await read('pregeneration_diagnostics');
const inspect=await p.inspect({kind:'slide,textbox,shape,image,table,chart,layout',maxChars:200000});
await fs.writeFile(path.join(build,'source-inspect.ndjson'),inspect.ndjson);
const records=inspect.ndjson.split('\n').filter(Boolean).map(line=>JSON.parse(line));
function text(slide,value,left,top,width,height,size=27,color=ink,bold=false){
 const s=slide.shapes.add({geometry:'textbox',position:{left,top,width,height},fill:'none',line:{fill:'none',width:0}});
 s.text=value;s.text.insets=0;s.text.style={fontSize:size,typeface:font,color,bold};return s;
}
function replaceAt(index,predicate,value){
 const r=records.find(r=>r.slideIndex===index&&r.kind==='textbox'&&predicate(r));
 if(!r)throw Error('Missing source text on slide'+(index+1));
 const s=p.resolve(r.id);s.text=value;s.text.style={...r.style,typeface:font};return s;
}
function foot(slide,value){text(slide,value,72,620,1110,50,18,grey);}
function fresh(index,title,note){
 const ids=[...new Set(records.filter(r=>r.slideIndex===index&&/^(sh|im|tb|ch)\//.test(r.id)).map(r=>r.id))];
 p.delete(ids);const slide=p.slides.items[index];
 text(slide,title,64,42,1152,80,44,ink,true);
 text(slide,String(index+1),1175,676,45,24,18,grey);
 slide.speakerNotes.text=note;return slide;
}
function table(slide,values,top,widths){
 const t=slide.tables.add({rows:values.length,columns:values[0].length,left:72,top,width:1136,height:values.length*66,columnWidths:widths,values});
 t.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length}).assign({textStyle:{fontSize:25,typeface:font,color:ink},margins:{left:12,right:12,top:12,bottom:10}});
 t.cells.block({row:0,column:0,rowCount:1,columnCount:values[0].length}).assign({fill:'#e9f1f5',textStyle:{fontSize:25,typeface:font,color:ink,bold:true}});
 t.borders.assign({fill:'#d4dfe5',width:1,style:'solid'});return t;
}
// Keep the original diagram, data split/settings table, cost table and two native charts.
replaceAt(0,r=>r.position.top===42,'Adaptive confidence acquisition\nfor cost-aware LLM routing');
replaceAt(0,r=>r.position.top===400,'A negative confirmatory result\nwith benefit-aware and architectural follow-ups');
replaceAt(0,r=>r.position.top===620,'COMP4901B · 9 November 2026 · Authors: [complete author list]');
p.slides.items[0].speakerNotes.text='Consolidated completed research, including the development-only audit at ffe94cf. The original512-question test is confirmatory only for its frozen primary comparison. Independent model preselection is post-test exploratory. Benefit-aware and pre-generation studies reuse development assessment. No reserved-test evaluation ran. Full paper: deliverables/REPORT.md.';
replaceAt(1,r=>r.position.top===42,'Original answer-first adaptive IC');
p.slides.items[1].speakerNotes.text+=' The IC score itself is question-only and can precede generation. This diagram describes the original answer-first implementation, not an intrinsic IC requirement. Ordinary QA prefill is required for the cheap probe. Released code: https://github.com/tigerchen52/query_level_uncertainty/blob/5c15b32a6f9f36688098d0a749cb68bad11e9f23/ql_uncertainty/internal_confidence.py';
const splitTable=records.find(r=>r.slideIndex===2&&r.kind==='table');
p.resolve(splitTable.id).cells.set(splitTable.rows-1,2,'Unused after development STOP');
p.slides.items[2].speakerNotes.text+=' The reserved fresh512 test remains unused. The audit and benefit extension use the same already-inspected128 assessment questions and cannot supply fresh confirmation.';
replaceAt(3,r=>r.position.top===620,'Original-test component means. Warm replay excludes loading, queueing and residency.\nEstimator/model calls are charged; deployment latency was not measured.');
p.slides.items[3].speakerNotes.text+=' For an IC-only rule with identical choices, answer-first costs exceed IC-first by E[aS]. This subtraction is invalid for a cheap-prefill method unless the required QA prefill cost is retained. RouterBench already uses probabilistic model mixtures as a cost-quality control: https://arxiv.org/html/2403.12031v2';
replaceAt(4,r=>r.position.top===465,'Neither IC-using rule has a positive familywise lower bound or meets the precommitted 3-point practical target.');
replaceAt(6,r=>r.position.left===820,'Net benefit / strong call\nSelected 0.360 · global 0.352\n\nStrong service cost\nSelected 2.080 s\nOverall 1.865 s\n\nMean-cost illustration\nLift >1.218 before IC overhead');
p.slides.items[6].speakerNotes.text+=' Constant-cost break-even: g_selected/g > t/(t-s)+h/[r(t-s)], assuming positive global benefit and an exact random-preselection cost match. The1.218 value uses original-test mean QA costs illustratively. It does not replace per-query costs or prove deployment performance.';

let s=fresh(7,'Benefit-aware extension: development STOP',
 'Multioutput ridge predicts signed strong-minus-small correctness benefit and small/strong QA cost. Question features are train-only TFIDF and hidden features are the original1536d QA-prefill states. Four ridge alphas selected by calibration MSE:10question and1000hidden. Sixteen fixed rules. Gate requires exact random-preselection cost match,>=8assessment queries per branch,calibration and assessment gains>=3pp and positive Bonferroni16 lower bound. No point passes. Largest calibrationgain1.075pp and matched assessmentgain.128pp [0,.720]. This is a descriptive maximum over the fixed grid, not an assessment-selected new policy. No fresh-test inference. results/benefit_development.json and benefit_diagnostics.json.');
text(s,'Signed benefit target: strong correctness − small correctness',72,150,1136,55,29);
const bd=diag.assessment_diagnostics;
table(s,[['Predictor','Features','Rescue AUROC*','Max. matched gain'],
 ['Query preselection','TF–IDF',bd.preselection.positive_benefit_auroc.toFixed(3),'+.128 pp [0, .720]'],
 ['Post-answer cascade','QA prefill',bd.cascade.positive_benefit_auroc.toFixed(3),'No positive gain']],235,[320,220,230,366]);
text(s,'All 16 points fail the fixed gate.\nLargest calibration gain 1.075 pp, below the 3 pp target.\nNo point retains eight assessment queries in each branch.',72,470,1136,125,29);
foot(s,'DEVELOPMENT ONLY, n=128. *Rescue versus all other queries, not correctness AUROC.\nGain interval: Bonferroni-16. The reserved fresh test remains unused.');

s=fresh(8,'IC-first: lower cost for one fixed rule',
 'Development-only audit at ffe94cf. The .5 threshold was retrospectively highlighted after replay and was not calibration-selected. It cannot be promoted as a new operating point or confirmatory test. Cached IC is independent of the answer and uses one separate prompt forward. At weighted .5: assessmentaccuracy.5625, answer-firstcost1.6818570224531246, IC-firstcost1.3990305253281239, saving.28282649712500074 (16.8163%). Randomcostmatchedaccuracy.5334191878311934; gain2.908pp pointwise95[-1.4737,7.1213]. Calibrationgain-5.2815pp. All three calibration-selected IC-first families bypass IC with globally constant endpoint rules. Original adaptive-versus-cheap contrast remains conditional on answer-first architecture and is not invalidated. Sources results/pregeneration_audit.json and pregeneration_diagnostics.json. Chen paper https://arxiv.org/html/2506.09669v4');
const illustration=auditDiag.illustrative_unselected_threshold_05.weighted;
const ia=illustration.assessment;
table(s,[['Unselected .5 rule','Assessment EM','s/query'],
 ['Answer-first IC',(100*ia.accuracy).toFixed(2)+'%',ia.answer_first_seconds.toFixed(4)],
 ['IC-first, same decisions',(100*ia.accuracy).toFixed(2)+'%',ia.mean_seconds.toFixed(4)],
 ['Cost-matched random',(100*ia.random.accuracy).toFixed(2)+'%',ia.mean_seconds.toFixed(4)]],160,[540,300,296]);
text(s,'16.8% lower replayed cost, with identical answers\nGain vs random: +2.91 pp [−1.47, +7.12], pointwise 95%\nCalibration gain: −5.28 pp',72,455,1136,135,29);
foot(s,'DEVELOPMENT ONLY. Retrospectively highlighted cut, not a selected operating point.\nThe saving applies to this IC-only rule, not measured deployment-wide performance.');

s=fresh(9,'Pre-generation curves: development only',
 'All thin curves are unselected cuts from the pre-analysis .05 confidence grid, using identical measured full IC forward cost. They are descriptive assessment curves, not fitted or test-selected operating points. The selected IC families are direct small/strong endpoints and acquire IC0%. Their cost-matched mixtures equal random preselection algebraically, which does not establish signal equivalence. Assessment small-correctnessAUROC weighted.6966,top.6832,unweighted.7455. The positive weighted .5 assessment result conflicts with calibration. Prefill/selective-IC costs are hypothetical because total QA timing was not split. Across four F=rho*S sensitivity scenarios the largest selected-policy gain is1.33pp, includingzeroF. QA KV must retain the same prefix/model/cache/logits. A separate IC prompt cannot supply QA KV. No additional measurement or confirmation was launched. Source figure: deliverables/figures/pregeneration_audit.png; exact data: results/pregeneration_diagnostics.json.');
const chartNumber=v=>Number(v.toPrecision(12));
const c=audit.components.assessment,curve=auditDiag.grid.assessment;
const sr=c.strong_seconds,ss=c.small_seconds,as=audit.oracle_preselection.accuracy[0],at=audit.families['ic-first-weighted'].points[0].assessment.accuracy;
const series=[{name:'Random preselection',xValues:[ss,sr].map(chartNumber),values:[as,at],line:{fill:ink,width:2},marker:{symbol:'none'}}];
for(const [name,color,label] of [['weighted',blue,'Weighted IC'],['top_right',rust,'Top-right P(Yes)'],['unweighted','#398455','Unweighted mean']]){
 series.push({name:label,xValues:curve[name].map(r=>chartNumber(r.mean_seconds)),values:curve[name].map(r=>chartNumber(r.accuracy)),line:{fill:color,width:2},marker:{symbol:'none'}});
}
series.push({name:'Selected endpoints',xValues:[ss,sr].map(chartNumber),values:[as,at],line:{fill:'none',width:0},marker:{symbol:'diamond',size:10,fill:ink}});
const ch=s.charts.add('scatter',{position:{left:72,top:145,width:1136,height:430},series,scatterOptions:{style:'lineWithMarkers'},hasLegend:true,
 legend:{position:'bottom',textStyle:{fontSize:22}},xAxis:{min:.2,max:2.0,title:{text:'Warm component seconds / query',textStyle:{fontSize:23}},textStyle:{fontSize:22},numberFormatCode:'0.0'},
 yAxis:{min:.3,max:.65,title:{text:'Assessment exact match',textStyle:{fontSize:23}},textStyle:{fontSize:22},numberFormatCode:'0%',majorGridlines:{fill:'#e0e7eb',width:1}}});
applyPresentationChartFont(ch,{fontFamily:font});
foot(s,'Assessment n=128. Curves are unselected cuts; diamonds are calibration-selected endpoints.\nSelective-prefill costs remain hypothetical. IC superiority is not established.');

s=fresh(10,'Synthesis and scientific limits',
 'Assessment complementarity:42bothcorrect,50bothwrong,31rescues,5harms. Free label oracle78/128=60.9375%, strong73/128=57.03125%. Oracleminimumcost.6123preselection/.7039answerfirst with free perfect labels, unattainable. Original confidence AUROCs do not predict signed routing benefit. Originaltest512quantizedonepair/dataset, transferredIC,coarsegrid. Laterassessment128reused,illustrativecutposthoc andpointwise intervals. Additivewarmreplayisnotdeploymentlatency. Answer-first overhead does not invalidate the original adaptive-versus-cheap comparison because both pay small QA. Further confirmation is not justified by the completed development evidence.');
text(s,'Assessment oracle ceiling 60.94% versus strong-only 57.03%\n31 rescues and five harms show complementarity, not an attainable router.',72,150,1136,100,29);
text(s,'Supported by this study',72,300,510,50,31,blue,true);
text(s,'Adaptive IC misses its frozen target\n\nPrediction quality differs from routing benefit\n\nEarlier decisions can avoid answer cost',72,375,520,210,28);
text(s,'Limits on the claims',700,300,510,50,31,rust,true);
text(s,'One dataset and quantized model pair\n512-question original test\nReused development assessment\nPost-test exploratory comparisons\nWarm replay and unmeasured KV costs',700,375,510,210,28);
foot(s,'The oracle uses unavailable labels. IC-first superiority is not established.\nNo further confirmation is warranted by these data. The reserved test remains unused.');

s=fresh(11,'Related work and reproducibility',
 'Verified primary references in deliverables/REPORT.md. Chen etal Query-LevelUncertainty ICLR2026 https://arxiv.org/html/2506.09669v4; LingjiaoChen,Zaharia,Zou FrugalGPT2023 https://arxiv.org/abs/2305.05176; Ong etal RouteLLM ICLR2025 https://arxiv.org/html/2406.18665v4; Dekoninck,Baader,Vechev UnifiedRouting/Cascading revised2026 https://arxiv.org/html/2410.10347v4; Huetal RouterBench2024 https://arxiv.org/html/2403.12031v2; Lietal LLMRouterBench2026 https://arxiv.org/abs/2601.07206v1. Dataset Joshi etal TriviaQA ACL2017 https://aclanthology.org/P17-1147/. RouterBench ZeroRouter already uses probabilistic model mixtures. No novelty claim for that baseline, learned routing, benefit prediction or quality-minus-cost optimization. Reproducibility guide identifies immutable protocols, numericalJSON and private cached outputs.');
table(s,[['Prior work','Established idea'],['Chen et al., IC','Answerability before generation'],['FrugalGPT','Quality-scored, cost-aware cascades'],['RouteLLM','Learned query-only model selection'],['Dekoninck et al.','Unified quality/cost routing framework'],['Routing benchmarks','Model mixtures, oracles and fair evaluation']],150,[400,736]);
foot(s,'Our contribution is the preserved negative result and cost/target diagnosis.\nReport, editable slides, six figure pairs and reproducibility guide share one repository.');

await fs.mkdir(path.join(build,'previews'),{recursive:true});
for(const [i,sl] of p.slides.items.entries()){
 const blob=await p.export({slide:sl,format:'png',scale:1});
 await fs.writeFile(path.join(build,'previews',`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await blob.arrayBuffer()));
}
const candidate=path.join(build,'candidate.pptx');await(await PresentationFile.exportPptx(p)).save(candidate);
// The importer drops embedded chart workbooks. Restore the actual source chart
// parts and workbook bytes for unchanged slides6/7, rather than inventing lineage.
const preserved=path.join(build,'source-preserved-candidate.pptx');
const repair=spawnSync(process.env.RUNTIME_PYTHON,['-c',String.raw`
import json,posixpath,sys,hashlib
from zipfile import ZipFile,ZIP_DEFLATED
import xml.etree.ElementTree as E
source,candidate,output=sys.argv[1:]
C='http://schemas.openxmlformats.org/drawingml/2006/chart'
R='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
REL='http://schemas.openxmlformats.org/package/2006/relationships'
CT='http://schemas.openxmlformats.org/package/2006/content-types'
def relpart(part):return posixpath.join(posixpath.dirname(part),'_rels',posixpath.basename(part)+'.rels')
def chart(z,n):
    slide=f'ppt/slides/slide{n}.xml'
    refs=E.fromstring(z.read(slide)).findall('.//{'+C+'}chart');assert len(refs)==1
    rid=refs[0].get('{'+R+'}id')
    rels=E.fromstring(z.read(relpart(slide)))
    target=next(r.get('Target') for r in rels if r.get('Id')==rid)
    return posixpath.normpath(posixpath.join(posixpath.dirname(slide),target)).lstrip('/')
with ZipFile(source) as src,ZipFile(candidate) as dst:
    parts={n:dst.read(n) for n in dst.namelist()};records=[]
    for n in [6,7]:
        old,new=chart(src,n),chart(dst,n)
        xml=src.read(old);rels=E.fromstring(src.read(relpart(old)))
        assert len(rels)==1 and rels[0].get('Type')==R+'/package'
        workbook=posixpath.normpath(posixpath.join(posixpath.dirname(old),rels[0].get('Target'))).lstrip('/')
        newbook=f'ppt/embeddings/preserved-source-slide{n}.xlsx'
        blob=src.read(workbook);parts[newbook]=blob;parts[new]=xml
        rels[0].set('Target',posixpath.relpath(newbook,posixpath.dirname(new)))
        E.register_namespace('',REL)
        parts[relpart(new)]=E.tostring(rels,encoding='utf-8',xml_declaration=True)
        records.append(dict(slide=n,chart_sha256=hashlib.sha256(xml).hexdigest(),workbook_sha256=hashlib.sha256(blob).hexdigest()))
    types=E.fromstring(parts['[Content_Types].xml'])
    if not any(x.get('Extension')=='xlsx' for x in types):
        E.SubElement(types,'{'+CT+'}Default',Extension='xlsx',ContentType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    E.register_namespace('',CT)
    parts['[Content_Types].xml']=E.tostring(types,encoding='utf-8',xml_declaration=True)
with ZipFile(output,'w',ZIP_DEFLATED) as z:
    for name,data in parts.items():z.writestr(name,data)
print(json.dumps(records))
`,sourceCopy,candidate,preserved],{encoding:'utf8',maxBuffer:1024*1024});
if(repair.status!==0)throw Error(repair.stderr||repair.error?.message||'Source chart preservation failed');
await fs.writeFile(path.join(build,'preserved-source-charts.json'),repair.stdout);
const finalPath=process.argv[2]||path.join(repo,'outputs/report-finalization',path.basename(build),'COMP4901B_research.pptx');
await fs.mkdir(path.dirname(finalPath),{recursive:true});
const result=await finalizePresentation({workspaceDir:repo,candidatePath:preserved,finalPath,
 explicitTotalSlideCount:12,pythonExecutable:process.env.RUNTIME_PYTHON,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',...([3,4,5,8,9,12].flatMap(n=>['--require-native-table-slide',String(n)]))],
 requiredNativeTableOwnerSlides:[3,4,5,8,9,12],requiredNativeChartOwnerSlides:[6,7,10],requiredEmbeddedWorkbookChartOwnerSlides:[6,7],
 materializeLiteralChartWorkbooks:true,
 fontPolicy:{basis:'reference',families:[font],referencePath:sourceCopy,referenceSha256:sourceSha256},
 verifyArtifactToolImport:true,receiptPath:path.join(build,'validation.json')});
console.log(JSON.stringify({finalPath,build,slides:p.slides.items.length,font,sha256:result.finalSha256}));
