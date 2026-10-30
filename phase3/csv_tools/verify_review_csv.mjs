import fs from 'node:fs/promises';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
const dependencies='C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1025-760-0_26.1025.760.0_x64__3k8sg7r9htsxt/dependencies/node';
const require=createRequire(dependencies+'/artifact-loader.cjs');
const {Workbook}=await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const base=path.resolve('phase3/collection_review_outputs'),schema=JSON.parse(await fs.readFile(path.join(base,'CSV_SCHEMA.json'),'utf8'));
let tables=0,rows=0;
for(const [name,info] of Object.entries(schema)){
 const original=await fs.readFile(path.join(base,name),'utf8'),workbook=await Workbook.fromCSV(original,{sheetName:name.slice(0,31)});workbook.recalculate();
 const values=workbook.worksheets.getItemAt(0).getRangeByIndexes(0,0,info.rows+1,info.columns.length).values;
 if(JSON.stringify(values[0])!==JSON.stringify(info.columns))throw new Error('Column mismatch: '+name);
 const quote=value=>{const s=value==null?'':String(value);return /[",\r\n]/.test(s)?'"'+s.replaceAll('"','""')+'"':s;};
 const rendered=values.map(r=>r.map(quote).join(',')).join('\n')+'\n';if(rendered!==original)throw new Error('CSV values changed: '+name);tables++;rows+=info.rows;
}
const report={status:'passed',library:'@oai/artifact-tool',CSV_tables:tables,CSV_rows_including_control:rows,all_values_headers_and_missing_cells_exact:true,scientific_new_subjects:0};
await fs.writeFile('phase3/data_gate_validation/collection_review/ARTIFACT_CSV_VALIDATION.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
