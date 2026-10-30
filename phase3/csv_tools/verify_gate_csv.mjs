import fs from 'node:fs/promises';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
const dependencies='C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1025-760-0_26.1025.760.0_x64__3k8sg7r9htsxt/dependencies/node';
const require=createRequire(dependencies+'/artifact-loader.cjs');
const {Workbook}=await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const root=path.resolve('phase3'),latest=JSON.parse(await fs.readFile(path.join(root,'data_gate_outputs/LATEST.json'),'utf8'));
const schema=JSON.parse(await fs.readFile(path.join(latest.path,'CSV_SCHEMA.json'),'utf8'));
const out=path.join(root,'data_gate_validation/artifact_csv');await fs.mkdir(out,{recursive:true});
let tables=0,records=0;
for(const [name,info] of Object.entries(schema)){
  const source=await fs.readFile(path.join(latest.path,'csv',name+'.csv'),'utf8');
  const workbook=await Workbook.fromCSV(source,{sheetName:name.slice(0,31)});workbook.recalculate();
  const sheet=workbook.worksheets.getItemAt(0),values=sheet.getRangeByIndexes(0,0,info.rows+1,info.columns.length).values;
  if(JSON.stringify(values[0])!==JSON.stringify(info.columns))throw new Error('Header changed: '+name);
  const quote=value=>{const text=value==null?'':String(value);return /[",\r\n]/.test(text)?'"'+text.replaceAll('"','""')+'"':text;};
  const newline=source.includes('\r\n')?'\r\n':'\n';const result=values.map(row=>row.map(quote).join(',')).join(newline)+newline;
  if(result!==source)throw new Error('Artifact CSV round-trip changed source data: '+name);
  await fs.writeFile(path.join(out,name+'.csv'),result);tables++;records+=info.rows;
  if(name==='historical_class_support'){
    sheet.getRangeByIndexes(0,0,info.rows+1,info.columns.length).format.autofitColumns();
    const preview=await workbook.render({sheetName:name.slice(0,31),range:'A1:H10',scale:1.5,format:'png'});
    await fs.writeFile(path.join(root,'data_gate_validation/csv_support_preview.png'),new Uint8Array(await preview.arrayBuffer()));
  }
}
const result={status:'passed',library:'@oai/artifact-tool',CSV_tables:tables,CSV_rows:records,all_values_headers_and_missing_cells_exact:true,CSV_only_user_requirement:true};
await fs.writeFile(path.join(root,'data_gate_validation/ARTIFACT_CSV_VALIDATION.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result));
