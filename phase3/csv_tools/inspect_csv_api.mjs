import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
const dependencies='C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1025-760-0_26.1025.760.0_x64__3k8sg7r9htsxt/dependencies/node';
const require=createRequire(dependencies+'/artifact-loader.cjs');
const {Workbook}=await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const wb=Workbook.create();
console.log(JSON.stringify(wb.help('csv',{include:'index,examples,notes',maxChars:4000})).slice(0,6000));
