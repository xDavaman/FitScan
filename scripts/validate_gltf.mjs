// Prüft GLB-Dateien mit dem offiziellen Khronos-Validator. Fehler oder Warnungen -> Exit 1.
import fs from "fs";
import validator from "gltf-validator";

let bad = 0;
for (const file of process.argv.slice(2)) {
  const report = await validator.validateBytes(new Uint8Array(fs.readFileSync(file)), { maxIssues: 20 });
  const { numErrors, numWarnings } = report.issues;
  console.log(`${file}: ${numErrors} Fehler, ${numWarnings} Warnungen`);
  for (const m of report.issues.messages) {
    if (m.severity <= 1) console.log(`  ${m.code}: ${m.message} (${m.pointer})`);
  }
  if (numErrors || numWarnings) bad++;
}
process.exit(bad ? 1 : 0);
