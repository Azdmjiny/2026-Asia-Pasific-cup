import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const inputPath = "/Users/jinyu/workspace/2026亚太杯/data/merged.xlsx";
const tempPath = "/tmp/zero-audit/merged_audited.xlsx";
const previewPath = "/tmp/zero-audit/zero_audit.png";

const input = await FileBlob.load(inputPath);
const workbook = await SpreadsheetFile.importXlsx(input);
const sheet = workbook.worksheets.add("Zero Audit");
sheet.showGridLines = false;
sheet.freezePanes.freezeRows(7);

sheet.getRange("A1:G1").merge();
sheet.getRange("A1").values = [["缺失值与零值判定审计"]];
sheet.getRange("A1:G1").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF", size: 16 },
  horizontalAlignment: "center",
  verticalAlignment: "center",
};
sheet.getRange("A1:G1").format.rowHeight = 30;

sheet.getRange("A3:B6").values = [
  ["审计结论", "没有空白单元格能以足够置信度判定为 0"],
  ["新增零值数量", 0],
  ["原有明确零值", "2 个，均位于 CLR 字段"],
  ["处理原则", "保留 Merged 原始合并数据；不把缺测、短横线或结构性未提供改成 0"],
];
sheet.getRange("A3:A6").format = {
  fill: "#D9EAF7",
  font: { bold: true, color: "#17365D" },
};
sheet.getRange("B3:B6").format.wrapText = true;
sheet.getRange("B3").format = {
  fill: "#E2F0D9",
  font: { bold: true, color: "#375623" },
  wrapText: true,
};

const headers = [
  "字段",
  "Merged 缺失数",
  "原始空白数",
  "原始短横线数",
  "原有数字 0",
  "补 0 数量",
  "判定与依据",
];
const rows = [
  ["RIVER LEVEL", 236, 30, 206, 0, 0, "缺测；前后水位均为正常正值"],
  ["R/W PUMP DUTY", 1748, 183, 89, 0, 0, "缺测/字段未提供；缺失时原水流量仍为正"],
  ["R/W FLOW", 11, 5, 5, 0, 0, "孤立或短段缺测；前后流量约 44–54"],
  ["R/W NTU", 1, 0, 1, 0, 0, "孤立缺测；相邻值为 2 和 40"],
  ["R/W CLR", 1, 0, 1, 0, 0, "孤立缺测；相邻值为 81 和 400"],
  ["R/W PH", 1644, 168, 0, 0, 0, "部分月份字段未提供，不代表 pH 为 0"],
  ["FILT. NTU", 0, 0, 0, 0, 0, "完整记录，无需处理"],
  ["C/W WELL LEVEL", 2, 1, 0, 0, 0, "孤立缺测；相邻水位约 3.6–3.8"],
  ["PH", 1644, 168, 0, 0, 0, "部分月份字段未提供，不代表 pH 为 0"],
  ["NTU", 341, 336, 5, 0, 0, "含 2026-02 预测目标及少量缺测，必须保留为空"],
  ["CLR", 2, 2, 0, 2, 0, "两个明确数字 0 已保留；两个空白相邻值均为 5，属于缺测"],
  ["CL2", 1726, 183, 67, 0, 0, "缺测/字段未提供；缺失时出厂流量仍为正"],
  ["F/RIDE", 4132, 662, 1994, 0, 0, "采集字段启停；短横线期间 ALUM 仍为正，不能解释为停加"],
  ["ALUM", 1645, 169, 0, 0, 0, "部分月份字段未提供；已记录值始终为正"],
  ["T/W PUMP DUTY", 1652, 172, 4, 0, 0, "缺测/字段未提供；缺失时出厂流量仍为正"],
  ["T/W FLOW", 16, 15, 0, 0, 0, "孤立或短段缺测；相邻流量均为正"],
  ["18ML LEVEL", 5460, 2731, 1253, 0, 0, "全期结构性未提供；水位不可能在正常运行期持续为 0"],
  ["18ML FLOW", 5460, 2731, 1253, 0, 0, "全期结构性未提供；不能将未提供解释为零流量"],
  ["REMARKS", 5237, 3761, 0, 0, 0, "文本备注留空，不属于数值零"],
];

sheet.getRange("A8:G8").values = [headers];
sheet.getRange(`A9:G${8 + rows.length}`).values = rows;
sheet.getRange("A8:G8").format = {
  fill: "#4472C4",
  font: { bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
};
sheet.getRange(`A8:G${8 + rows.length}`).format.borders = {
  preset: "all",
  style: "thin",
  color: "#D9E2F3",
};
sheet.getRange(`B9:F${8 + rows.length}`).format.horizontalAlignment = "center";
sheet.getRange(`G9:G${8 + rows.length}`).format.wrapText = true;
sheet.getRange(`A9:G${8 + rows.length}`).conditionalFormats.addCustom(
  "=$F9>0",
  { fill: "#FFF2CC", font: { bold: true, color: "#7F6000" } },
);

const detailStart = 11 + rows.length;
sheet.getRange(`A${detailStart}:G${detailStart}`).merge();
sheet.getRange(`A${detailStart}`).values = [["原有明确零值明细（未作修改）"]];
sheet.getRange(`A${detailStart}:G${detailStart}`).format = {
  fill: "#D9EAD3",
  font: { bold: true, color: "#274E13" },
};
sheet.getRange(`A${detailStart + 1}:D${detailStart + 1}`).values = [[
  "日期", "时间", "字段", "原始值",
]];
sheet.getRange(`A${detailStart + 2}:D${detailStart + 3}`).values = [
  ["2025-02-27", "03:00", "CLR", 0],
  ["2025-02-27", "05:00", "CLR", 0],
];
sheet.getRange(`A${detailStart + 1}:D${detailStart + 1}`).format = {
  fill: "#70AD47",
  font: { bold: true, color: "#FFFFFF" },
};
sheet.getRange(`A${detailStart + 1}:D${detailStart + 3}`).format.borders = {
  preset: "all",
  style: "thin",
  color: "#C6E0B4",
};

sheet.getRange("A:A").format.columnWidth = 22;
sheet.getRange("B:F").format.columnWidth = 16;
sheet.getRange("G:G").format.columnWidth = 55;
sheet.getRange(`A3:G${detailStart + 3}`).format.verticalAlignment = "center";

const check = await workbook.inspect({
  kind: "table",
  range: `Zero Audit!A1:G${detailStart + 3}`,
  include: "values,formulas",
  tableMaxRows: 40,
  tableMaxCols: 7,
});
console.log(check.ndjson);

const preview = await workbook.render({
  sheetName: "Zero Audit",
  range: `A1:G${detailStart + 3}`,
  scale: 1,
  format: "png",
});
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(tempPath);
await fs.copyFile(tempPath, inputPath);
console.log(JSON.stringify({ inputPath, previewPath, rows: rows.length }));
