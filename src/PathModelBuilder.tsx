import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Share2,
  ListPlus,
  Table as TableIcon,
  Network,
  Calculator,
  CheckCircle2,
  Download,
  ChevronRight,
  ChevronLeft,
  AlertCircle,
  Settings2
} from "lucide-react";

// --- Types ---

type VarName = string;

type Cell = {
  r: number; // correlation
  n: number; // pairwise sample size, NaN for diagonal
};

type CellMatrix = Record<VarName, Record<VarName, Cell>>;

type Edge = { from: VarName; to: VarName };

type Coef = { 
  from: VarName; 
  to: VarName; 
  beta: number; 
  se: number;      
  t: number;       
  p: number;       
  vif: number;     
};

type Fit = {
  totalN: number;
  N_method: "harmonic" | "min";
  SRMR: number;
  df: number;
  observedMoments: number;
  freeParams: number;
  endogenousCount: number;
  chi2?: number;
  rmsea?: number;
  cfi?: number;
  tli?: number;
};

type EstResult = {
  coeffs: Coef[];
  r2: Record<VarName, number>;
  resid: Record<VarName, number>;
  fit: Fit;
};

type NodePos = Record<VarName, { x: number; y: number }>;

type SampleType = "All" | "Lodging" | "Restaurant" | "Tourism and travel";

const BASE_VARS: VarName[] = ["loyalty", "satisfaction", "value", "quality"];

// --- Data Presets ---
const SAMPLE_PRESETS: Record<SampleType, { pair: Record<string, { r: number; n: number }> }> = {
  All: {
    pair: {
      "loyalty|satisfaction": { r: 0.734, n: 63671 },
      "loyalty|value": { r: 0.545, n: 81110 },
      "loyalty|quality": { r: 0.575, n: 52764 },
      "satisfaction|value": { r: 0.708, n: 37150 },
      "quality|satisfaction": { r: 0.711, n: 34677 },
      "quality|value": { r: 0.561, n: 58390 },
    },
  },
  Lodging: {
    pair: {
      "loyalty|satisfaction": { r: 0.726, n: 19271 },
      "loyalty|value": { r: 0.675, n: 11993 },
      "loyalty|quality": { r: 0.547, n: 12251 },
      "satisfaction|value": { r: 0.77, n: 12705 },
      "quality|satisfaction": { r: 0.922, n: 8859 },
      "quality|value": { r: 0.784, n: 9268 },
    },
  },
  Restaurant: {
    pair: {
      "loyalty|satisfaction": { r: 0.707, n: 10048 },
      "loyalty|value": { r: 0.693, n: 6599 },
      "loyalty|quality": { r: 0.532, n: 10941 },
      "satisfaction|value": { r: 0.72, n: 6710 },
      "quality|satisfaction": { r: 0.714, n: 7936 },
      "quality|value": { r: 0.6, n: 7301 },
    },
  },
  "Tourism and travel": {
    pair: {
      "loyalty|satisfaction": { r: 0.739, n: 32943 },
      "loyalty|value": { r: 0.503, n: 62257 },
      "loyalty|quality": { r: 0.598, n: 28903 },
      "satisfaction|value": { r: 0.661, n: 17474 },
      "quality|satisfaction": { r: 0.601, n: 16283 },
      "quality|value": { r: 0.501, n: 41821 },
    },
  },
};

// --- Math & Stats Helpers ---

function pairKey(a: VarName, b: VarName) {
  const x = [a, b].sort();
  return `${x[0]}|${x[1]}`;
}

function deepClone<T>(obj: T): T {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const sc = (globalThis as any).structuredClone as undefined | ((x: T) => T);
  if (typeof sc === "function") return sc(obj);
  return JSON.parse(JSON.stringify(obj)) as T;
}

function getNormalPValue(z: number): number {
  const absZ = Math.abs(z);
  const p = 0.3275911;
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  
  const sign = (z < 0) ? -1 : 1;
  const t = 1.0 / (1.0 + p * absZ);
  const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-absZ * absZ);
  return (1.0 - y) * 2;
}

function getSignificanceStar(p: number) {
  if (p < 0.001) return "***";
  if (p < 0.01) return "**";
  if (p < 0.05) return "*";
  return "";
}

// --- Matrix Algebra ---

function matIdentity(n: number): number[][] {
  return Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
}
function matClone(A: number[][]): number[][] {
  return A.map((r) => r.slice());
}
function transpose(A: number[][]): number[][] {
  return A[0].map((_, j) => A.map((row) => row[j]));
}
function matMul(A: number[][], B: number[][]): number[][] {
  const n = A.length,
    m = B[0].length,
    k = B.length;
  const out = Array.from({ length: n }, () => Array(m).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      let s = 0;
      for (let t = 0; t < k; t++) s += A[i][t] * B[t][j];
      out[i][j] = s;
    }
  }
  return out;
}
function matVecMul(A: number[][], v: number[]): number[] {
  return A.map((row) => row.reduce((s, x, i) => s + x * v[i], 0));
}
function vecDot(a: number[], b: number[]): number {
  return a.reduce((s, x, i) => s + x * b[i], 0);
}

function matInverse(A: number[][]): number[][] {
  const n = A.length;
  let M = matClone(A);
  let I = matIdentity(n);

  for (let col = 0; col < n; col++) {
    let pivotRow = col;
    for (let r = col; r < n; r++) {
      if (Math.abs(M[r][col]) > Math.abs(M[pivotRow][col])) pivotRow = r;
    }
    if (Math.abs(M[pivotRow][col]) < 1e-12) {
      throw new Error("Matrix is singular (cannot invert). Consider removing/adjusting variables or paths.");
    }

    [M[col], M[pivotRow]] = [M[pivotRow], M[col]];
    [I[col], I[pivotRow]] = [I[pivotRow], I[col]];

    const piv = M[col][col];
    for (let j = 0; j < n; j++) {
      M[col][j] /= piv;
      I[col][j] /= piv;
    }

    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = M[r][col];
      for (let j = 0; j < n; j++) {
        M[r][j] -= factor * M[col][j];
        I[r][j] -= factor * I[col][j];
      }
    }
  }
  return I;
}

function matDet(A: number[][]): number {
  const n = A.length;
  let M = matClone(A);
  let det = 1;
  for (let i = 0; i < n; i++) {
    let pivot = i;
    for (let r = i; r < n; r++) if (Math.abs(M[r][i]) > Math.abs(M[pivot][i])) pivot = r;
    if (Math.abs(M[pivot][i]) < 1e-12) return 0;
    if (pivot !== i) {
      [M[i], M[pivot]] = [M[pivot], M[i]];
      det *= -1;
    }
    det *= M[i][i];
    const piv = M[i][i];
    for (let r = i + 1; r < n; r++) {
      const f = M[r][i] / piv;
      for (let c = i; c < n; c++) M[r][c] -= f * M[i][c];
    }
  }
  return det;
}

function matTrace(A: number[][]): number {
  return A.reduce((s, row, i) => s + row[i], 0);
}

// --- Parsing Helpers ---

function splitLine(line: string): string[] {
  return line.split(/,|\t|;/).map((s) => s.trim());
}

function parseCellToken(token: string, isDiag: boolean): Cell {
  const t = String(token ?? "").trim();
  if (!t) return { r: Number.NaN, n: Number.NaN };

  let m = t.match(/^([+-]?\d*\.?\d+)\s*\|\s*([+-]?\d*\.?\d+)\s*$/);
  if (m) {
    const r = Number(m[1]);
    const n = Number(m[2]);
    return { r, n: isDiag ? Number.NaN : n };
  }
  m = t.match(/^([+-]?\d*\.?\d+)\s*[\(\[\{]\s*([+-]?\d*\.?\d+)\s*[\)\]\}]\s*$/);
  if (m) {
    const r = Number(m[1]);
    const n = Number(m[2]);
    return { r, n: isDiag ? Number.NaN : n };
  }
  m = t.match(/^([+-]?\d*\.?\d+)\s*$/);
  if (m) {
    const r = Number(m[1]);
    return { r, n: isDiag ? Number.NaN : Number.NaN };
  }

  return { r: Number.NaN, n: Number.NaN };
}

function parseCombinedMatrixText(text: string): { vars: VarName[]; cellMatrix: CellMatrix } {
  const raw = (text || "").trim();
  if (!raw) throw new Error("Matrix text is empty.");

  const lines = raw
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean);

  if (lines.length < 2) throw new Error("Matrix needs at least 2 rows.");

  const rows = lines.map(splitLine);
  const header = rows[0];
  if (header.length < 2) throw new Error('Header row must include variable names (e.g., ",A,B,C").');

  const vars = header.slice(1);
  if (vars.some((v) => !v)) throw new Error("Header contains empty variable name.");
  if (new Set(vars).size !== vars.length) throw new Error("Duplicate variable names in header.");

  const cellMatrix: CellMatrix = {};

  for (let i = 1; i < rows.length; i++) {
    const rowName = rows[i][0];
    if (!rowName) throw new Error(`Row ${i + 1} has empty variable name.`);
    cellMatrix[rowName] = {};

    for (let j = 1; j < header.length; j++) {
      const colName = vars[j - 1];
      const token = rows[i][j] ?? "";
      const isDiag = rowName === colName;
      const cell = parseCellToken(token, isDiag);
      cellMatrix[rowName][colName] = cell;
    }
  }

  for (const r of vars) {
    if (!cellMatrix[r]) throw new Error(`Missing row for "${r}". Row names must match header.`);
    for (const c of vars) {
      if (!cellMatrix[r][c]) throw new Error(`Missing cell (${r}, ${c}).`);
    }
  }

  for (const v of vars) {
    cellMatrix[v][v] = { r: 1, n: Number.NaN };
  }

  return { vars, cellMatrix };
}

function makeEmptyCellMatrix(vars: VarName[]): CellMatrix {
  const M: CellMatrix = {};
  for (const r of vars) {
    M[r] = {};
    for (const c of vars) {
      if (r === c) M[r][c] = { r: 1, n: Number.NaN };
      else M[r][c] = { r: Number.NaN, n: Number.NaN };
    }
  }
  return M;
}

function carryOverCellMatrix(oldVars: VarName[], oldM: CellMatrix, newVars: VarName[], newM: CellMatrix): CellMatrix {
  const ov = new Set(oldVars);
  for (const r of newVars) {
    if (!ov.has(r)) continue;
    for (const c of newVars) {
      if (!ov.has(c)) continue;
      const cell = oldM?.[r]?.[c];
      if (cell) newM[r][c] = { r: cell.r, n: cell.n };
    }
  }
  for (const v of newVars) newM[v][v] = { r: 1, n: Number.NaN };
  return newM;
}

function validateCellMatrix(vars: VarName[], M: CellMatrix): { ok: boolean; errors: string[]; warnings: string[] } {
  const errors: string[] = [];
  const warnings: string[] = [];

  for (const r of vars) {
    if (!M[r]) errors.push(`Missing row "${r}".`);
  }
  if (errors.length) return { ok: false, errors, warnings };

  for (const r of vars) {
    for (const c of vars) {
      const cell = M[r]?.[c];
      if (!cell) {
        errors.push(`Missing cell (${r}, ${c}).`);
        continue;
      }

      const isDiag = r === c;
      const rr = cell.r;

      if (!Number.isFinite(rr)) {
        errors.push(`Invalid correlation r at (${r}, ${c}).`);
      } else {
        if (rr < -1 || rr > 1) errors.push(`Correlation r out of bounds [-1,1] at (${r}, ${c}): ${rr}.`);
        if (isDiag && Math.abs(rr - 1) > 1e-6) errors.push(`Diagonal must be 1. Found (${r},${c})=${rr}.`);
      }

      if (!isDiag) {
        const nn = cell.n;
        if (!Number.isFinite(nn)) {
          errors.push(`Missing/invalid sample size n at (${r}, ${c}). Use r|n or r(n).`);
        } else if (nn <= 2) {
          errors.push(`Sample size n must be > 2 at (${r}, ${c}). Found n=${nn}.`);
        }
      }
    }
  }

  for (let i = 0; i < vars.length; i++) {
    for (let j = i + 1; j < vars.length; j++) {
      const a = vars[i],
        b = vars[j];
      const c1 = M[a][b],
        c2 = M[b][a];

      if (Number.isFinite(c1?.r) && Number.isFinite(c2?.r) && Math.abs(c1.r - c2.r) > 1e-6) {
        errors.push(`Matrix not symmetric in r: r(${a},${b}) != r(${b},${a}).`);
      }
      if (Number.isFinite(c1?.n) && Number.isFinite(c2?.n) && Math.abs(c1.n - c2.n) > 1e-6) {
        warnings.push(`n not symmetric: n(${a},${b}) != n(${b},${a}). Consider making them equal.`);
      }
      const rr = M[a][b].r;
      if (Number.isFinite(rr) && Math.abs(rr) >= 0.999999) {
        warnings.push(`Very high |r|≈1 between ${a} and ${b}. This can make estimation unstable (singular).`);
      }
    }
  }

  return { ok: errors.length === 0, errors, warnings };
}

function buildRMatrix(vars: VarName[], M: CellMatrix): number[][] {
  return vars.map((r) => vars.map((c) => M[r][c].r));
}

function symN(M: CellMatrix, a: VarName, b: VarName): number {
  const n1 = M[a]?.[b]?.n;
  const n2 = M[b]?.[a]?.n;
  if (Number.isFinite(n1) && Number.isFinite(n2)) return Math.min(n1, n2);
  if (Number.isFinite(n1)) return n1;
  if (Number.isFinite(n2)) return n2;
  return Number.NaN;
}

function computeTotalN(vars: VarName[], M: CellMatrix, method: "harmonic" | "min"): number {
  const ns: number[] = [];
  for (let i = 0; i < vars.length; i++) {
    for (let j = i + 1; j < vars.length; j++) {
      const n = symN(M, vars[i], vars[j]);
      if (Number.isFinite(n)) ns.push(n);
    }
  }
  if (ns.length === 0) return Number.NaN;
  if (method === "min") return Math.min(...ns);

  let denom = 0;
  for (const n of ns) denom += 1 / n;
  return ns.length / denom;
}

// --- Stats Estimation Core ---

function parentsOf(node: VarName, edges: Edge[]): VarName[] {
  return edges.filter((e) => e.to === node).map((e) => e.from);
}

function estimatePathsFromCorrelation(vars: VarName[], cellM: CellMatrix, edges: Edge[], totalN: number): {
  coeffs: Coef[];
  r2: Record<VarName, number>;
  resid: Record<VarName, number>;
} {
  const coeffs: Coef[] = [];
  const r2: Record<VarName, number> = {};
  const resid: Record<VarName, number> = {};
  const endogenous = new Set(edges.map((e) => e.to));

  for (const y of vars) {
    const X = parentsOf(y, edges);
    if (X.length === 0) continue;

    const Rxx = X.map((a) => X.map((b) => cellM[a][b].r));
    const rXy = X.map((a) => cellM[a][y].r);

    const inv = matInverse(Rxx);
    const beta = matVecMul(inv, rXy);

    // Calculate R2
    const R2 = vecDot(rXy, beta);
    r2[y] = R2;
    resid[y] = Math.max(1e-8, 1 - R2);

    const k = X.length;
    const df = totalN - k - 1;
    const errorVar = (1 - R2) / df;

    X.forEach((x, i) => {
      const vif = inv[i][i];
      const se = Math.sqrt(errorVar * vif);
      const t = Math.abs(beta[i] / se);
      const p = getNormalPValue(t);

      coeffs.push({ 
        from: x, 
        to: y, 
        beta: beta[i],
        se,
        t,
        p,
        vif
      });
    });
  }

  for (const v of vars) {
    if (endogenous.has(v) && !(v in resid)) resid[v] = 1;
  }

  return { coeffs, r2, resid };
}

function impliedSigmaRecursive(
  vars: VarName[],
  S: number[][],
  edges: Edge[],
  coeffs: Coef[],
  residVar: Record<VarName, number>
): number[][] {
  const p = vars.length;
  const idx: Record<VarName, number> = Object.fromEntries(vars.map((v, i) => [v, i]));
  const endogenous = new Set(edges.map((e) => e.to));

  const B = Array.from({ length: p }, () => Array(p).fill(0));
  for (const e of coeffs) {
    B[idx[e.to]][idx[e.from]] = e.beta;
  }

  const exoIdx = vars.map((v, i) => (!endogenous.has(v) ? i : -1)).filter((i) => i >= 0);
  const Psi = Array.from({ length: p }, () => Array(p).fill(0));

  for (let a = 0; a < exoIdx.length; a++) {
    for (let b = 0; b < exoIdx.length; b++) {
      const i = exoIdx[a],
        j = exoIdx[b];
      Psi[i][j] = S[i][j];
    }
  }

  for (const v of vars) {
    const i = idx[v];
    if (endogenous.has(v)) Psi[i][i] = Math.max(1e-8, residVar[v] ?? 1);
  }

  const I = matIdentity(p);
  const IminusB = I.map((row, i) => row.map((x, j) => x - B[i][j]));
  const inv = matInverse(IminusB);
  const invT = transpose(inv);

  return matMul(matMul(inv, Psi), invT);
}

function srmrOffDiag(S: number[][], Sigma: number[][]): number {
  const p = S.length;
  let sum = 0,
    k = 0;
  for (let i = 0; i < p; i++) {
    for (let j = i + 1; j < p; j++) {
      const d = S[i][j] - Sigma[i][j];
      sum += d * d;
      k++;
    }
  }
  return Math.sqrt(sum / Math.max(1, k));
}

function fitML(S: number[][], Sigma: number[][], N: number): { chi2: number; chi2_0: number } {
  const p = S.length;
  const detS = matDet(S);
  const detSig = matDet(Sigma);
  if (detS <= 0 || detSig <= 0) {
    throw new Error("Observed/implied matrix not positive definite (det<=0). Check correlations or model constraints.");
  }

  const SigInv = matInverse(Sigma);
  const tr = matTrace(matMul(S, SigInv));
  const Fml = Math.log(detSig) + tr - Math.log(detS) - p;
  const chi2 = (N - 1) * Fml;

  const Sigma0 = Array.from({ length: p }, (_, i) =>
    Array.from({ length: p }, (_, j) => (i === j ? S[i][i] : 0))
  );
  const det0 = matDet(Sigma0);
  const inv0 = matInverse(Sigma0);
  const tr0 = matTrace(matMul(S, inv0));
  const F0 = Math.log(det0) + tr0 - Math.log(detS) - p;
  const chi2_0 = (N - 1) * F0;

  return { chi2, chi2_0 };
}

function computeFitIndices(args: { chi2: number; df: number; chi2_0: number; df0: number; N: number }) {
  const { chi2, df, chi2_0, df0, N } = args;
  const cfi = 1 - Math.max(0, chi2 - df) / Math.max(1e-12, chi2_0 - df0);
  const tli = 1 - (chi2 / Math.max(1e-12, df) - 1) / (chi2_0 / Math.max(1e-12, df0) - 1);
  const rmsea = Math.sqrt(Math.max(0, (chi2 - df) / (Math.max(1e-12, df) * (N - 1))));
  return { cfi, tli, rmsea };
}

function countDF(vars: VarName[], edges: Edge[]) {
  const observedMoments = (vars.length * (vars.length + 1)) / 2;
  const endogenous = new Set(edges.map((e) => e.to));
  const endoCount = vars.filter((v) => endogenous.has(v)).length;
  const freeParams = edges.length + endoCount;
  const df = Math.max(0, observedMoments - freeParams);
  const df0 = (vars.length * (vars.length - 1)) / 2;
  return { df, df0, observedMoments, freeParams, endoCount };
}

// --- UI Components & Helpers ---

function generateDefaultMatrixText(currentVars: VarName[]) {
  const header = ["", ...currentVars].join(",");
  const lines = [header];
  const allPreset = SAMPLE_PRESETS["All"].pair;

  for (let i = 0; i < currentVars.length; i++) {
    const rowName = currentVars[i];
    const row = [rowName];
    for (let j = 0; j < currentVars.length; j++) {
      const colName = currentVars[j];
      if (i === j) {
        row.push("1");
      } else if (i > j) {
        const k = pairKey(rowName, colName);
        const val = allPreset[k];
        if (val) {
          row.push(`${val.r}|${val.n}`);
        } else {
          row.push("");
        }
      } else {
        row.push("");
      }
    }
    lines.push(row.join(","));
  }
  return lines.join("\n");
}

function ErrorBox({
  title,
  items,
  tone,
}: {
  title: string;
  items: string[];
  tone: "error" | "warn";
}) {
  if (!items || items.length === 0) return null;
  const colors =
    tone === "error"
      ? "bg-rose-50 border-rose-200 text-rose-900"
      : "bg-amber-50 border-amber-200 text-amber-900";
  return (
    <div className={`rounded-xl border p-4 text-sm ${colors} mb-4`}>
      <div className="font-semibold mb-2 flex items-center gap-2">
         {tone === 'error' ? <AlertCircle size={16}/> : <AlertCircle size={16}/>}
         {title}
      </div>
      <ul className="list-disc pl-5 space-y-1">
        {items.map((x, i) => (
          <li key={i}>{x}</li>
        ))}
      </ul>
    </div>
  );
}

function initialPositions(vars: VarName[], width: number, height: number): NodePos {
  const cx = width / 2;
  const cy = height / 2;
  const rad = Math.min(width, height) * 0.35;
  const pos: NodePos = {};
  vars.forEach((v, i) => {
    const ang = (2 * Math.PI * i) / Math.max(1, vars.length) - Math.PI/2;
    pos[v] = { x: cx + rad * Math.cos(ang), y: cy + rad * Math.sin(ang) };
  });
  return pos;
}

function buildPresetBaseMatrix(sample: SampleType): CellMatrix {
  const M = makeEmptyCellMatrix(BASE_VARS);
  const preset = SAMPLE_PRESETS[sample];

  for (let i = 0; i < BASE_VARS.length; i++) {
    for (let j = i + 1; j < BASE_VARS.length; j++) {
      const a = BASE_VARS[i];
      const b = BASE_VARS[j];
      const k = pairKey(a, b);
      const v = preset.pair[k];
      if (v) {
        M[a][b] = { r: v.r, n: v.n };
        M[b][a] = { r: v.r, n: v.n };
      }
    }
  }
  for (const v of BASE_VARS) M[v][v] = { r: 1, n: Number.NaN };
  return M;
}

// --- Main Component ---

export default function PathModelBuilder() {
  const [step, setStep] = useState<number>(0); 
  
  const [sampleType, setSampleType] = useState<SampleType>("All");
  const [customVars, setCustomVars] = useState<VarName[]>([]);
  const [pendingCount, setPendingCount] = useState<number>(0);
  const [pendingNames, setPendingNames] = useState<string[]>([]);
  const vars = useMemo<VarName[]>(() => [...BASE_VARS, ...customVars], [customVars]);

  const [cellM, setCellM] = useState<CellMatrix>(() => {
    const baseM = buildPresetBaseMatrix("All");
    return carryOverCellMatrix(BASE_VARS, baseM, BASE_VARS, baseM);
  });
  const [matrixText, setMatrixText] = useState<string>(() => generateDefaultMatrixText([...BASE_VARS]));
  const [inputMode, setInputMode] = useState<"text" | "grid">("grid");
  const [matrixErrors, setMatrixErrors] = useState<string[]>([]);
  const [matrixWarnings, setMatrixWarnings] = useState<string[]>([]);

  const [edges, setEdges] = useState<Edge[]>([]);
  const [nodePos, setNodePos] = useState<NodePos>({});
  const [connectFrom, setConnectFrom] = useState<VarName | null>(null);

  const [nMethod, setNMethod] = useState<"harmonic" | "min">("harmonic");
  const [lastEst, setLastEst] = useState<EstResult | null>(null);
  const [estError, setEstError] = useState<string | null>(null);

  useEffect(() => {
    setCellM((prev) => {
      const next = deepClone(prev);
      const base = buildPresetBaseMatrix(sampleType);
      for (const a of BASE_VARS) {
        for (const b of BASE_VARS) {
          next[a] ??= {};
          next[b] ??= {};
          next[a][b] = deepClone(base[a][b]);
        }
      }
      for (const v of vars) {
        next[v] ??= {};
        next[v][v] = { r: 1, n: Number.NaN };
      }
      return next;
    });
    setLastEst(null);
  }, [sampleType]);

  useEffect(() => {
    const width = 800;
    const height = 450;
    setNodePos((prev) => {
      const next: NodePos = { ...(prev || {}) };
      let missing = false;
      for (const v of vars) if (!next[v]) missing = true;
      if (!missing) return next;

      const init = initialPositions(vars, width, height);
      for (const v of vars) if (!next[v]) next[v] = init[v];
      return next;
    });
  }, [vars.join("|"), step]);

  useEffect(() => { setLastEst(null); }, [edges]);

  const setCell = (r: VarName, c: VarName, patch: Partial<Cell>) => {
    setCellM((prev) => {
      const next = deepClone(prev);
      const old = next[r]?.[c] ?? { r: Number.NaN, n: Number.NaN };
      if (!next[r]) next[r] = {};
      next[r][c] = { ...old, ...patch };

      if (r !== c) {
        if (!next[c]) next[c] = {};
        const old2 = next[c]?.[r] ?? { r: Number.NaN, n: Number.NaN };
        next[c][r] = { ...old2, ...patch };
      } else {
        next[r][c] = { r: 1, n: Number.NaN };
      }
      return next;
    });
    setLastEst(null);
  };

  const validateMatrixState = () => {
    const v = validateCellMatrix(vars, cellM);
    setMatrixErrors(v.errors);
    setMatrixWarnings(v.warnings);
    return v.ok;
  };

  const applyVars = () => {
    const names = pendingNames.map((s) => s.trim()).filter(Boolean);
    if (names.length !== pendingCount) {
      alert(`You specified ${pendingCount} new variables, but provided ${names.length} valid names.`);
      return;
    }
    const all = [...BASE_VARS, ...names];
    const dup = all.find((x, i) => all.indexOf(x) !== i);
    if (dup) {
      alert(`Duplicate variable name detected: "${dup}".`);
      return;
    }
    const newVars = [...BASE_VARS, ...names];
    const newM = makeEmptyCellMatrix(newVars);
    carryOverCellMatrix(vars, cellM, newVars, newM);
    const basePreset = buildPresetBaseMatrix(sampleType);
    for (const a of BASE_VARS) for (const b of BASE_VARS) newM[a][b] = deepClone(basePreset[a][b]);
    setCustomVars(names);
    setCellM(newM);
    setEdges([]);
    setLastEst(null);
    setMatrixText(generateDefaultMatrixText(newVars)); 
    setMatrixErrors([]);
  };

  const runEstimation = () => {
    const v = validateCellMatrix(vars, cellM);
    const errors: string[] = [];
    const warnings: string[] = [...v.warnings];
    setEstError(null);

    if (!edges.length) errors.push("No paths defined. Create at least one directed edge.");
    if (!v.ok) errors.push(...v.errors);

    const N = v.ok ? computeTotalN(vars, cellM, nMethod) : Number.NaN;
    if (v.ok && (!Number.isFinite(N) || N <= 2)) errors.push("Total N is invalid (check pairwise n values).");

    if (errors.length) {
      setMatrixErrors(errors);
      setMatrixWarnings(warnings);
      setLastEst(null);
      return false;
    }

    try {
      const S = buildRMatrix(vars, cellM);
      const { coeffs, r2, resid } = estimatePathsFromCorrelation(vars, cellM, edges, N);
      const Sigma = impliedSigmaRecursive(vars, S, edges, coeffs, resid);
      const srmr = srmrOffDiag(S, Sigma);
      const { df, df0, observedMoments, freeParams, endoCount } = countDF(vars, edges);

      const fit: Fit = {
        totalN: N,
        N_method: nMethod,
        SRMR: srmr,
        df,
        observedMoments,
        freeParams,
        endogenousCount: endoCount,
      };

      if (df > 0) {
        const { chi2, chi2_0 } = fitML(S, Sigma, N);
        const { cfi, tli, rmsea } = computeFitIndices({ chi2, df, chi2_0, df0, N });
        fit.chi2 = chi2;
        fit.rmsea = rmsea;
        fit.cfi = cfi;
        fit.tli = tli;
      } else {
        warnings.push("df = 0. Model is saturated. Fit indices (RMSEA/CFI) are not computed.");
      }

      setMatrixWarnings(warnings);
      setLastEst({ coeffs, r2, resid, fit });
      return true;
    } catch (e) {
      setEstError(e instanceof Error ? e.message : String(e));
      setLastEst(null);
      return false;
    }
  };

  const dragRef = useRef<{ active: boolean; varName: VarName | null; dx: number; dy: number }>({
    active: false, varName: null, dx: 0, dy: 0,
  });
  
  // FIX: Corrected function with right variable names
  const onNodeClick = (v: VarName) => {
    if (!connectFrom) { setConnectFrom(v); return; }
    if (connectFrom === v) { setConnectFrom(null); return; }
    
    // Only add edge in step 2
    if (step === 2) {
      // Correct variable names used here
      if (connectFrom === v) return;
      if (edges.some((e) => e.from === connectFrom && e.to === v)) return;
      setEdges([...edges, { from: connectFrom, to: v }]);
    }
    setConnectFrom(null);
  };
  
  const onPointerDownNode = (e: React.PointerEvent<SVGCircleElement>, v: VarName) => {
    if (step !== 2) return;
    e.preventDefault();
    const svg = e.currentTarget.ownerSVGElement;
    if (!svg) return;
    const pt = svg.createSVGPoint();
    pt.x = e.clientX; pt.y = e.clientY;
    const cursor = pt.matrixTransform(svg.getScreenCTM()?.inverse());
    const p = nodePos[v];
    if (!p) return;
    dragRef.current = { active: true, varName: v, dx: p.x - cursor.x, dy: p.y - cursor.y };
  };
  
  const onPointerMoveSvg = (e: React.PointerEvent<SVGSVGElement>) => {
    if (!dragRef.current.active || !dragRef.current.varName) return;
    const svg = e.currentTarget;
    const pt = svg.createSVGPoint();
    pt.x = e.clientX; pt.y = e.clientY;
    const cursor = pt.matrixTransform(svg.getScreenCTM()?.inverse());
    const nx = cursor.x + dragRef.current.dx;
    const ny = cursor.y + dragRef.current.dy;
    setNodePos((prev) => ({
      ...prev,
      [dragRef.current.varName!]: {
        x: Math.max(50, Math.min(800 - 50, nx)),
        y: Math.max(50, Math.min(450 - 50, ny)),
      },
    }));
  };
  
  const Steps = ["Setup Variables", "Input Data", "Build Model", "Analysis Results"];
  
  const goNext = () => {
    if (step === 0) {
      applyVars();
      setStep(1);
    } else if (step === 1) {
       if (validateMatrixState()) setStep(2);
    } else if (step === 2) {
       if (runEstimation()) setStep(3);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 font-sans text-slate-800 pb-20">
      
      <div className="bg-white border-b border-slate-200 sticky top-0 z-40 shadow-sm">
        <div className="max-w-6xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Share2 className="text-indigo-600" size={24} />
            <div>
               <h1 className="font-bold text-lg leading-tight">Extended MASEM Analysis from SO, YANG AND LI (2025)</h1>
               <p className="text-[10px] text-slate-500 font-medium">Programmed by Dr. Yang Yang, Temple University.</p>
            </div>
          </div>
          
          <div className="hidden md:flex items-center gap-1">
             {Steps.map((label, i) => (
                <div key={i} className="flex items-center">
                   <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-semibold transition-colors
                      ${step === i ? 'bg-indigo-50 text-indigo-700 border border-indigo-200' : 
                        step > i ? 'text-indigo-600' : 'text-slate-400'}`}>
                      <div className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px]
                         ${step === i ? 'bg-indigo-600 text-white' : 
                           step > i ? 'bg-indigo-100 text-indigo-700' : 'bg-slate-100 text-slate-500'}`}>
                         {i + 1}
                      </div>
                      {label}
                   </div>
                   {i < Steps.length - 1 && <div className="w-4 h-[1px] bg-slate-200 mx-1"></div>}
                </div>
             ))}
          </div>

          <div className="flex items-center gap-3">
		<a
  		href="./MASEM_User_Manual.pdf"
  		download
  		className="inline-flex items-center gap-2 rounded-xl bg-slate-800 border border-slate-700 px-4 py-2 text-sm font-bold text-white shadow-md transition-all hover:bg-slate-700 hover:-translate-y-0.5 hover:shadow-lg"
  		title="Download the user manual PDF"
		>
  		<Download size={14} />
  		Download User Manual
		</a>
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-4 py-8">
        
        <div className="bg-white rounded-2xl shadow-sm border border-slate-200 min-h-[500px] p-6 md:p-8">
          
          {step === 0 && (
            <div className="max-w-3xl mx-auto animate-in fade-in slide-in-from-bottom-4 duration-500">
               <div className="text-center mb-8">
                  <div className="inline-flex items-center justify-center w-12 h-12 rounded-xl bg-indigo-100 text-indigo-600 mb-4">
                     <ListPlus size={24}/>
                  </div>
                  <h2 className="text-2xl font-bold mb-2">Configure Analysis Variables</h2>
                  <p className="text-slate-500">Select your base dataset and add any custom variables needed for your MASEM model.</p>
               </div>

               <div className="grid md:grid-cols-2 gap-8">
                 <div className="space-y-4">
                    <label className="block text-sm font-semibold text-slate-700">Meta-analytical Dataset Preset</label>
                    <div className="relative">
                      <select 
                        className="w-full appearance-none bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 pr-8 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                        value={sampleType}
                        onChange={(e) => setSampleType(e.target.value as SampleType)}
                      >
                         <option value="All">All Samples</option>
                         <option value="Lodging">Lodging Industry</option>
                         <option value="Restaurant">Restaurant Industry</option>
                         <option value="Tourism and travel">Tourism & Travel</option>
                      </select>
                      <div className="absolute right-3 top-3.5 text-slate-400 pointer-events-none"><ChevronRight size={16} className="rotate-90"/></div>
                    </div>
                 </div>

                 <div className="space-y-4">
                    <div className="flex justify-between items-center">
                       <label className="block text-sm font-semibold text-slate-700">Custom Variables</label>
                       <span className="text-xs bg-slate-100 text-slate-600 px-2 py-1 rounded-full">{pendingCount} added</span>
                    </div>
                    
                    <div className="flex gap-4">
                       <div className="w-24">
                          <input type="number" min="0" max="10" className="w-full border border-slate-200 rounded-xl px-3 py-2 text-center" 
                            value={pendingCount}
                            onChange={(e) => {
                                const v = Math.max(0, Math.min(10, Number(e.target.value)));
                                setPendingCount(v);
                                setPendingNames(prev => {
                                   const arr = [...prev];
                                   while(arr.length < v) arr.push("");
                                   while(arr.length > v) arr.pop();
                                   return arr;
                                });
                            }}
                          />
                          <span className="text-[10px] text-slate-400 block text-center mt-1">Count</span>
                       </div>
                       <div className="flex-1">
                          <textarea 
                             className="w-full border border-slate-200 rounded-xl px-3 py-2 text-sm font-mono h-32 resize-none focus:ring-2 focus:ring-indigo-500 focus:outline-none"
                             placeholder={`Enter names...\none per line`}
                             value={pendingNames.join("\n")}
                             onChange={(e) => {
                                const lines = e.target.value.split("\n");
                                setPendingNames(lines.slice(0, pendingCount));
                             }}
                             disabled={pendingCount === 0}
                          />
                       </div>
                    </div>
                 </div>
               </div>
            </div>
          )}

          {step === 1 && (
             <div className="h-full flex flex-col animate-in fade-in slide-in-from-right-4 duration-500">
                <div className="flex items-center justify-between mb-4">
                   <div>
                     <h2 className="text-xl font-bold flex items-center gap-2"><TableIcon className="text-indigo-600"/> Correlation Matrix</h2>
                     <p className="text-sm text-slate-500">Input Pearson's <i>r</i> and pairwise sample size <i>N</i> after meta analysis on correlation effect size.</p>
                   </div>
                   <div className="flex bg-slate-100 p-1 rounded-lg">
                      <button onClick={() => setInputMode("grid")} className={`px-3 py-1 rounded-md text-xs font-medium transition-all ${inputMode==='grid'?'bg-white shadow-sm text-indigo-700':'text-slate-500'}`}>Grid Input</button>
                      <button onClick={() => setInputMode("text")} className={`px-3 py-1 rounded-md text-xs font-medium transition-all ${inputMode==='text'?'bg-white shadow-sm text-indigo-700':'text-slate-500'}`}>Text Paste</button>
                   </div>
                </div>

                <ErrorBox title="Validation Issues" items={matrixErrors} tone="error" />
                <ErrorBox title="Warnings" items={matrixWarnings} tone="warn" />

                <div className="flex-1 border border-slate-200 rounded-xl overflow-hidden bg-slate-50 relative">
                   {inputMode === 'grid' ? (
                      <div className="overflow-auto h-[400px]">
                        <table className="w-full text-xs border-collapse">
                           <thead className="bg-white sticky top-0 z-10 shadow-sm">
                              <tr>
                                 <th className="p-3 text-left bg-slate-50 border-b border-r sticky left-0 z-20">Variables</th>
                                 {vars.map(v => <th key={v} className="p-3 text-left font-medium text-slate-600 border-b min-w-[140px]">{v}</th>)}
                              </tr>
                           </thead>
                           <tbody>
                              {vars.map(r => (
                                 <tr key={r} className="border-b border-slate-100 last:border-0 hover:bg-white transition-colors">
                                    <td className="p-3 font-semibold text-slate-700 bg-slate-50 border-r sticky left-0 z-10">{r}</td>
                                    {vars.map(c => {
                                       const isDiag = r === c;
                                       const cell = cellM?.[r]?.[c] ?? {r:NaN, n:NaN};
                                       return (
                                          <td key={c} className="p-2">
                                             {isDiag ? (
                                                <div className="w-full h-full p-2 bg-slate-100 rounded text-center text-slate-400">1.00</div>
                                             ) : (
                                                <div className={`p-2 rounded border ${Number.isFinite(cell.r) ? 'bg-white border-indigo-100 shadow-sm' : 'bg-rose-50 border-rose-100'}`}>
                                                   <div className="flex items-center gap-2 mb-1">
                                                      <span className="text-[10px] font-bold text-slate-400 w-3">r</span>
                                                      <input type="number" step="0.01" className="w-full bg-transparent border-b border-slate-200 focus:border-indigo-500 outline-none text-slate-700 font-medium"
                                                         value={Number.isFinite(cell.r) ? cell.r : ''}
                                                         onChange={e => setCell(r,c,{r: parseFloat(e.target.value)})}
                                                         placeholder="-"
                                                      />
                                                   </div>
                                                   <div className="flex items-center gap-2">
                                                      <span className="text-[10px] font-bold text-slate-400 w-3">N</span>
                                                      <input type="number" step="1" className="w-full bg-transparent border-b border-slate-200 focus:border-indigo-500 outline-none text-slate-600"
                                                         value={Number.isFinite(cell.n) ? cell.n : ''}
                                                         onChange={e => setCell(r,c,{n: parseFloat(e.target.value)})}
                                                         placeholder="-"
                                                      />
                                                   </div>
                                                </div>
                                             )}
                                          </td>
                                       )
                                    })}
                                 </tr>
                              ))}
                           </tbody>
                        </table>
                      </div>
                   ) : (
                      <div className="p-4 h-full flex flex-col">
                         <div className="text-xs text-slate-500 mb-2">
                            Paste matrix (lower triangular entries automatically filled for base variables).
                         </div>
                         <textarea 
                           className="flex-1 w-full border border-slate-300 rounded-lg p-3 font-mono text-xs resize-none"
                           value={matrixText}
                           onChange={e => setMatrixText(e.target.value)}
                           placeholder="Paste matrix here..."
                         />
                         <div className="mt-3 flex gap-2">
                           <button onClick={() => {
                              try {
                                 const p = parseCombinedMatrixText(matrixText);
                                 setCellM(p.cellMatrix);
                              } catch(e) { setMatrixErrors([(e as Error).message]) }
                           }} className="bg-slate-800 text-white px-4 py-2 rounded-lg text-sm">Load Text</button>
                           <button onClick={() => setMatrixText(generateDefaultMatrixText(vars))} className="text-slate-600 px-4 py-2 text-sm">Reset Template</button>
                         </div>
                      </div>
                   )}
                </div>
             </div>
          )}

          {step === 2 && (
             <div className="h-full flex flex-col animate-in fade-in slide-in-from-right-4 duration-500">
                <div className="flex items-center justify-between mb-4">
                  <div>
                     <h2 className="text-xl font-bold flex items-center gap-2"><Network className="text-indigo-600"/> Path Diagram</h2>
                     <p className="text-sm text-slate-500">Click a node to start a connection. Click another to finish. Drag nodes to rearrange.</p>
                  </div>
                  <div className="flex items-center gap-2">
                     <div className="text-xs font-medium px-3 py-1 bg-slate-100 rounded-lg text-slate-600">
                        Sample size method: 
                        <select value={nMethod} onChange={e => setNMethod(e.target.value as any)} className="bg-transparent ml-1 outline-none font-bold text-indigo-700">
                           <option value="harmonic">Harmonic Mean N</option>
                           <option value="min">Minimum N</option>
                        </select>
                     </div>
                     <button onClick={() => setEdges([])} className="text-xs text-rose-600 font-semibold px-3 py-1 border border-rose-200 rounded-lg hover:bg-rose-50">Clear Edges</button>
                  </div>
                </div>
                
                <div className="flex-1 bg-slate-50 border border-slate-200 rounded-xl relative overflow-hidden select-none">
                  {connectFrom && (
                     <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-indigo-600 text-white text-xs px-4 py-2 rounded-full shadow-lg z-50 animate-bounce">
                        Select target for <b>{connectFrom}</b>
                     </div>
                  )}
                  
                  <svg
                     viewBox="0 0 800 450"
                     className="w-full h-full cursor-crosshair"
                     onPointerMove={onPointerMoveSvg}
                     onPointerUp={() => dragRef.current.active = false}
                     onPointerLeave={() => dragRef.current.active = false}
                  >
                     <defs>
                        <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="28" refY="3.5" orient="auto">
                           <polygon points="0 0, 10 3.5, 0 7" fill="#64748b" />
                        </marker>
                     </defs>

                     {edges.map((e, idx) => {
                        const p1 = nodePos[e.from], p2 = nodePos[e.to];
                        if(!p1 || !p2) return null;
                        const mx = (p1.x + p2.x)/2, my = (p1.y + p2.y)/2;
                        const dx = p2.x - p1.x, dy = p2.y - p1.y;
                        const norm = Math.sqrt(dx*dx + dy*dy) || 1;
                        const off = 20; 
                        const cx = mx - (dy/norm)*off, cy = my + (dx/norm)*off;
                        const d = `M ${p1.x} ${p1.y} Q ${cx} ${cy} ${p2.x} ${p2.y}`;
                        
                        return (
                           <g key={idx} onClick={() => setEdges(prev => prev.filter((_,i) => i !== idx))} className="group cursor-pointer">
                              <path d={d} stroke="#94a3b8" strokeWidth="2" fill="none" markerEnd="url(#arrowhead)" className="group-hover:stroke-rose-400 transition-colors"/>
                              <path d={d} stroke="transparent" strokeWidth="15" fill="none" /> 
                           </g>
                        )
                     })}

                     {vars.map(v => {
                        const p = nodePos[v];
                        if(!p) return null;
                        const isSelected = connectFrom === v;
                        return (
                           <g key={v} transform={`translate(${p.x},${p.y})`}>
                              <circle 
                                 r="40" 
                                 className={`transition-all duration-300 ${isSelected ? 'fill-indigo-100 stroke-indigo-500 stroke-[3px]' : 'fill-white stroke-slate-300 stroke-2 hover:stroke-indigo-400'}`}
                                 onPointerDown={e => onPointerDownNode(e, v)}
                                 onClick={(e) => { e.stopPropagation(); onNodeClick(v); }}
                                 style={{ cursor: 'grab' }}
                              />
                              <text dy="4" textAnchor="middle" className={`text-[11px] font-bold select-none pointer-events-none ${isSelected ? 'fill-indigo-700' : 'fill-slate-700'}`}>
                                 {v}
                              </text>
                           </g>
                        )
                     })}
                  </svg>
                </div>
             </div>
          )}

          {step === 3 && lastEst && (
             <div className="animate-in fade-in slide-in-from-right-4 duration-500">
                <div className="flex items-center gap-3 mb-6">
                   <div className="p-3 bg-emerald-100 text-emerald-700 rounded-xl"><Calculator size={24}/></div>
                   <div>
                      <h2 className="text-2xl font-bold text-slate-800">Estimation Results</h2>
                      <p className="text-sm text-slate-500">Based on Recursive Path Analysis (OLS per equation).</p>
                   </div>
                   <div className="ml-auto flex gap-2">
                       <div className="text-right">
                          <div className="text-xs text-slate-400 uppercase font-bold tracking-wider">Total N</div>
                          <div className="font-mono font-bold text-lg text-slate-700">{lastEst.fit.totalN.toFixed(0)}</div>
                       </div>
                   </div>
                </div>

                <ErrorBox title="Fit Warnings" items={matrixWarnings} tone="warn" />

                <div className="mb-6 bg-white border border-slate-200 rounded-xl relative overflow-hidden h-[450px]">
                   <div className="absolute top-2 left-2 z-10 bg-white/90 px-2 py-1 rounded text-xs text-slate-500 font-medium">
                      Path Diagram Results (β shown on paths, R² above nodes)
                   </div>
                   <svg viewBox="0 0 800 450" className="w-full h-full">
                     <defs>
                        <marker id="arrowhead-res" markerWidth="10" markerHeight="7" refX="28" refY="3.5" orient="auto">
                           <polygon points="0 0, 10 3.5, 0 7" fill="#475569" />
                        </marker>
                     </defs>
                     {lastEst.coeffs.map((c, idx) => {
                        const p1 = nodePos[c.from], p2 = nodePos[c.to];
                        if(!p1 || !p2) return null;
                        
                        const mx = (p1.x + p2.x)/2, my = (p1.y + p2.y)/2;
                        const dx = p2.x - p1.x, dy = p2.y - p1.y;
                        const norm = Math.sqrt(dx*dx + dy*dy) || 1;
                        const off = 20; 
                        const cx = mx - (dy/norm)*off, cy = my + (dx/norm)*off;
                        const d = `M ${p1.x} ${p1.y} Q ${cx} ${cy} ${p2.x} ${p2.y}`;

                        const sig = getSignificanceStar(c.p);
                        
                        return (
                           <g key={idx}>
                              <path d={d} stroke="#475569" strokeWidth="2" fill="none" markerEnd="url(#arrowhead-res)"/>
                              <rect x={cx - 24} y={cy - 10} width="48" height="20" rx="4" fill="white" stroke="#e2e8f0" />
                              <text x={cx} y={cy} dy="4" textAnchor="middle" fontSize="11" fill="#1e293b" fontWeight="bold">
                                 {c.beta.toFixed(2)}{sig}
                              </text>
                           </g>
                        )
                     })}

                     {vars.map(v => {
                        const p = nodePos[v];
                        if(!p) return null;
                        const r2Val = lastEst.r2[v];
                        
                        return (
                           <g key={v} transform={`translate(${p.x},${p.y})`}>
                              <circle 
                                 r="40" 
                                 fill="#f1f5f9"
                                 stroke="#cbd5e1" 
                                 strokeWidth="2"
                              />
                              <text dy="4" textAnchor="middle" className="text-[11px] font-bold fill-slate-700">
                                 {v}
                              </text>
                              {r2Val !== undefined && (
                                 <text dy="-50" textAnchor="middle" className="text-[10px] fill-indigo-600 font-bold">
                                    R² = {r2Val.toFixed(2)}
                                 </text>
                              )}
                           </g>
                        )
                     })}
                   </svg>
                </div>
                
                <div className="grid lg:grid-cols-3 gap-6">
                   <div className="lg:col-span-1 space-y-4">
                      <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm">
                         <h3 className="font-bold text-slate-800 mb-4 flex items-center gap-2"><Settings2 size={16}/> Model Fit Indices</h3>
                         <div className="space-y-3">
                            <div className="flex justify-between items-center pb-2 border-b border-slate-50">
                               <span className="text-sm text-slate-500">SRMR</span>
                               <span className={`font-mono font-bold ${lastEst.fit.SRMR < 0.08 ? 'text-emerald-600' : 'text-amber-600'}`}>{lastEst.fit.SRMR.toFixed(4)}</span>
                            </div>
                            {lastEst.fit.rmsea !== undefined ? (
                               <>
                                 <div className="flex justify-between items-center pb-2 border-b border-slate-50">
                                    <span className="text-sm text-slate-500">RMSEA</span>
                                    <span className={`font-mono font-bold ${lastEst.fit.rmsea < 0.08 ? 'text-emerald-600' : 'text-amber-600'}`}>{lastEst.fit.rmsea.toFixed(4)}</span>
                                 </div>
                                 <div className="flex justify-between items-center pb-2 border-b border-slate-50">
                                    <span className="text-sm text-slate-500">CFI</span>
                                    <span className={`font-mono font-bold ${lastEst.fit.cfi! > 0.9 ? 'text-emerald-600' : 'text-amber-600'}`}>{lastEst.fit.cfi!.toFixed(4)}</span>
                                 </div>
                                 <div className="flex justify-between items-center pb-2 border-b border-slate-50">
                                    <span className="text-sm text-slate-500">TLI</span>
                                    <span className={`font-mono font-bold ${lastEst.fit.tli! > 0.9 ? 'text-emerald-600' : 'text-amber-600'}`}>{lastEst.fit.tli!.toFixed(4)}</span>
                                 </div>
                                 <div className="flex justify-between items-center pt-2">
                                    <span className="text-sm text-slate-500">Chi-Square (df={lastEst.fit.df})</span>
                                    <span className="font-mono font-bold text-slate-700">{lastEst.fit.chi2!.toFixed(2)}</span>
                                 </div>
                               </>
                            ) : (
                               <div className="text-xs text-slate-400 italic">Saturated model (df=0). Perfect fit assumed.</div>
                            )}
                         </div>
                      </div>
                   </div>

                   <div className="lg:col-span-2">
                      <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">
                         <div className="bg-slate-50 px-5 py-3 border-b border-slate-200">
                            <h3 className="font-bold text-slate-700">Path Coefficients</h3>
                         </div>
                         <div className="overflow-x-auto">
                           <table className="w-full text-sm text-left">
                              <thead className="bg-white text-xs uppercase text-slate-500 font-semibold">
                                 <tr>
                                    <th className="px-5 py-3">Path</th>
                                    <th className="px-5 py-3 text-right">Std. β</th>
                                    <th className="px-5 py-3 text-right">S.E.</th>
                                    <th className="px-5 py-3 text-right">t-value</th>
                                    <th className="px-5 py-3 text-right">p-value</th>
                                    <th className="px-5 py-3 text-right">VIF</th>
                                 </tr>
                              </thead>
                              <tbody className="divide-y divide-slate-100">
                                 {lastEst.coeffs.map((c, i) => (
                                    <tr key={i} className="hover:bg-slate-50 transition-colors">
                                       <td className="px-5 py-3 font-medium text-slate-700">
                                          <span className="text-slate-500">{c.from}</span> <span className="text-slate-300">→</span> <span className="text-indigo-600">{c.to}</span>
                                       </td>
                                       <td className="px-5 py-3 text-right font-mono font-bold">{c.beta.toFixed(3)}{getSignificanceStar(c.p)}</td>
                                       <td className="px-5 py-3 text-right font-mono text-slate-500">{c.se.toFixed(3)}</td>
                                       <td className="px-5 py-3 text-right font-mono text-slate-500">{c.t.toFixed(2)}</td>
                                       <td className="px-5 py-3 text-right font-mono">
                                          {c.p < 0.001 ? <span className="text-emerald-600 font-bold">&lt; .001</span> : 
                                           c.p < 0.05 ? <span className="text-emerald-600 font-bold">{c.p.toFixed(3)}</span> : 
                                           <span className="text-slate-400">{c.p.toFixed(3)}</span>}
                                       </td>
                                       <td className="px-5 py-3 text-right font-mono text-xs text-slate-400">{c.vif.toFixed(2)}</td>
                                    </tr>
                                 ))}
                              </tbody>
                           </table>
                         </div>
                      </div>
                   </div>
                </div>
             </div>
          )}
        </div>

        <div className="fixed bottom-0 left-0 right-0 bg-white border-t border-slate-200 p-4 z-50">
           <div className="max-w-6xl mx-auto flex justify-between items-center">
              <button 
                onClick={() => setStep(s => Math.max(0, s-1))}
                disabled={step === 0}
                className="px-6 py-2.5 rounded-xl font-medium text-slate-600 hover:bg-slate-100 disabled:opacity-30 disabled:cursor-not-allowed flex items-center gap-2 transition-colors"
              >
                <ChevronLeft size={18}/> Back
              </button>

              <div className="flex gap-2">
                 {step === 2 && (
                    <button onClick={runEstimation} className="md:hidden bg-indigo-600 text-white px-4 py-2 rounded-xl">Run</button>
                 )}
              </div>

              {step < 3 ? (
                 <button 
                  onClick={goNext}
                  className="px-6 py-2.5 rounded-xl font-bold bg-indigo-600 text-white hover:bg-indigo-700 shadow-lg shadow-indigo-200 flex items-center gap-2 transition-all hover:scale-105 active:scale-95"
                 >
                   {step === 2 ? 'Run Estimation' : 'Next Step'} <ChevronRight size={18}/>
                 </button>
              ) : (
                 <button 
                  onClick={() => setStep(0)}
                  className="px-6 py-2.5 rounded-xl font-bold text-indigo-600 border border-indigo-200 hover:bg-indigo-50 transition-colors"
                 >
                   Start New Analysis
                 </button>
              )}
           </div>
        </div>

      </div>
    </div>
  );
}