import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Share2,
  ListPlus,
  Table as TableIcon,
  Network,
  Calculator,
  CheckCircle2,
  Download,
} from "lucide-react";

type VarName = string;

type Cell = {
  r: number; // correlation
  n: number; // pairwise sample size, NaN for diagonal
};

type CellMatrix = Record<VarName, Record<VarName, Cell>>;

type Edge = { from: VarName; to: VarName };

type Coef = { from: VarName; to: VarName; beta: number };

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

// ------------------------------
// Default pairwise r and N from the provided meta table (k ignored)
// Order: loyalty, satisfaction, value, quality
// ------------------------------
const SAMPLE_PRESETS: Record<
  SampleType,
  {
    pair: Record<string, { r: number; n: number }>;
  }
> = {
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

// ------------------------------
// Parsing: combined matrix text
// Accepted formats per cell:
//  - r|n
//  - r(n) or r[n] or r{n}
// Diagonal can be "1" (n ignored)
// ------------------------------
function splitLine(line: string): string[] {
  return line.split(/,|\t|;/).map((s) => s.trim());
}

function parseCellToken(token: string, isDiag: boolean): Cell {
  const t = String(token ?? "").trim();
  if (!t) return { r: Number.NaN, n: Number.NaN };

  // r|n
  let m = t.match(/^([+-]?\d*\.?\d+)\s*\|\s*([+-]?\d*\.?\d+)\s*$/);
  if (m) {
    const r = Number(m[1]);
    const n = Number(m[2]);
    return { r, n: isDiag ? Number.NaN : n };
  }

  // r(n) or r[n] or r{n}
  m = t.match(/^([+-]?\d*\.?\d+)\s*[\(\[\{]\s*([+-]?\d*\.?\d+)\s*[\)\]\}]\s*$/);
  if (m) {
    const r = Number(m[1]);
    const n = Number(m[2]);
    return { r, n: isDiag ? Number.NaN : n };
  }

  // just r
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

  // Validate square coverage and matching row names
  for (const r of vars) {
    if (!cellMatrix[r]) throw new Error(`Missing row for "${r}". Row names must match header.`);
    for (const c of vars) {
      if (!cellMatrix[r][c]) throw new Error(`Missing cell (${r}, ${c}).`);
    }
  }

  // Force diagonal to 1
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

// ------------------------------
// Small matrix algebra
// ------------------------------
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

// ------------------------------
// SEM-lite estimation
// ------------------------------
function parentsOf(node: VarName, edges: Edge[]): VarName[] {
  return edges.filter((e) => e.to === node).map((e) => e.from);
}

function estimatePathsFromCorrelation(vars: VarName[], cellM: CellMatrix, edges: Edge[]): {
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

    const R2 = vecDot(rXy, beta);
    r2[y] = R2;
    resid[y] = Math.max(1e-8, 1 - R2);

    X.forEach((x, i) => coeffs.push({ from: x, to: y, beta: beta[i] }));
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

  // B matrix
  const B = Array.from({ length: p }, () => Array(p).fill(0));
  for (const e of coeffs) {
    B[idx[e.to]][idx[e.from]] = e.beta;
  }

  // exogenous indices
  const exoIdx = vars.map((v, i) => (!endogenous.has(v) ? i : -1)).filter((i) => i >= 0);

  // Psi
  const Psi = Array.from({ length: p }, () => Array(p).fill(0));

  // exogenous cov fixed to observed
  for (let a = 0; a < exoIdx.length; a++) {
    for (let b = 0; b < exoIdx.length; b++) {
      const i = exoIdx[a],
        j = exoIdx[b];
      Psi[i][j] = S[i][j];
    }
  }

  // endogenous residual variances
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

  // independence model
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

// ------------------------------
// UI helpers
// ------------------------------
function defaultSampleText(vars: VarName[]) {
  const header = ["", ...vars].join(",");
  const lines = [header];
  for (const r of vars) {
    const row = [r];
    for (const c of vars) {
      if (r === c) row.push("1");
      else row.push(""); // expecting r|n
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
    <div className={`rounded-xl border p-4 text-sm ${colors}`}>
      <div className="font-semibold mb-2">{title}</div>
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
  const rad = Math.min(width, height) * 0.33;
  const pos: NodePos = {};
  vars.forEach((v, i) => {
    const ang = (2 * Math.PI * i) / Math.max(1, vars.length);
    pos[v] = { x: cx + rad * Math.cos(ang), y: cy + rad * Math.sin(ang) };
  });
  return pos;
}

function cellBgForR(r: number) {
  if (!Number.isFinite(r)) return "bg-rose-50";
  if (r === 1) return "bg-slate-50";
  if (r > 0) return "bg-emerald-50";
  return "bg-rose-50";
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

// ------------------------------
// Main Component
// ------------------------------
export default function PathModelBuilder() {
  const [sampleType, setSampleType] = useState<SampleType>("All");

  // variables
  const [customVars, setCustomVars] = useState<VarName[]>([]);
  const [pendingCount, setPendingCount] = useState<number>(0);
  const [pendingNames, setPendingNames] = useState<string[]>([]);

  const vars = useMemo<VarName[]>(() => [...BASE_VARS, ...customVars], [customVars]);

  // matrix state
  const [cellM, setCellM] = useState<CellMatrix>(() => {
    // start with the selected sample preset for base vars
    const baseM = buildPresetBaseMatrix("All");
    return carryOverCellMatrix(BASE_VARS, baseM, BASE_VARS, baseM);
  });
  const [matrixText, setMatrixText] = useState<string>(() => defaultSampleText([...BASE_VARS]));
  const [inputMode, setInputMode] = useState<"text" | "grid">("grid");

  // errors/warnings
  const [matrixErrors, setMatrixErrors] = useState<string[]>([]);
  const [matrixWarnings, setMatrixWarnings] = useState<string[]>([]);

  // edges + positions
  const [edges, setEdges] = useState<Edge[]>([]);
  const [nodePos, setNodePos] = useState<NodePos>({});
  const [connectFrom, setConnectFrom] = useState<VarName | null>(null);

  // estimation
  const [nMethod, setNMethod] = useState<"harmonic" | "min">("harmonic");
  const [lastEst, setLastEst] = useState<EstResult | null>(null);

  // Apply base preset when sampleType changes (keep custom vars untouched)
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sampleType]);

  // ensure node positions exist for all vars
  useEffect(() => {
    const width = 720;
    const height = 440;
    setNodePos((prev) => {
      const next: NodePos = { ...(prev || {}) };
      let missing = false;
      for (const v of vars) if (!next[v]) missing = true;
      if (!missing) return next;

      const init = initialPositions(vars, width, height);
      for (const v of vars) if (!next[v]) next[v] = init[v];
      return next;
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [vars.join("|")]);

  // whenever edges change, clear lastEst
  useEffect(() => {
    setLastEst(null);
  }, [edges]);

  // grid edit: enforce symmetry
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

  const validateOnly = () => {
    const v = validateCellMatrix(vars, cellM);
    setMatrixErrors(v.errors);
    setMatrixWarnings(v.warnings);
  };

  const applyVars = () => {
    const names = pendingNames.map((s) => s.trim()).filter(Boolean);

    if (names.length !== pendingCount) {
      setMatrixErrors([
        `You specified ${pendingCount} new variables, but provided ${names.length} valid names. Fill all names (non-empty).`,
      ]);
      return;
    }

    const all = [...BASE_VARS, ...names];
    const dup = all.find((x, i) => all.indexOf(x) !== i);
    if (dup) {
      setMatrixErrors([`Duplicate variable name detected: "${dup}". Variable names must be unique.`]);
      return;
    }

    const newVars = [...BASE_VARS, ...names];
    const newM = makeEmptyCellMatrix(newVars);

    // keep current values if possible
    carryOverCellMatrix(vars, cellM, newVars, newM);

    // ensure base preset always exists
    const basePreset = buildPresetBaseMatrix(sampleType);
    for (const a of BASE_VARS) for (const b of BASE_VARS) newM[a][b] = deepClone(basePreset[a][b]);

    setCustomVars(names);
    setCellM(newM);
    setEdges([]);
    setLastEst(null);
    setMatrixText(defaultSampleText(newVars));
    setMatrixErrors([]);
    setMatrixWarnings([]);
  };

  const resetBase = () => {
    const newVars = [...BASE_VARS];
    setCustomVars([]);
    setPendingCount(0);
    setPendingNames([]);
    setCellM(buildPresetBaseMatrix(sampleType));
    setEdges([]);
    setLastEst(null);
    setMatrixText(defaultSampleText(newVars));
    setMatrixErrors([]);
    setMatrixWarnings([]);
  };

  const runEstimationAndStore = () => {
    const v = validateCellMatrix(vars, cellM);
    const errors: string[] = [];
    const warnings: string[] = [...v.warnings];

    if (!edges.length) errors.push("No paths defined. Create at least one directed edge.");
    if (!v.ok) errors.push(...v.errors);

    const N = v.ok ? computeTotalN(vars, cellM, nMethod) : Number.NaN;
    if (v.ok && (!Number.isFinite(N) || N <= 2)) errors.push("Total N is invalid (check pairwise n values).");

    setMatrixErrors(errors);
    setMatrixWarnings(warnings);

    if (errors.length) {
      setLastEst(null);
      return;
    }

    try {
      const S = buildRMatrix(vars, cellM);
      const { coeffs, r2, resid } = estimatePathsFromCorrelation(vars, cellM, edges);
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
        warnings.push("df = 0. Fit indices like χ²/RMSEA/CFI/TLI are not meaningful.");
      }

      setMatrixWarnings(warnings);
      setLastEst({ coeffs, r2, resid, fit });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setMatrixErrors([msg]);
      setLastEst(null);
    }
  };

  // Diagram state: dragging
  const width = 720;
  const height = 440;
  const dragRef = useRef<{ active: boolean; varName: VarName | null; dx: number; dy: number }>({
    active: false,
    varName: null,
    dx: 0,
    dy: 0,
  });

  const betaMap = useMemo(() => {
    const m = new Map<string, number>();
    (lastEst?.coeffs || []).forEach((c) => m.set(`${c.from}→${c.to}`, c.beta));
    return m;
  }, [lastEst]);

  const rOf = (a: VarName, b: VarName) => cellM?.[a]?.[b]?.r;

  const addEdge = (from: VarName, to: VarName) => {
    if (from === to) return;
    const exists = edges.some((e) => e.from === from && e.to === to);
    if (exists) return;
    setEdges([...edges, { from, to }]);
  };

  const onNodeClick = (v: VarName) => {
    if (!connectFrom) {
      setConnectFrom(v);
      return;
    }
    if (connectFrom === v) {
      setConnectFrom(null);
      return;
    }
    addEdge(connectFrom, v);
    setConnectFrom(null);
  };

  const removeEdgeAt = (idx: number) => {
    setEdges(edges.filter((_, i) => i !== idx));
  };

  const onPointerDownNode = (e: React.PointerEvent<SVGCircleElement>, v: VarName) => {
    e.preventDefault();
    const svg = e.currentTarget.ownerSVGElement;
    if (!svg) return;

    const pt = svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const cursor = pt.matrixTransform(svg.getScreenCTM()?.inverse());

    const p = nodePos[v];
    if (!p) return;

    dragRef.current = {
      active: true,
      varName: v,
      dx: p.x - cursor.x,
      dy: p.y - cursor.y,
    };
  };

  const onPointerMoveSvg = (e: React.PointerEvent<SVGSVGElement>) => {
    if (!dragRef.current.active) return;
    const v = dragRef.current.varName;
    if (!v) return;

    const svg = e.currentTarget;
    const pt = svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const cursor = pt.matrixTransform(svg.getScreenCTM()?.inverse());

    const nx = cursor.x + dragRef.current.dx;
    const ny = cursor.y + dragRef.current.dy;

    setNodePos((prev) => ({
      ...prev,
      [v]: {
        x: Math.max(50, Math.min(width - 50, nx)),
        y: Math.max(50, Math.min(height - 50, ny)),
      },
    }));
  };

  const endDrag = () => {
    dragRef.current.active = false;
    dragRef.current.varName = null;
  };

  return (
    <div className="min-h-screen p-4 md:p-8 max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-8 flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4">
        <div className="flex items-start gap-3">
          <div className="mt-1 text-indigo-600">
            <Share2 size={22} />
          </div>
          <div>
            <h1 className="text-2xl md:text-3xl font-bold text-slate-800">
              Expand the MESEM analysis of So, Yang and Li (2025)
            </h1>
            <p className="text-slate-600 mt-1 text-sm">
              Programmed by Dr. Yang Yang, Temple University.
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <a
            href="/MESEM_User_Manual.pdf"
            download
            className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700 shadow-sm hover:bg-slate-50"
            title="Download the user manual PDF (place it in /public)."
          >
            <Download size={16} />
            Download User Manual
          </a>

          <div className="flex items-center gap-2 bg-white border border-slate-200 rounded-xl p-2">
            <span className="text-xs text-slate-600">Sample</span>
            <select
              value={sampleType}
              onChange={(e) => setSampleType(e.target.value as SampleType)}
              className="text-sm border border-slate-200 rounded-lg px-2 py-1"
            >
              <option value="All">All</option>
              <option value="Lodging">Lodging</option>
              <option value="Restaurant">Restaurant</option>
              <option value="Tourism and travel">Tourism and travel</option>
            </select>
          </div>

          <div className="flex items-center gap-2 bg-white border border-slate-200 rounded-xl p-2">
            <span className="text-xs text-slate-600">Total N</span>
            <select
              value={nMethod}
              onChange={(e) => setNMethod(e.target.value as "harmonic" | "min")}
              className="text-sm border border-slate-200 rounded-lg px-2 py-1"
            >
              <option value="harmonic">Harmonic mean</option>
              <option value="min">Minimum</option>
            </select>
          </div>
        </div>
      </div>

      {/* Global errors/warnings */}
      <div className="mb-6 space-y-3">
        <ErrorBox title="Errors (matrix/model)" items={matrixErrors} tone="error" />
        <ErrorBox title="Warnings" items={matrixWarnings} tone="warn" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Left */}
        <div className="space-y-6">
          {/* Variables panel */}
          <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
            <h2 className="text-lg font-semibold text-slate-800 mb-3 flex items-center gap-2">
              <ListPlus size={18} /> Variables
            </h2>

            <div className="text-xs text-slate-500 mb-3">
              Base variables are fixed (loyalty, satisfaction, value, quality). Specify how many new variables you want, then provide their names.
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 items-end">
              <div>
                <label className="text-xs text-slate-600"># new variables</label>
                <input
                  type="number"
                  min={0}
                  max={12}
                  value={pendingCount}
                  onChange={(e) => {
                    const v = Math.max(0, Math.min(12, Number(e.target.value || 0)));
                    setPendingCount(v);
                    setPendingNames((prev) => {
                      const arr = [...prev];
                      while (arr.length < v) arr.push("");
                      while (arr.length > v) arr.pop();
                      return arr;
                    });
                  }}
                  className="w-full mt-1 px-3 py-2 border border-slate-300 rounded-lg"
                />
              </div>

              <div className="md:col-span-2">
                <label className="text-xs text-slate-600">New variable names (one per line)</label>
                <textarea
                  value={pendingNames.join("\n")}
                  onChange={(e) => {
                    const lines = e.target.value.split(/\r?\n/).map((s) => s.trim());
                    const fixed = lines.slice(0, pendingCount);
                    while (fixed.length < pendingCount) fixed.push("");
                    setPendingNames(fixed);
                  }}
                  placeholder={"trust\nengagement\nprice_fairness"}
                  className="w-full mt-1 h-24 px-3 py-2 border border-slate-300 rounded-lg text-sm font-mono"
                />
              </div>
            </div>

            <div className="flex flex-wrap gap-2 mt-4">
              <button
                onClick={applyVars}
                className="bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-lg text-sm font-medium"
              >
                Apply variables
              </button>
              <button
                onClick={resetBase}
                className="px-4 py-2 rounded-lg text-sm border border-slate-200 hover:bg-slate-50"
              >
                Reset to base 4
              </button>

              <div className="ml-auto text-xs text-slate-500 self-center">
                Total variables: <b>{vars.length}</b>
              </div>
            </div>

            <div className="mt-3 flex flex-wrap gap-2">
              {BASE_VARS.map((v) => (
                <span
                  key={v}
                  className="text-xs px-2 py-1 rounded-full bg-slate-100 border border-slate-200 text-slate-700"
                >
                  {v}
                </span>
              ))}
              {customVars.map((v) => (
                <span
                  key={v}
                  className="text-xs px-2 py-1 rounded-full bg-indigo-50 border border-indigo-200 text-indigo-800"
                >
                  {v}
                </span>
              ))}
            </div>
          </div>

          {/* Matrix input panel */}
          <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
            <div className="flex items-center justify-between gap-2 mb-3">
              <h2 className="text-lg font-semibold text-slate-800 flex items-center gap-2">
                <TableIcon size={18} /> Matrix (r + pairwise N)
              </h2>

              <div className="flex gap-1 bg-slate-50 border border-slate-200 rounded-lg p-1">
                <button
                  onClick={() => setInputMode("text")}
                  className={`px-3 py-1.5 text-xs rounded-md ${
                    inputMode === "text" ? "bg-slate-900 text-white" : "text-slate-700 hover:bg-white"
                  }`}
                >
                  Text
                </button>
                <button
                  onClick={() => setInputMode("grid")}
                  className={`px-3 py-1.5 text-xs rounded-md ${
                    inputMode === "grid" ? "bg-slate-900 text-white" : "text-slate-700 hover:bg-white"
                  }`}
                >
                  Grid
                </button>
              </div>
            </div>

            {inputMode === "text" ? (
              <>
                <div className="text-xs text-slate-500 mb-2">
                  Paste a combined matrix. Off-diagonal cells must include both correlation and sample size:{" "}
                  <span className="font-mono">r|n</span> or <span className="font-mono">r(n)</span>.
                </div>

                <textarea
                  value={matrixText}
                  onChange={(e) => setMatrixText(e.target.value)}
                  className="w-full h-52 px-3 py-2 border border-slate-300 rounded-lg text-xs font-mono"
                  placeholder={`,loyalty,satisfaction,value,quality
loyalty,1,0.734|63671,0.545|81110,0.575|52764
satisfaction,0.734|63671,1,0.708|37150,0.711|34677
value,0.545|81110,0.708|37150,1,0.561|58390
quality,0.575|52764,0.711|34677,0.561|58390,1`}
                />

                <div className="flex flex-wrap gap-2 mt-3">
                  <button
                    onClick={() => {
                      const parsed = parseCombinedMatrixText(matrixText);
                      const pv = parsed.vars;

                      const same = pv.length === vars.length && pv.every((v, i) => v === vars[i]);
                      if (!same) {
                        throw new Error(
                          "Variable names in pasted matrix do not match current variables. Apply variables first, then paste a matrix with the same header."
                        );
                      }
                      setCellM(parsed.cellMatrix);
                      setLastEst(null);
                    }}
                    className="bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-lg text-sm font-medium"
                  >
                    Load from text
                  </button>

                  <button
                    onClick={() => setMatrixText(defaultSampleText(vars))}
                    className="px-4 py-2 rounded-lg text-sm border border-slate-200 hover:bg-slate-50"
                  >
                    Fill template
                  </button>

                  <button
                    onClick={validateOnly}
                    className="ml-auto px-4 py-2 rounded-lg text-sm border border-slate-200 hover:bg-slate-50"
                  >
                    Validate matrix
                  </button>
                </div>
              </>
            ) : (
              <>
                <div className="text-xs text-slate-500 mb-3">
                  Enter <b>r</b> and <b>N</b>. Edits mirror automatically (symmetry enforced). Header row/first column are sticky for easier scrolling.
                </div>

                {/* Improved matrix scroller: horizontal + vertical, sticky header + first column */}
                <div className="overflow-x-auto max-h-[420px] overflow-y-auto rounded-lg border border-slate-200">
                  <table className="text-xs border-collapse min-w-max w-full">
                    <thead>
                      <tr className="border-b border-slate-200">
                        <th className="sticky top-0 left-0 z-30 bg-white p-2 text-left text-slate-600"></th>
                        {vars.map((v) => (
                          <th
                            key={v}
                            className="sticky top-0 z-20 bg-white p-2 text-left text-slate-600 whitespace-nowrap"
                          >
                            {v}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {vars.map((r) => (
                        <tr key={r} className="border-t border-slate-100">
                          <td className="sticky left-0 z-10 bg-white p-2 font-semibold text-slate-700 whitespace-nowrap border-r border-slate-100">
                            {r}
                          </td>

                          {vars.map((c) => {
                            const isDiag = r === c;
                            const cell = cellM?.[r]?.[c] ?? { r: Number.NaN, n: Number.NaN };

                            return (
                              <td key={c} className="p-2 align-top">
                                {isDiag ? (
                                  <div className="text-slate-500 bg-slate-50 border border-slate-200 rounded-lg px-2 py-2">
                                    1.00
                                  </div>
                                ) : (
                                  <div className={`border border-slate-200 rounded-lg p-1.5 ${cellBgForR(cell.r)}`}>
                                    <div className="flex items-center gap-2">
                                      <label className="text-[10px] text-slate-500 w-4">r</label>
                                      <input
                                        type="number"
                                        step="0.001"
                                        value={Number.isFinite(cell.r) ? cell.r : ""}
                                        onChange={(e) => setCell(r, c, { r: Number(e.target.value) })}
                                        className="w-16 px-2 py-1 border border-slate-300 rounded-md text-xs"
                                        placeholder="0.734"
                                      />
                                    </div>
                                    <div className="flex items-center gap-2 mt-2">
                                      <label className="text-[10px] text-slate-500 w-4">N</label>
                                      <input
                                        type="number"
                                        step={1}
                                        value={Number.isFinite(cell.n) ? cell.n : ""}
                                        onChange={(e) => setCell(r, c, { n: Number(e.target.value) })}
                                        className="w-16 px-2 py-1 border border-slate-300 rounded-md text-xs"
                                        placeholder="63671"
                                      />
                                    </div>
                                  </div>
                                )}
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="flex mt-3">
                  <button
                    onClick={validateOnly}
                    className="ml-auto px-4 py-2 rounded-lg text-sm border border-slate-200 hover:bg-slate-50"
                  >
                    Validate matrix
                  </button>
                </div>
              </>
            )}
          </div>

          {/* Quick actions */}
          <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
            <div className="flex items-center justify-between gap-2">
              <h3 className="text-sm font-semibold text-slate-800 flex items-center gap-2">
                <CheckCircle2 size={18} /> Quick actions
              </h3>
              <button
                onClick={runEstimationAndStore}
                className="bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-lg text-sm font-medium"
              >
                Run estimation
              </button>
            </div>
            <div className="text-xs text-slate-500 mt-2">
              This runs validation + estimation and stores results so the diagram shows β on edges.
            </div>
          </div>
        </div>

        {/* Right */}
        <div className="space-y-6">
          {/* Diagram panel */}
          <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-lg font-semibold text-slate-800 flex items-center gap-2">
                <Network size={18} /> Diagram (drag + connect)
              </h2>
              <div className="text-xs text-slate-500">Click node A then node B to add A→B. Click an edge to remove.</div>
            </div>

            {connectFrom && (
              <div className="mb-3 text-xs text-indigo-800 bg-indigo-50 border border-indigo-200 rounded-lg p-2">
                Connecting from: <b>{connectFrom}</b> (click a target node)
              </div>
            )}

            <div className="border border-slate-100 rounded-lg bg-slate-50 overflow-hidden">
              <svg
                viewBox={`0 0 ${width} ${height}`}
                className="w-full h-auto"
                onPointerMove={onPointerMoveSvg}
                onPointerUp={endDrag}
                onPointerLeave={endDrag}
              >
                <defs>
                  <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="26" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" fill="#64748b" />
                  </marker>
                </defs>

                {/* edges */}
                {edges.map((e, idx) => {
                  const p1 = nodePos[e.from];
                  const p2 = nodePos[e.to];
                  if (!p1 || !p2) return null;

                  const key = `${e.from}→${e.to}`;
                  const label = betaMap.has(key)
                    ? betaMap.get(key)!.toFixed(2)
                    : Number.isFinite(rOf(e.from, e.to))
                    ? rOf(e.from, e.to).toFixed(2)
                    : "?";

                  const mx = (p1.x + p2.x) / 2;
                  const my = (p1.y + p2.y) / 2;

                  // curve offset
                  const dx = p2.x - p1.x;
                  const dy = p2.y - p1.y;
                  const norm = Math.sqrt(dx * dx + dy * dy) || 1;
                  const off = 18;
                  const cx = mx - (dy / norm) * off;
                  const cy = my + (dx / norm) * off;

                  const d = `M ${p1.x} ${p1.y} Q ${cx} ${cy} ${p2.x} ${p2.y}`;

                  return (
                    <g key={idx} className="cursor-pointer" onClick={() => removeEdgeAt(idx)}>
                      <path
                        d={d}
                        stroke="#64748b"
                        strokeWidth="2.2"
                        fill="none"
                        markerEnd="url(#arrowhead)"
                        opacity="0.9"
                      />
                      <rect x={cx - 18} y={cy - 11} width="36" height="22" rx="6" fill="white" stroke="#e2e8f0" />
                      <text x={cx} y={cy} dy="4" textAnchor="middle" fontSize="10" fill="#334155" fontWeight="bold">
                        {label}
                      </text>
                    </g>
                  );
                })}

                {/* nodes */}
                {vars.map((v) => {
                  const p = nodePos[v];
                  if (!p) return null;
                  const isSelected = connectFrom === v;

                  return (
                    <g key={v} className="select-none">
                      <circle
                        cx={p.x}
                        cy={p.y}
                        r="38"
                        fill={isSelected ? "#c7d2fe" : "#e0e7ff"}
                        stroke={isSelected ? "#4f46e5" : "#64748b"}
                        strokeWidth={isSelected ? "3" : "2"}
                        onPointerDown={(e) => onPointerDownNode(e, v)}
                        onClick={() => onNodeClick(v)}
                        style={{ cursor: "grab" }}
                      />
                      <text x={p.x} y={p.y} dy="-3" textAnchor="middle" fontSize="10" fontWeight="bold" fill="#1e293b">
                        {v}
                      </text>
                    </g>
                  );
                })}
              </svg>
            </div>

            <div className="mt-3 flex items-center gap-2">
              <button onClick={() => setEdges([])} className="text-xs text-rose-700 hover:text-rose-900 font-medium">
                Clear edges
              </button>
              <span className="ml-auto text-xs text-slate-500">
                Edges: <b>{edges.length}</b>
              </span>
            </div>
          </div>

          {/* Estimation panel */}
          <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
            <h2 className="text-lg font-semibold text-slate-800 mb-3 flex items-center gap-2">
              <Calculator size={18} /> Estimate model
            </h2>

            <div className="text-xs text-slate-500 mb-3">
              Total N is derived from pairwise N using: <b>{nMethod === "harmonic" ? "harmonic mean" : "minimum"}</b>.
            </div>

            <div className="flex flex-wrap gap-2">
              <button
                onClick={runEstimationAndStore}
                className="bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-lg text-sm font-medium"
              >
                Run estimation
              </button>
              <button
                onClick={validateOnly}
                className="px-4 py-2 rounded-lg text-sm border border-slate-200 hover:bg-slate-50"
              >
                Validate matrix
              </button>
            </div>

            {lastEst && (
              <div className="mt-4 space-y-5">
                <div>
                  <div className="text-sm font-semibold text-slate-800 mb-2">Path coefficients (standardized β)</div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="text-slate-500">
                          <th className="text-left py-1">From</th>
                          <th className="text-left py-1">To</th>
                          <th className="text-right py-1">β</th>
                        </tr>
                      </thead>
                      <tbody>
                        {lastEst.coeffs.map((c, i) => (
                          <tr key={i} className="border-t border-slate-100">
                            <td className="py-1">{c.from}</td>
                            <td className="py-1">{c.to}</td>
                            <td className="py-1 text-right font-semibold">{c.beta.toFixed(3)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                <div>
                  <div className="text-sm font-semibold text-slate-800 mb-2">R² (endogenous variables)</div>
                  <div className="flex flex-wrap gap-2">
                    {Object.keys(lastEst.r2).length === 0 ? (
                      <span className="text-xs text-slate-500">No endogenous variables (no incoming paths).</span>
                    ) : (
                      Object.entries(lastEst.r2).map(([k, v]) => (
                        <span
                          key={k}
                          className="text-xs px-2 py-1 rounded-full bg-slate-100 border border-slate-200 text-slate-700"
                        >
                          {k}: <b>{v.toFixed(3)}</b>
                        </span>
                      ))
                    )}
                  </div>
                </div>

                <div>
                  <div className="text-sm font-semibold text-slate-800 mb-2">Goodness of fit</div>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div className="bg-slate-50 border border-slate-200 rounded-lg p-3">
                      Total N: <b>{lastEst.fit.totalN.toFixed(1)}</b>
                      <div className="text-xs text-slate-500 mt-1">({lastEst.fit.N_method})</div>
                    </div>
                    <div className="bg-slate-50 border border-slate-200 rounded-lg p-3">df: <b>{lastEst.fit.df}</b></div>

                    <div className="bg-slate-50 border border-slate-200 rounded-lg p-3">
                      SRMR: <b>{lastEst.fit.SRMR.toFixed(4)}</b>
                    </div>
                    {"chi2" in lastEst.fit ? (
                      <div className="bg-slate-50 border border-slate-200 rounded-lg p-3">
                        χ²: <b>{lastEst.fit.chi2!.toFixed(2)}</b>
                      </div>
                    ) : (
                      <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 text-slate-500">χ²: n/a</div>
                    )}

                    {"rmsea" in lastEst.fit ? (
                      <>
                        <div className="bg-slate-50 border border-slate-200 rounded-lg p-3">
                          RMSEA: <b>{lastEst.fit.rmsea!.toFixed(4)}</b>
                        </div>
                        <div className="bg-slate-50 border border-slate-200 rounded-lg p-3">
                          CFI: <b>{lastEst.fit.cfi!.toFixed(4)}</b>
                        </div>
                        <div className="bg-slate-50 border border-slate-200 rounded-lg p-3">
                          TLI: <b>{lastEst.fit.tli!.toFixed(4)}</b>
                        </div>
                        <div className="bg-slate-50 border border-slate-200 rounded-lg p-3">
                          Params: <b>{lastEst.fit.freeParams}</b>
                          <div className="text-xs text-slate-500 mt-1">Moments: {lastEst.fit.observedMoments}</div>
                        </div>
                      </>
                    ) : (
                      <div className="col-span-2 text-xs text-slate-500">
                        RMSEA/CFI/TLI require df &gt; 0. Currently only SRMR is shown.
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {!lastEst && (
              <div className="mt-4 text-xs text-slate-500">
                Tip: create at least one arrow, fill the matrix (including pairwise N), then click Run estimation.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
