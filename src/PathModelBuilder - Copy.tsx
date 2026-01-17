import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Network,
  Download,
  Grid3X3,
  FileText,
  Info,
  AlertTriangle,
  Trash2,
  Play,
  CheckCircle2,
  MousePointerClick,
} from "lucide-react";

/**
 * Path Model Builder
 * - Default variables: loyalty, satisfaction, value, quality
 * - Load default correlation (r) + pairwise N from presets (All/Lodging/Restaurant/Tourism and travel)
 * - Grid view and Text view both default to the 4-variable matrices
 * - Allow at most 3 new variables (total max 7 variables)
 * - Symmetry enforced
 */

type SampleType = "All" | "Lodging" | "Restaurant" | "Tourism and travel";
type ViewMode = "grid" | "text";

type Cell = { r: number; n: number };
type Matrix = Record<string, Record<string, Cell>>;

const BASE_VARS = ["loyalty", "satisfaction", "value", "quality"] as const;
const MAX_NEW_VARS = 3;

const pairKey = (a: string, b: string) => {
  const x = a.trim().toLowerCase();
  const y = b.trim().toLowerCase();
  return [x, y].sort().join("|");
};

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

function clampR(x: number) {
  if (!Number.isFinite(x)) return Number.NaN;
  if (x > 0.999) return 0.999;
  if (x < -0.999) return -0.999;
  return x;
}

function deepCloneMatrix(m: Matrix): Matrix {
  const out: Matrix = {};
  for (const r of Object.keys(m)) {
    out[r] = {};
    for (const c of Object.keys(m[r])) out[r][c] = { r: m[r][c].r, n: m[r][c].n };
  }
  return out;
}

function makeEmptyMatrix(vars: string[]): Matrix {
  const out: Matrix = {};
  for (const r of vars) {
    out[r] = {};
    for (const c of vars) {
      out[r][c] = r === c ? { r: 1, n: Number.NaN } : { r: Number.NaN, n: Number.NaN };
    }
  }
  return out;
}

function applyPresetBasePairs(m: Matrix, sampleType: SampleType) {
  const preset = SAMPLE_PRESETS[sampleType]?.pair;
  for (let i = 0; i < BASE_VARS.length; i++) {
    for (let j = i + 1; j < BASE_VARS.length; j++) {
      const a = BASE_VARS[i];
      const b = BASE_VARS[j];
      const k = pairKey(a, b);
      const v = preset?.[k];
      // 强制：如果 preset 缺失，写 NaN（避免沿用旧值造成“看起来没加载”）
      if (v) {
        m[a][b] = { r: v.r, n: v.n };
        m[b][a] = { r: v.r, n: v.n };
      } else {
        m[a][b] = { r: Number.NaN, n: Number.NaN };
        m[b][a] = { r: Number.NaN, n: Number.NaN };
      }
    }
  }
  // 对角线
  for (const v of BASE_VARS) {
    m[v][v] = { r: 1, n: Number.NaN };
  }
}

function defaultTextMatrices(sampleType: SampleType) {
  const vars = [...BASE_VARS];
  const preset = SAMPLE_PRESETS[sampleType].pair;

  // build r matrix text (tab-separated)
  const header = ["", ...vars].join("\t");
  const rRows: string[] = [header];
  const nRows: string[] = [header];

  for (const r of vars) {
    const rLine: string[] = [r];
    const nLine: string[] = [r];

    for (const c of vars) {
      if (r === c) {
        rLine.push("1.000");
        nLine.push("");
      } else {
        const k = pairKey(r, c);
        const v = preset[k];
        rLine.push(v ? v.r.toFixed(3) : "");
        nLine.push(v ? String(v.n) : "");
      }
    }
    rRows.push(rLine.join("\t"));
    nRows.push(nLine.join("\t"));
  }

  return {
    rText: rRows.join("\n"),
    nText: nRows.join("\n"),
  };
}

function parseMatrixText(rText: string, nText: string) {
  // Expect tab or comma separated matrix with header row/col
  const splitLine = (line: string) => line.split(/\t|,/).map((s) => s.trim());

  const rLines = rText.split(/\r?\n/).map((s) => s.trim()).filter(Boolean);
  const nLines = nText.split(/\r?\n/).map((s) => s.trim()).filter(Boolean);

  if (rLines.length < 2) throw new Error("Correlation matrix text is too short.");
  if (nLines.length < 2) throw new Error("Sample size matrix text is too short.");

  const rHeader = splitLine(rLines[0]).filter((_, i) => i > 0);
  const nHeader = splitLine(nLines[0]).filter((_, i) => i > 0);

  if (rHeader.length === 0) throw new Error("Correlation matrix header row is missing variable names.");
  if (nHeader.length === 0) throw new Error("N matrix header row is missing variable names.");

  const vars = rHeader.map((v) => v.toLowerCase());
  const varsN = nHeader.map((v) => v.toLowerCase());

  if (vars.join("|") !== varsN.join("|")) throw new Error("Variable headers do not match between r and N matrices.");

  const k = vars.length;
  if (rLines.length !== k + 1) throw new Error("Correlation matrix row count does not match header size.");
  if (nLines.length !== k + 1) throw new Error("N matrix row count does not match header size.");

  // Enforce max variables: base 4 + 3 new
  if (k > BASE_VARS.length + MAX_NEW_VARS) {
    throw new Error(`Too many variables in text matrices. Maximum is ${BASE_VARS.length + MAX_NEW_VARS}.`);
  }

  const m = makeEmptyMatrix(vars);

  for (let i = 0; i < k; i++) {
    const rRow = splitLine(rLines[i + 1]);
    const nRow = splitLine(nLines[i + 1]);

    const rowNameR = (rRow[0] || "").toLowerCase();
    const rowNameN = (nRow[0] || "").toLowerCase();

    if (rowNameR !== vars[i]) throw new Error(`Correlation matrix row name mismatch at row ${i + 1}: expected ${vars[i]}, got ${rowNameR || "(blank)"}.`);
    if (rowNameN !== vars[i]) throw new Error(`N matrix row name mismatch at row ${i + 1}: expected ${vars[i]}, got ${rowNameN || "(blank)"}.`);

    // values start at index 1
    for (let j = 0; j < k; j++) {
      const rr = rRow[j + 1] ?? "";
      const nn = nRow[j + 1] ?? "";

      if (i === j) {
        m[vars[i]][vars[j]] = { r: 1, n: Number.NaN };
        continue;
      }

      const rVal = rr === "" ? Number.NaN : Number(rr);
      const nVal = nn === "" ? Number.NaN : Number(nn);

      if (rr !== "" && !Number.isFinite(rVal)) throw new Error(`Invalid r at (${vars[i]}, ${vars[j]}): "${rr}"`);
      if (nn !== "" && (!Number.isFinite(nVal) || nVal <= 0)) throw new Error(`Invalid N at (${vars[i]}, ${vars[j]}): "${nn}"`);

      m[vars[i]][vars[j]] = { r: clampR(rVal), n: nVal };
    }
  }

  // Symmetry enforce: mirror upper to lower if mismatch, and warn outside.
  // We'll average r if both provided and different; N will take min (conservative).
  const warnings: string[] = [];
  for (let i = 0; i < k; i++) {
    for (let j = i + 1; j < k; j++) {
      const a = vars[i];
      const b = vars[j];
      const ab = m[a][b];
      const ba = m[b][a];

      const r1 = ab.r;
      const r2 = ba.r;
      const n1 = ab.n;
      const n2 = ba.n;

      const r1ok = Number.isFinite(r1);
      const r2ok = Number.isFinite(r2);
      const n1ok = Number.isFinite(n1);
      const n2ok = Number.isFinite(n2);

      let rFinal = Number.NaN;
      let nFinal = Number.NaN;

      if (r1ok && r2ok && Math.abs(r1 - r2) > 1e-6) {
        warnings.push(`r mismatch for (${a}, ${b}). Averaged ${r1.toFixed(3)} and ${r2.toFixed(3)}.`);
        rFinal = clampR((r1 + r2) / 2);
      } else if (r1ok) rFinal = r1;
      else if (r2ok) rFinal = r2;

      if (n1ok && n2ok && Math.abs(n1 - n2) > 1e-6) {
        warnings.push(`N mismatch for (${a}, ${b}). Used min(${n1}, ${n2}).`);
        nFinal = Math.min(n1, n2);
      } else if (n1ok) nFinal = n1;
      else if (n2ok) nFinal = n2;

      m[a][b] = { r: rFinal, n: nFinal };
      m[b][a] = { r: rFinal, n: nFinal };
    }
  }

  // diag
  for (const v of vars) m[v][v] = { r: 1, n: Number.NaN };

  return { vars, matrix: m, warnings };
}

function cellBgForR(r: number) {
  if (!Number.isFinite(r) || r === 1) return "bg-white";
  if (r >= 0.5) return "bg-emerald-50";
  if (r >= 0.3) return "bg-emerald-50/50";
  if (r <= -0.3) return "bg-rose-50/60";
  return "bg-white";
}

type Edge = { from: string; to: string };

type EstimationResult = {
  message: string;
};

function uniqueVars(names: string[]) {
  const cleaned = names.map((s) => s.trim().toLowerCase()).filter(Boolean);
  const seen = new Set<string>();
  const out: string[] = [];
  for (const v of cleaned) {
    if (!seen.has(v)) {
      seen.add(v);
      out.push(v);
    }
  }
  return out;
}

function harmonicMean(xs: number[]) {
  const vals = xs.filter((x) => Number.isFinite(x) && x > 0);
  if (vals.length === 0) return Number.NaN;
  const inv = vals.reduce((a, x) => a + 1 / x, 0);
  return vals.length / inv;
}

function minPositive(xs: number[]) {
  const vals = xs.filter((x) => Number.isFinite(x) && x > 0);
  if (vals.length === 0) return Number.NaN;
  return Math.min(...vals);
}

// Placeholder “estimation” logic (kept lightweight):
// This tool focuses on matrix construction + model diagram. You can replace with MESEM estimation later.
function runEstimation(vars: string[], m: Matrix, edges: Edge[], totalNMode: "harmonic" | "min") {
  if (edges.length < 1) throw new Error("Please create at least one arrow (edge) before estimation.");

  // Basic validation: ensure all edges have r and N
  const missing: string[] = [];
  const Ns: number[] = [];
  for (const e of edges) {
    const cell = m[e.from]?.[e.to];
    if (!cell || !Number.isFinite(cell.r)) missing.push(`Missing r for ${e.from} → ${e.to}`);
    if (!cell || !Number.isFinite(cell.n)) missing.push(`Missing N for ${e.from} → ${e.to}`);
    if (cell && Number.isFinite(cell.n)) Ns.push(cell.n);
  }
  if (missing.length) throw new Error(missing.join("; "));

  const totalN = totalNMode === "harmonic" ? harmonicMean(Ns) : minPositive(Ns);
  if (!Number.isFinite(totalN)) throw new Error("Could not derive total N from pairwise N (no valid Ns found).");

  return {
    message: `Estimation ran successfully (demo). Total N (${totalNMode}) = ${Math.round(totalN)}. Edges = ${edges.length}.`,
  } as EstimationResult;
}

export default function PathModelBuilder() {
  const [sampleType, setSampleType] = useState<SampleType>("All");
  const [viewMode, setViewMode] = useState<ViewMode>("grid");

  const [vars, setVars] = useState<string[]>([...BASE_VARS]);
  const [cellM, setCellM] = useState<Matrix>(() => {
    const m = makeEmptyMatrix([...BASE_VARS]);
    applyPresetBasePairs(m, "All");
    return m;
  });

  const [newVarCount, setNewVarCount] = useState<number>(0);
  const [newVarNames, setNewVarNames] = useState<string>("");

  const [matrixErrors, setMatrixErrors] = useState<string[]>([]);
  const [matrixWarnings, setMatrixWarnings] = useState<string[]>([]);

  const [totalNMode, setTotalNMode] = useState<"harmonic" | "min">("harmonic");

  const [edges, setEdges] = useState<Edge[]>([]);
  const [selectFrom, setSelectFrom] = useState<string | null>(null);

  const [est, setEst] = useState<EstimationResult | null>(null);
  const [estError, setEstError] = useState<string | null>(null);

  // Text mode defaults (always reflect current sample preset for the base 4)
  const [rText, setRText] = useState<string>(() => defaultTextMatrices("All").rText);
  const [nText, setNText] = useState<string>(() => defaultTextMatrices("All").nText);

  // Diagram nodes positions (simple draggable layout)
  type NodePos = { x: number; y: number };
  const [pos, setPos] = useState<Record<string, NodePos>>(() => {
    // Default layout for 4 nodes
    return {
      loyalty: { x: 350, y: 70 },
      satisfaction: { x: 220, y: 170 },
      value: { x: 480, y: 170 },
      quality: { x: 350, y: 280 },
    };
  });

  // Whenever sample type changes: reset base correlations (preserve any extra vars & their cells)
  useEffect(() => {
    setCellM((prev) => {
      const next = deepCloneMatrix(prev);
      // ensure structure
      for (const r of vars) {
        next[r] ??= {};
        for (const c of vars) {
          if (!next[r][c]) next[r][c] = r === c ? { r: 1, n: Number.NaN } : { r: Number.NaN, n: Number.NaN };
        }
        next[r][r] = { r: 1, n: Number.NaN };
      }
      // apply base preset pairs (overwrites 6 pairs, and never leaves old values)
      applyPresetBasePairs(next, sampleType);
      return next;
    });

    // refresh text defaults to the base 4 preset
    const d = defaultTextMatrices(sampleType);
    setRText(d.rText);
    setNText(d.nText);

    setMatrixErrors([]);
    setMatrixWarnings([]);
    setEst(null);
    setEstError(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sampleType]);

  const totalVars = vars.length;

  function ensureMatrixForVars(nextVars: string[]) {
    // start from old matrix and expand/contract
    setCellM((prev) => {
      const next = makeEmptyMatrix(nextVars);
      // copy over overlapping
      for (const r of nextVars) {
        for (const c of nextVars) {
          if (prev[r]?.[c]) next[r][c] = { ...prev[r][c] };
        }
      }
      // enforce diag
      for (const v of nextVars) next[v][v] = { r: 1, n: Number.NaN };
      // always ensure base preset pairs for current sample type
      for (const v of BASE_VARS) {
        if (!next[v]) return next; // if base missing, leave
      }
      applyPresetBasePairs(next, sampleType);
      return next;
    });
  }

  function applyVariables() {
    const errs: string[] = [];

    const count = Math.max(0, Math.min(MAX_NEW_VARS, Math.floor(newVarCount)));
    if (newVarCount > MAX_NEW_VARS) {
      errs.push(`You can add at most ${MAX_NEW_VARS} new variables. (You requested ${newVarCount}.)`);
    }

    const names = uniqueVars(newVarNames.split(/\r?\n/));
    if (count > 0 && names.length < count) {
      errs.push(`Please provide at least ${count} new variable name(s), one per line.`);
    }

    const base = [...BASE_VARS];
    const extras = names.slice(0, count);

    // Prevent collision with base vars
    for (const ex of extras) {
      if (base.includes(ex as any)) errs.push(`New variable "${ex}" duplicates a base variable name.`);
    }

    // Total limit
    if (extras.length > MAX_NEW_VARS) errs.push(`Too many new variables. Max is ${MAX_NEW_VARS}.`);

    if (errs.length) {
      setMatrixErrors(errs);
      return;
    }

    setMatrixErrors([]);
    setMatrixWarnings([]);

    const nextVars = [...base, ...extras];
    setVars(nextVars);

    // init/expand matrix
    ensureMatrixForVars(nextVars);

    // expand default positions for new vars
    setPos((prev) => {
      const next = { ...prev };
      const w = 700;
      const h = 340;
      const centerX = w / 2;
      const centerY = h / 2;
      const radius = 140;

      const existing = new Set(Object.keys(next));
      const toAdd = extras.filter((v) => !existing.has(v));
      const startAngle = Math.PI / 6;

      toAdd.forEach((v, idx) => {
        const ang = startAngle + idx * (Math.PI / 3);
        next[v] = {
          x: Math.round(centerX + radius * Math.cos(ang)),
          y: Math.round(centerY + radius * Math.sin(ang)),
        };
      });
      // keep base positions if missing
      for (const v of BASE_VARS) {
        if (!next[v]) next[v] = { x: 350, y: 70 };
      }
      return next;
    });

    // reset edges & estimation when structure changes
    setEdges([]);
    setSelectFrom(null);
    setEst(null);
    setEstError(null);
  }

  function resetToBase4() {
    setNewVarCount(0);
    setNewVarNames("");
    setVars([...BASE_VARS]);
    const m = makeEmptyMatrix([...BASE_VARS]);
    applyPresetBasePairs(m, sampleType);
    setCellM(m);
    setEdges([]);
    setSelectFrom(null);
    setEst(null);
    setEstError(null);

    setPos({
      loyalty: { x: 350, y: 70 },
      satisfaction: { x: 220, y: 170 },
      value: { x: 480, y: 170 },
      quality: { x: 350, y: 280 },
    });

    // refresh text defaults
    const d = defaultTextMatrices(sampleType);
    setRText(d.rText);
    setNText(d.nText);

    setMatrixErrors([]);
    setMatrixWarnings([]);
  }

  function setCell(a: string, b: string, patch: Partial<Cell>) {
    setCellM((prev) => {
      const next = deepCloneMatrix(prev);
      const cur = next[a]?.[b] ?? { r: Number.NaN, n: Number.NaN };
      const upd: Cell = {
        r: patch.r !== undefined ? clampR(patch.r) : cur.r,
        n: patch.n !== undefined ? patch.n : cur.n,
      };
      if (!next[a]) next[a] = {};
      if (!next[b]) next[b] = {};
      if (!next[a][b]) next[a][b] = { r: Number.NaN, n: Number.NaN };
      if (!next[b][a]) next[b][a] = { r: Number.NaN, n: Number.NaN };

      next[a][b] = upd;
      next[b][a] = upd; // symmetry
      if (a === b) next[a][b] = { r: 1, n: Number.NaN };
      return next;
    });
  }

  function validateMatrix() {
    const errs: string[] = [];
    const warns: string[] = [];

    // Check symmetric + bounds + diag
    for (const r of vars) {
      for (const c of vars) {
        const cell = cellM[r]?.[c];
        if (!cell) continue;

        if (r === c) {
          if (cell.r !== 1) warns.push(`Diagonal r for "${r}" should be 1. (Forced to 1.)`);
          continue;
        }

        if (Number.isFinite(cell.r) && Math.abs(cell.r) >= 1) errs.push(`Invalid r at (${r}, ${c}): must be between -1 and 1.`);
        if (Number.isFinite(cell.n) && cell.n <= 0) errs.push(`Invalid N at (${r}, ${c}): must be > 0.`);
      }
    }

    setMatrixErrors(errs);
    setMatrixWarnings(warns);

    return errs.length === 0;
  }

  function loadFromText() {
    try {
      setMatrixErrors([]);
      setMatrixWarnings([]);

      const parsed = parseMatrixText(rText, nText);

      // Enforce base vars exist
      for (const v of BASE_VARS) {
        if (!parsed.vars.includes(v)) {
          throw new Error(`Text matrices must include the base variable "${v}".`);
        }
      }

      // Enforce max new vars
      const extras = parsed.vars.filter((v) => !BASE_VARS.includes(v as any));
      if (extras.length > MAX_NEW_VARS) {
        throw new Error(`Text matrices include ${extras.length} new variables. Maximum is ${MAX_NEW_VARS}.`);
      }

      setVars(parsed.vars);
      setCellM(parsed.matrix);
      setMatrixWarnings(parsed.warnings);

      // sync “new vars” UI box
      setNewVarCount(extras.length);
      setNewVarNames(extras.join("\n"));

      // reset edges
      setEdges([]);
      setSelectFrom(null);
      setEst(null);
      setEstError(null);
    } catch (e: any) {
      setMatrixErrors([e?.message ?? String(e)]);
    }
  }

  function clearEdges() {
    setEdges([]);
    setSelectFrom(null);
    setEst(null);
    setEstError(null);
  }

  function toggleEdge(from: string, to: string) {
    if (from === to) return;
    setEdges((prev) => {
      const exists = prev.some((e) => e.from === from && e.to === to);
      if (exists) return prev.filter((e) => !(e.from === from && e.to === to));
      return [...prev, { from, to }];
    });
  }

  function onNodeClick(v: string) {
    if (!selectFrom) {
      setSelectFrom(v);
      return;
    }
    if (selectFrom === v) {
      setSelectFrom(null);
      return;
    }
    toggleEdge(selectFrom, v);
    setSelectFrom(null);
  }

  function run() {
    setEstError(null);
    setEst(null);
    try {
      const ok = validateMatrix();
      if (!ok) throw new Error("Matrix validation failed. Please fix errors before estimation.");
      const result = runEstimation(vars, cellM, edges, totalNMode);
      setEst(result);
    } catch (e: any) {
      setEstError(e?.message ?? String(e));
    }
  }

  // ---------- Diagram rendering ----------
  const svgW = 740;
  const svgH = 360;

  const dragRef = useRef<{ id: string; ox: number; oy: number } | null>(null);

  function onMouseDownNode(e: React.MouseEvent, id: string) {
    const p = pos[id] ?? { x: 100, y: 100 };
    dragRef.current = { id, ox: e.clientX - p.x, oy: e.clientY - p.y };
  }

  function onMouseMoveCanvas(e: React.MouseEvent) {
    if (!dragRef.current) return;
    const { id, ox, oy } = dragRef.current;
    const x = Math.max(40, Math.min(svgW - 40, e.clientX - ox));
    const y = Math.max(40, Math.min(svgH - 40, e.clientY - oy));
    setPos((prev) => ({ ...prev, [id]: { x, y } }));
  }

  function onMouseUpCanvas() {
    dragRef.current = null;
  }

  const nodeList = useMemo(() => {
    // Ensure all vars have a position
    return vars.map((v) => {
      const p = pos[v] ?? { x: 100, y: 100 };
      return { id: v, x: p.x, y: p.y };
    });
  }, [vars, pos]);

  // ---------- UI ----------
  return (
    <div className="min-h-screen bg-gray-100 text-slate-900">
      <div className="max-w-7xl mx-auto px-4 py-6">
        {/* Header */}
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div className="flex-1 text-center md:text-left space-y-1">
            <div className="flex items-center justify-center md:justify-start gap-2">
              <Network className="w-5 h-5 text-indigo-600" />
              <h1 className="text-2xl font-bold text-slate-800">
                Expand the MESEM analysis of So, Yang and Li (2025)
              </h1>
            </div>
            <p className="text-sm text-slate-600">Programmed by Dr. Yang Yang, Temple University.</p>
          </div>

          <div className="flex items-center justify-center md:justify-end gap-3">
            <div className="flex items-center gap-2 text-sm">
              <span className="text-slate-600">Sample</span>
              <select
                value={sampleType}
                onChange={(e) => setSampleType(e.target.value as SampleType)}
                className="rounded-lg border border-slate-300 bg-white px-2 py-1.5 text-sm shadow-sm"
              >
                <option value="All">All</option>
                <option value="Lodging">Lodging</option>
                <option value="Restaurant">Restaurant</option>
                <option value="Tourism and travel">Tourism and travel</option>
              </select>
            </div>

            <div className="flex items-center gap-2 text-sm">
              <span className="text-slate-600">Total N</span>
              <select
                value={totalNMode}
                onChange={(e) => setTotalNMode(e.target.value as any)}
                className="rounded-lg border border-slate-300 bg-white px-2 py-1.5 text-sm shadow-sm"
              >
                <option value="harmonic">Harmonic mean</option>
                <option value="min">Minimum</option>
              </select>
            </div>

            <a
              href="/MESEM_User_Manual.pdf"
              download
              className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm text-slate-700 shadow-sm hover:bg-slate-50"
              title="Download User Manual (PDF)"
            >
              <Download className="w-4 h-4" />
              User manual
            </a>
          </div>
        </div>

        {/* Alerts */}
        {(matrixErrors.length > 0 || matrixWarnings.length > 0 || estError) && (
          <div className="mt-4 space-y-2">
            {matrixErrors.length > 0 && (
              <div className="rounded-xl border border-rose-200 bg-rose-50 p-3 text-sm text-rose-900">
                <div className="flex items-center gap-2 font-semibold">
                  <AlertTriangle className="w-4 h-4" /> Errors
                </div>
                <ul className="mt-2 list-disc pl-5 space-y-1">
                  {matrixErrors.map((x, i) => (
                    <li key={i}>{x}</li>
                  ))}
                </ul>
              </div>
            )}

            {matrixWarnings.length > 0 && (
              <div className="rounded-xl border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
                <div className="flex items-center gap-2 font-semibold">
                  <Info className="w-4 h-4" /> Warnings
                </div>
                <ul className="mt-2 list-disc pl-5 space-y-1">
                  {matrixWarnings.map((x, i) => (
                    <li key={i}>{x}</li>
                  ))}
                </ul>
              </div>
            )}

            {estError && (
              <div className="rounded-xl border border-rose-200 bg-rose-50 p-3 text-sm text-rose-900">
                <div className="flex items-center gap-2 font-semibold">
                  <AlertTriangle className="w-4 h-4" /> Estimation failed
                </div>
                <div className="mt-1">{estError}</div>
              </div>
            )}
          </div>
        )}

        <div className="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left column */}
          <div className="space-y-6">
            {/* Variables card */}
            <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
              <div className="flex items-center justify-between">
                <div className="font-semibold text-slate-800 flex items-center gap-2">
                  <Grid3X3 className="w-4 h-4 text-slate-500" />
                  Variables
                </div>
                <div className="text-xs text-slate-500">Total variables: {totalVars}</div>
              </div>

              <p className="mt-2 text-xs text-slate-500">
                Base variables are fixed. You may add up to <b>{MAX_NEW_VARS}</b> new variables.
              </p>

              <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-3">
                <div>
                  <label className="text-xs font-medium text-slate-700"># new variables (0–{MAX_NEW_VARS})</label>
                  <input
                    type="number"
                    min={0}
                    max={MAX_NEW_VARS}
                    value={newVarCount}
                    onChange={(e) => setNewVarCount(Number(e.target.value))}
                    className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm"
                  />
                </div>

                <div className="md:col-span-2">
                  <label className="text-xs font-medium text-slate-700">New variable names (one per line)</label>
                  <textarea
                    value={newVarNames}
                    onChange={(e) => setNewVarNames(e.target.value)}
                    rows={3}
                    placeholder="e.g.\ntrust\nengagement\n..."
                    className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm"
                  />
                </div>
              </div>

              <div className="mt-3 flex flex-wrap gap-2 items-center">
                <button
                  onClick={applyVariables}
                  className="rounded-lg bg-indigo-600 px-3 py-2 text-sm font-medium text-white hover:bg-indigo-700"
                >
                  Apply variables
                </button>
                <button
                  onClick={resetToBase4}
                  className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 hover:bg-slate-50"
                >
                  Reset to base 4
                </button>

                <div className="ml-auto flex flex-wrap gap-2">
                  {vars.map((v) => (
                    <span
                      key={v}
                      className={`rounded-full border px-2 py-1 text-xs ${
                        (BASE_VARS as readonly string[]).includes(v)
                          ? "border-slate-200 bg-slate-100 text-slate-700"
                          : "border-indigo-200 bg-indigo-50 text-indigo-700"
                      }`}
                    >
                      {v}
                    </span>
                  ))}
                </div>
              </div>

              <div className="mt-3 text-xs text-slate-500 flex items-center gap-2">
                <MousePointerClick className="w-4 h-4" />
                Tip: In the diagram, click node A then node B to add A→B. Click again to remove.
              </div>
            </div>

            {/* Matrix card */}
            <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
              <div className="flex items-center justify-between">
                <div className="font-semibold text-slate-800 flex items-center gap-2">
                  <Grid3X3 className="w-4 h-4 text-slate-500" />
                  Matrix (r + pairwise N)
                </div>
                <div className="inline-flex rounded-lg border border-slate-200 bg-slate-50 p-1 text-xs">
                  <button
                    onClick={() => setViewMode("text")}
                    className={`rounded-md px-2 py-1 ${viewMode === "text" ? "bg-white shadow-sm" : "text-slate-600"}`}
                  >
                    Text
                  </button>
                  <button
                    onClick={() => setViewMode("grid")}
                    className={`rounded-md px-2 py-1 ${viewMode === "grid" ? "bg-white shadow-sm" : "text-slate-600"}`}
                  >
                    Grid
                  </button>
                </div>
              </div>

              <p className="mt-2 text-xs text-slate-500">
                Symmetry is enforced. Header row/first column are sticky for easier scrolling.
              </p>

              {viewMode === "grid" ? (
                <div className="mt-4 overflow-x-auto max-h-[420px] overflow-y-auto rounded-lg border border-slate-200">
                  <table className="text-xs border-collapse min-w-max w-full">
                    <thead>
                      <tr className="border-b border-slate-200">
                        <th className="sticky top-0 z-20 bg-white p-2 text-left text-slate-600"></th>
                        {vars.map((v) => (
                          <th
                            key={v}
                            className="sticky top-0 z-10 bg-white p-2 text-left text-slate-600 whitespace-nowrap"
                          >
                            {v}
                          </th>
                        ))}
                      </tr>
                    </thead>

                    <tbody>
                      {vars.map((r) => (
                        <tr key={r} className="border-b border-slate-100">
                          <td className="sticky left-0 z-10 bg-white p-2 font-semibold text-slate-700 whitespace-nowrap border-r border-slate-100">
                            {r}
                          </td>

                          {vars.map((c) => {
                            const cell = cellM[r]?.[c] ?? { r: Number.NaN, n: Number.NaN };
                            const isDiag = r === c;
                            return (
                              <td key={`${r}|${c}`} className="p-2 align-top">
                                <div
                                  className={`border border-slate-200 rounded-lg p-1.5 ${cellBgForR(cell.r)} ${
                                    isDiag ? "bg-slate-50" : ""
                                  }`}
                                >
                                  {isDiag ? (
                                    <div className="flex items-center justify-center">
                                      <input
                                        value={"1.00"}
                                        disabled
                                        className="w-16 px-2 py-1 border border-slate-200 rounded-md text-xs bg-slate-50 text-slate-500"
                                      />
                                    </div>
                                  ) : (
                                    <div className="space-y-1">
                                      <div className="flex items-center gap-2">
                                        <span className="text-[10px] text-slate-500 w-4">r</span>
                                        <input
                                          value={Number.isFinite(cell.r) ? String(cell.r) : ""}
                                          onChange={(e) => setCell(r, c, { r: e.target.value === "" ? Number.NaN : Number(e.target.value) })}
                                          className="w-16 px-2 py-1 border border-slate-300 rounded-md text-xs bg-white"
                                          placeholder="0.000"
                                        />
                                      </div>
                                      <div className="flex items-center gap-2">
                                        <span className="text-[10px] text-slate-500 w-4">N</span>
                                        <input
                                          value={Number.isFinite(cell.n) ? String(Math.round(cell.n)) : ""}
                                          onChange={(e) =>
                                            setCell(r, c, { n: e.target.value === "" ? Number.NaN : Number(e.target.value) })
                                          }
                                          className="w-16 px-2 py-1 border border-slate-300 rounded-md text-xs bg-white"
                                          placeholder="N"
                                        />
                                      </div>
                                    </div>
                                  )}
                                </div>
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="mt-4 space-y-3">
                  <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                    <div className="text-xs font-semibold text-slate-700 flex items-center gap-2">
                      <FileText className="w-4 h-4" />
                      Correlation matrix (r)
                    </div>
                    <textarea
                      value={rText}
                      onChange={(e) => setRText(e.target.value)}
                      rows={7}
                      className="mt-2 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs font-mono"
                    />
                  </div>

                  <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                    <div className="text-xs font-semibold text-slate-700 flex items-center gap-2">
                      <FileText className="w-4 h-4" />
                      Sample size matrix (N)
                    </div>
                    <textarea
                      value={nText}
                      onChange={(e) => setNText(e.target.value)}
                      rows={7}
                      className="mt-2 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs font-mono"
                    />
                  </div>

                  <div className="flex items-center gap-2">
                    <button
                      onClick={loadFromText}
                      className="rounded-lg bg-indigo-600 px-3 py-2 text-sm font-medium text-white hover:bg-indigo-700"
                    >
                      Load matrices
                    </button>
                    <button
                      onClick={() => {
                        const d = defaultTextMatrices(sampleType);
                        setRText(d.rText);
                        setNText(d.nText);
                      }}
                      className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 hover:bg-slate-50"
                    >
                      Reset text to default
                    </button>
                    <span className="text-xs text-slate-500">
                      Use tab-separated (recommended) or comma-separated matrices with header row/col.
                    </span>
                  </div>
                </div>
              )}

              <div className="mt-4 flex items-center justify-end gap-2">
                <button
                  onClick={validateMatrix}
                  className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 hover:bg-slate-50"
                >
                  Validate matrix
                </button>
              </div>
            </div>

            {/* Quick actions */}
            <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
              <div className="flex items-center justify-between">
                <div className="font-semibold text-slate-800 flex items-center gap-2">
                  <Info className="w-4 h-4 text-slate-500" />
                  Quick actions
                </div>
                <button
                  onClick={run}
                  className="inline-flex items-center gap-2 rounded-lg bg-indigo-600 px-3 py-2 text-sm font-medium text-white hover:bg-indigo-700"
                >
                  <Play className="w-4 h-4" />
                  Run estimation
                </button>
              </div>
              <p className="mt-2 text-xs text-slate-500">
                This runs validation + estimation and stores results so the diagram can display outputs (demo).
              </p>

              {est && (
                <div className="mt-3 rounded-xl border border-emerald-200 bg-emerald-50 p-3 text-sm text-emerald-900">
                  <div className="flex items-center gap-2 font-semibold">
                    <CheckCircle2 className="w-4 h-4" />
                    Success
                  </div>
                  <div className="mt-1">{est.message}</div>
                </div>
              )}
            </div>
          </div>

          {/* Right column: Diagram + Estimate */}
          <div className="space-y-6">
            {/* Diagram */}
            <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
              <div className="flex items-center justify-between">
                <div className="font-semibold text-slate-800 flex items-center gap-2">
                  <Network className="w-4 h-4 text-slate-500" />
                  Diagram (drag + connect)
                </div>
                <div className="text-xs text-slate-500">Edges: {edges.length}</div>
              </div>

              <p className="mt-2 text-xs text-slate-500">
                Click node A then node B to add A→B. Click the same edge again to remove. Drag nodes to reposition.
              </p>

              <div className="mt-3 rounded-xl border border-slate-200 bg-slate-50 overflow-hidden">
                <svg
                  width="100%"
                  height={svgH}
                  viewBox={`0 0 ${svgW} ${svgH}`}
                  onMouseMove={onMouseMoveCanvas}
                  onMouseUp={onMouseUpCanvas}
                  onMouseLeave={onMouseUpCanvas}
                  className="select-none"
                >
                  <defs>
                    <marker id="arrow" markerWidth="10" markerHeight="7" refX="22" refY="3.5" orient="auto">
                      <polygon points="0 0, 10 3.5, 0 7" fill="#64748b" />
                    </marker>
                  </defs>

                  {/* edges */}
                  {edges.map((e, idx) => {
                    const from = nodeList.find((n) => n.id === e.from);
                    const to = nodeList.find((n) => n.id === e.to);
                    if (!from || !to) return null;

                    // offset for circle radius
                    const R = 22;
                    const dx = to.x - from.x;
                    const dy = to.y - from.y;
                    const dist = Math.max(1, Math.sqrt(dx * dx + dy * dy));
                    const ux = dx / dist;
                    const uy = dy / dist;

                    const x1 = from.x + ux * R;
                    const y1 = from.y + uy * R;
                    const x2 = to.x - ux * R;
                    const y2 = to.y - uy * R;

                    // label = r
                    const cell = cellM[e.from]?.[e.to];
                    const label = cell && Number.isFinite(cell.r) ? cell.r.toFixed(3) : "NA";

                    const lx = (x1 + x2) / 2;
                    const ly = (y1 + y2) / 2;

                    return (
                      <g key={idx}>
                        <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="#64748b" strokeWidth={2} markerEnd="url(#arrow)" />
                        <rect x={lx - 16} y={ly - 10} width={32} height={18} rx={6} fill="#ffffff" stroke="#e2e8f0" />
                        <text x={lx} y={ly + 3} fontSize={10} textAnchor="middle" fill="#334155" fontWeight="bold">
                          {label}
                        </text>
                      </g>
                    );
                  })}

                  {/* nodes */}
                  {nodeList.map((n) => {
                    const isSelected = selectFrom === n.id;
                    return (
                      <g
                        key={n.id}
                        onMouseDown={(e) => onMouseDownNode(e, n.id)}
                        onClick={() => onNodeClick(n.id)}
                        style={{ cursor: "pointer" }}
                      >
                        <circle
                          cx={n.x}
                          cy={n.y}
                          r={22}
                          fill={isSelected ? "#fde68a" : "#e0e7ff"}
                          stroke={isSelected ? "#ca8a04" : "#64748b"}
                          strokeWidth={2}
                        />
                        <text x={n.x} y={n.y + 3} fontSize={11} textAnchor="middle" fill="#0f172a" fontWeight="bold">
                          {n.id}
                        </text>
                      </g>
                    );
                  })}
                </svg>
              </div>

              <div className="mt-3 flex items-center justify-between">
                <button
                  onClick={clearEdges}
                  className="inline-flex items-center gap-2 text-sm text-rose-700 hover:text-rose-900"
                >
                  <Trash2 className="w-4 h-4" />
                  Clear edges
                </button>
                <div className="text-xs text-slate-500">Selected: {selectFrom ?? "none"}</div>
              </div>
            </div>

            {/* Estimate model */}
            <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
              <div className="font-semibold text-slate-800">Estimate model</div>
              <p className="mt-2 text-sm text-slate-600">
                Total N is derived from pairwise N using <b>{totalNMode === "harmonic" ? "harmonic mean" : "minimum"}</b>.
              </p>

              <div className="mt-4 flex items-center gap-2">
                <button
                  onClick={run}
                  className="inline-flex items-center gap-2 rounded-lg bg-indigo-600 px-3 py-2 text-sm font-medium text-white hover:bg-indigo-700"
                >
                  <Play className="w-4 h-4" />
                  Run estimation
                </button>
                <button
                  onClick={validateMatrix}
                  className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 hover:bg-slate-50"
                >
                  Validate matrix
                </button>
              </div>

              <p className="mt-3 text-xs text-slate-500">
                Tip: create at least one arrow, fill the matrix (including pairwise N), then click Run estimation.
              </p>
            </div>
          </div>
        </div>

        <div className="mt-10 text-center text-xs text-slate-500">
          © {new Date().getFullYear()} — Path Model Builder. For research/teaching use.
        </div>
      </div>
    </div>
  );
}
