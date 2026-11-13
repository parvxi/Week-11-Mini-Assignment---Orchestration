/**
 * Live-graph config for the dust-visibility forecasting agent workflow (student_loop/graph.py).
 *
 * Compared with the class graph (simpleLoopConfig.ts):
 *   + leakage_auditor, baseline_benchmark, threshold_tuner, train_router   (new)
 *   ~ ml_engineer, judge, memory_writer, holdout_eval  (modified)
 *   - context_builder                                  (removed)
 */

export type LoopNodeKind = 'file' | 'llm' | 'code' | 'code_llm' | 'store'

export interface LoopNode {
  id: string
  title: string
  role: string
  kind: LoopNodeKind
  service: string
  /** Shown as a small tag so the diagram itself documents what changed. */
  badge?: 'new' | 'modified'
}

export interface LoopEdge {
  id: string
  source: string
  target: string
  kind: 'control' | 'data'
  label?: string
  sourceHandle?: string
  targetHandle?: string
}

export interface LoopGraphConfig {
  nodes: LoopNode[]
  edges: LoopEdge[]
  positions: Record<string, { x: number; y: number }>
  subtitle?: string
}

const NODES: LoopNode[] = [
  {
    id: 'dataset',
    title: 'saudi_dust_daily_latest.csv',
    role: 'Station-days exported from Postgres',
    kind: 'file',
    service: 'postgres export',
  },
  {
    id: 'data_scientist',
    title: 'Data Scientist',
    role: 'Profile table · name suspects',
    kind: 'llm',
    service: 'llm agent',
  },
  {
    id: 'leakage_auditor',
    title: 'Leakage Auditor',
    role: 'Measures every column · owns the drop list',
    kind: 'code',
    service: 'code agent',
    badge: 'new',
  },
  {
    id: 'baseline_benchmark',
    title: 'Baseline Benchmark',
    role: 'Persistence forecast: tomorrow = today',
    kind: 'code',
    service: 'code agent',
    badge: 'new',
  },
  {
    id: 'hypothesis_generator',
    title: 'Hypothesis Generator',
    role: 'Conceptual next step (no HPT)',
    kind: 'llm',
    service: 'llm agent',
  },
  {
    id: 'ml_engineer',
    title: 'ML Engineer',
    role: 'Family + hyperparameters · train · self-repair',
    kind: 'code_llm',
    service: 'llm + sklearn',
    badge: 'modified',
  },
  {
    id: 'train_router',
    title: 'Training router',
    role: 'trained → tune · failed → judge',
    kind: 'code',
    service: 'langgraph edge',
    badge: 'new',
  },
  {
    id: 'threshold_tuner',
    title: 'Threshold Tuner',
    role: 'Decision cut-off for rare events',
    kind: 'code',
    service: 'code agent',
    badge: 'new',
  },
  {
    id: 'judge',
    title: 'Judge',
    role: 'Verdict, skill vs persistence, guardrails',
    kind: 'code_llm',
    service: 'llm + rules',
    badge: 'modified',
  },
  {
    id: 'router',
    title: 'Router',
    role: 'accept · revise · stop',
    kind: 'code',
    service: 'langgraph edge',
  },
  {
    id: 'memory_writer',
    title: 'Memory writer',
    role: 'Files lesson + writes next brief',
    kind: 'code',
    service: 'code agent',
    badge: 'modified',
  },
  {
    id: 'research_memory',
    title: 'Research memory',
    role: 'history · lessons · brief',
    kind: 'store',
    service: 'graph state',
  },
  {
    id: 'best_candidate',
    title: 'Best candidate',
    role: 'Best spec + threshold on validation',
    kind: 'store',
    service: 'graph state',
  },
  {
    id: 'holdout_eval',
    title: 'Final holdout',
    role: 'Future days · full metrics · ablation',
    kind: 'code',
    service: 'code agent',
    badge: 'modified',
  },
]

const EDGES: LoopEdge[] = [
  { id: 'e-csv-ds', source: 'dataset', target: 'data_scientist', kind: 'data', label: 'csv', sourceHandle: 'right-out', targetHandle: 'left-in' },
  { id: 'e-ds-audit', source: 'data_scientist', target: 'leakage_auditor', kind: 'control', label: 'suspects', sourceHandle: 'right-out', targetHandle: 'left-in' },
  { id: 'e-audit-bench', source: 'leakage_auditor', target: 'baseline_benchmark', kind: 'control', label: 'drop list', sourceHandle: 'right-out', targetHandle: 'left-in' },
  { id: 'e-bench-hyp', source: 'baseline_benchmark', target: 'hypothesis_generator', kind: 'control', label: 'floor', sourceHandle: 'right-out', targetHandle: 'left-in' },
  { id: 'e-hyp-ml', source: 'hypothesis_generator', target: 'ml_engineer', kind: 'control', sourceHandle: 'right-out', targetHandle: 'left-in' },
  { id: 'e-ml-trouter', source: 'ml_engineer', target: 'train_router', kind: 'control', sourceHandle: 'right-out', targetHandle: 'left-in' },
  { id: 'e-trouter-tune', source: 'train_router', target: 'threshold_tuner', kind: 'control', label: 'trained', sourceHandle: 'right-out', targetHandle: 'left-in' },
  { id: 'e-trouter-judge', source: 'train_router', target: 'judge', kind: 'control', label: 'failed', sourceHandle: 'top-out', targetHandle: 'top-in' },
  { id: 'e-tune-judge', source: 'threshold_tuner', target: 'judge', kind: 'control', sourceHandle: 'right-out', targetHandle: 'left-in' },
  { id: 'e-judge-router', source: 'judge', target: 'router', kind: 'control', sourceHandle: 'bottom-out', targetHandle: 'top-in' },
  { id: 'e-router-mem', source: 'router', target: 'memory_writer', kind: 'control', label: 'revise', sourceHandle: 'left-out', targetHandle: 'right-in' },
  { id: 'e-mem-store', source: 'memory_writer', target: 'research_memory', kind: 'data', sourceHandle: 'left-out', targetHandle: 'right-in' },
  { id: 'e-store-hyp', source: 'research_memory', target: 'hypothesis_generator', kind: 'control', label: 'brief', sourceHandle: 'top-out', targetHandle: 'bottom-in' },
  { id: 'e-router-best', source: 'router', target: 'best_candidate', kind: 'control', label: 'accept / stop', sourceHandle: 'bottom-out', targetHandle: 'top-in' },
  { id: 'e-best-hold', source: 'best_candidate', target: 'holdout_eval', kind: 'control', sourceHandle: 'left-out', targetHandle: 'right-in' },
]

/**
 * Grid: x step 260, rows at y = 60 / 280 / 500.
 *   top:    dataset · scientist · auditor · hypothesis · engineer · train router · tuner · judge
 *   mid:                                    memory     · writer                            router
 *   bottom:                                                                      holdout · best
 */
const POSITIONS: Record<string, { x: number; y: number }> = {
  dataset: { x: 0, y: 60 },
  data_scientist: { x: 260, y: 60 },
  leakage_auditor: { x: 520, y: 60 },
  baseline_benchmark: { x: 780, y: 60 },
  hypothesis_generator: { x: 1040, y: 60 },
  ml_engineer: { x: 1300, y: 60 },
  train_router: { x: 1560, y: 60 },
  threshold_tuner: { x: 1820, y: 60 },
  judge: { x: 2080, y: 60 },
  router: { x: 2080, y: 280 },
  memory_writer: { x: 1300, y: 280 },
  research_memory: { x: 1040, y: 280 },
  best_candidate: { x: 2080, y: 500 },
  holdout_eval: { x: 1820, y: 500 },
}

export const STUDENT_LOOP_CONFIG: LoopGraphConfig = {
  nodes: NODES,
  edges: EDGES,
  positions: POSITIONS,
  subtitle: 'Dust-visibility forecast workflow · nodes and edges read from the compiled LangGraph',
}

export const STUDENT_LOOP_NODES = NODES

/** LangGraph node name → box on the diagram (identity for real nodes). */
export function visualNodeId(langGraphNode: string): string {
  return langGraphNode
}

type GraphEdge = { source: string; target: string; conditional?: boolean }

/**
 * Boxes to draw for a compiled graph. Routers and stores are not LangGraph nodes,
 * so they appear only when the edges that imply them exist.
 */
export function visualIdsFromGraph(nodes: string[], edges: GraphEdge[]): string[] {
  const ids = new Set<string>(nodes)
  const has = (s: string, t: string) => edges.some((e) => e.source === s && e.target === t)
  if (ids.has('data_scientist')) ids.add('dataset')
  if (has('ml_engineer', 'threshold_tuner') && has('ml_engineer', 'judge')) ids.add('train_router')
  if (has('judge', 'memory_writer') || has('judge', 'holdout_eval')) ids.add('router')
  if (ids.has('memory_writer')) ids.add('research_memory')
  if (ids.has('holdout_eval')) ids.add('best_candidate')
  return Array.from(ids)
}

/** Translate compiled LangGraph edges into `source->target` keys of the diagram. */
export function visualEdgeKeysFromGraph(edges: GraphEdge[]): string[] {
  const keys = new Set<string>()
  const add = (a: string, b: string) => keys.add(`${a}->${b}`)
  const has = (s: string, t: string) => edges.some((e) => e.source === s && e.target === t)
  const trainRouter = has('ml_engineer', 'threshold_tuner') && has('ml_engineer', 'judge')

  for (const { source: s, target: t } of edges) {
    if (s === '__start__') add('dataset', t)
    else if (t === '__end__') continue
    else if (s === 'ml_engineer' && trainRouter) {
      add('ml_engineer', 'train_router')
      add('train_router', t)
    } else if (s === 'judge' && t === 'memory_writer') {
      add('judge', 'router')
      add('router', 'memory_writer')
    } else if (s === 'judge' && t === 'holdout_eval') {
      add('judge', 'router')
      add('router', 'best_candidate')
      add('best_candidate', 'holdout_eval')
    } else if (s === 'memory_writer' && t === 'hypothesis_generator') {
      add('memory_writer', 'research_memory')
      add('research_memory', 'hypothesis_generator')
    } else add(s, t)
  }
  return Array.from(keys)
}
