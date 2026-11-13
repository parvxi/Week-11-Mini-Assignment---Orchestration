/** Final evaluation panel for the forecasting agent workflow: holdout metrics + leakage audit. */

export type HoldoutMetrics = {
  f1?: number
  precision?: number
  recall?: number
  accuracy?: number
  roc_auc?: number | null
  pr_auc?: number | null
  threshold?: number
  n_holdout?: number
  positive_rate?: number
  majority_class_accuracy?: number
  n_features?: number
  model_family?: string
  holdout_period?: string
  persistence_baseline?: { f1: number; precision: number; recall: number }
  confusion_matrix?: { tn: number; fp: number; fn: number; tp: number }
  leakage_ablation?: {
    columns_restored?: string[]
    f1?: number
    roc_auc?: number | null
    accuracy?: number
  }
}

export type LeakageAudit = {
  findings?: { column: string; reason: string; evidence: string }[]
  leak_columns?: string[]
  unconfirmed_suspects?: string[]
  strongest_legit_features?: { column: string; single_feature_auc: number }[]
  n_columns_checked?: number
}

const fmt = (v: number | null | undefined, digits = 3) =>
  typeof v === 'number' ? v.toFixed(digits) : '—'

function Metric({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="border border-line bg-ink-2/60 px-3 py-2">
      <p className="font-mono text-[10px] uppercase tracking-wider text-mist">{label}</p>
      <p className="mt-0.5 font-display text-2xl text-paper">{value}</p>
      {hint && <p className="text-[11px] leading-snug text-mist">{hint}</p>}
    </div>
  )
}

const REASON_LABEL: Record<string, string> = {
  target_leak: 'target leak',
  declared_outcome: 'next-day outcome',
  identifier: 'identifier',
  raw_date: 'raw date',
  high_cardinality: 'high cardinality',
}

export function HoldoutReport({
  metrics,
  audit,
  stopReason,
}: {
  metrics: HoldoutMetrics | null
  audit: LeakageAudit | null
  stopReason?: string | null
}) {
  if (!metrics && !audit) return null
  const cm = metrics?.confusion_matrix
  const ablation = metrics?.leakage_ablation
  const findings = audit?.findings ?? []

  return (
    <section className="border border-line bg-panel/40">
      <div className="border-b border-line px-3 py-2">
        <h2 className="font-display text-xl text-paper">Evaluation results</h2>
        <p className="text-xs text-mist">
          Sealed holdout{metrics?.holdout_period ? ` (${metrics.holdout_period})` : ''}, scored once
          {metrics?.model_family ? ` · ${metrics.model_family}` : ''}
          {typeof metrics?.n_holdout === 'number' ? ` · ${metrics.n_holdout.toLocaleString()} rows` : ''}
          {stopReason ? ` · stop reason: ${stopReason}` : ''}
        </p>
      </div>

      {metrics && (
        <div className="grid gap-4 p-3 lg:grid-cols-[1.4fr_1fr]">
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
            <Metric label="F1 (low-vis day)" value={fmt(metrics.f1)} hint={`threshold ${fmt(metrics.threshold, 2)}`} />
            <Metric label="Precision" value={fmt(metrics.precision)} />
            <Metric label="Recall" value={fmt(metrics.recall)} />
            <Metric label="ROC-AUC" value={fmt(metrics.roc_auc)} hint="0.5 = coin flip" />
            <Metric
              label="PR-AUC"
              value={fmt(metrics.pr_auc)}
              hint={`event rate ${fmt(metrics.positive_rate)}`}
            />
            <Metric
              label="Accuracy"
              value={fmt(metrics.accuracy)}
              hint={`always "clear" scores ${fmt(metrics.majority_class_accuracy)}`}
            />
          </div>

          {cm && (
            <div>
              <p className="mb-1 font-mono text-[10px] uppercase tracking-wider text-mist">
                Confusion matrix (holdout)
              </p>
              <table className="w-full border-collapse text-center text-sm">
                <thead>
                  <tr className="text-[11px] text-mist">
                    <th className="p-1" />
                    <th className="p-1 font-normal">predicted clear</th>
                    <th className="p-1 font-normal">predicted low-vis</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <th className="p-1 text-right text-[11px] font-normal text-mist">actual clear</th>
                    <td className="border border-line bg-ok/10 p-2 text-paper">{cm.tn.toLocaleString()}</td>
                    <td className="border border-line bg-danger/10 p-2 text-paper">{cm.fp.toLocaleString()}</td>
                  </tr>
                  <tr>
                    <th className="p-1 text-right text-[11px] font-normal text-mist">actual low-vis</th>
                    <td className="border border-line bg-danger/10 p-2 text-paper">{cm.fn.toLocaleString()}</td>
                    <td className="border border-line bg-ok/10 p-2 text-paper">{cm.tp.toLocaleString()}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {metrics?.persistence_baseline && (
        <p className="mx-3 mb-3 border border-line bg-ink-2/60 px-3 py-2 text-sm text-paper">
          <span className="font-medium">Persistence baseline</span> (tomorrow = today) on the same
          holdout: F1 {fmt(metrics.persistence_baseline.f1)}, precision{' '}
          {fmt(metrics.persistence_baseline.precision)}, recall {fmt(metrics.persistence_baseline.recall)}.
          Model skill over persistence:{' '}
          <span className="font-medium">
            {typeof metrics.f1 === 'number'
              ? `${metrics.f1 - metrics.persistence_baseline.f1 >= 0 ? '+' : ''}${(metrics.f1 - metrics.persistence_baseline.f1).toFixed(3)} F1`
              : '—'}
          </span>
        </p>
      )}

      {ablation && (
        <p className="mx-3 mb-3 border border-danger/40 bg-danger/10 px-3 py-2 text-sm text-paper">
          <span className="font-medium">Leakage ablation:</span> the same model with{' '}
          <code>{(ablation.columns_restored ?? []).join(', ')}</code> let back in scores F1{' '}
          {fmt(ablation.f1)} / ROC-AUC {fmt(ablation.roc_auc)}. Those columns hold
          tomorrow's outcome, so the honest score is the one above.
        </p>
      )}

      {audit && (
        <div className="border-t border-line p-3">
          <p className="mb-2 font-mono text-[10px] uppercase tracking-wider text-mist">
            Leakage audit · {audit.n_columns_checked ?? '—'} columns measured · {findings.length} blocked
          </p>
          <table className="w-full text-left text-xs">
            <tbody>
              {findings.map((f) => (
                <tr key={f.column} className="border-t border-line/60">
                  <td className="py-1 pr-3 font-mono text-paper">{f.column}</td>
                  <td
                    className={`py-1 pr-3 ${f.reason === 'target_leak' || f.reason === 'declared_outcome' ? 'font-medium text-danger' : 'text-mist'}`}
                  >
                    {REASON_LABEL[f.reason] ?? f.reason}
                  </td>
                  <td className="py-1 text-mist">{f.evidence}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {(audit.strongest_legit_features?.length ?? 0) > 0 && (
            <p className="mt-2 text-xs text-mist">
              Strongest remaining columns (single-feature ROC-AUC):{' '}
              {audit.strongest_legit_features!
                .map((f) => `${f.column} ${f.single_feature_auc.toFixed(3)}`)
                .join(' · ')}
            </p>
          )}
          {(audit.unconfirmed_suspects?.length ?? 0) > 0 && (
            <p className="mt-1 text-xs text-mist">
              Suspected by the Data Scientist but not confirmed by measurement (kept as features):{' '}
              {audit.unconfirmed_suspects!.join(', ')}
            </p>
          )}
        </div>
      )}
    </section>
  )
}
