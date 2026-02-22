import { useState, useEffect, useCallback } from 'react'
import apiClient from '../api/client'

interface ParamChangeInfo {
  param_name: string
  old_value: number | string
  new_value: number | string
  reason: string
}

interface ResearchReport {
  experiment_name: string
  hypothesis: string
  methodology: string
  data_period: string
  results: Record<string, unknown>
  conclusion: string
  recommended_changes: ParamChangeInfo[]
  improvement_significant: boolean
  timestamp: string
}

interface ExperimentInfo {
  name: string
  target_engine: string
  status: string
}

function ExperimentCard({ exp }: { exp: ExperimentInfo }) {
  return (
    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
      <div className="flex items-center justify-between">
        <div>
          <h4 className="font-semibold">{exp.name}</h4>
          <p className="text-xs text-gray-500 mt-1">
            Target: {exp.target_engine}
          </p>
        </div>
        <span className="text-xs px-2 py-1 rounded bg-gray-700 text-gray-300">
          {exp.status}
        </span>
      </div>
    </div>
  )
}

function ReportCard({ report }: { report: ResearchReport }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div
      className={`bg-gray-800 rounded-lg p-4 border ${
        report.improvement_significant
          ? 'border-green-700'
          : 'border-gray-700'
      }`}
    >
      <div
        className="flex items-center justify-between cursor-pointer"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-2">
          <span
            className={`w-2 h-2 rounded-full ${
              report.improvement_significant
                ? 'bg-green-500'
                : 'bg-gray-500'
            }`}
          />
          <h4 className="font-semibold">{report.experiment_name}</h4>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs text-gray-500">
            {new Date(report.timestamp).toLocaleString()}
          </span>
          <span className="text-gray-400 text-sm">
            {expanded ? '\u25B2' : '\u25BC'}
          </span>
        </div>
      </div>

      {expanded && (
        <div className="mt-4 space-y-3 text-sm">
          <div>
            <p className="text-xs text-gray-500 mb-1">Hypothesis</p>
            <p className="text-gray-300">{report.hypothesis}</p>
          </div>
          <div>
            <p className="text-xs text-gray-500 mb-1">Methodology</p>
            <p className="text-gray-300">{report.methodology}</p>
          </div>
          <div>
            <p className="text-xs text-gray-500 mb-1">Data Period</p>
            <p className="text-gray-300">{report.data_period}</p>
          </div>
          <div>
            <p className="text-xs text-gray-500 mb-1">Results</p>
            <div className="bg-gray-900 rounded p-3 font-mono text-xs">
              {Object.entries(report.results).map(([key, val]) => (
                <div key={key} className="flex justify-between py-0.5">
                  <span className="text-gray-400">{key}</span>
                  <span className="text-gray-200">
                    {typeof val === 'object'
                      ? JSON.stringify(val)
                      : String(val)}
                  </span>
                </div>
              ))}
            </div>
          </div>
          <div>
            <p className="text-xs text-gray-500 mb-1">Conclusion</p>
            <p
              className={
                report.improvement_significant
                  ? 'text-green-400'
                  : 'text-gray-300'
              }
            >
              {report.conclusion}
            </p>
          </div>
          {report.recommended_changes.length > 0 && (
            <div>
              <p className="text-xs text-gray-500 mb-1">
                Recommended Changes
              </p>
              <div className="space-y-1">
                {report.recommended_changes.map((c, i) => (
                  <div
                    key={i}
                    className="bg-gray-900 rounded p-2 text-xs font-mono"
                  >
                    <span className="text-gray-400">{c.param_name}: </span>
                    <span className="text-red-400">
                      {String(c.old_value)}
                    </span>
                    <span className="text-gray-500"> {'\u2192'} </span>
                    <span className="text-green-400">
                      {String(c.new_value)}
                    </span>
                    <p className="text-gray-500 mt-1">{c.reason}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function Research() {
  const [experiments, setExperiments] = useState<ExperimentInfo[]>([])
  const [reports, setReports] = useState<ResearchReport[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchData = useCallback(async () => {
    try {
      const [expRes, repRes] = await Promise.all([
        apiClient.get<{ experiments: ExperimentInfo[] }>(
          '/research/experiments'
        ),
        apiClient.get<{ reports: ResearchReport[] }>('/research/reports'),
      ])
      setExperiments(expRes.data.experiments)
      setReports(repRes.data.reports)
      setError(null)
    } catch (err: unknown) {
      if (err && typeof err === 'object' && 'response' in err) {
        const axiosErr = err as { response?: { status?: number } }
        if (axiosErr.response?.status === 503) {
          setError('Engine mode is not enabled.')
        } else {
          setError('Failed to fetch research data')
        }
      } else {
        setError('Failed to fetch research data')
      }
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, 30000)
    return () => clearInterval(interval)
  }, [fetchData])

  if (loading) {
    return (
      <div className="space-y-6">
        <h2 className="text-2xl font-bold">Research</h2>
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700 animate-pulse h-32" />
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700 animate-pulse h-64" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="space-y-6">
        <h2 className="text-2xl font-bold">Research</h2>
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <p className="text-yellow-400">{error}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Research</h2>

      {/* Experiments */}
      <div>
        <h3 className="text-lg font-semibold mb-3">
          Registered Experiments
        </h3>
        {experiments.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {experiments.map((exp) => (
              <ExperimentCard key={exp.name} exp={exp} />
            ))}
          </div>
        ) : (
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700 text-center text-gray-400">
            No experiments registered yet.
          </div>
        )}
      </div>

      {/* Reports */}
      <div>
        <h3 className="text-lg font-semibold mb-3">Research Reports</h3>
        {reports.length > 0 ? (
          <div className="space-y-3">
            {reports.map((report, i) => (
              <ReportCard key={`${report.experiment_name}-${i}`} report={report} />
            ))}
          </div>
        ) : (
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700 text-center text-gray-400">
            No research reports available yet. Reports will appear after
            the first research cycle runs.
          </div>
        )}
      </div>
    </div>
  )
}

export default Research
