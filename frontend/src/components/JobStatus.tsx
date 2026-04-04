import { useCallback, useEffect, useState } from 'react'
import { fetchJobSummary, fetchJobs } from '../api/client'
import type { Job, JobSummary } from '../types'

const STATUS_COLORS: Record<string, string> = {
  pending: 'text-yellow-400',
  running: 'text-blue-400',
  done: 'text-green-400',
  failed: 'text-red-400',
}

const STATUSES = ['pending', 'running', 'done', 'failed']
const PAGE_SIZE = 50

export function JobStatus() {
  const [summary, setSummary] = useState<JobSummary>({})
  const [jobs, setJobs] = useState<Job[]>([])
  const [total, setTotal] = useState(0)
  const [filter, setFilter] = useState('')
  const [offset, setOffset] = useState(0)

  useEffect(() => {
    fetchJobSummary().then(setSummary).catch(console.error)
  }, [])

  const loadJobs = useCallback(() => {
    fetchJobs(filter || undefined, PAGE_SIZE, offset)
      .then((r) => {
        setJobs(r.jobs)
        setTotal(r.total)
      })
      .catch(console.error)
  }, [filter, offset])

  useEffect(() => {
    loadJobs()
  }, [loadJobs])

  const changeFilter = (s: string) => {
    setFilter(s)
    setOffset(0)
  }

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      {/* Summary cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        {STATUSES.map((s) => (
          <div
            key={s}
            className="bg-gray-900 rounded-lg p-4 border border-gray-800"
          >
            <div className="text-xs text-gray-500 uppercase tracking-wide">{s}</div>
            <div className={`text-3xl font-bold mt-1 ${STATUS_COLORS[s]}`}>
              {summary[s as keyof JobSummary] ?? 0}
            </div>
          </div>
        ))}
      </div>

      {/* Jobs table */}
      <div className="bg-gray-900 rounded-lg border border-gray-800 overflow-hidden">
        {/* Filters */}
        <div className="px-4 py-3 border-b border-gray-800 flex flex-wrap items-center gap-2">
          <span className="text-xs text-gray-500">Status:</span>
          {['', ...STATUSES].map((s) => (
            <button
              key={s || 'all'}
              onClick={() => changeFilter(s)}
              className={`text-xs px-2.5 py-1 rounded-full transition-colors ${
                filter === s
                  ? 'bg-indigo-600 text-white'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              {s || 'All'}
            </button>
          ))}
          <span className="ml-auto text-xs text-gray-500">{total.toLocaleString()} jobs</span>
        </div>

        {/* Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs text-gray-500 border-b border-gray-800">
                <th className="px-4 py-2 font-medium">ID</th>
                <th className="px-4 py-2 font-medium">Symbol</th>
                <th className="px-4 py-2 font-medium">Interval</th>
                <th className="px-4 py-2 font-medium">Month</th>
                <th className="px-4 py-2 font-medium">Status</th>
                <th className="px-4 py-2 font-medium">Completed</th>
                <th className="px-4 py-2 font-medium">Error</th>
              </tr>
            </thead>
            <tbody>
              {jobs.length === 0 && (
                <tr>
                  <td colSpan={7} className="px-4 py-8 text-center text-gray-600 text-xs">
                    No jobs found.
                  </td>
                </tr>
              )}
              {jobs.map((j) => (
                <tr
                  key={j.id}
                  className="border-b border-gray-800/60 hover:bg-gray-800/40 transition-colors"
                >
                  <td className="px-4 py-2 text-gray-500">{j.id}</td>
                  <td className="px-4 py-2">{j.symbol}</td>
                  <td className="px-4 py-2">{j.interval}</td>
                  <td className="px-4 py-2">
                    {j.year}-{String(j.month).padStart(2, '0')}
                  </td>
                  <td className={`px-4 py-2 font-medium ${STATUS_COLORS[j.status] ?? ''}`}>
                    {j.status}
                  </td>
                  <td className="px-4 py-2 text-gray-500 text-xs">
                    {j.completed_at
                      ? new Date(j.completed_at).toLocaleString()
                      : '—'}
                  </td>
                  <td
                    className="px-4 py-2 text-red-400/80 text-xs max-w-xs truncate"
                    title={j.error ?? ''}
                  >
                    {j.error ? j.error.split('\n')[0] : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        <div className="px-4 py-3 flex items-center justify-end gap-3 border-t border-gray-800">
          <button
            disabled={offset === 0}
            onClick={() => setOffset((o) => Math.max(0, o - PAGE_SIZE))}
            className="text-xs px-3 py-1 text-gray-400 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed"
          >
            ← Previous
          </button>
          <span className="text-xs text-gray-500">
            {total === 0 ? '0' : `${offset + 1}–${Math.min(offset + PAGE_SIZE, total)}`} of {total.toLocaleString()}
          </span>
          <button
            disabled={offset + PAGE_SIZE >= total}
            onClick={() => setOffset((o) => o + PAGE_SIZE)}
            className="text-xs px-3 py-1 text-gray-400 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed"
          >
            Next →
          </button>
        </div>
      </div>
    </div>
  )
}
