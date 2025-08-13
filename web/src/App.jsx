import React, { useState } from 'react'
import './App.css'

function App() {
  const [text, setText] = useState('')
  const [maxTokens, setMaxTokens] = useState(256)
  const [temperature, setTemperature] = useState(0)
  const [summary, setSummary] = useState('')
  const [latencyMs, setLatencyMs] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [includeHints, setIncludeHints] = useState(true)
  const [compareMode, setCompareMode] = useState(true)

  const examples = [
    'Patient admitted for pneumonia. Treated with antibiotics and discharged in stable condition.',
    '76-year-old with HTN and OA presents for med check. No fever/chills, no CP. BP stable.'
  ]

  const loadSample = (i) => setText(examples[i])
  const clearAll = () => { setText(''); setSummary(''); setLatencyMs(null); setError(null) }

  const summarize = async () => {
    setLoading(true); setError(null); setSummary('');
    const start = performance.now()
    try {
      const base = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000')
      const effectiveText = includeHints
        ? `Please highlight key diagnoses, medications, and follow-ups.\n\n${text}`
        : text
      const res = await fetch(base + '/summarize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: effectiveText, max_new_tokens: maxTokens, temperature })
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err?.error?.message || res.statusText)
      }
      const data = await res.json()
      setSummary(data.summary || '')
      setLatencyMs(Math.round(performance.now() - start))
      console.log('summarize_ok', { inputChars: effectiveText.length, latencyMs: Math.round(performance.now() - start) })
    } catch (e) {
      setError(e.message || 'Request failed')
      console.warn('summarize_error', e)
    } finally {
      setLoading(false)
    }
  }

  const download = (ext) => {
    if (!summary) return
    const blob = new Blob([summary], { type: ext === 'md' ? 'text/markdown;charset=utf-8' : 'text/plain;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `summary.${ext}`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div style={{ maxWidth: 900, margin: '2rem auto', padding: '0 1rem', fontFamily: 'system-ui, Arial' }}>
      <h1>Clinical Note Summarizer</h1>
      <p style={{ color: '#666' }}>
        Informational only; not medical advice. Do not paste PHI or real patient identifiers.
      </p>

      <div style={{ marginBottom: 8 }}>
        <button onClick={() => loadSample(0)}>Load sample 1</button>
        <button onClick={() => loadSample(1)} style={{ marginLeft: 8 }}>Load sample 2</button>
      </div>

      <div style={{ marginBottom: 12 }}>
        <label>Clinical Note</label>
        <textarea
          rows={10}
          style={{ width: '100%', boxSizing: 'border-box' }}
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
      </div>

      <div style={{ display: 'flex', gap: 16, marginBottom: 12 }}>
        <div>
          <label>Max tokens</label>
          <input type="number" value={maxTokens} min={1} max={1024} onChange={(e) => setMaxTokens(parseInt(e.target.value || '0', 10))} />
        </div>
        <div>
          <label>Temperature</label>
          <input type="number" step="0.1" min={0} max={1} value={temperature} onChange={(e) => setTemperature(parseFloat(e.target.value || '0'))} />
        </div>
        <div>
          <label>Model</label>
          <input value="Service default" readOnly />
        </div>
      </div>

      <div style={{ marginBottom: 12 }}>
        <button onClick={summarize} disabled={loading || !text || text.length < 5}>Summarize</button>
        <button onClick={clearAll} style={{ marginLeft: 8 }}>Clear</button>
        {summary && <button style={{ marginLeft: 8 }} onClick={() => navigator.clipboard.writeText(summary)}>Copy Summary</button>}
        {summary && <button style={{ marginLeft: 8 }} onClick={() => download('txt')}>Download .txt</button>}
        {summary && <button style={{ marginLeft: 8 }} onClick={() => download('md')}>Download .md</button>}
      </div>

      <div style={{ display: 'flex', gap: 16, alignItems: 'center', marginBottom: 8 }}>
        <label><input type="checkbox" checked={includeHints} onChange={(e) => setIncludeHints(e.target.checked)} /> Include prompt hints</label>
        <label style={{ marginLeft: 16 }}><input type="checkbox" checked={compareMode} onChange={(e) => setCompareMode(e.target.checked)} /> Compare mode</label>
      </div>

      {loading && <p>Summarizing...</p>}
      {error && <p style={{ color: 'crimson' }}>{error}</p>}

      {!!summary && (
        <div style={{ marginTop: 16 }}>
          {compareMode ? (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
              <div>
                <h3>Original</h3>
                <pre style={{ whiteSpace: 'pre-wrap', background: '#fafafa', padding: 8 }}>{text}</pre>
              </div>
              <div>
                <h3>Summary</h3>
                <pre style={{ whiteSpace: 'pre-wrap', background: '#fafafa', padding: 8 }}>{summary}</pre>
              </div>
            </div>
          ) : (
            <>
              <h3>Summary</h3>
              <pre style={{ whiteSpace: 'pre-wrap', background: '#fafafa', padding: 8 }}>{summary}</pre>
            </>
          )}
          <small>
            {(() => {
              const inWords = text.trim() ? text.trim().split(/\s+/).length : 0
              const outWords = summary.trim() ? summary.trim().split(/\s+/).length : 0
              const approxTokensIn = Math.round(inWords * 1.3)
              const approxTokensOut = Math.round(outWords * 1.3)
              return `Latency: ${latencyMs ?? '-'} ms | Input: ${inWords} words (~${approxTokensIn} tokens) | Output: ${outWords} words (~${approxTokensOut} tokens)`
            })()}
          </small>
        </div>
      )}
    </div>
  )
}

export default App
