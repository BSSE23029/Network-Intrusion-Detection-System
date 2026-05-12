import { useState } from "react"

import Navbar from "../components/Navbar"
import UploadBox from "../components/UploadBox"
import ResultTable from "../components/ResultTable"
import StatsCards from "../components/StatsCards"
import Charts from "../components/Charts"
import { predictFile } from "../services/api"

function Dashboard() {

  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleUpload = async (file) => {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const res = await predictFile(file)
      setResult(res.data)
    } catch (err) {
      const detail = err?.response?.data?.detail || err?.message || "Prediction failed"
      setError(detail)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="dashboard-shell">
      <div className="dashboard-container">
        <Navbar />

        <div className="hero-block">
          <h1 className="hero-title">Cyber Threat Detection Dashboard</h1>
          <p className="hero-subtitle">
            Real-time AI powered intrusion monitoring with confident threat classification.
          </p>
        </div>

        <div className="panel upload-panel">
          <UploadBox onUpload={handleUpload} />
        </div>

        {loading && (
          <div className="panel status-panel">
            <p className="status-text">Analyzing traffic, please wait...</p>
          </div>
        )}

        {error && !loading && (
          <div className="panel error-banner">
            <p className="status-text">Error: {error}</p>
          </div>
        )}

        {result && !loading && !error && (
          <div className="results-wrap">
            <StatsCards summary={result.summary} />
            <Charts summary={result.summary} />
            <ResultTable predictions={result.predictions} />
          </div>
        )}
      </div>
    </div>
  )
}

export default Dashboard
