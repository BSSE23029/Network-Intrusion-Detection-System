import { useState } from "react"

function UploadBox({ onUpload }) {
  const [file, setFile] = useState(null)

  const handleSubmit = (e) => {
    e.preventDefault()
    if (!file) {
      return
    }
    onUpload?.(file)
  }

  const handleClear = () => {
    setFile(null)
  }

  return (
    <form onSubmit={handleSubmit} className="upload-form">
      <label className="upload-label">
        Upload Network Traffic File
      </label>

      <div className="upload-dropzone">
        <input
          className="upload-input"
          type="file"
          accept=".csv,.pcap,.txt"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
        <p className="file-meta">
          {file ? `Selected: ${file.name}` : "Drop a CSV/PCAP file or choose from device"}
        </p>
      </div>

      <div className="upload-actions">
        <button type="submit" className="btn-primary">Analyze</button>
        <button type="button" className="btn-secondary" onClick={handleClear}>Clear</button>
      </div>
    </form>
  )
}

export default UploadBox