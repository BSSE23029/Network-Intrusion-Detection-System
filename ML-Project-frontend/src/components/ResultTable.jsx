const MAX_VISIBLE_ROWS = 100

function ResultTable({ predictions }) {

  if (!predictions || predictions.length === 0) {
    return null
  }

  const total = predictions.length
  const rows = predictions.slice(0, MAX_VISIBLE_ROWS)

  const riskClass = (level) => {
    if (level === "HIGH") return "badge badge-attack"
    if (level === "MEDIUM") return "badge badge-medium"
    return "badge badge-normal"
  }

  const statusClass = (status) => {
    if (status === "Attack") return "badge badge-attack"
    if (status === "Uncertain") return "badge badge-uncertain"
    return "badge badge-normal"
  }

  return (
    <div className="panel table-panel">
      <h2 className="panel-title">Detection Results</h2>
      <table className="result-table">
        <thead>
          <tr>
            <th>Traffic ID</th>
            <th>Status</th>
            <th title="Model certainty in the predicted label">Confidence</th>
            <th title="Bucketed attack probability">Risk Level</th>
          </tr>
        </thead>

        <tbody>
          {rows.map((row) => (
            <tr key={row.id}>
              <td>{row.id}</td>
              <td>
                <span className={statusClass(row.status)}>
                  {row.status}
                </span>
              </td>
              <td>{row.confidence}%</td>
              <td>
                <span className={riskClass(row.risk_level)}>
                  {row.risk_level}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {total > MAX_VISIBLE_ROWS && (
        <p className="table-footnote">
          Showing {MAX_VISIBLE_ROWS} of {total.toLocaleString()} records.
        </p>
      )}
    </div>
  )
}

export default ResultTable
