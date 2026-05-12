import {
  PieChart,
  Pie,
  Tooltip,
  Cell,
  ResponsiveContainer
} from "recharts"

function Charts({ summary }) {

  if (!summary) {
    return null
  }

  const data = [
    { name: "Normal", value: summary.normal ?? 0 },
    { name: "Attack", value: summary.attack ?? 0 },
    { name: "Uncertain", value: summary.uncertain ?? 0 }
  ]

  return (
    <div className="panel chart-panel">
      <h2 className="panel-title">Prediction Distribution</h2>
      <p className="chart-meta">
        Attack traffic represents {summary.attack_percentage}% of analyzed records.
        {typeof summary.uncertain_percentage === "number" && summary.uncertain_percentage > 0 && (
          <> Borderline rows: {summary.uncertain_percentage}%.</>
        )}
      </p>

      <ResponsiveContainer width="100%" height={320}>
        <PieChart>

          <Pie
            data={data}
            dataKey="value"
            outerRadius={120}
            label
          >
            <Cell fill="#3fd3c6" />
            <Cell fill="#ff9f45" />
            <Cell fill="#ffc857" />
          </Pie>

          <Tooltip />

        </PieChart>
      </ResponsiveContainer>

      <div className="legend-row">
        <span><span className="legend-dot" style={{ background: "#3fd3c6" }}></span>Normal</span>
        <span><span className="legend-dot" style={{ background: "#ff9f45" }}></span>Attack</span>
        <span><span className="legend-dot" style={{ background: "#ffc857" }}></span>Uncertain</span>
      </div>

    </div>
  )
}

export default Charts
