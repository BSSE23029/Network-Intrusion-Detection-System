function StatsCards({ summary }) {

  if (!summary) {
    return null
  }

  const uncertain = summary.uncertain ?? 0
  const uncertainPct = summary.uncertain_percentage ?? 0
  const normalPct = Math.max(
    0,
    100 - (summary.attack_percentage ?? 0) - uncertainPct
  )

  const cards = [
    {
      label: "Total Records",
      value: summary.total?.toLocaleString?.() ?? summary.total,
      footnote: "Rows analyzed"
    },
    {
      label: "Attacks Detected",
      value: summary.attack?.toLocaleString?.() ?? summary.attack,
      footnote: `${summary.attack_percentage}% of traffic`
    },
    {
      label: "Normal Traffic",
      value: summary.normal?.toLocaleString?.() ?? summary.normal,
      footnote: `${normalPct.toFixed(2)}% of traffic`
    },
    {
      label: "Uncertain",
      value: uncertain.toLocaleString?.() ?? uncertain,
      footnote: `${uncertainPct}% borderline rows`
    },
    {
      label: "Attack Rate",
      value: `${summary.attack_percentage}%`,
      footnote: summary.attack_percentage > 30 ? "Elevated threat level" : "Within normal range"
    }
  ]

  return (
    <div className="stats-grid">
      {cards.map((card) => (
        <div key={card.label} className="stat-card">
          <p className="stat-label">{card.label}</p>
          <p className="stat-value">{card.value}</p>
          <p className="stat-footnote">{card.footnote}</p>
        </div>
      ))}
    </div>
  )
}

export default StatsCards
