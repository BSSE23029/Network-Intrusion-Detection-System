
import { Routes, Route, useNavigate } from "react-router-dom"
import { useEffect } from "react"

import LandingPage from "./pages/landingPage"
import Dashboard from "./pages/Dashboard"

function App() {
  const navigate = useNavigate()

  // Run once on initial mount. If the page load was a hard reload, ensure we land on the
  // root landing page. Use an empty deps array to avoid rerunning when navigation changes.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    try {
      const navEntry = performance.getEntriesByType
        ? performance.getEntriesByType("navigation")[0]
        : null
      const navType = navEntry?.type || (performance?.navigation?.type === 1 ? "reload" : "navigate")

      if (navType === "reload" && window.location.pathname !== "/") {
        navigate("/")
      }
    } catch (err) {
      // keep silent if performance API is unavailable
    }
  }, [])

  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/dashboard" element={<Dashboard />} />
    </Routes>
  )
}

export default App
